#!/usr/bin/env python3
"""Pick vllm-ascend performance PRs that can seed SRV-Bench class-3 (code) tasks.

Standard library only, and specifically no requests/docker import: the GitHub
REST API is reached through urllib.request, and --fixture swaps the transport
for a local JSON file so selection stays testable offline.

Selection pipeline
------------------
1. resolve --since <commit> into a commit timestamp;
2. walk merged (state=closed) pull requests newer than that timestamp;
3. keep the ones whose labels or title mention performance wording;
4. read the changed files of each remaining PR and derive needs_recompile
   (touches csrc/ or cmake/) and py_only;
5. drop needs_recompile candidates: a class-3 task has a 2400 s budget and no
   toolchain to rebuild Ascend operators, so such a PR cannot be retested;
6. --py-only additionally keeps 100%-Python candidates only.

The selection core (select_candidates) is transport-agnostic: the live client
and the fixture client only answer "closed pulls", "commit date" and "changed
files", which is exactly what --self-test exercises on an inline fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

DEFAULT_REPO = "vllm-project/vllm-ascend"
API_ROOT = "https://api.github.com"
USER_AGENT = "srv-bench-pick-perf-prs/1.0"

# Latin keywords match on token boundaries so "perf" cannot fire inside an
# unrelated word; CJK keywords match as substrings because those languages do
# not delimit words.
PERF_KEYWORDS = (
    "performance",
    "perf",
    "fast",
    "faster",
    "fastest",
    "speed",
    "speedup",
    "accelerate",
    "accelerating",
    "acceleration",
    "latency",
    "throughput",
    "optimize",
    "optimise",
    "optimization",
    "optimisation",
    "efficiency",
    "slow",
    "slowdown",
)
PERF_KEYWORDS_CJK = ("性能", "优化", "加速", "提速", "吞吐", "时延", "延迟")
PERF_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(PERF_KEYWORDS) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

RECOMPILE_PREFIXES = ("csrc/", "cmake/")

MAX_SCAN_DEFAULT = 300
MAX_FILE_PAGES = 3
PER_PAGE = 100


class PickError(RuntimeError):
    """Any setup, transport or fixture problem; main() turns it into an exit code."""


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def norm_path(path):
    """Repo-relative path with forward slashes and no leading "./" or "/"."""
    text = str(path).replace(chr(92), "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def perf_keyword_hit(*texts):
    """True when any text mentions performance wording."""
    for text in texts:
        if not text:
            continue
        if PERF_RE.search(text):
            return True
        if any(tok in text for tok in PERF_KEYWORDS_CJK):
            return True
    return False


def needs_recompile(changed_files):
    """A PR touching csrc/ or cmake/ needs an operator rebuild."""
    return any(norm_path(p).startswith(RECOMPILE_PREFIXES) for p in changed_files)


def is_py_only(changed_files):
    """Every changed file is Python (an empty change set is not py_only)."""
    return bool(changed_files) and all(
        norm_path(p).lower().endswith(".py") for p in changed_files
    )


def parse_timestamp(value):
    """Parse a GitHub ISO-8601 timestamp; naive values are taken as UTC."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _label_names(pull):
    names = []
    for label in pull.get("labels") or []:
        name = label.get("name") if isinstance(label, dict) else label
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def select_candidates(pulls, files_for, since_date, *, limit, py_only=False,
                      allow_recompile=False):
    """Filter pulls down to retestable perf candidates.

    files_for(number) -> list of repo-relative paths is injected, so the same
    code runs against the live API and against a fixture.
    Returns (candidates, stats).
    """
    since = parse_timestamp(since_date)
    stats = {
        "closed_pulls": len(pulls),
        "unmerged": 0,
        "before_since": 0,
        "no_perf_keyword": 0,
        "keyword_matched": 0,
        "dropped_needs_recompile": [],
        "dropped_not_py_only": [],
        "truncated_changed_files": [],
    }
    matched = []
    for pull in pulls:
        merged_at = pull.get("merged_at")
        if not merged_at:
            stats["unmerged"] += 1
            continue
        if parse_timestamp(merged_at) < since:
            stats["before_since"] += 1
            continue
        title = pull.get("title") or ""
        if not perf_keyword_hit(title, " ".join(_label_names(pull))):
            stats["no_perf_keyword"] += 1
            continue
        stats["keyword_matched"] += 1
        matched.append(pull)

    candidates = []
    for pull in matched:
        number = int(pull["number"])
        changed = [norm_path(p) for p in files_for(number)]
        if pull.get("files_truncated"):
            stats["truncated_changed_files"].append(number)
        candidate = {
            "number": number,
            "title": pull.get("title") or "",
            "merged_at": pull.get("merged_at"),
            "merge_commit": pull.get("merge_commit_sha"),
            "changed_files": sorted(changed),
            "needs_recompile": needs_recompile(changed),
            "py_only": is_py_only(changed),
            "labels": _label_names(pull),
            "url": pull.get("html_url"),
        }
        if candidate["needs_recompile"] and not allow_recompile:
            stats["dropped_needs_recompile"].append(number)
            continue
        if py_only and not candidate["py_only"]:
            stats["dropped_not_py_only"].append(number)
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda c: (parse_timestamp(c["merged_at"]), c["number"]),
                    reverse=True)
    return (candidates[:limit] if limit is not None else candidates), stats


# --------------------------------------------------------------------------- #
# transports
# --------------------------------------------------------------------------- #
class FixtureClient:
    """Offline transport: a JSON file shaped like the API answers we need."""

    def __init__(self, data, source):
        if not isinstance(data, dict):
            raise PickError("%s: fixture must be a JSON object" % source)
        self.data = data
        self.source = source

    def since_date(self, since):
        commits = self.data.get("commits") or {}
        record = commits.get(since)
        if record is not None:
            return record["commit"]["committer"]["date"]
        if self.data.get("since"):
            return self.data["since"]
        raise PickError(
            "%s: commit %s absent from the fixture; add it under 'commits'"
            % (self.source, since)
        )

    def closed_pulls(self, max_scan):
        pulls = self.data.get("pulls") or []
        if not isinstance(pulls, list):
            raise PickError("%s: 'pulls' must be a list" % self.source)
        return pulls[:max_scan]

    def files_for(self, number):
        entry = (self.data.get("files") or {}).get(str(number))
        if entry is None:
            raise PickError(
                "%s: no files entry for PR #%d; the fixture must carry every "
                "keyword-matched PR" % (self.source, number)
            )
        return [item["filename"] if isinstance(item, dict) else item
                for item in entry]


class LiveClient:
    """GitHub REST API transport (urllib only, one attempt per request)."""

    def __init__(self, repo, token, timeout):
        self.repo = repo
        self.token = token
        self.timeout = timeout

    def _get(self, path, params=None):
        url = API_ROOT + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise PickError("GET %s -> HTTP %s %s" % (url, exc.code, exc.reason))
        except urllib.error.URLError as exc:
            raise PickError("GET %s -> %s" % (url, exc.reason))
        except (ValueError, UnicodeDecodeError) as exc:
            raise PickError("GET %s -> unparsable JSON: %s" % (url, exc))

    def since_date(self, since):
        payload = self._get("/repos/%s/commits/%s" % (self.repo, since))
        return payload["commit"]["committer"]["date"]

    def closed_pulls(self, max_scan):
        pulls = []
        pages = max(1, (max_scan + PER_PAGE - 1) // PER_PAGE)
        for page in range(1, pages + 1):
            chunk = self._get(
                "/repos/%s/pulls" % self.repo,
                {"state": "closed", "sort": "updated", "direction": "desc",
                 "per_page": PER_PAGE, "page": page},
            )
            if not isinstance(chunk, list):
                raise PickError("pull listing page %d is not a list" % page)
            pulls.extend(chunk)
            if len(chunk) < PER_PAGE or len(pulls) >= max_scan:
                break
        return pulls[:max_scan]

    def files_for(self, number):
        names = []
        for page in range(1, MAX_FILE_PAGES + 1):
            chunk = self._get(
                "/repos/%s/pulls/%d/files" % (self.repo, number),
                {"per_page": PER_PAGE, "page": page},
            )
            names.extend(item["filename"] for item in chunk)
            if len(chunk) < PER_PAGE:
                break
        return names


# --------------------------------------------------------------------------- #
# inline fixture used by --self-test
# --------------------------------------------------------------------------- #
SINCE_COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
SELF_TEST_FIXTURE = {
    "repo": DEFAULT_REPO,
    "commits": {
        SINCE_COMMIT: {
            "sha": SINCE_COMMIT,
            "commit": {"committer": {"date": "2025-06-01T00:00:00Z"}},
        }
    },
    "pulls": [
        {   # kept: keyword in the label, pure python
            "number": 101,
            "title": "Optimize block table slot mapping rebuild",
            "state": "closed",
            "merged_at": "2025-06-03T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000001",
            "labels": [{"name": "performance"}],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/101",
        },
        {   # dropped: touches csrc/ -> an operator rebuild, impossible in budget
            "number": 102,
            "title": "Speedup MoE token dispatcher",
            "state": "closed",
            "merged_at": "2025-06-05T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000002",
            "labels": [],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/102",
        },
        {   # dropped at the keyword stage
            "number": 103,
            "title": "Fix docstring typo in worker",
            "state": "closed",
            "merged_at": "2025-06-06T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000003",
            "labels": [{"name": "documentation"}],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/103",
        },
        {   # dropped: cmake/ change -> a rebuild
            "number": 104,
            "title": "Improve scheduler throughput via build flags",
            "state": "closed",
            "merged_at": "2025-06-07T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000004",
            "labels": [{"name": "perf"}],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/104",
        },
        {   # dropped: merged before the baseline commit
            "number": 105,
            "title": "Faster attention mask build",
            "state": "closed",
            "merged_at": "2025-05-20T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000005",
            "labels": [{"name": "performance"}],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/105",
        },
        {   # kept by default, dropped by --py-only (also ships a markdown doc)
            "number": 106,
            "title": "[perf] Reduce attention mask pre-processing",
            "state": "closed",
            "merged_at": "2025-06-08T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000006",
            "labels": [],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/106",
        },
        {   # kept: python-only, keyword only in the title
            "number": 107,
            "title": "\u6027\u80fd\u4f18\u5316: batch token dispatch bookkeeping",
            "state": "closed",
            "merged_at": "2025-06-09T08:00:00Z",
            "merge_commit_sha": "cc00000000000000000000000000000000000007",
            "labels": [],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/107",
        },
        {   # dropped: still open
            "number": 108,
            "title": "Performance: nothing merged yet",
            "state": "open",
            "merged_at": None,
            "merge_commit_sha": None,
            "labels": [{"name": "performance"}],
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/108",
        },
    ],
    "files": {
        "101": [{"filename": "vllm_ascend/worker/block_table.py"}],
        "102": [{"filename": "vllm_ascend/ops/fused_moe/token_dispatcher.py"},
                {"filename": "csrc/mc2/dispatch_ffn_combine/op_host/tiling.cpp"}],
        "103": [{"filename": "vllm_ascend/worker/utils.py"}],
        "104": [{"filename": "cmake/ascend.cmake"},
                {"filename": "vllm_ascend/core/scheduler_dynamic_batch.py"}],
        "105": [{"filename": "vllm_ascend/attention/attention_mask.py"}],
        "106": [{"filename": "vllm_ascend/attention/utils.py"},
                {"filename": "docs/source/perf/attention.md"}],
        "107": [{"filename": "vllm_ascend/ops/fused_moe/token_dispatcher.py"},
                {"filename": "./vllm_ascend/core/profiling_chunk_predictor.py"}],
    },
}


def run_self_test(stream=sys.stdout):
    """Prove the filter logic against the inline fixture; return an exit code."""
    def say(message):
        print(message, file=stream)

    client = FixtureClient(SELF_TEST_FIXTURE, "<inline fixture>")
    since = client.since_date(SINCE_COMMIT)
    failures = []

    def expect(label, got, want):
        if got != want:
            failures.append(label)
            say("FAIL %s: got %r want %r" % (label, got, want))
        else:
            say("ok   %s = %r" % (label, got))

    pulls = client.closed_pulls(MAX_SCAN_DEFAULT)
    candidates, stats = select_candidates(
        pulls, client.files_for, since, limit=None, py_only=False,
    )
    say("[self-test] resolved since date: %s" % since)
    say("[self-test] closed pulls=%d keyword_matched=%d "
        "dropped_needs_recompile=%s"
        % (stats["closed_pulls"], stats["keyword_matched"],
           stats["dropped_needs_recompile"]))

    expect("unmerged", stats["unmerged"], 1)
    expect("before_since", stats["before_since"], 1)
    expect("no_perf_keyword", stats["no_perf_keyword"], 1)
    expect("keyword_matched", stats["keyword_matched"], 5)
    expect("needs_recompile dropped", stats["dropped_needs_recompile"], [102, 104])
    expect("default survivors", [c["number"] for c in candidates], [107, 106, 101])

    by_number = {c["number"]: c for c in candidates}
    expect("101 needs_recompile", by_number[101]["needs_recompile"], False)
    expect("101 py_only", by_number[101]["py_only"], True)
    expect("101 merge_commit", by_number[101]["merge_commit"],
           "cc00000000000000000000000000000000000001")
    expect("101 changed_files", by_number[101]["changed_files"],
           ["vllm_ascend/worker/block_table.py"])
    # a "./"-prefixed fixture path must be normalised before scoring
    expect("107 changed_files", by_number[107]["changed_files"],
           ["vllm_ascend/core/profiling_chunk_predictor.py",
            "vllm_ascend/ops/fused_moe/token_dispatcher.py"])
    expect("106 py_only", by_number[106]["py_only"], False)

    py_candidates, py_stats = select_candidates(
        pulls, client.files_for, since, limit=None, py_only=True,
    )
    expect("--py-only dropped_not_py_only", py_stats["dropped_not_py_only"], [106])
    expect("--py-only survivors", [c["number"] for c in py_candidates], [107, 101])

    limited, _ = select_candidates(pulls, client.files_for, since, limit=1)
    expect("--limit 1", [c["number"] for c in limited], [107])

    kept_all, _ = select_candidates(
        pulls, client.files_for, since, limit=None, allow_recompile=True,
    )
    expect("--allow-recompile survivors", [c["number"] for c in kept_all],
           [107, 106, 104, 102, 101])

    if failures:
        say("[self-test] FAILED: %d assertion(s): %s"
            % (len(failures), ", ".join(failures)))
        return 1
    say("[self-test] OK: all assertions passed; %d candidates after filtering, "
        "%d with --py-only" % (len(candidates), len(py_candidates)))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        prog="pick_perf_prs.py",
        description="Pick vllm-ascend performance PRs merged after a baseline "
                    "commit, dropping candidates that need an operator rebuild.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="owner/name of the GitHub repository (default: %(default)s)")
    parser.add_argument("--since",
                        help="baseline commit SHA that merged PRs must be newer than")
    parser.add_argument("--limit", type=int, default=20,
                        help="maximum candidates to emit (default: %(default)s)")
    parser.add_argument("--fixture", default=None,
                        help="local JSON file used instead of the GitHub API (offline)")
    parser.add_argument("--out", default=None, help="write the JSON report to this path")
    parser.add_argument("--py-only", action="store_true",
                        help="keep only candidates whose every changed file is Python")
    parser.add_argument("--allow-recompile", action="store_true",
                        help="do not drop candidates that touch csrc/ or cmake/")
    parser.add_argument("--token", default=None,
                        help="GitHub token (default: $GITHUB_TOKEN or $GH_TOKEN)")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--max-scan", type=int, default=MAX_SCAN_DEFAULT,
                        help="maximum closed pull requests to scan (default: %(default)s)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the inline-fixture filter self-test and exit")
    return parser


def _make_client(args):
    if args.fixture:
        try:
            with open(args.fixture, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as exc:
            raise PickError("cannot read fixture %s: %s" % (args.fixture, exc))
        except ValueError as exc:
            raise PickError("fixture %s is not valid JSON: %s" % (args.fixture, exc))
        return FixtureClient(data, args.fixture)
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise PickError(
            "no GitHub token (pass --token, or set GITHUB_TOKEN/GH_TOKEN) and no "
            "--fixture given; offline runs need --fixture <path.json>"
        )
    return LiveClient(args.repo, token, args.timeout)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.since:
        sys.stderr.write("pick_perf_prs: --since <commit> is required\n")
        return 2
    try:
        client = _make_client(args)
    except PickError as exc:
        sys.stderr.write("pick_perf_prs: %s\n" % exc)
        return 2
    try:
        since = client.since_date(args.since)
        pulls = client.closed_pulls(args.max_scan)
        candidates, stats = select_candidates(
            pulls, client.files_for, since,
            limit=args.limit, py_only=args.py_only,
            allow_recompile=args.allow_recompile,
        )
    except PickError as exc:
        sys.stderr.write("pick_perf_prs: %s\n" % exc)
        return 3

    report = {
        "repo": args.repo,
        "since_commit": args.since,
        "since_date": since,
        "limit": args.limit,
        "filters": {
            "perf_keywords": list(PERF_KEYWORDS) + list(PERF_KEYWORDS_CJK),
            "drop_needs_recompile": not args.allow_recompile,
            "py_only": bool(args.py_only),
            "recompile_prefixes": list(RECOMPILE_PREFIXES),
        },
        "stats": stats,
        "candidates": candidates,
    }
    mode = "fixture" if args.fixture else "api"
    print("repo=%s since=%s resolved=%s mode=%s"
          % (args.repo, args.since, since, mode))
    print("scanned=%d keyword_matched=%d dropped_needs_recompile=%s "
          "dropped_not_py_only=%s candidates=%d"
          % (stats["closed_pulls"], stats["keyword_matched"],
             stats["dropped_needs_recompile"],
             stats["dropped_not_py_only"], len(candidates)))
    for candidate in candidates:
        print("#%-6d recompile=%-5s py_only=%-5s files=%-3d %s"
              % (candidate["number"], candidate["needs_recompile"],
                 candidate["py_only"], len(candidate["changed_files"]),
                 candidate["title"]))
    if stats["truncated_changed_files"]:
        print("note: file listings truncated for %s"
              % stats["truncated_changed_files"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
