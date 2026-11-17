"""
Keeps dataset/results.csv current against upstream.

    python refresh_data.py            # fetch only if the local copy has gone stale
    python refresh_data.py --force    # fetch regardless

`get_model()` calls refresh() before fitting. That is the only moment a refresh
can take effect: the fitted model is cached for the life of the process, so a
cron job rewriting the CSV at 3am would be invisible to a server that has been
up since Tuesday. Startup is where the data is read, so startup is where it is
refreshed.

Not on a daily schedule, because the data is not daily. International football
runs in FIFA windows -- in 2025, March/June/September/October/November carried
~90% of the year's matches while February had 4 and April had 2. A staleness
check costs nothing in the quiet months and self-heals during a tournament,
when the app is being started often anyway.
"""

import argparse
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RESULTS_CSV = Path("dataset/results.csv")

STALE_AFTER_DAYS = 3
TIMEOUT_SECONDS = 30

EXPECTED_HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"
# Upstream sat at ~49,500 rows in Aug 2026 and only grows. A floor well under
# that catches a truncated download or an error page served as 200 without
# tripping on ordinary variation.
MIN_ROWS = 40_000


def is_stale(path=RESULTS_CSV, stale_after=STALE_AFTER_DAYS):
    """True when the file is missing or was last fetched more than `stale_after` days ago.

    Deliberately keyed on file mtime -- when we last fetched -- rather than on
    the newest match date. In a quiet month the data legitimately does not
    advance, and keying on match dates would refetch on every single start.
    """
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > stale_after


def _looks_like_results(text):
    """Guard against saving an error page or a truncated download over good data."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != EXPECTED_HEADER:
        return f"unexpected header: {lines[0][:80] if lines else '(empty)'!r}"
    if len(lines) - 1 < MIN_ROWS:
        return f"only {len(lines) - 1} rows, expected at least {MIN_ROWS}"
    return None


def refresh(path=RESULTS_CSV, force=False, stale_after=STALE_AFTER_DAYS, url=RESULTS_URL):
    """Fetch upstream if needed. Returns True when the file was replaced.

    Never raises and never leaves a partial file: on any failure the existing
    copy is left untouched and the caller carries on with it. Stale data beats
    an app that will not start because GitHub is having a bad morning.
    """
    if not (force or is_stale(path, stale_after)):
        return False

    # certifi's CA bundle, not the system default: a stock python.org install on
    # macOS ships no trust store, so urlopen fails CERTIFICATE_VERIFY_FAILED on
    # a machine where curl works fine. Verification stays on either way.
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS, context=context) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as e:
        print(f"refresh_data: fetch failed ({e}); keeping {path}", file=sys.stderr)
        return False

    problem = _looks_like_results(text)
    if problem:
        print(f"refresh_data: rejected download -- {problem}; keeping {path}", file=sys.stderr)
        return False

    # Write beside the target and move into place, so an interrupted write
    # cannot truncate a good dataset.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(text)
        staged = Path(tmp.name)
    shutil.move(str(staged), str(path))
    print(f"refresh_data: updated {path} ({len(text.splitlines()) - 1:,} matches)")
    return True


def check():
    """Self-check for the validation and staleness rules. No network."""
    good = EXPECTED_HEADER + "\n" + "\n".join("row" for _ in range(MIN_ROWS))
    assert _looks_like_results(good) is None

    assert "unexpected header" in _looks_like_results("<!DOCTYPE html>\n<html>")
    assert "unexpected header" in _looks_like_results("")
    short = EXPECTED_HEADER + "\n" + "\n".join("row" for _ in range(10))
    assert "only 10 rows" in _looks_like_results(short)

    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "nope.csv"
        assert is_stale(missing), "a missing file is always stale"
        fresh = Path(d) / "fresh.csv"
        fresh.write_text("x")
        assert not is_stale(fresh, stale_after=1), "just-written file is not stale"
        assert is_stale(fresh, stale_after=-1), "negative threshold forces staleness"
    print("refresh_data: checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="fetch even if the copy is fresh")
    parser.add_argument("--check", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.check:
        check()
    elif not refresh(force=args.force):
        print(f"refresh_data: {RESULTS_CSV} is current (or upstream unreachable)")
