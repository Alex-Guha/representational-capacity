"""
Lightweight dependency check used by scripts that consume artifacts produced
by other scripts in this directory. Keeps the failure mode explicit: a script
that needs `embd_ortho_stats.txt` or `optimized_packing.json` will tell the
reviewer exactly which upstream script to run, rather than crashing inside a
file-open call.
"""

import os
import sys


def require_paths(prereqs):
    """
    prereqs: iterable of (path, producer_script) tuples. Each path may be a
    file or a directory. If any are missing, print a clear error listing the
    upstream script(s) to run, then exit.
    """
    missing = [(p, s) for p, s in prereqs if not os.path.exists(p)]
    if not missing:
        return
    print("ERROR: missing prerequisite artifact(s):", file=sys.stderr)
    for path, producer in missing:
        print(f"  - {path}  (produced by: {producer})", file=sys.stderr)
    print("\nRun the listed script(s) first, then re-run this one.", file=sys.stderr)
    sys.exit(1)
