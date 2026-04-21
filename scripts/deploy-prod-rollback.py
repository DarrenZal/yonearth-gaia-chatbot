#!/usr/bin/env python3
"""deploy-prod-rollback.py — list and restore from deploy-prod.sh snapshots.

Usage:
    scripts/deploy-prod-rollback.py list [--host user@host] [-n 20]
    scripts/deploy-prod-rollback.py show <snapshot-name> [--host user@host]
    scripts/deploy-prod-rollback.py restore <snapshot-name> [--host user@host] [--yes]

The snapshot-name is the directory under /root/snapshots/ that was printed by
deploy-prod.sh (e.g. ``20260421-110928-deploy``).

Examples:
    scripts/deploy-prod-rollback.py list
    scripts/deploy-prod-rollback.py show 20260421-110928-deploy
    scripts/deploy-prod-rollback.py restore 20260421-110928-deploy
"""

import argparse
import json
import subprocess
import sys

DEFAULT_HOST = "claudeuser@152.53.194.214"
LOG_PATH = "/root/deploy-log/deploys.jsonl"
SNAP_ROOT = "/root/snapshots"


def ssh(host: str, cmd: str, *, check: bool = True, capture: bool = True) -> str:
    """Run a command on the remote host and return stdout."""
    result = subprocess.run(
        ["ssh", host, cmd],
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise SystemExit(f"ssh error ({result.returncode}): {stderr}")
    return result.stdout


def iter_log_entries(host: str):
    """Yield each JSONL entry from /root/deploy-log/deploys.jsonl."""
    raw = ssh(host, f"sudo cat {LOG_PATH} 2>/dev/null", check=False)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed lines rather than bail — log might have a partial write
            continue


def cmd_list(host: str, limit: int) -> None:
    entries = list(iter_log_entries(host))
    if not entries:
        print(f"no deploy log entries on {host}:{LOG_PATH}")
        return
    shown = entries[-limit:]
    print(f"# Most recent {len(shown)} of {len(entries)} deploys on {host}")
    print()
    print(f"{'timestamp (UTC)':<20}  {'branch':<20}  {'snapshot':<34}  target")
    print("-" * 110)
    for e in shown:
        ts = e.get("ts", "")[:19]
        branch = (e.get("git_branch") or "-")[:20]
        snap = (e.get("snapshot") or "").split("/")[-1]
        target = e.get("target") or ""
        msg = (e.get("message") or "").strip()
        print(f"{ts:<20}  {branch:<20}  {snap:<34}  {target}")
        if msg:
            print(f"    └ {msg}")


def find_entry(host: str, snap_name: str):
    matches = [e for e in iter_log_entries(host)
               if (e.get("snapshot") or "").rstrip("/").endswith(snap_name)]
    if not matches:
        raise SystemExit(f"no log entry references snapshot {snap_name!r}")
    return matches[-1]  # last-write-wins if duplicate


def cmd_show(host: str, snap_name: str) -> None:
    entry = find_entry(host, snap_name)
    print(json.dumps(entry, indent=2))
    print()
    print(f"# {entry['snapshot']}/ contents:")
    sys.stdout.write(ssh(host, f"sudo ls -la {entry['snapshot']}/"))


def cmd_restore(host: str, snap_name: str, assume_yes: bool) -> None:
    entry = find_entry(host, snap_name)
    target_full = entry["target"]
    if ":" in target_full:
        _, target_path = target_full.split(":", 1)
    else:
        target_path = target_full

    print(f"# Restoring {entry['snapshot']} → {target_path}")
    print(f"  deploy ts      : {entry.get('ts')}")
    print(f"  deploy message : {entry.get('message') or '(none)'}")
    print(f"  git commit     : {entry.get('git_commit')}")
    print()
    print("# Files in snapshot:")
    sys.stdout.write(ssh(host, f"sudo ls -la {entry['snapshot']}/"))
    print()

    if not assume_yes:
        answer = input(f"Restore these files to {target_path}? [y/N] ").strip().lower()
        if answer != "y":
            print("aborted.")
            return

    # Take a pre-rollback snapshot of the current state so rollback itself is reversible.
    pre_snap = f"{SNAP_ROOT}/{entry['snapshot'].split('/')[-1]}-prerollback"
    ssh(
        host,
        f"sudo mkdir -p {pre_snap} && "
        f"cd {target_path} && "
        f"for f in $(sudo ls {entry['snapshot']}/); do "
        f"  if sudo test -e \"$f\"; then sudo cp -r \"$f\" {pre_snap}/; fi; "
        f"done",
    )
    print(f"→ current state saved to {pre_snap} (rollback-of-rollback)")

    ssh(host, f"sudo cp -r {entry['snapshot']}/. {target_path}/", capture=False)
    print(f"✓ restored {snap_name} → {target_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"default: {DEFAULT_HOST}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="show recent deploys")
    sp.add_argument("-n", "--limit", type=int, default=20, help="how many to show (default 20)")

    sp = sub.add_parser("show", help="show details of one snapshot")
    sp.add_argument("snap_name")

    sp = sub.add_parser("restore", help="restore a snapshot to its original target")
    sp.add_argument("snap_name")
    sp.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")

    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list(args.host, args.limit)
    elif args.cmd == "show":
        cmd_show(args.host, args.snap_name)
    elif args.cmd == "restore":
        cmd_restore(args.host, args.snap_name, args.yes)


if __name__ == "__main__":
    main()
