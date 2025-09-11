#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

GITHUB_API = "https://api.github.com"


def run(cmd: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return result.stdout


def github_request(method: str, url: str, token: str, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/vnd.github+json"
    return requests.request(method, url, headers=headers, **kwargs)


def fetch_issue(owner: str, repo: str, issue_number: int, token: str) -> dict:
    r = github_request("GET", f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}", token)
    r.raise_for_status()
    return r.json()


def create_branch(branch: str) -> None:
    run(["git", "switch", "-c", branch])


def commit_all(message: str) -> None:
    run(["git", "add", "-A"]) 
    run(["git", "commit", "-m", message])


def push_branch(branch: str) -> None:
    run(["git", "push", "-u", "origin", branch])


def open_pr(owner: str, repo: str, title: str, head: str, base: str, body: str, token: str) -> dict:
    r = github_request(
        "POST",
        f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
        token,
        json={"title": title, "head": head, "base": base, "body": body},
    )
    r.raise_for_status()
    return r.json()


def suggest_fix_with_llm(issue: dict) -> dict:
    # Minimal stub: propose a small change (e.g., updating a README) to demonstrate flow.
    title = issue.get("title", "")
    body = issue.get("body", "")
    # Extract a filename hint if present in issue body
    m = re.search(r"`([^`]+\.(py|md))`", body)
    target = m.group(1) if m else "README.md"
    content = f"AI remediation placeholder for: {title}.\n\nDetails:\n{body[:500]}\n"
    return {"file": target, "content": content}


def apply_change(change: dict) -> None:
    path = Path(change["file"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(change["content"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue", required=True, type=int)
    parser.add_argument("--base", default=os.environ.get("GITHUB_BASE_REF", "main"))
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    issue = fetch_issue(args.owner, args.repo, args.issue, token)

    branch = f"ai-fix/issue-{args.issue}"
    create_branch(branch)

    change = suggest_fix_with_llm(issue)
    apply_change(change)

    commit_all(f"[AI FIX] {issue.get('title','Issue')} (#{args.issue})")
    push_branch(branch)

    pr = open_pr(
        args.owner,
        args.repo,
        title=f"[AI FIX] Attempt to resolve #{args.issue}",
        head=branch,
        base=args.base,
        body=f"Automated AI remediation attempt for #{args.issue}. Please review.",
        token=token,
    )

    print(json.dumps({"pull_request_url": pr.get("html_url")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())