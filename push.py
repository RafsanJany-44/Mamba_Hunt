#!/usr/bin/env python3
"""Commit and push the Mamba_Hunt repository with progress and diagnostics.

Place this file directly inside Mamba_Hunt and run: python push.py
No third-party Python packages are required.
"""

import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
REMOTE = "origin"
DIAGNOSTIC_DIR = REPOSITORY / "push_diagnostics"
REPORT_FILE = DIAGNOSTIC_DIR / "last_push_report.txt"
MAX_GITHUB_FILE_BYTES = 95 * 1024 * 1024


def redact(text):
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", text)
    text = re.sub(r"\bghp_[A-Za-z0-9]+\b", "ghp_***", text)
    text = re.sub(r"\bgithub_pat_[A-Za-z0-9_]+\b", "github_pat_***", text)
    return text


class Report:
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        clean = redact(str(text).rstrip())
        self.lines.append(clean)

    def command(self, args, output, code):
        self.add("$ " + " ".join(args))
        # Git uses NUL separators for machine-readable filename lists. Convert
        # them only in the human-readable report so copied diagnostics do not
        # appear truncated; the command caller still receives the raw output.
        self.add(output.replace("\0", "\n"))
        self.add("Exit code: " + str(code))
        self.add()

    def save(self):
        DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


REPORT = Report()


def heading(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)
    REPORT.add("=" * 78)
    REPORT.add(text)
    REPORT.add("=" * 78)


def run_git(arguments, check=True):
    command = ["git", *arguments]
    process = subprocess.run(
        command,
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    output = process.stdout.rstrip()
    REPORT.command(command, output, process.returncode)
    if check and process.returncode != 0:
        raise RuntimeError(
            "Git command failed:\n"
            + " ".join(command)
            + "\n\n"
            + output
        )
    return process.returncode, output


def progress_bar(percent):
    width = 36
    filled = round(width * percent / 100)
    bar = "█" * filled + "─" * (width - filled)
    print(f"\rPushing [{bar}] {percent:3d}%", end="", flush=True)


def push_with_progress(branch):
    command = [
        "git",
        "push",
        "--progress",
        REMOTE,
        f"HEAD:{branch}",
    ]
    REPORT.add("$ " + " ".join(command))
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "1"

    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=environment,
    )

    complete_output = []
    current = []
    last_percent = -1
    assert process.stdout is not None

    while True:
        character = process.stdout.read(1)
        if character == "" and process.poll() is not None:
            break
        if character not in ("\r", "\n"):
            current.append(character)
            continue

        line = "".join(current).strip()
        current = []
        if not line:
            continue
        complete_output.append(line)
        match = re.search(r"(\d{1,3})%", line)
        if match:
            percent = min(100, int(match.group(1)))
            if percent != last_percent:
                progress_bar(percent)
                last_percent = percent
        else:
            if last_percent >= 0:
                print()
                last_percent = -1
            print(redact(line))

    remaining = "".join(current).strip()
    if remaining:
        complete_output.append(remaining)
        print(redact(remaining))

    code = process.wait()
    if last_percent >= 0:
        print()
    output = "\n".join(complete_output)
    REPORT.add(redact(output))
    REPORT.add("Exit code: " + str(code))
    REPORT.add()
    if code != 0:
        raise RuntimeError("git push failed:\n\n" + output)


def staged_files_too_large(git_root):
    _, output = run_git(
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    )
    paths = [item for item in output.split("\0") if item]
    problems = []
    for relative in paths:
        path = git_root / relative
        if not path.is_file() or path.stat().st_size <= MAX_GITHUB_FILE_BYTES:
            continue
        _, attribute = run_git(["check-attr", "filter", "--", relative], check=False)
        if not attribute.rstrip().endswith(": lfs"):
            problems.append((relative, path.stat().st_size))
    return problems


def running_training_processes():
    try:
        process = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []

    matches = []
    training_names = (
        "train_ubfc_cross_matched",
        "train_pure_cross_matched",
        "train_ubfc.py",
        "train_pure.py",
    )
    for line in process.stdout.splitlines():
        if str(os.getpid()) in line.split(maxsplit=1)[:1]:
            continue
        if any(name in line for name in training_names):
            matches.append(line.strip())
    return matches


def ensure_local_artifacts_are_not_staged(git_root):
    exclude = git_root / ".git" / "info" / "exclude"
    if not exclude.parent.exists():
        return
    current = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    scope = REPOSITORY.relative_to(git_root).as_posix()
    prefix = "/" + scope + "/" if scope != "." else "/"
    # Remove the broad rule written by version 1 of this script. Best and
    # official checkpoints must remain eligible for Git staging.
    legacy_rule = prefix + "results/models/**/*.pth"
    lines = [line for line in current.splitlines() if line != legacy_rule]
    rules = [
        prefix + "push_diagnostics/",
        prefix + "results/models/**/*_Epoch*.pth",
    ]
    for rule in rules:
        if rule not in lines:
            lines.append(rule)
    exclude.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    REPORT.add("Generated: " + datetime.datetime.now().isoformat(timespec="seconds"))
    REPORT.add("Repository: " + str(REPOSITORY))

    try:
        if shutil.which("git") is None:
            raise RuntimeError("Git is not installed or is not available in PATH.")

        os.chdir(REPOSITORY)
        heading("STEP 1/5 — VERIFYING REPOSITORY")
        _, git_root_text = run_git(["rev-parse", "--show-toplevel"])
        git_root = Path(git_root_text.strip()).resolve()
        _, branch = run_git(["branch", "--show-current"])
        branch = branch.strip()
        if not branch:
            raise RuntimeError("The repository is in detached-HEAD state.")

        _, remote_url = run_git(["remote", "get-url", REMOTE])
        print("Branch :", branch)
        print("Remote :", redact(remote_url))
        ensure_local_artifacts_are_not_staged(git_root)

        training = running_training_processes()
        if training:
            raise RuntimeError(
                "Training is still running. Wait until training finishes before "
                "creating a Git commit. Active process(es):\n\n"
                + "\n".join(training)
            )

        heading("STEP 2/5 — STAGING CHANGES")
        # Per-epoch weights are excluded. Released official checkpoints and each
        # validation-selected *_Best.pth checkpoint remain eligible for pushing.
        run_git(
            [
                "add",
                "-A",
                "--",
                ".",
                ":(exclude,glob)results/models/**/*_Epoch*.pth",
            ]
        )
        # Remove per-epoch weights staged by an earlier interrupted script run.
        run_git(
            [
                "restore",
                "--staged",
                "--",
                ":(glob)results/models/**/*_Epoch*.pth",
            ],
            check=False,
        )
        _, status = run_git(["status", "--short"])
        print(status if status else "No uncommitted changes.")

        large_files = staged_files_too_large(git_root)
        if large_files:
            details = [
                f"{name} ({size / 1024 / 1024:.1f} MB)"
                for name, size in large_files
            ]
            raise RuntimeError(
                "These staged files exceed the safe GitHub file limit and are not "
                "tracked with Git LFS:\n\n" + "\n".join(details)
            )

        heading("STEP 3/5 — CREATING COMMIT")
        staged_code, _ = run_git(["diff", "--cached", "--quiet"], check=False)
        if staged_code == 0:
            print("Nothing new to commit.")
            REPORT.add("Nothing new to commit.")
        else:
            message = "Update Mamba_Hunt " + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            run_git(["commit", "-m", message])
            print("Created commit:", message)

        heading("STEP 4/5 — PUSHING TO GITHUB")
        print("There is no Python timeout. Large uploads may continue as long as needed.")
        push_with_progress(branch)

        heading("STEP 5/5 — VERIFYING PUSH")
        run_git(["status", "--short", "--branch"])
        print("Push completed successfully.")
        REPORT.add("RESULT: PUSH COMPLETED SUCCESSFULLY")

    except Exception as error:
        heading("PUSH FAILED")
        message = redact(str(error))
        print(message)
        REPORT.add("RESULT: PUSH FAILED")
        REPORT.add(message)
        REPORT.save()
        print()
        print("Error details saved to:")
        print(REPORT_FILE)
        print("Upload that text file if you need help diagnosing the failure.")
        return 1

    REPORT.save()
    print("Push report saved to:", REPORT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
