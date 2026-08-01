#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


def write_line(log: TextIO, message: str) -> None:
    print(message, flush=True)
    print(message, file=log, flush=True)


def copy_output(stream: TextIO, log: TextIO) -> None:
    for line in stream:
        print(line, end="", flush=True)
        print(line, end="", file=log, flush=True)


def terminate_process_tree(process: subprocess.Popen[str], log: TextIO) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        for output in (result.stdout, result.stderr):
            if output:
                write_line(log, output.rstrip())
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def describe_status(return_code: int) -> str:
    if return_code >= 0:
        return f"exit code {return_code}"
    try:
        return f"signal {-return_code} ({signal.Signals(-return_code).name})"
    except ValueError:
        return f"signal {-return_code}"


def run_command(command: list[str], timeout_seconds: float, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log:
        write_line(log, f"command: {subprocess.list2cmdline(command)}")
        write_line(log, f"timeout: {timeout_seconds:g}s")
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
        except OSError as error:
            write_line(log, f"could not start command: {error}")
            return 127

        if process.stdout is None:
            write_line(log, "could not capture command output")
            terminate_process_tree(process, log)
            return 127

        reader = threading.Thread(target=copy_output, args=(process.stdout, log), daemon=True)
        reader.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            write_line(log, f"command timed out after {timeout_seconds:g}s")
            terminate_process_tree(process, log)
            process.wait()
            reader.join(timeout=5)
            process.stdout.close()
            return 124

        reader.join(timeout=5)
        process.stdout.close()
        duration = time.monotonic() - started
        write_line(log, f"command finished after {duration:.3f}s with {describe_status(return_code)}")
        return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--success-on-timeout", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    status = run_command(command, args.timeout_seconds, args.log)
    if status == 124 and args.success_on_timeout:
        return 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
