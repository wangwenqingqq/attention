"""Bounded, fail-closed launch sequence; stop only task-owned subprocesses."""
import csv
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GPU = os.environ["CUDA_VISIBLE_DEVICES"]
PYTHON = sys.executable
STATE = ROOT / "runs" / "campaign_status.json"
active = None


def record(stage, **extra):
    row = {"stage": stage, "time_unix": time.time(), "pid": os.getpid(), **extra}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2))
    temporary.replace(STATE)
    with (ROOT / "runs" / "campaign_events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def gpu_processes():
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"
    ], text=True)
    return [row for row in csv.reader(output.splitlines()) if row and row[0].strip() == GPU]


def terminate_child():
    if active is not None and active.poll() is None:
        # Dedicated child session: never kill by name or kill another GPU process.
        try:
            os.killpg(active.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            active.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait()


def stop(signum, frame):
    terminate_child()
    record("stopped", signal=signum)
    raise SystemExit(128 + signum)


def run(stage, command, log, cwd=ROOT, cap_seconds=3600):
    global active
    others = gpu_processes()
    if others:
        raise RuntimeError(f"GPU isolation failed before {stage}: {others}")
    record(stage, command=command, log=str(log))
    with log.open("x") as stream:
        active = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                  env=os.environ.copy(), start_new_session=True)
        started = time.monotonic()
        while active.poll() is None:
            if time.monotonic() - started > cap_seconds:
                stop(signal.SIGTERM, None)
            time.sleep(15)
            unexpected = [row for row in gpu_processes() if int(row[1]) != active.pid]
            if unexpected:
                terminate_child()
                raise RuntimeError(f"GPU contention during {stage}; observation retained: {unexpected}")
        if active.returncode:
            raise RuntimeError(f"{stage} failed with exit {active.returncode}; see {log}")
    active = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
        raise ValueError("Tag must be a simple nonempty directory name")
    logs = ROOT / "logs" / args.tag
    outputs = ROOT / "runs" / args.tag
    if args.tag:
        logs.mkdir(exist_ok=False)
        outputs.mkdir(exist_ok=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    (ROOT / "campaign.pid").write_text(str(os.getpid()) + "\n")
    record("preflight", gpu=GPU, tag=args.tag, lr=args.lr, steps=args.steps,
           train_examples=args.train_examples, dropout=args.dropout,
           checkpoint="new task directory; original inputs immutable")
    environment = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,driver_version,pstate,power.draw,clocks.sm,memory.used",
        "--format=csv"
    ], text=True)
    (logs / "gpu_preflight.csv").write_text(environment)
    run("phase0", [PYTHON, "-m", "unittest", "-v", "test_reference", "test_batched"],
        logs / "phase0.log", cap_seconds=600)
    run("official_basic", [PYTHON, "-m", "zoology.launch", "zoology/experiments/basic_examples/basic.py"],
        logs / "official_basic.log", ROOT / "upstream" / "zoology", cap_seconds=1800)
    raw = (logs / "official_basic.log").read_text()
    matches = re.findall(r"valid/accuracy[\s'=:\"]+([0-9.]+)", raw)
    accuracies = [float(x) for x in matches]
    if not accuracies or max(accuracies) <= 0.99:
        raise RuntimeError("Official basic accuracy admission not met; halt before long-context training")
    record("official_basic_admitted", max_accuracy=max(accuracies))
    runs = {}
    for group in "FMBCD":
        output = outputs / f"{group}_seed123"
        run(f"mqar_{group}", [PYTHON, "train_mqar.py", "--group", group, "--output", str(output),
                             "--steps", str(args.steps), "--lr", str(args.lr),
                             "--train-examples", str(args.train_examples), "--dropout", str(args.dropout)],
            logs / f"mqar_{group}.log", cap_seconds=3600)
        result = json.loads((output / "result.json").read_text())
        runs[group] = result
        if group == "F" and result["best_validation"]["query_accuracy"] < 0.90:
            record("halted_full_baseline_quality", result=result,
                   pending="diagnose learning recipe; sparse groups were not started")
            return
    initials = {row["initial_state_sha256"] for row in runs.values()}
    if len(initials) != 1:
        raise RuntimeError("Initialization identity gate failed")
    record("screen_completed", results=runs, pending=["three seeds", "exact/greedy oracle",
           "real-corpus LM", "optimized system kernels and external comparators"],
           allowed_claim="one-seed synthetic screening; no publication or GPU speed claim")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        terminate_child()
        record("failed", error=repr(error))
        raise
