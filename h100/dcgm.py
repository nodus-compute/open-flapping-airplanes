"""DCGM 4.x discovery, profiling probe, and timestamped sampler."""

import argparse
import json
import math
import os
import pathlib
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

DCGM_IMAGE = (
    "nvcr.io/nvidia/k8s/dcgm-exporter@sha256:"
    "17e9d49093f2d86d90260eec3c07b18b3f78fc5eadef3c8ab26a24fa5a9c1b2c"
)
DCGM_FIELDS = ("1002", "1004", "1005")
DCGM_INTERVAL_MS = 100
DCGM_PROBE_ROWS = 50
DCGM_PROBE_TIMEOUT_S = 20
STOP = False


def _run(command, timeout=60, check=False):
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=check,
    )


def _write_json(path, value):
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _detect_host_engine():
    engines = _run(["pgrep", "-a", "nv-hostengine"], timeout=10).stdout.splitlines()
    systemctl = shutil.which("systemctl")
    if systemctl:
        result = _run(
            [systemctl, "is-active", "nvidia-dcgm"], timeout=10
        )
        systemd_state = result.stdout.strip() or f"exit_{result.returncode}"
    else:
        systemd_state = "systemctl_not_found"
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        port_open = probe.connect_ex(("127.0.0.1", 5555)) == 0
    finally:
        probe.close()
    host_engines = []
    for line in engines:
        pid, _, command = line.partition(" ")
        host_engines.append({"pid": pid, "cmdline": command})
    return {
        "host_engines": host_engines,
        "systemd_nvidia_dcgm": systemd_state,
        "port_5555_open": port_open,
        "pgrep_output": engines,
    }


def _dcgmi_command(state, arguments):
    if state["mode"] == "host_client":
        client_arguments = [
            arguments[0],
            "--host",
            "127.0.0.1:5555",
            *arguments[1:],
        ]
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--entrypoint",
            "dcgmi",
            DCGM_IMAGE,
            *client_arguments,
        ]
    return ["docker", "exec", state["engine_container"], "dcgmi", *arguments]


def _discovery_ids(text):
    candidates = set()
    for line in text.splitlines():
        match = re.match(r"^\s*\|\s*(\d+)\s*\|", line)
        if match:
            candidates.add(int(match.group(1)))
            continue
        match = re.search(r"\bGPU\s+ID\s*[:=]\s*(\d+)\b", line, re.I)
        if match:
            candidates.add(int(match.group(1)))
            continue
        match = re.match(r"^\s*GPU\s+(\d+)\s*:", line, re.I)
        if match:
            candidates.add(int(match.group(1)))
    return sorted(candidates)


def _discovery_uuid(text):
    match = re.search(r"\bGPU-[0-9a-f-]{16,}\b", text, re.I)
    return match.group(0) if match else None


def _parse_row(line):
    text = line.strip()
    fields = [field.rstrip(",") for field in text.split()]
    if not fields or fields[0].startswith("#"):
        return None
    if fields[0].upper() == "GPU":
        if len(fields) < 5:
            return None
        try:
            gpu_id = int(fields[1])
        except ValueError:
            return None
        values = fields[2:]
    else:
        try:
            gpu_id = int(fields[0])
        except ValueError:
            return None
        if len(fields) < 4:
            return None
        values = fields[1:]

    keyed = {}
    for field_id, key in zip(DCGM_FIELDS, ("sm_active", "pipe_tensor_active", "dram_active")):
        match = re.search(
            rf"{field_id}\s*[=:]\s*([-+]?\d+(?:\.\d+)?)",
            text,
        )
        if match:
            keyed[key] = float(match.group(1))
    if keyed:
        if len(keyed) != 3:
            return None
        metrics = keyed
    else:
        if len(values) >= 6 and values[0] == "1002":
            values = [values[1], values[3], values[5]]
        if len(values) < 3:
            return None
        try:
            numbers = [float(value) for value in values[:3]]
        except ValueError:
            return None
        metrics = dict(
            zip(("sm_active", "pipe_tensor_active", "dram_active"), numbers)
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        return None
    return {"dcgm_id": gpu_id, **metrics}


def _dcgm_failure(detection, detail):
    engines = detection["host_engines"] or [{"pid": "none", "cmdline": "none"}]
    engine_text = ", ".join(
        f"pid={row['pid']} cmdline={row['cmdline']}" for row in engines
    )
    return (
        f"error=dcgm_profiling_unavailable: {detail}. Detected host engine(s): "
        f"{engine_text}; systemd nvidia-dcgm state="
        f"{detection['systemd_nvidia_dcgm']}; 127.0.0.1:5555 open="
        f"{detection['port_5555_open']}. The system nvidia-dcgm nv-hostengine "
        "may hold the profiling counters, so a private engine can get zero rows. "
        "Stop the nvidia-dcgm service or run from the host engine. This harness "
        "does not stop the service."
    )


def _container_running(name):
    result = _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _start_monitor(out, state):
    monitor_log = out / "dcgm_monitor.log"
    with monitor_log.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "monitor",
                "--out",
                str(out),
                "--state",
                str(out / "dcgm_state.json"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    state["monitor_pid"] = process.pid
    _write_json(out / "dcgm_state.json", state)
    deadline = time.monotonic() + DCGM_PROBE_TIMEOUT_S + 3
    probe_path = out / "dcgm_probe.json"
    while time.monotonic() < deadline:
        if probe_path.exists():
            probe = json.loads(probe_path.read_text())
            if probe.get("passed"):
                return
            raise RuntimeError(
                _dcgm_failure(
                    state["host_detection"],
                    f"received {probe.get('rows', 0)} numeric rows, "
                    f"required {DCGM_PROBE_ROWS} within "
                    f"{DCGM_PROBE_TIMEOUT_S}s",
                )
            )
        try:
            os.kill(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    raise RuntimeError(
        _dcgm_failure(
            state["host_detection"],
            "DCGM dmon process exited or probe timed out",
        )
    )


def start(args):
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    detection = _detect_host_engine()
    _write_json(out / "dcgm_host_detection.json", detection)
    engine_container = f"ofa-h100-dcgm-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    state = {
        "mode": "host_client" if detection["port_5555_open"] else "private",
        "engine_container": None,
        "host_detection": detection,
        "gpu_index": args.gpu,
        "gpu_uuid": None,
        "dcgm_id": None,
        "monitor_pid": None,
        "dcgm_image": DCGM_IMAGE,
    }
    raw = []
    try:
        pulled = _run(["docker", "pull", DCGM_IMAGE], timeout=900)
        if pulled.returncode:
            raise RuntimeError(f"failed to pull DCGM image: {pulled.stdout.strip()}")
        if state["mode"] == "private":
            state["engine_container"] = engine_container
            result = _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    engine_container,
                    "--gpus",
                    f"device={args.uuid}",
                    "--cap-add",
                    "SYS_ADMIN",
                    "--entrypoint",
                    "nv-hostengine",
                    DCGM_IMAGE,
                    "-n",
                ],
                timeout=60,
            )
            if result.returncode or not _container_running(engine_container):
                raise RuntimeError(
                    "private nv-hostengine container did not start: "
                    f"{result.stdout.strip()}"
                )
            inspect = _run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", engine_container],
                timeout=10,
            )
            if inspect.returncode == 0 and inspect.stdout.strip().isdigit():
                state["owned_gpu_pids"] = [int(inspect.stdout.strip())]
        gpu = _run(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            timeout=15,
        )
        if gpu.returncode:
            raise RuntimeError(f"nvidia-smi UUID query failed: {gpu.stdout.strip()}")
        state["gpu_uuid"] = gpu.stdout.strip().splitlines()[0].strip()
        list_command = _dcgmi_command(state, ["discovery", "-l"])
        raw.append(f"$ {' '.join(list_command)}")
        listing = _run(list_command, timeout=30)
        raw[-1] += f"\n{listing.stdout}"
        candidates = _discovery_ids(listing.stdout)
        matches = []
        for candidate in candidates:
            command = _dcgmi_command(
                state,
                ["discovery", "--gpuid", str(candidate), "-i", "a"],
            )
            raw.append(f"$ {' '.join(command)}")
            result = _run(command, timeout=30)
            raw[-1] += f"\n{result.stdout}"
            discovered_uuid = _discovery_uuid(result.stdout)
            if result.returncode == 0 and discovered_uuid:
                if discovered_uuid.lower() == state["gpu_uuid"].lower():
                    matches.append(candidate)
        (out / "dcgm_discovery.txt").write_text(
            "\n".join(raw), encoding="utf-8"
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"DCGM GPU UUID mapping failed: selected NVIDIA GPU "
                f"{state['gpu_uuid']}, matching DCGM IDs={matches}; "
                f"raw discovery saved to {out / 'dcgm_discovery.txt'}"
            )
        state["dcgm_id"] = matches[0]
        _write_json(out / "dcgm_state.json", state)
        _start_monitor(out, state)
        print(
            f"DCGM profiling probe passed: GPU UUID={state['gpu_uuid']} "
            f"DCGM ID={state['dcgm_id']} mode={state['mode']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        if raw:
            (out / "dcgm_discovery.txt").write_text(
                "\n".join(raw), encoding="utf-8"
            )
        detail = str(exc)
        if "dcgm_profiling_unavailable" not in detail:
            detail = _dcgm_failure(detection, detail)
        _write_json(out / "dcgm_failure.json", {"error": detail})
        print(detail, file=sys.stderr, flush=True)
        stop_state = state.copy()
        stop_state["monitor_pid"] = (
            json.loads((out / "dcgm_state.json").read_text()).get("monitor_pid")
            if (out / "dcgm_state.json").exists()
            else None
        )
        _stop_state(stop_state)
        _write_json(
            out / "summary.json",
            {
                "evidence": "PREREGISTERED",
                "verdict": False,
                "error": detail,
                "errors": [detail],
                "dcgm": detection,
            },
        )
        return 7


def _monitor_command(state):
    args = [
        "dmon",
        "-i",
        str(state["dcgm_id"]),
        "-e",
        ",".join(DCGM_FIELDS),
        "-d",
        str(DCGM_INTERVAL_MS),
    ]
    if state["mode"] == "host_client":
        return [
            "docker",
            "run",
            "--rm",
            "--init",
            "--network",
            "host",
            "--entrypoint",
            "dcgmi",
            DCGM_IMAGE,
            *args[:1],
            "--host",
            "127.0.0.1:5555",
            *args[1:],
        ]
    return ["docker", "exec", state["engine_container"], "dcgmi", *args]


def _handle_signal(_signum, _frame):
    global STOP
    STOP = True


def monitor(args):
    global STOP
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    out = pathlib.Path(args.out)
    state = json.loads(pathlib.Path(args.state).read_text())
    command = _monitor_command(state)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        _write_json(
            out / "dcgm_probe.json",
            {"passed": False, "rows": 0, "error": f"{type(exc).__name__}: {exc}"},
        )
        return 1
    if process.stdout is None:
        _write_json(out / "dcgm_probe.json", {"passed": False, "rows": 0})
        return 1
    raw_path = out / "dcgm_raw.txt"
    probe_raw_path = out / "dcgm_probe_raw.txt"
    parsed_path = out / "dcgm.jsonl"
    rows = 0
    deadline = time.monotonic() + DCGM_PROBE_TIMEOUT_S
    probing = True
    try:
        with (
            raw_path.open("a", encoding="utf-8") as raw,
            probe_raw_path.open("w", encoding="utf-8") as probe_raw,
            parsed_path.open("a", encoding="utf-8") as parsed,
        ):
            while not STOP:
                if probing and time.monotonic() >= deadline:
                    _write_json(
                        out / "dcgm_probe.json",
                        {
                            "passed": False,
                            "rows": rows,
                            "required_rows": DCGM_PROBE_ROWS,
                            "timeout_seconds": DCGM_PROBE_TIMEOUT_S,
                        },
                    )
                    return 1
                readable, _, _ = select.select([process.stdout], [], [], 0.1)
                if not readable:
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                wall = time.time()
                monotonic = time.monotonic()
                raw.write(f"{wall:.6f} {monotonic:.6f} {line}")
                raw.flush()
                if probing:
                    probe_raw.write(f"{wall:.6f} {monotonic:.6f} {line}")
                    probe_raw.flush()
                sample = _parse_row(line)
                if not sample or sample["dcgm_id"] != state["dcgm_id"]:
                    continue
                row = {
                    "wall_time": wall,
                    "monotonic_time": monotonic,
                    **sample,
                }
                parsed.write(json.dumps(row, allow_nan=False) + "\n")
                parsed.flush()
                if probing:
                    rows += 1
                    if rows >= DCGM_PROBE_ROWS:
                        _write_json(
                            out / "dcgm_probe.json",
                            {
                                "passed": True,
                                "rows": rows,
                                "target_dcgm_id": state["dcgm_id"],
                                "gpu_uuid": state["gpu_uuid"],
                                "mode": state["mode"],
                                "host_detection": state["host_detection"],
                            },
                        )
                        probing = False
                        probe_raw.flush()
        if probing:
            _write_json(
                out / "dcgm_probe.json",
                {
                    "passed": False,
                    "rows": rows,
                    "required_rows": DCGM_PROBE_ROWS,
                    "timeout_seconds": DCGM_PROBE_TIMEOUT_S,
                    "process_exit_code": process.poll(),
                },
            )
        return 0 if not probing else 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _stop_state(state):
    pid = state.get("monitor_pid")
    if pid:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    container = state.get("engine_container")
    if container:
        _run(["docker", "rm", "-f", container], timeout=30)


def stop(args):
    state_path = pathlib.Path(args.out) / "dcgm_state.json"
    if not state_path.exists():
        return 0
    try:
        _stop_state(json.loads(state_path.read_text()))
    except Exception as exc:
        print(f"DCGM cleanup warning: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def mark(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    row = {
        "phase": args.phase,
        "event": args.event,
        "wall_time": time.time(),
        "monotonic_time": time.monotonic(),
    }
    if args.exit_code is not None:
        row["exit_code"] = args.exit_code
    if args.reason:
        row["reason"] = args.reason
    with (out / "phase_marks.jsonl").open("a", encoding="utf-8") as marks:
        marks.write(json.dumps(row, allow_nan=False) + "\n")
        marks.flush()
        os.fsync(marks.fileno())
    return 0


def parser():
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--gpu", type=int, required=True)
    start_parser.add_argument("--uuid", required=True)
    start_parser.add_argument("--out", required=True)
    start_parser.set_defaults(function=start)
    monitor_parser = commands.add_parser("monitor")
    monitor_parser.add_argument("--out", required=True)
    monitor_parser.add_argument("--state", required=True)
    monitor_parser.set_defaults(function=monitor)
    stop_parser = commands.add_parser("stop")
    stop_parser.add_argument("--out", required=True)
    stop_parser.set_defaults(function=stop)
    mark_parser = commands.add_parser("mark")
    mark_parser.add_argument("--out", required=True)
    mark_parser.add_argument("--phase", required=True)
    mark_parser.add_argument("--event", choices=("start", "end", "skipped"), required=True)
    mark_parser.add_argument("--exit-code", type=int)
    mark_parser.add_argument("--reason")
    mark_parser.set_defaults(function=mark)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
