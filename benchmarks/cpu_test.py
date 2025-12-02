import os
import csv
import time
import datetime
import statistics
import math

import numpy as np
import psutil
import matplotlib.pyplot as plt


def cpu_benchmark_with_history(
    N=2048,
    target_duration_s=1.0,   # aim for ~1s of compute
    max_repetitions=50,
    dtype=np.float32,
):

    # === store inside folder next to this .py file ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "cpu_benchmarks")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "cpu_benchmark_log.csv")
    plot_path = os.path.join(log_dir, "cpu_benchmark_history.png")

    # --- Load history ---
    history = []
    if os.path.exists(log_path):
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = datetime.datetime.fromisoformat(row["timestamp"])
                history.append(
                    {
                        "timestamp": ts,
                        "gflops": float(row["gflops"]),
                        "elapsed": float(row["elapsed"]),
                        "N": int(row["N"]),
                        "repetitions": int(row["repetitions"]),
                        "sys_cpu": float(row["sys_cpu"]),
                    }
                )

    # --- Allocate matrices ---
    A = np.random.rand(N, N).astype(dtype)
    B = np.random.rand(N, N).astype(dtype)

    _ = A @ B  # warm-up

    # --- estimate time for one matmul ---
    start_single = time.time()
    _ = A @ B
    elapsed_single = time.time() - start_single
    elapsed_single = max(elapsed_single, 1e-6)

    # choose repetitions
    estimated_reps = math.ceil(target_duration_s / elapsed_single)
    repetitions = max(3, min(max_repetitions, estimated_reps))

    # --- Run benchmark ---
    psutil.cpu_percent(interval=None)
    start = time.time()
    for _ in range(repetitions):
        _ = A @ B
    elapsed = time.time() - start
    sys_cpu = psutil.cpu_percent(interval=0.1)

    flops_total = 2.0 * (N ** 3) * repetitions
    gflops = flops_total / elapsed / 1e9
    now = datetime.datetime.now()

    print(f"\nBenchmark at {now.isoformat(timespec='seconds')}")
    print(f"N={N}, repetitions={repetitions}")
    print(f"Elapsed: {elapsed:.4f}s")
    print(f"Throughput: {gflops:.1f} GFLOP/s")
    print(f"System CPU usage during run: {sys_cpu:.1f}%")

    # --- append to csv ---
    new_row = {
        "timestamp": now.isoformat(timespec="seconds"),
        "gflops": f"{gflops:.6f}",
        "elapsed": f"{elapsed:.6f}",
        "N": str(N),
        "repetitions": str(repetitions),
        "sys_cpu": f"{sys_cpu:.2f}",
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        fieldnames = ["timestamp", "gflops", "elapsed", "N", "repetitions", "sys_cpu"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_row)

    # --- recompute history including new point ---
    history.append(
        {
            "timestamp": now,
            "gflops": gflops,
            "elapsed": elapsed,
            "N": N,
            "repetitions": repetitions,
            "sys_cpu": sys_cpu,
        }
    )

    history.sort(key=lambda r: r["timestamp"])
    dates = [h["timestamp"] for h in history]
    gvals = [h["gflops"] for h in history]

    current_value = gflops
    if len(gvals) > 1:
        median_previous = statistics.median(gvals[:-1])
    else:
        median_previous = current_value

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dates, gvals, "o-", label="Measured GFLOP/s")

    ax.axhline(
        current_value,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Current: {current_value:.1f} GFLOP/s",
    )

    ax.axhline(
        median_previous,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Median (previous): {median_previous:.1f} GFLOP/s",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("CPU benchmark history")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(plot_path, dpi=150)

    print(f"Saved log:  {log_path}")
    print(f"Saved plot: {plot_path}")

    plt.show()   # blocking show
    plt.close(fig)

    return {
        "timestamp": now,
        "gflops": gflops,
        "elapsed": elapsed,
        "N": N,
        "repetitions": repetitions,
        "sys_cpu": sys_cpu,
        "log_path": log_path,
        "plot_path": plot_path,
    }


def main():
    cpu_benchmark_with_history()


if __name__ == "__main__":
    main()
