import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import csv
import datetime
import statistics


# === Parameters ===
max_gb = 32         # maximum total memory to test (GB)
dtype = np.float32   # 4 bytes per value
hold_seconds = 2     # seconds to keep allocation before freeing
enable_plotting = False  # toggle live plotting (per-run GB sweep) on/off


def memory_fill_test_with_history():
    # === Setup ===
    bytes_per_value = np.dtype(dtype).itemsize
    proc = psutil.Process(os.getpid())

    sizes = []        # GB
    times = []        # seconds per GB
    mem_usages = []   # GB
    swap_usages = []  # GB

    # Per-run live plot (GB vs s/GB), if enabled
    if enable_plotting:
        plt.ion()
        fig_live, ax_live = plt.subplots(figsize=(8, 5))
        line_live, = ax_live.plot([], [], "o-", label="fill time per GB")
        ax_live.set_xlabel("Total allocation size (GB)")
        ax_live.set_ylabel("Seconds per GB (fill speed)")
        ax_live.set_title("Memory fill performance (exponential scaling)")
        ax_live.grid(True)
        ax_live.legend()
        plt.show(block=False)
    else:
        fig_live = None

    print(f"\n--- Exponential memory fill test up to {max_gb} GB ---\n")

    gb = 1
    while gb <= max_gb:
        print(f"\nAllocating {gb} GB...")
        values_needed = int((gb * (1024**3)) / bytes_per_value)
        try:
            arr = np.empty(values_needed, dtype=dtype)

            start = time.time()
            arr.fill(1.0)
            elapsed = time.time() - start
            per_gb = elapsed / gb  # seconds per GB

            mem = proc.memory_info().rss / 1024**3
            swap = psutil.swap_memory().used / 1024**3

            sizes.append(gb)
            times.append(per_gb)
            mem_usages.append(mem)
            swap_usages.append(swap)

            print(f"Filled {gb} GB in {elapsed:.2f}s → {per_gb:.3f} s/GB")
            print(f"Process memory usage: {mem:.2f} GB")
            print(f"System swap usage:   {swap:.2f} GB")

            # Print all results so far
            print("\nProgress so far:")
            print(f"{'GB':>6} | {'s/GB':>8} | {'Proc Mem (GB)':>14} | {'Swap Used (GB)':>14}")
            print("-" * 60)
            for g, t, m, s in zip(sizes, times, mem_usages, swap_usages):
                print(f"{g:6.0f} | {t:8.3f} | {m:14.2f} | {s:14.2f}")

            # Update live plot if enabled
            if enable_plotting:
                line_live.set_data(sizes, times)
                ax_live = fig_live.axes[0]
                ax_live.set_xlim(0, max(sizes) * 1.1)
                ymin, ymax = min(times), max(times)
                margin = (ymax - ymin) * 0.2 if ymax > ymin else 0.1
                ax_live.set_ylim(max(0, ymin - margin), ymax + margin)
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()

            time.sleep(hold_seconds)

        except MemoryError:
            print(f"❌ Memory allocation failed at {gb} GB")
            break
        finally:
            # ensure freeing
            try:
                del arr
            except NameError:
                pass
            time.sleep(1)

        gb *= 2  # exponentially increase

    print("\n--- Test complete ---")
    if enable_plotting:
        plt.ioff()
        plt.show()

    # === Aggregate metric for this run and log history ===
    if not sizes:
        print("No successful allocations; skipping history logging.")
        return

    # times = seconds per GB; convert to speeds in GB/s (higher is better)
    speeds_gb_per_s = [1.0 / t for t in times]
    median_speed = statistics.median(speeds_gb_per_s)
    max_gb_reached = max(sizes)
    min_s_per_gb = min(times)
    max_s_per_gb = max(times)

    # Folder next to this .py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "memory_benchmarks")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "memory_benchmark_log.csv")
    plot_path = os.path.join(log_dir, "memory_benchmark_history.png")

    # Load existing history
    history = []
    if os.path.exists(log_path):
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = datetime.datetime.fromisoformat(row["timestamp"])
                history.append(
                    {
                        "timestamp": ts,
                        "median_gb_per_s": float(row["median_gb_per_s"]),
                        "max_gb": float(row["max_gb"]),
                        "min_s_per_gb": float(row["min_s_per_gb"]),
                        "max_s_per_gb": float(row["max_s_per_gb"]),
                    }
                )

    # Append this run
    now = datetime.datetime.now()
    new_row = {
        "timestamp": now.isoformat(timespec="seconds"),
        "median_gb_per_s": f"{median_speed:.6f}",
        "max_gb": f"{max_gb_reached:.2f}",
        "min_s_per_gb": f"{min_s_per_gb:.6f}",
        "max_s_per_gb": f"{max_s_per_gb:.6f}",
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        fieldnames = ["timestamp", "median_gb_per_s", "max_gb",
                      "min_s_per_gb", "max_s_per_gb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_row)

    # Update history list with current run
    history.append(
        {
            "timestamp": now,
            "median_gb_per_s": median_speed,
            "max_gb": max_gb_reached,
            "min_s_per_gb": min_s_per_gb,
            "max_s_per_gb": max_s_per_gb,
        }
    )
    history.sort(key=lambda r: r["timestamp"])

    dates = [h["timestamp"] for h in history]
    med_speeds = [h["median_gb_per_s"] for h in history]

    current_value = median_speed
    if len(med_speeds) > 1:
        median_previous = statistics.median(med_speeds[:-1])
    else:
        median_previous = current_value

    print(f"\nRun summary:")
    print(f"  Max GB reached:          {max_gb_reached:.2f} GB")
    print(f"  Median fill speed:       {median_speed:.3f} GB/s")
    print(f"  Fastest s/GB this run:   {min_s_per_gb:.3f} s/GB")
    print(f"  Slowest s/GB this run:   {max_s_per_gb:.3f} s/GB")
    print(f"  Log written to:          {log_path}")

    # === History plot (blocking) ===
    fig_hist, ax_hist = plt.subplots(figsize=(9, 5))

    ax_hist.plot(dates, med_speeds, "o-", label="Median fill speed (GB/s)")

    # Current value: blue horizontal line
    ax_hist.axhline(
        current_value,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Current: {current_value:.3f} GB/s",
    )

    # Median of previous values: red horizontal line
    ax_hist.axhline(
        median_previous,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Median (previous): {median_previous:.3f} GB/s",
    )

    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Median fill speed (GB/s)")
    ax_hist.set_title("Memory fill benchmark history")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend()

    fig_hist.autofmt_xdate()
    fig_hist.tight_layout()

    fig_hist.savefig(plot_path, dpi=150)
    print(f"History plot saved to:     {plot_path}")

    plt.show()
    plt.close(fig_hist)


if __name__ == "__main__":
    memory_fill_test_with_history()
