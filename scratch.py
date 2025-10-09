import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import gc

# === Parameters ===
max_gb = 128         # maximum total memory to test (GB)
dtype = np.float32   # 4 bytes per value
hold_seconds = 2     # seconds to keep allocation before freeing
enable_plotting = False  # toggle live plotting on/off

# === Setup ===
bytes_per_value = np.dtype(dtype).itemsize
proc = psutil.Process(os.getpid())

sizes = []
times = []
proc_mem_usages = []
sys_mem_used = []
swap_usages = []

if enable_plotting:
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], "o-", label="fill time per GB")
    ax.set_xlabel("Total allocation size (GB)")
    ax.set_ylabel("Seconds per GB (fill speed)")
    ax.set_title("Memory fill performance (exponential scaling)")
    ax.grid(True)
    ax.legend()
    plt.show(block=False)

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
        per_gb = elapsed / gb

        # Collect memory stats
        mem = proc.memory_info().rss / 1024**3
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory().used / 1024**3

        proc_mem_usages.append(mem)
        sys_mem_used.append(vm.used / 1024**3)
        swap_usages.append(swap)

        sizes.append(gb)
        times.append(per_gb)

        print(f"Filled {gb} GB in {elapsed:.2f}s → {per_gb:.3f} s/GB")
        print(f"Process memory usage: {mem:.2f} GB")
        print(f"System memory used:   {vm.used / 1024**3:.2f} / {vm.total / 1024**3:.0f} GB ({vm.percent:.1f}%)")
        print(f"System swap used:     {swap:.2f} GB")

        # Print table
        print("\nProgress so far:")
        print(f"{'GB':>6} | {'s/GB':>8} | {'Proc Mem (GB)':>14} | {'Sys Used (GB)':>14} | {'Swap Used (GB)':>14}")
        print("-" * 80)
        for g, t, pm, sm, sw in zip(sizes, times, proc_mem_usages, sys_mem_used, swap_usages):
            print(f"{g:6.0f} | {t:8.3f} | {pm:14.2f} | {sm:14.2f} | {sw:14.2f}")

        # Update live plot if enabled
        if enable_plotting:
            line.set_data(sizes, times)
            ax.set_xlim(0, max(sizes) * 1.1)
            ymin, ymax = min(times), max(times)
            margin = (ymax - ymin) * 0.2 if ymax > ymin else 0.1
            ax.set_ylim(max(0, ymin - margin), ymax + margin)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        time.sleep(hold_seconds)

    except MemoryError:
        print(f"❌ Memory allocation failed at {gb} GB")
        break
    finally:
        del arr
        gc.collect()
        time.sleep(1)

    gb *= 2  # exponentially increase

print("\n--- Test complete ---")
if enable_plotting:
    plt.ioff()
    plt.show()
