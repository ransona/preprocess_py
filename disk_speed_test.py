import subprocess
import os
import matplotlib.pyplot as plt

# Directories to test
test_directories = {
    "/data/fast": [],
    "/data/common": []
}

iterations = 5
block_size = "1G"
testfile_name = "testfile"

def run_dd(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "copied" in line and "s," in line:
                return extract_speed(line)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def extract_speed(line):
    try:
        parts = line.strip().split(",")
        speed = parts[-1].strip()  # e.g., '850 MB/s'
        return speed
    except:
        return None

def parse_speed(speed_str):
    if speed_str is None:
        return 0
    value, unit = speed_str.split()
    value = float(value)
    if unit == "kB/s":
        return value / 1024
    elif unit == "GB/s":
        return value * 1024
    return value  # assume MB/s

def run_tests():
    results = {dir: {"write": [], "read": []} for dir in test_directories}

    for i in range(iterations):
        print(f"\n--- Iteration {i + 1} ---")
        for dir in test_directories:
            testfile = os.path.join(dir, testfile_name)

            print(f"Write test for {dir}")
            write_speed = run_dd(f"dd if=/dev/zero of={testfile} bs={block_size} count=1 oflag=direct")
            results[dir]["write"].append(parse_speed(write_speed))

            print(f"Read test for {dir}")
            read_speed = run_dd(f"dd if={testfile} of=/dev/null bs={block_size} iflag=direct")
            results[dir]["read"].append(parse_speed(read_speed))

            print("Cleaning up...")
            try:
                os.remove(testfile)
            except Exception as e:
                print(f"Failed to delete test file: {e}")
    return results

def plot_results(results):
    x = list(range(1, iterations + 1))

    # Find common Y-axis limit
    all_speeds = []
    for dir in results:
        all_speeds += results[dir]["write"]
        all_speeds += results[dir]["read"]
    y_max = max(all_speeds) * 1.1 if all_speeds else 100  # 10% headroom

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Write plot
    for dir in results:
        ax1.plot(x, results[dir]["write"], label=dir)
    ax1.set_title("Write Speed (MB/s)")
    ax1.set_ylabel("Speed (MB/s)")
    ax1.set_ylim(0, y_max)
    ax1.grid(True)
    ax1.legend()

    # Read plot
    for dir in results:
        ax2.plot(x, results[dir]["read"], label=dir)
    ax2.set_title("Read Speed (MB/s)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Speed (MB/s)")
    ax2.set_ylim(0, y_max)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_tests()
    plot_results(results)
