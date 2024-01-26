import matrix_msg
import scratch2
import time

while True:
    start_time = time.time()
    matrix_msg.main('adamranson','Matrix','Server queue notifications')
    # scratch2.main('adamranson','T1')
    exe_time = time.time() - start_time
    print(f"{exe_time} secs")