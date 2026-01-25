import subprocess
import os
import sys
import signal
import tomllib
import tomli_w
import time

TARGET_WORD = "Data Collect Done"
COMMAND = ["pixi", "run", "main", "--seed", "42"]
STORING_DIRECTORY = "vision_data/"
CONFIG_FILENAME = "filter_config.toml"

def SET_SEED(seed: str):
    global STORING_DIRECTORY

    COMMAND[4] = seed
    STORING_DIRECTORY += seed+"/"
    if not os.path.exists(STORING_DIRECTORY):
        os.makedirs(STORING_DIRECTORY, exist_ok=True)
        print(f"[Supervisor] Created directory: {STORING_DIRECTORY}")

def run_and_monitor():
    time.sleep(1)
    print(f"[Supervisor] Starting: {' '.join(COMMAND)}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        COMMAND,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        preexec_fn=os.setsid  # Required for killpg to work correctly
    )

    try:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush() 
            
            if TARGET_WORD in line:
                print(f"\n[Supervisor] Match found! Terminating...")
                break # Exit loop to trigger finally block
                
    except KeyboardInterrupt:
        print("\n[Supervisor] Interrupted.")
    finally:
        # Close the pipe first
        process.stdout.close()
        
        # Kill the entire group safely
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass # Already gone
            
        # Give the OS a moment to clean up, then stop waiting
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
            
        print("[Supervisor] Cleanup finished.")
        

def main(seed: int, noisy_x: float = None, noisy_y: float = None):
    global STORING_DIRECTORY

    SET_SEED(str(seed))
    
    with open(CONFIG_FILENAME, "rb") as f:
        data = tomllib.load(f)

    # CLEAN DATA Collection
    data["OUTPUT_FILE"] = STORING_DIRECTORY + "clean_dwa.csv"
    data["OUTPUT_FILE_2"] = STORING_DIRECTORY + "clean_dwa.csv"
    data["CLEAN"] = True
    with open(CONFIG_FILENAME, "wb") as f:
        tomli_w.dump(data, f)
    run_and_monitor()
    print("CLEAN DATA COLLECT DONE")

    #NOISY DATA Collection
    data["OUTPUT_FILE"] = STORING_DIRECTORY + "noisy_dwa.csv"
    data["OUTPUT_FILE_2"] = STORING_DIRECTORY + "noisy_dwa.csv"
    data["CLEAN"] = False
    COMMAND.append("--noisy_x")
    COMMAND.append(str(noisy_x))
    COMMAND.append("--noisy_y")
    COMMAND.append(str(noisy_y))
    with open(CONFIG_FILENAME, "wb") as f:
        tomli_w.dump(data, f)
    run_and_monitor()
    print("NOISY DATA COLLECT DONE")

    #KALMAN FILTER DATA
    data["OUTPUT_FILE"] = STORING_DIRECTORY + "kalman_filtered_dwa.csv"
    data["OUTPUT_FILE_2"] = STORING_DIRECTORY + "kalman_filtered_dwa.csv"
    data["FILTER"] = "KALMAN"
    with open(CONFIG_FILENAME, "wb") as f:
        tomli_w.dump(data, f)
    run_and_monitor()
    print("KALMAN FILTER DATA COLLECT DONE")

    # #FIR FILTER DATA
    # data["OUTPUT_FILE"] = STORING_DIRECTORY + "fir_filtered_dwa.csv"
    # data["OUTPUT_FILE_2"] = STORING_DIRECTORY + "fir_filtered_dwa.csv"
    # data["FILTER"] = "FIR"
    # with open(CONFIG_FILENAME, "wb") as f:
    #     tomli_w.dump(data, f)
    # run_and_monitor()
    # print("FIR FILTER DATA COLLECT DONE")



if __name__ == "__main__":
    main(32, 0.01, 0.01)
    print("Test Ends")