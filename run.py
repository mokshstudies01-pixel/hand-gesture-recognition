import sys
import subprocess
import os

os.environ['PYTHONPATH'] = os.getcwd()

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [prepare|train|app]")
        print("  prepare [dataset_path] - Process dataset or generate synthetic data")
        print("  train                  - Train the model")
        print("  app                    - Run Streamlit app")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "prepare":
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
        if dataset_path:
            run_cmd(f"python -m src.prepare_data {dataset_path}")
        else:
            run_cmd("python -m src.prepare_data")
    
    elif cmd == "train":
        run_cmd("python -m src.train")
    
    elif cmd == "app":
        run_cmd("streamlit run src/app.py")
    
    else:
        print(f"Unknown command: {cmd}")