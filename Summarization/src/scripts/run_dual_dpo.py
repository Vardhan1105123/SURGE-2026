import os
import subprocess
import time
import sys
import argparse

def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    if process.returncode != 0:
        print("Error: Command failed!")
        sys.exit(1)

def get_python_cmd(env_type, env_name, script_path):
    if env_type == "conda":
        return f"conda run --no-capture-output -n {env_name} python {script_path}"
    else:
        if os.name == 'nt':
            python_exec = os.path.join(env_name, "Scripts", "python.exe")
        else:
            python_exec = os.path.join(env_name, "bin", "python")
        return f"{python_exec} {script_path}"

def main():
    parser = argparse.ArgumentParser(description="Run Dual DPO Pipeline")
    parser.add_argument("--env-type", choices=["conda", "venv"], default="conda", help="Environment manager type")
    parser.add_argument("--legalsum-env", default="legalsum", help="Name or path for the LegalSum environment")
    args = parser.parse_args()

    run_command(get_python_cmd(args.env_type, args.legalsum_env, "src/dpo_baseline/train_dpo_baseline.py"))
    
    print("Optimized DPO finished! Pausing execution for 60 minutes to let the laptop GPU cool down...")
    time.sleep(3600)
    
    run_command(get_python_cmd(args.env_type, args.legalsum_env, "src/dpo_rhetorical/train_dpo_rhetorical.py"))

if __name__ == "__main__":
    main()
