import time
import subprocess
import sys
import os
import argparse

def get_python_cmd_list(env_type, env_name, script_path):
    if env_type == "conda":
        return ["conda", "run", "--no-capture-output", "-n", env_name, "python", script_path]
    else:
        if os.name == 'nt':
            python_exec = os.path.join(env_name, "Scripts", "python.exe")
        else:
            python_exec = os.path.join(env_name, "bin", "python")
        return [python_exec, script_path]

def main():
    parser = argparse.ArgumentParser(description="Run Final Tests")
    parser.add_argument("--env-type", choices=["conda", "venv"], default="conda", help="Environment manager type")
    parser.add_argument("--legalsum-env", default="legalsum", help="Name or path for the LegalSum environment")
    args = parser.parse_args()

    print("Starting 30-minute cooldown to let the GPU rest...")
    time.sleep(1800)
    
    print("Running evaluate_dpo_models.py...")
    subprocess.run(
        get_python_cmd_list(args.env_type, args.legalsum_env, "src/common/evaluate_dpo_models.py"), 
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )
    
    print("Running run_chunking_on_all.py...")
    subprocess.run(
        get_python_cmd_list(args.env_type, args.legalsum_env, "src/dynamic_chunking/run_chunking_on_all.py"), 
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )

if __name__ == "__main__":
    main()
