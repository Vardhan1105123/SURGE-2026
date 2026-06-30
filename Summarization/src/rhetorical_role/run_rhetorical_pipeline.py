import os
import subprocess
import sys
import argparse

def run_command(cmd, step_name):
    print(f"\n{'='*60}")
    print(f"=== {step_name.upper()} ===")
    print(f"{'='*60}")
    print(f"Executing: {cmd}")
    
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    
    if process.returncode != 0:
        print(f"\nError: Pipeline aborted at step: {step_name}")
        sys.exit(1)
    
    print(f"[SUCCESS] Completed step: {step_name}")

def get_python_cmd(env_type, env_name, script_path):
    if env_type == "conda":
        return f"conda run --no-capture-output -n {env_name} python {script_path}"
    else:
        # For venv, we assume env_name is the path to the virtual environment folder
        if os.name == 'nt':
            python_exec = os.path.join(env_name, "Scripts", "python.exe")
        else:
            python_exec = os.path.join(env_name, "bin", "python")
        return f"{python_exec} {script_path}"

def main():
    parser = argparse.ArgumentParser(description="Run the End-to-End Rhetorical Pipeline")
    parser.add_argument("--env-type", choices=["conda", "venv"], default="conda", 
                        help="Environment manager type (default: conda)")
    parser.add_argument("--opennyai-env", default="opennyai_env", 
                        help="Name (conda) or path (venv) for the OpenNYAI environment")
    parser.add_argument("--legalsum-env", default="legalsum", 
                        help="Name (conda) or path (venv) for the LegalSum environment")
    
    args = parser.parse_args()

    print("""END-TO-END RHETORICAL PIPELINE

This script chains together the full rhetorical pipeline:
  1. Rhetorical Tagging (Semantic Parsing via OpenNYAI)
  2. Supervised Fine-Tuning (SFT) of the Rhetorical Model
  3. Direct Preference Optimization (DPO) of the Rhetorical Model

Note: Due to hardware constraints, this unified script was modularized, 
but is provided here as the definitive end-to-end production pipeline.
""")

    # Step 1: Tag the raw dataset using OpenNYAI
    run_command(
        cmd=get_python_cmd(args.env_type, args.opennyai_env, "src/rhetorical_role/run_opennyai_local.py"),
        step_name="Phase 1: Semantic Rhetorical Tagging"
    )

    # Step 2: SFT on the tagged dataset
    run_command(
        cmd=get_python_cmd(args.env_type, args.legalsum_env, "src/rhetorical_role/train_rhetorical_role.py"),
        step_name="Phase 2: Supervised Fine-Tuning (Rhetorical Model)"
    )

    # Step 3: DPO Alignment
    run_command(
        cmd=get_python_cmd(args.env_type, args.legalsum_env, "src/dpo_rhetorical/train_dpo_rhetorical.py"),
        step_name="Phase 3: Direct Preference Optimization (Rhetorical DPO)"
    )

    print("\n[SUCCESS] The final models have finished training and aligning.")

if __name__ == "__main__":
    main()
