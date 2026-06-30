import os
import subprocess
import time
import sys

TOTAL_STEPS = 2650
BATCH_SIZE = 500
COOLDOWN_MINUTES = 30
current_steps = 0 # Starting completely fresh on the full dataset (change if you want to resume from a specific checkpoint)

def log(msg):
    print(msg)

log("=== Starting Automated Batch Orchestrator ===")
log(f"Target Total Steps: {TOTAL_STEPS}")
log(f"Current Steps: {current_steps}")

while current_steps < TOTAL_STEPS:
    next_steps = min(current_steps + BATCH_SIZE, TOTAL_STEPS)
    
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting training batch: {current_steps} to {next_steps} steps ---")
    
    env = os.environ.copy()
    env["MAX_STEPS"] = str(next_steps)
    
    process = subprocess.Popen(
        [sys.executable, "src/training/train_optimized_baseline.py"],
        env=env
    )
    
    process.wait()
    
    # Check if checkpoint exists
    checkpoint_dir = f"led_model_weights_optimized/checkpoint-{next_steps}"
    if not os.path.exists(checkpoint_dir):
        log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {checkpoint_dir} was not created. The script may have crashed unexpectedly!")
        log("Halting orchestrator.")
        sys.exit(1)
        
    log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch complete! Checkpoint {next_steps} finished saved.")
    
    current_steps = next_steps
    
    if current_steps < TOTAL_STEPS:
        log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sleeping for {COOLDOWN_MINUTES} minutes for thermal cooldown...")
        time.sleep(COOLDOWN_MINUTES * 60)

log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === ORCHESTRATOR FULLY COMPLETE! ===")
log("Target total steps achieved.")
