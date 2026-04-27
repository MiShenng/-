import os
from pathlib import Path
import subprocess
import sys

def run_phase(script_path):
    print(f"\n{'='*50}\nRunning {script_path}...\n{'='*50}")
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
    if result.returncode != 0:
        print(f"Error executing {script_path}")
        sys.exit(1)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    scripts_dir = base_dir / 'scripts'
    
    phases = [
        os.path.join(scripts_dir, 'phase1_ingest', 'ingest.py'),
        os.path.join(scripts_dir, 'phase2_preprocess', 'preprocess.py'),
        os.path.join(scripts_dir, 'phase3_core', 'evaluator.py'),
        os.path.join(scripts_dir, 'phase4_heuristic', 'heuristic.py'),
        os.path.join(scripts_dir, 'phase5_postopt', 'lp_solve.py')
    ]
    
    for phase in phases:
        run_phase(phase)
        
    print("\nAll phases executed successfully. Check the 'results' directory for outputs.")
