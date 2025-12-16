import subprocess
import sys
import os
from pathlib import Path

#!/usr/bin/env python3

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent.parent
    venv_dir = script_dir / "venv"
    requirements_file = script_dir / "requirements.txt"
    
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    
    # Get the pip executable path
    pip_executable = venv_dir / "bin" / "pip"
    
    print("Upgrading pip...")
    subprocess.check_call([str(pip_executable), "install", "--upgrade", "pip"])
    
    if requirements_file.exists():
        print(f"Installing requirements from {requirements_file}...")
        subprocess.check_call([str(pip_executable), "install", "-r", str(requirements_file)])
        print("Installation completed successfully!")
    else:
        print(f"Warning: {requirements_file} not found")
    
    print(f"\nTo activate the virtual environment, run:")
    print(f"source {venv_dir}/bin/activate")

if __name__ == "__main__":
    main()