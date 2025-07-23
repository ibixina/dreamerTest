import sys
import subprocess

print("Starting DreamerV3 with PolyominoEnv...")

# Forward all command-line args to DreamerV3
subprocess.run(["python", "dreamerv3/main.py"] + sys.argv[1:])
