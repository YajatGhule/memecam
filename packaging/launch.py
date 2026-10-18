"""PyInstaller entry point: memecam(.exe) runs v1, memecamv2(.exe) — the same program copied — runs v2."""
import multiprocessing
import sys
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from memecam.main import main, main_v2
    (main_v2 if Path(sys.argv[0]).stem.lower().endswith("v2") else main)()
