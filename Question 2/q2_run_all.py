import subprocess
import sys

from q2_config import OUT_DIR

SCRIPTS = [
    "q2_1_spmo_umd_beta.py",
    "q2_2_methodology.py",
    "q2_3_long_leg.py",
    "q2_4_ff6_controls.py",
    "q2_5_other_etfs.py",
    "q2_report.py",
]


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("=" * 60)
    print("MFIN 7037 Q2: Running all questions then report")
    print("=" * 60)
    for i, name in enumerate(SCRIPTS, 1):
        path = os.path.join(script_dir, name)
        if not os.path.isfile(path):
            print(f"[Skip] {name} not found")
            continue
        print(f"\n[{i}/{len(SCRIPTS)}] Running {name} ...")
        rc = subprocess.call([sys.executable, path])
        if rc != 0:
            print(f"Warning: {name} exited with code {rc}")
    print("\n" + "=" * 60)
    print("All done. Outputs (including REPORT_Q2.md and REPORT_Q2.pdf) in:")
    print(OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
