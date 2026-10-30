"""One-time local installation. Run with 64-bit Python 3.11; see INSTALL.md."""
import argparse
import platform
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute",choices=("auto","cpu","mps","cuda124"),default="auto")
    args=parser.parse_args()
    if sys.version_info[:2] != (3,11) or sys.maxsize <= 2**32:
        raise SystemExit("Please run this installer using 64-bit Python 3.11. See INSTALL.md.")
    if platform.system()=="Darwin" and platform.machine()!="arm64":
        raise SystemExit("This pinned PyTorch 2.6 environment targets Apple Silicon, Windows or Linux. Intel Macs need a separately validated environment.")
    if args.compute=="cuda124" and platform.system()=="Darwin":
        raise SystemExit("CUDA is for compatible NVIDIA GPUs on Windows/Linux. Use --compute auto or mps on supported macOS.")
    if args.compute=="mps" and platform.system()!="Darwin":
        raise SystemExit("The MPS PyTorch build is for supported macOS systems. Use cpu or cuda124 on this computer.")
    root=Path(__file__).resolve().parent
    env=root/".venv"
    python=env/("Scripts/python.exe" if platform.system()=="Windows" else "bin/python")
    if not python.exists():
        print("Creating the isolated BAL environment...",flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    def call(*command):
        subprocess.run([str(python),*command],check=True,cwd=root)
    call("-m","pip","install","--upgrade","pip")
    command=["-m","pip","install","torch==2.6.0","torchvision==0.21.0"]
    if platform.system()!="Darwin":
        command += ["--index-url","https://download.pytorch.org/whl/"+("cu124" if args.compute=="cuda124" else "cpu")]
    call(*command)
    call("-m","pip","install","-r",str(root/"requirements.txt"))
    call("-m","pip","check")
    call("-m","ipykernel","install","--user","--name","bal-lab","--display-name","BAL analysis (Python 3.11)")
    if platform.system()=="Darwin":
        print("The standard macOS PyTorch build includes MPS support. Notebook DEVICE=auto selects it when available.")
    print("\nInstallation completed. Start Jupyter with:")
    print(f'"{python}" -m jupyterlab "{root / "BAL_End_to_End_Lab.ipynb"}"')


if __name__=="__main__":
    main()
