command = 'pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com'
import os
import subprocess
def install_rapidsai():
    try:
        print("Installing RAPIDS AI...")
        subprocess.check_call(command, shell=True)
        print("RAPIDS AI installation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during RAPIDS AI installation: {e}")
        exit(1)
if __name__ == "__main__":
    install_rapidsai()
