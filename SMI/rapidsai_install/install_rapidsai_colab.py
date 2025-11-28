colab_command = 'git clone https://github.com/rapidsai/rapidsai-csp-utils.git'
colab_command = '& python rapidsai-csp-utils/colab/pip-install.py'
#run if you are on colab
if __name__ == "__main__":
    import os
    if 'COLAB_GPU' in os.environ:
        print("Running on Google Colab, installing RAPIDS AI...")
        os.system(colab_command)
    else:
        print("Not running on Google Colab, skipping RAPIDS AI installation.")
        exit()