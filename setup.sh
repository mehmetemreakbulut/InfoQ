#Rapids Install for SMI Calculation
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==25.6.*"
pip install cupy-cuda12x
pip install -U datasets
#Install requirements.txt
pip install -r requirements.txt

