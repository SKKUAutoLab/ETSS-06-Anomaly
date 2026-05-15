sudo apt-get install -y libxcb-cursor-dev
uv venv --python 3.12
source .venv/bin/activate
uv pip install --pre "x-anylabeling-cvhub[gpu-cu11]"
uv pip install numpy==1.26.4
uv pip install onnx onnxruntime-gpu==1.19.2 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
# uv run anylabeling/app.py
