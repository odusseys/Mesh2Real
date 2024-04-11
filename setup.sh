### NVIDFFRAST REQS
set -e

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

pip install git+https://github.com/huggingface/diffusers.git
pip install ninja opencv-python transformers peft accelerate \
     imageio imageio-ffmpeg

cd FeatUp && pip install .
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast && pip install .

pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu121.html \
#  trimesh[all] \
 open3d \
 xatlas \
 rembg \
 imageio
