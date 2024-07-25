# AesMamba
[ACM MM'24] Oral, AesMamba


# env
requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+
# our version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade pip setuptools wheel


# install mamba
conda create -n Aesmamba python=3.8
conda activate Aesmamba


git clone https://github.com/state-spaces/mamba.git
cd mamba
MAMBA_FORCE_BUILD=TRUE pip install .

# other requirements
cd ../Aesmamba
pip install -r requirements.txt


# VIAA task
cd AesMamba_v && python train_viaa.py
# MIAA task
cd AesMamba_m && python train_viaa.py
# FIAA task
cd AesMamba_f && python train_viaa.py
# PIAA task
cd AesMamba_p && python train_viaa.py


you can change the config in their .py file 