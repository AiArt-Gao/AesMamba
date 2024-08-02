## AesMamba: Universal Image Aesthetic Assessment with State Space Models (ACM MM 2024)





<p align="center">
<img src="assets/result_sum.png" width="800px"/>
<img src="assets/Model.png" width="800px"/>
</p>



## TODO
- [x] Add inference code and config files
- [x] Add checkpoint and script for IAA task

# Results

## VIAA 
<img src="assets/VIAA.png" width="400px"/> 

## FIAA
<img src="assets/FIAA.png" width="400px"/> 

## MIAA
<img src="assets/MIAA.png" width="400px"/> 

## PIAA
<img src="assets/PIAA.png" width="400px"/> 

# Dependencies and Installation
requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+
## our version(advised)
```pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117```

```pip install --upgrade pip setuptools wheel```

## install mamba
```conda create -n Aesmamba python=3.8```
```conda activate Aesmamba```

```git clone https://github.com/state-spaces/mamba.git```
```cd mamba```
```MAMBA_FORCE_BUILD=TRUE pip install .```
# other requirements
```cd ../Aesmamba```
```pip install -r requirements.txt```


# VIAA task
```cd AesMamba_v && python train_viaa.py```
# MIAA task
```cd AesMamba_m && python train_miaa.py```
# FIAA task
```cd AesMamba_f && python train_multi_attr_add_balce.py```
# PIAA task
```cd AesMamba_p && python multi_attr_pred_model_add_human_attr.py.py```

# Noticing
You can change the config in their corresponding .py file. We will combine the four tasks in our later works.

In our code, we classified the image by its score in each dataset. We uploaded some of their csv files. As for other datasets, we only provide the method of classification because the csv file is large.

# Pretrain path
Visual Encoder:vmamba tiny and Text Encoder:bert base
We use old version of vmamba, the ckpt is here:

Link: https://pan.baidu.com/s/1REVTVD4w20G7lKnIM-Btjg   Passward: c1mk


Vmamba base and it's conda environment
please ref https://github.com/MzeroMiko/VMamba
