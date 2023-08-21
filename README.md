## Cloud Forcasting Transformer

### Introduction
Our work tackles the problem of cloud cover nowcasting using Deep Learning. Nowcasting is a field of meteorology which aims at forecasting weather for a short term of up to
a few hours. Cloud cover, as the name suggests, refers the fraction of the sky covered by clouds at some geographical location. Our work bases on the Earthformer, a transformer model created by Amazon science. 
We train the Earthformer on our original dataset composed of satellite images provided by the Israeli Meteorological Service.

The model's input is a sequence of 13 satellite images taken 15 minutes apart. The model's output is 12 images that
represent the prediction of the next satellite images following the input sequence.

<center>

|  Input  |  Target  | Prediction |
|:-------------------------:|:-------------------------:|:-------------------------:|
|  <img src="./images_for_readme/test_0_input.gif" width="200"/>  |  <img src="./images_for_readme/test_0_target.gif" width="200"/>  |  <img src="./images_for_readme/test_0_lpips_wd.gif" width="200"/>  |

</center>

## Installation
### Train
Install CUDA (We had CUDA 11.7 or 11.8 installed): https://developer.nvidia.com/cuda-downloads

Clone the repository and install requirements:
```
git clone https://github.com/leahandofir/cloud-forecasting-transformer.git
cd cloud-forecasting-transformer
pip install -r train_requirements.txt
python3 -m pip install -U -e . --no-build-isolation --extra-index-url --trusted-host
```

Set up wandb if used:
```
pip install wandb
wandb login
```

### Inference
The inference was tested on windows only.
Clone the repository and install requirements:
```
git clone https://github.com/leahandofir/cloud-forecasting-transformer.git
cd cloud-forecasting-transformer
pip install -r inference_requirements.txt
python3 -m pip install -U -e . --no-build-isolation --extra-index-url --trusted-host
```

## Cloudformer and Baselines Training
TODO: how to train? basic example on our model.

| Model       | Script Folder                                                                     | Pretrained Weights                                                                                                     | Config                                                        |
|---------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| cloud-forcasting-transformer         | [train_cuboid_ims.py](./scripts/cuboid_transformer/ims/train_cuboid_ims.py)       | TODO!        | [ims_cfg.yaml](./scripts/cuboid_transformer/ims/ims_cfg.yaml) |
| PredRNN | [train_predrnn_ims.py](./scripts/baselines/predrnn/train_predrnn_ims.py)          | | [ims_cfg.yaml](./scripts/baselines/predrnn/ims_cfg.yaml)      |
| Rainformer | [train_rainformer_ims.py](./scripts/baselines/rainformer/train_rainformer_ims.py) | | [ims_cfg.yaml](./scripts/baselines/rainformer/ims_cfg.yaml)   |

## Inference

The inference script is located at [ims_inference.py](scripts/cuboid_transformer/ims/ims_inference.py). Prior to execution, one must prepare:  
&ndash; A checkpoint file of a trained model.  
&ndash; A directory with the input sequence in PNG format and the naming convention *`%Y%m%d%H%M.png`*.

A simple example of inference is as follows:
```
python scripts/cuboid_transformer/ims/ims_inference.py --ckpt [path-to-ckpt-file] --data-dir [path-to-data-dir] --start-time [time-in-format-%Y%m%d%H%M]
```

You can read more about the possible arguments in [ims_inference](./scripts/cuboid_transformer/ims/README.md).


The script outputs a PNG file that contains the prediction for the given input (the first row is the input sequence, and the second row is the model's prediction):

![](./images_for_readme/inference_output.png)

## Credits
©EUMETSAT 2023

Earthformer: https://github.com/amazon-science/earth-forecasting-transformer

SEVIR: https://sevir.mit.edu/

PredRNN: https://github.com/thuml/predrnn-pytorch

Rainformer: https://github.com/Zjut-MultimediaPlus/Rainformer
