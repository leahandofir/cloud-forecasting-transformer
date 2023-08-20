## Cloud Forcasting Transformer

## Dataset
TODO: write

## Installation
### Training
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

### Inferencing
The inference was tested on windows only.
Clone the repository and install requirements:
```
git clone https://github.com/leahandofir/cloud-forecasting-transformer.git
cd cloud-forecasting-transformer
pip install -r inference_requirements.txt
python3 -m pip install -U -e . --no-build-isolation --extra-index-url --trusted-host
```
Run inference script at scripts/cuboid_transformer/ims/ims_inference.py.

## Cloudformer and Baselines Training
TODO: how to train? basic example on our model.

| Model       | Script Folder                                            | Pretrained Weights                                                                                                     | Config                                                                              |
|---------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| cloud-forcasting-transformer         | [train_cuboid_ims.py](./scripts/cuboid_transformer/ims/train_cuboid_ims.py)            | TODO!        | [ims_cfg.yaml](./scripts/cuboid_transformer/ims/ims_cfg.yaml)              |
| PredRNN | | | |
| Rainformer | | | |

## Inferencing
TODO: write basic example.

## Credits
©EUMETSAT 2023

Earthformer: https://github.com/amazon-science/earth-forecasting-transformer

SEVIR: https://sevir.mit.edu/

PredRNN: https://github.com/thuml/predrnn-pytorch

Rainformer: https://github.com/Zjut-MultimediaPlus/Rainformer
