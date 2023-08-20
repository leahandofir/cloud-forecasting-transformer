## Cloud Forcasting Transformer

## Tutorials

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

## Dataset
TODO: write

## Cloudformer and Baselines Training
Find detailed instructions in how to train the models with our pretrained models in the corresponding script folder.
| Model       | Script Folder                                            | Pretrained Weights                                                                                                     | Config                                                                              |
|---------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| cloud-forcasting-transformer         | [scripts](./scripts/cuboid_transformer/ims)            | TODO!        | [config](./scripts/cuboid_transformer/ims/ims_cfg.yaml)              |
| PredRNN | | | |
| Rainformer | | | |

## Credits
©EUMETSAT 2023
Copyright
Unless otherwise noted, you may download and copy content from the EUMETSAT website(s) for your own personal and non-commercial use. For any other use of the content, you must request authorisation from EUMETSAT. 

Earthformer: https://github.com/amazon-science/earth-forecasting-transformer

SEVIR: https://sevir.mit.edu/

PredRNN: https://github.com/thuml/predrnn-pytorch

Rainformer: https://github.com/Zjut-MultimediaPlus/Rainformer
