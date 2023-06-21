## Cloud Forcasting Transformer

## Tutorials

## Introduction

Conventionally, Earth system (e.g., weather and climate) forecasting relies on numerical simulation with complex physical models and are hence both 
expensive in computation and demanding on domain expertise. With the explosive growth of the spatiotemporal Earth observation data in the past decade, 
data-driven models that apply Deep Learning (DL) are demonstrating impressive potential for various Earth system forecasting tasks. 
The Transformer as an emerging DL architecture, despite its broad success in other domains, has limited adoption in this area. 
In this paper, we propose **Earthformer**, a space-time Transformer for Earth system forecasting. 
Earthformer is based on a generic, flexible and efficient space-time attention block, named **Cuboid Attention**. 
The idea is to decompose the data into cuboids and apply cuboid-level self-attention in parallel. These cuboids are further connected with a collection of global vectors.

Earthformer achieves strong results in synthetic datasets like MovingMNIST and N-body MNIST dataset, and also outperforms non-Transformer models (like ConvLSTM, CNN-U-Net) in SEVIR (precipitation nowcasting) and ICAR-ENSO2021 (El Nino/Southern Oscillation forecasting).


![teaser](figures/teaser.png)


### Cuboid Attention Illustration

![cuboid_attention_illustration](figures/cuboid_illustration.gif)

## Installation
### Training
### Inferencing

## Dataset

### EUMESAT

## Cloudformer Training
Find detailed instructions in the corresponding training script folder
- [N-body MNIST](./scripts/cuboid_transformer/ims/README.md) #TODO: write instructions

## Training Script and Pretrained Models

Find detailed instructions in how to train the models or running inference with our pretrained models in the corresponding script folder.

| Dataset       | Script Folder                                            | Pretrained Weights                                                                                                     | Config                                                                              |
|---------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| SEVIR         | [scripts](./scripts/cuboid_transformer/sevir)            | [link](https://deep-earth.s3.amazonaws.com/experiments/earthformer/pretrained_checkpoints/earthformer_sevir.pt)        | [config](./scripts/cuboid_transformer/sevir/earthformer_sevir_v1.yaml)              |

## Credits
©EUMETSAT 2023
Copyright
Unless otherwise noted, you may download and copy content from the EUMETSAT website(s) for your own personal and non-commercial use. For any other use of the content, you must request authorisation from EUMETSAT. 

Earthformer

SEVIR 
