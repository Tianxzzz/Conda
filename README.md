# Conda
Code for KDD2024 paper “Latent Diffusion-based Data Augmentation for Continuous-Time Dynamic Graph Model”



## Modified models 

Since Conda needs to perform diffusion on the historical neighbor sequence of a target node, we made some modifications to the model implemented in DyGFormer. Although the original model can also use conda, it essentially only allows diffusion after computing all node embeddings, which means the diffusion sequence length is equivalent to the entire set of neighbors. By using the modified model, Conda can more effectively control the sequence length for diffusion. The modified models are in the file ```models_for_diffusion```.

## Model Training

To use conda, run ```alter_train_DA.py```.
For example, to train model *TCL+conda* on dataset with 0.1 training ratio, we can run the following comands:
```{bash}
python alter_train_DA.py --dataset_name wikipedia --model_name TCL --num_runs 5 --gpu 0 --val_ratio 0.1 --test_ratio 0.8
```

For the original model, we can run the following comands:
```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name TCL --num_runs 5 --gpu 0 --val_ratio 0.1 --test_ratio 0.8
```

The code for the modified models in the current project is a bit disorganized. We will integrate the modified models into the training code and continue updating it within a week.


## Acknowledgments
The basic framework is based [DyGFormer](https://github.com/yule-BUAA/DyGLib), we are grateful to the authors for making their project codes publicly available.

## Citation
Please consider citing our paper when using this project.
```
@inproceedings{10.1145/3637528.3671863,
author = {Tian, Yuxing and Jiang, Aiwen and Huang, Qi and Guo, Jian and Qi, Yiyan},
title = {Latent Diffusion-based Data Augmentation for Continuous-Time Dynamic Graph Model},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671863},
doi = {10.1145/3637528.3671863},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {2900–2911},
numpages = {12},
keywords = {data augmentation, diffusion model, dynamic graph},
location = {Barcelona, Spain},
series = {KDD '24}
}
```
