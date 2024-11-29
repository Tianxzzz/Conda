# Conda
Code for KDD2024 paper “Latent Diffusion-based Data Augmentation for Continuous-Time Dynamic Graph Model”

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
