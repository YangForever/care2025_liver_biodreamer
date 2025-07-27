# care2025_liver_biodreamer

This project utilise [uv](https://docs.astral.sh/uv/) to manage the dependencies.

To install all the dependencies, run the commands:
````
cd care2025_liver_biodreamer/
uv sync
````

*The packages used are listed in the pyproject.toml which can used to create a environment by conda if required* 

# Runing the data-preprocessing for nnUNet training
````
uv run scripts/prepare_nnunet.py
````
*Replace the path to the folder*

# Runing the multi-modal registration
````
uv run scripts/registration_GED4_to_other.py
````
*Replace the path to the folder*

# Runing the nnUNet
Please refer to the official nnUNet repository for details.

We are still figuring out integrating the uv with nnUNet, so to run the nnUNet, we need a conda enviroment and run the code inside the environment.

## nnUNet installation
````
cd nnUNet
pip install -e .
````
## nnUNet data preprocessed
After running the code and obtain the nnUNet_raw data

run the following code
````
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
````
Example:
```Console
nnUNetv2_plan_and_preprocess -d 13 --verify_dataset_integrity
```

## nnUNet training
```Console
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```
*DATASET_NAME_OR_ID: dataset id without '00'* \
*UNET_CONFIGURATION: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres*\
*FOLD: 0, 1, 2, 3, 4 (default 5-fold)*

Example:
```Console
nnUNetv2_train 13 3d_fullres 0
```

## nnUNet inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f FOLD --save_probabilities
```

Example:
```Console
nnUNetv2_predict -i ./Dataset013_care2025_benchmark1/imagesTs -o ./nnUNet_results/Dataset013_care2025_benchmark1/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/inference_test -d 13 -c 3d_fullres -f 0
```

# Fibrosis Classfication
Using the jupyter notebook
```Console
script/classification.ipynb
```