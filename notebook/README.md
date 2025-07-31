# CARE-Liver Validation Phase Instructions

Welcome to the validation phase of the CARE-Liver Challenge! This document outlines the validation data usage and submission instructions for both tasks.


## 0. Dataset Description

The validation data format is the same as the training data. Each case is organized in a folder containing multiple modality images:

```
XXXX-X/
├── DWI_800.nii.gz
├── GED1.nii.gz
├── GED2.nii.gz
├── GED3.nii.gz
├── GED4.nii.gz
├── T1.nii.gz
├── T2.nii.gz
```

There are 60 cases in total, with 20 cases from each vendor (A, B1, B2). Some cases may have missing modalities.

An info.csv file is provided in the root directory, with the following columns:
* Case: e.g., 0001-A
* Vendor: e.g., A
* Missing Modality: e.g., DWI_800 (blank if none)


## 1. Submission Instructions

### 1.1 Liver Fibrosis Staging (LiFS)

For LiFS, you are required to submit probabilistic predictions for two binary classification subtasks:
* Subtask 1 (Cirrhosis Detection): S1–S3 vs. S4
* Subtask 2 (Substantial Fibrosis Detection): S1 vs. S2–S4

Under each of the following two settings:
* NonContrast: Using only T1WI, T2WI, and DWI
* Contrast: Using all available modalities (including GED1–GED4)

Please submit a single CSV file with the following format:

```
Case,Setting,Subtask1_prob_S4,Subtask2_prob_S1
XXXX-X,Contrast,0.2,0.5
XXXX-X,NonContrast,0.6,0.1
...
```
* Subtask1_prob_S4: Probability of class S4 (Cirrhosis)
* Subtask2_prob_S1: Probability of class S1 (Non-fibrotic)

We will evaluate predictions based on Accuracy and AUC for each subtask and setting.


### 1.2 Liver Segmentation (LiSeg)

Your submission should follow this folder structure per case, with one predicted file per segmented modality:

```
0001-A/
├── GED4_pred.nii.gz         # Contrast
├── DWI_800_pred.nii.gz      # NonContrast
├── T2_pred.nii.gz           # NonContrast
...
```

If you provide predictions for non-GED4 modalities(T1,T2,DWI), please also submit a CSV file named `Unsupervised_Segmentation_List.csv` with the following format:

```
Case,Modality
0001-A,T2
0001-A,DWI
0002-A,DWI
...
```

Also include a brief method description in `Method_Description.md`, explaining your segmentation strategy (e.g., registration to GED4, unsupervised clustering, etc.).


## 2. Evaluation Notes
* LiFS predictions will be evaluated automatically using Accuracy and AUC, with separate results for each subtask and setting.
* LiSeg predictions will be evaluated automatically using standard segmentation metrics (Dice, HD) for each modality.

## 3. Submission Instructions

Please organize your files as follows:

```
CARE-Liver-Submission.zip/
├── LiFS_pred.csv
├── LiSeg_pred/
│   ├── 0001-A/
│   │   ├── GED4_pred.nii.gz
│   │   ├── DWI_800_pred.nii.gz
│   │   └── T2_pred.nii.gz
│   └── ...
├── Unsupervised_Segmentation_List.csv   # Optional
├── Method_Description.md                # Optional
```

Then compress into a `.zip` file and upload to the submission platform.