import os
from pathlib import Path
import shutil
import nibabel as nib
import json
import natsort

modality_dict = {
    "GED4": 0,
    "GED3": 1,
    "GED2": 2,
    "GED1": 3,
    "T1": 4,
    "T2": 5,
    "DWI_800": 6
}

def moving_data(base_dir, nnunet_raw_data_dir):
    """
    Move the LIQA annotated data from the base directory to the target directory.
    """
    base_dir = Path(base_dir)
    nnunet_raw_data_dir = Path(nnunet_raw_data_dir)
    
    # Create the necessary directories
    nnunet_images_dir = nnunet_raw_data_dir / "imagesTr"
    nnunet_labels_dir = nnunet_raw_data_dir / "labelsTr"
    
    nnunet_images_dir.mkdir(parents=True, exist_ok=True)
    nnunet_labels_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for vendor in base_dir.iterdir():
        if vendor.is_dir():
            for patient in vendor.iterdir():
                if patient.is_dir():
                    for file in patient.iterdir():
                        if file.suffix == ".gz":
                            modality = file.name.split('.')[0]
                            if modality in modality_dict.keys():
                                modality_index = modality_dict[modality]
                                # moving images
                                shutil.copy(file, nnunet_images_dir / f"{patient.name}_{idx:03d}_{modality_index:04d}.nii.gz")
                                # moving labels
                                if modality != "GED4":
                                    label_file = 'mask_GED4_registered_to_' + file.name
                                elif modality == "GED4":
                                    label_file = 'mask_GED4.nii.gz'
                                else:
                                    print(f"Unknown modality: {modality}")
                                shutil.copy(patient / label_file, nnunet_labels_dir / f"{patient.name}_{idx:03d}.nii.gz")
                                    
                    idx += 1


def moving_augmented_data(base_dir, target_dir):
    """
    Moving the augmented data to benchmark2
    """
    base_dir = Path(base_dir)
    target_dir = Path(target_dir)
    target_images_dir = target_dir / "images"
    target_labels_dir = target_dir / "labels"
    if not target_images_dir.exists():
        target_images_dir.mkdir(parents=True, exist_ok=True)
    if not target_labels_dir.exists():
        target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for augmented_sample in base_dir.iterdir():
        if augmented_sample.is_dir():
            for file in augmented_sample.iterdir():
                if file.name.endswith('img.nii.gz'):
                    shutil.copy(file, target_images_dir / file.name)
                else:
                    shutil.copy(file, target_labels_dir / file.name)

                           
def nnunet_data_json(num_training, num_test):
    """
    Create the nnUNet data JSON file.
    """
    return {
        "name": "Care2025 Liver Benchmark",
        "description": "Liver segmentation benchmark dataset for Care2025.",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "CC BY-NC-SA 4.0",
        "release": "1.0",
        "channel_names": {
            "0": "GED4"
        },
        "labels": {
            "background": 0,
            "liver": 1
        },
        "numTraining": num_training,
        "numTest": num_test,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO"
    }

def prepare_nnunet_data_benchmark3(base_dir, nnunet_raw_data_dir):
    """
    Prepare the nnUNet data structure from the LIQA annotated data.
    """
    base_dir = Path(base_dir)
    nnunet_raw_data_dir = Path(nnunet_raw_data_dir)
    
    # Create the necessary directories
    nnunet_images_dir = nnunet_raw_data_dir / "imagesTr"
    nnunet_labels_dir = nnunet_raw_data_dir / "labelsTr"
    
    nnunet_images_dir.mkdir(parents=True, exist_ok=True)
    nnunet_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Move the data
    image_dir = base_dir / "images"
    label_dir = base_dir / "labels"

    images = natsort.natsorted(image_dir.glob("*.nii.gz"))
    labels = natsort.natsorted(label_dir.glob("*.nii.gz"))

    for idx, (image_file, label_file) in enumerate(zip(images, labels)):
        
        
        shutil.copy(image_file, nnunet_images_dir / (image_file.name.split('.')[0] + "_" + str(idx).zfill(3) + "_0000.nii.gz"))
        shutil.copy(label_file, nnunet_labels_dir / (image_file.name.split('.')[0] + "_" + str(idx).zfill(3) + ".nii.gz"))

    # Create the dataset JSON file
    num_training = len(list(nnunet_images_dir.glob("*.nii.gz")))
    num_test = 0  # Assuming no test data for now
    dataset_json = nnunet_data_json(num_training, num_test)
    with open(nnunet_raw_data_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)


if __name__ == "__main__":
    
    # Moving LIQA annotated data to the target directory
    # base_dir = "/hdd/yang/data/care2025/LiQA_training_data"
    # target_dir = "/hdd/yang/data/care2025/training_dataset/benchmark3"
    # moving_liqa_annotated_data(base_dir, target_dir)
    # print(f"Data has been moved from {base_dir} to {target_dir}.")
    # print("You can now run nnUNet training with the prepared data.")
    
    #Prepare nnUNet data structure
    base_dir = "/hdd/yang/data/care2025/training_dataset/benchmark3"
    nnunet_raw_data_dir = "/hdd/yang/data/kidney_seg_nnunet/nnUNet_raw/Dataset015_Care2025_benchmark3"
    prepare_nnunet_data_benchmark3(base_dir, nnunet_raw_data_dir)

    # base_dir = "/hdd/yang/data/care2025/augmented"
    # target_dir = "/hdd/yang/data/care2025/training_dataset/benchmark3"
    # moving_augmented_data(base_dir, target_dir)

    # base_dir = "/hdd/yang/data/care2025/LiQA_training_data/"
    # target_dir = "/hdd/yang/data/care2025/training_dataset/benchmark3"
    # moving_registered_data(base_dir, target_dir)

    pass
    


   