import os
from pathlib import Path
import shutil
from normalisation import normalise_image

def copying_validation_data(base_dir, target_dir):
    """
    copy the LIQA annotated data from the base directory to the target directory.
    """
    base_dir = Path(base_dir)
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for vendor in base_dir.iterdir():
        if vendor.is_dir():
            for patient in vendor.iterdir():
                if patient.is_dir():
                    for file in patient.iterdir():
                        if file.suffix == ".gz" and 'GED4' in file.name:
                            print(f"Copying {file} to {target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz')}")
                            shutil.copy(file, target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz'))
                            idx += 1

def copying_validation_data_all_modality(base_dir, target_dir):
    """
    copy the LIQA annotated data from the base directory to the target directory.
    """
    modality_dict = ["GED4","GED3","GED2","GED1","T1","T2","DWI_800"]
    
    base_dir = Path(base_dir)
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for vendor in base_dir.iterdir():
        if vendor.is_dir():
            for patient in vendor.iterdir():
                if patient.is_dir():
                    for file in patient.iterdir():
                        for modality in modality_dict:
                            if file.suffix == ".gz" and modality in file.name:
                                # Normalise the image before copying
                                norm_image = normalise_image(file)
                                # Copy the normalised image to the target directory
                                norm_image_path = target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz')
                                print(f"Saving normalised data to: {norm_image_path}")
                                norm_image.to_filename(norm_image_path)
                                idx += 1

def copying_train_GED4_data(base_dir, target_dir):
    """
    copy the LIQA annotated data from the base directory to the target directory.
    """
    base_dir = Path(base_dir)
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for vendor in base_dir.iterdir():
        if vendor.is_dir():
            for patient in vendor.iterdir():
                if patient.is_dir():
                    for file in patient.iterdir():
                        if file.suffix == ".gz" and file.name.startswith("GED4.nii"):
                            print(f"Copying {file} to {target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz')}")
                            shutil.copy(file, target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz'))
                            idx += 1

def copying_train_data_all_modality(base_dir, target_dir):
    """
    copy the LIQA annotated data from the base directory to the target directory.
    """
    modality_dict = ["GED4","GED3","GED2","GED1","T1","T2","DWI_800"]

    base_dir = Path(base_dir)
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for vendor in base_dir.iterdir():
        if vendor.is_dir():
            for patient in vendor.iterdir():
                if patient.is_dir():
                    for file in patient.iterdir():
                        if file.suffix == ".gz":
                            for modality in modality_dict:
                                if file.name.startswith(modality+'.nii'):
                                    print(f"Copying {file} to {target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz')}")
                                    shutil.copy(file, target_dir / (patient.name + '_' + file.name.split('.')[0] + '_' + str(idx).zfill(3) + '_0000.nii.gz'))
                                    idx += 1



if __name__ == "__main__":
    # copying_validation_data(
    #     base_dir = '/hdd/yang/data/care2025/LiQA_training_data',
    #     target_dir = '/hdd/yang/data/care2025/validation_dataset/val_GED4'
    # )

    # copying_train_GED4_data(
    #     base_dir = '/hdd/yang/data/care2025/LiQA_training_data',
    #     target_dir = '/hdd/yang/data/care2025/validation_dataset/train_GED4'
    # )

    # copying_validation_data_all_modality(
    #     base_dir = '/hdd/yang/data/care2025/LiQA_val/Data',
    #     target_dir = '/hdd/yang/data/care2025/validation_dataset/val_all'
    # )

    copying_train_data_all_modality(
        base_dir = '/hdd/yang/data/care2025/LiQA_training_data',
        target_dir = '/hdd/yang/data/care2025/validation_dataset/train_all'
    )
    


   