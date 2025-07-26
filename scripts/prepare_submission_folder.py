from pathlib import Path

def prepare_submission_folder(submission_dir: Path, validation_dir: Path) -> None:
    """
    Prepares the submission folder by creating it if it does not exist.
    
    Args:
        submission_path (Path): The path to the submission folder.
    """
    if not submission_dir.exists():
        submission_dir.mkdir(parents=True, exist_ok=True)
    
    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation path {validation_dir} does not exist.")
    
    for file in validation_dir.glob('*.gz'):
        file_name = file.name
        folder_name = file_name.split('_')[0]
        if not (submission_dir / folder_name).exists():
            (submission_dir / folder_name).mkdir(parents=True, exist_ok=True)
        pred_modality = file_name.split('_')[1]
        saving_name = pred_modality + '_pred.nii.gz'
        file.rename(submission_dir / folder_name / saving_name)
    
    print(f"Submission folder prepared at {submission_dir}.")

def prepare_submission_folder_DWI(submission_dir: Path, validation_dir: Path) -> None:
    """
    Prepares the submission folder by creating it if it does not exist.
    
    Args:
        submission_path (Path): The path to the submission folder.
    """
    if not submission_dir.exists():
        submission_dir.mkdir(parents=True, exist_ok=True)
    
    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation path {validation_dir} does not exist.")
    
    for patient_dir in validation_dir.glob('*/'):
        print(f"Processing patient directory: {patient_dir}")
        for file in patient_dir.glob('*.gz'):
            file_name = file.name
            folder_name = patient_dir.name
            if not (submission_dir / folder_name).exists():
                (submission_dir / folder_name).mkdir(parents=True, exist_ok=True)
            pred_modality = file_name.split('_')[0]
            if pred_modality == 'DWI':
                print(f"Renaming file {file_name} to DWI_800_pred.nii.gz")
                pred_modality = 'DWI_800'
                saving_name = pred_modality + '_pred.nii.gz'
                file.rename(submission_dir / folder_name / saving_name)
    
    print(f"Submission folder prepared at {submission_dir}.")

if __name__ == "__main__":
    submission_dir = '/hdd/yang/data/care2025/Submission/val_all_modalities_DWI_800/'
    validation_dir = '/hdd/yang/data/care2025/Submission/CARE-Liver-Submission2/LiSeg_pred'
    prepare_submission_folder_DWI(Path(submission_dir), Path(validation_dir))
    
