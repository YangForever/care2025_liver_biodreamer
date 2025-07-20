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

if __name__ == "__main__":
    submission_dir = '/hdd/yang/data/care2025/Submission/train_all/'
    validation_dir = '/hdd/yang/results/glomeruli_segmentation/nnUNet_results/Dataset014_Care2025_benchmark2/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/train_GED4_all/'
    prepare_submission_folder(Path(submission_dir), Path(validation_dir))
    
