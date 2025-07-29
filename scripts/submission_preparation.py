from pathlib import Path
import shutil
import argparse

def prepare_submission_folder(submission_dir: str, validation_dir: str) -> None:
    """
    Prepares the submission folder by creating it if it does not exist.
    
    Args:
        submission_path (Path): The path to the submission folder.
    """
    submission_dir = Path(submission_dir)
    submission_dir = submission_dir / 'LiSeg_pred'
    validation_dir = Path(validation_dir)
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
        if pred_modality == 'DWI':
            pred_modality = 'DWI_800'
        saving_name = pred_modality + '_pred.nii.gz'
        shutil.copy(file, submission_dir / folder_name / saving_name)
    print(f"Submission folder prepared at {submission_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare submission folder for validation data.")
    parser.add_argument("--submission_dir", type=str, required=True, help="Path to the submission directory.")
    parser.add_argument("--validation_dir", type=str, required=True, help="Path to the validation directory.")
    args = parser.parse_args()

    prepare_submission_folder(args.submission_dir, args.validation_dir)
    
