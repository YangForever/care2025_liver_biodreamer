from pathlib import Path
import nibabel as nib
import numpy as np
import skimage as ski
import argparse

def post_processing(inference_dir, output_dir):
    inference_dir = Path(inference_dir)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for pred in inference_dir.glob('*.nii.gz'):
        prediction_nift = nib.load(pred)
        prediction = prediction_nift.get_fdata()
        
        # Apply morphological operations
        closed_pred = ski.morphology.closing(prediction, ski.morphology.footprint_rectangle((4,4,4)))
        closed_pred = ski.morphology.erosion(closed_pred, ski.morphology.footprint_rectangle((1,1,1)))

        print(f'Processing {pred.name}')
        # Save the processed prediction
        new_prediction = nib.Nifti1Image(closed_pred.astype(np.float64), prediction_nift.affine, prediction_nift.header)
        pred_name = pred.stem.split('.')[0]
        new_prediction.to_filename(output_dir / f'{pred_name}_processed.nii.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-processing for inference results.')
    parser.add_argument('--inference_dir', type=str, required=True, help='Directory containing inference results.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed results.')
    
    args = parser.parse_args()
    
    post_processing(args.inference_dir, args.output_dir)