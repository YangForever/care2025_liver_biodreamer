from pathlib import Path
import argparse
import shutil


def segmentation_inference():
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation inference on a dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory, organised as released validation dataset.")
    # output_dir default parent folder
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output results.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the segmentation inference function
    segmentation_inference(args.input_dir, output_dir)