import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
from PIL import Image
from pytesseract import pytesseract
import numpy as np

def extract_metrics(text):
    psnr_pattern = r'PSNR: (\d+\.\d+)'
    ssim_pattern = r'SSIM: (\d+\.\d+)'
    rmse_pattern = r'RMSE: (\d+\.\d+)'
    
    try:
        psnr = float(re.search(psnr_pattern, text).group(1))
        ssim = float(re.search(ssim_pattern, text).group(1))
        rmse = float(re.search(rmse_pattern, text).group(1))
        return f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\nRMSE: {rmse:.4f}"
    except (AttributeError, ValueError):
        raise ValueError("Could not extract metrics from the text")

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from image {image_path}: {str(e)}")

def find_image_pairs(base_path, img_index):
    pairs = {}
    
    folders = {
        "Fixed-Point + RED-CNN": "fixed_point_red_cnn",
        "ADMM + RED-CNN": "admm_red_cnn",
        "SART": "sart_1000",
        "Fixed-Point + DenoMamba": "fixed_point_denomamba_red_test_1000_sigma_0.3_lam_0.5",
        "FBP": "fbp_1000",
        "P3 + DenoMamba": "admm_denomamba_p3_1000",
        "ADMM + DenoMamba": "admm_denomamba_red_1000",
        "ADMM TV": "admm_tv_1000"
    }
    
    for display_name, folder in folders.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder}")
            
        all_files = os.listdir(folder_path)
        comparison_files = [f for f in all_files if f.startswith(f"img_{img_index}_comparison_")]
        reconstruction_files = [f for f in all_files if f.startswith(f"img_{img_index}_reconstruction_")]
        
        if not comparison_files or not reconstruction_files:
            raise ValueError(f"Missing required images for index {img_index} in {folder}")
            
        comparison_file = sorted(comparison_files)[-1]
        reconstruction_file = sorted(reconstruction_files)[-1]
        
        pairs[display_name] = {
            "comparison": os.path.join(folder_path, comparison_file),
            "reconstruction": os.path.join(folder_path, reconstruction_file)
        }
    
    return pairs

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def create_comparison_diagram(base_path, img_index, output_filename='algo_comp_diagram.png'):
    image_pairs = find_image_pairs(base_path, img_index)
    
    images = {}
    for name, paths in image_pairs.items():
        try:
            comparison_text = extract_text_from_image(paths["comparison"])
            metrics = extract_metrics(comparison_text)
            
            images[name] = {
                "path": paths["reconstruction"],
                "metrics": metrics
            }
        except Exception as e:
            print(f"Warning: Failed to process {name}: {str(e)}")
            continue
    
    if not images:
        raise ValueError("No valid images were processed")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (name, details) in enumerate(images.items()):
        img = mpimg.imread(details["path"])
        gray = rgb2gray(img)
        ax = axes[idx // 4, idx % 4]
        ax.imshow(gray, cmap='gray')
        ax.axis('off')
        ax.set_title(name, fontsize=10, pad=5, weight='bold')
        ax.text(0.5, -0.15, details["metrics"], fontsize=8, ha='center', transform=ax.transAxes)
    
    plt.tight_layout(h_pad=3, rect=[0, 0, 1, 1])
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comparison diagram for reconstruction algorithms')
    parser.add_argument('img_index', type=int, help='Image index to process')
    parser.add_argument('--path', type=str, default="./", help='Base path containing algorithm folders')
    parser.add_argument('--output', type=str, default='algo_comp_diagram.png', 
                      help='Output filename (default: algo_comp_diagram.png)')
    
    args = parser.parse_args()
    
    try:
        create_comparison_diagram(args.path, args.img_index, args.output)
        print(f"Successfully created comparison diagram: {args.output}")
    except Exception as e:
        print(f"Error: {e}")