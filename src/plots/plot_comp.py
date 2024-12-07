import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import cv2

def get_metrics_from_npy(folder_path, img_index):
    try:
        data_dict = {}
        npy_count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(folder_path, filename)
                
                if filename.startswith('psnrs'):
                    data_dict['psnrs'] = np.load(file_path)
                elif filename.startswith('ssims'):
                    data_dict['ssims'] = np.load(file_path)
                elif filename.startswith('rmses'):
                    data_dict['rmses'] = np.load(file_path)

                npy_count += 1

        assert(npy_count == 3)
        
        psnr = data_dict['psnrs'][img_index]
        ssim = data_dict['ssims'][img_index]
        rmse = data_dict['rmses'][img_index]
        
        return f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\nRMSE: {rmse:.4f}"
    
    except FileNotFoundError:
        raise ValueError(f"Missing required npy files in {folder_path}")
    except Exception as e:
        raise ValueError(f"Error processing npy files in {folder_path}: {str(e)}")

def find_image_pairs(base_path, img_index):
    """Find comparison and reconstruction image pairs for a given index."""
    pairs = {}
    
    # Dictionary mapping algorithm names to their folder names
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
        comparison_files = [f for f in all_files if '_comparison_' in f and f.endswith('.png')]
        reconstruction_files = [f for f in all_files if '_reconstruction_' in f and f.endswith('.png')]
        
        num_pairs = len(comparison_files)
        if num_pairs == 0:
            raise ValueError(f"No image pairs found in {folder}")
            
        comparison_file = [f for f in comparison_files if f.startswith(f"img_{img_index}_comparison_")][0]
        reconstruction_file = [f for f in reconstruction_files if f.startswith(f"img_{img_index}_reconstruction_")][0]
        
        pairs[display_name] = {
            "folder_path": folder_path,
            "reconstruction": os.path.join(folder_path, reconstruction_file),
            "comparison": os.path.join(folder_path, comparison_file)
        }
    
    return pairs

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def extract_right_image(input_path, output_path="ground_truth.png"):
    """
    Helper function that extracts the ground truth image from the comparison plot
    """
    img = cv2.imread(input_path)
    
    total_width = img.shape[1]
    mid = total_width // 2
    right_img = img[:, mid + 1:]
    
    gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    row_std = np.std(gray, axis=1)
    
    threshold = 5
    content_rows = np.where(row_std > threshold)[0]
    
    gaps = np.diff(content_rows)
    significant_gap = np.where(gaps > 10)[0]
    if len(significant_gap) > 0:
        start_row = content_rows[significant_gap[0] + 1]
    else:
        start_row = content_rows[0]
    
    start_row = start_row + 5 # add offset to move past header
    
    right_img = right_img[start_row:, :]
    
    gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    
    content_img = right_img[y:y+h, x:x+w]
    
    square_size = min(content_img.shape[0], content_img.shape[1])
    
    center_y = content_img.shape[0] // 2
    center_x = content_img.shape[1] // 2
    
    half_size = square_size // 2
    start_y = max(0, center_y - half_size)
    start_x = max(0, center_x - half_size)
    
    final_img = content_img[start_y:start_y+square_size, start_x:start_x+square_size]
    
    return final_img

def create_comparison_diagram(base_path, img_index, output_filename='algo_comp_diagram.png'):
    image_pairs = find_image_pairs(base_path, img_index)
    
    images = {}
    got_ground_truth = False
    comp_img_path = None
    for name, paths in image_pairs.items():
        try:
            metrics = get_metrics_from_npy(paths["folder_path"], img_index)
            
            images[name] = {
                "path": paths["reconstruction"],
                "metrics": metrics
            }

            if not got_ground_truth:
                comp_img_path = paths["comparison"]
                got_ground_truth = True

        except Exception as e:
            print(f"Warning: Failed to process {name}: {str(e)}")
            continue
    
    if not images:
        raise ValueError("No valid images were processed")
    
    if not comp_img_path:
        raise ValueError("Comparison image not processed properly")
    
    _, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()
    gt_img = extract_right_image(comp_img_path) # get the ground truth image
    axes[0].imshow(gt_img, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Ground Truth", fontsize=10, pad=5, weight='bold')
    for idx, (name, details) in enumerate(images.items()):
        img = mpimg.imread(details["path"])
        gray = rgb2gray(img)
        ax = axes[idx + 1] # +1 b/c ground truth plotted
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
    parser.add_argument('index', type=int, help='Image index to process')
    parser.add_argument('--path', type=str, default='./', help='Base path containing algorithm folders')
    parser.add_argument('--output', type=str, default='algo_comp_diagram.png', 
                      help='Output filename (default: algo_comp_diagram.png)')
    
    args = parser.parse_args()
    
    try:
        create_comparison_diagram(args.path, args.index, args.output)
        print(f"Successfully created comparison diagram: {args.output}")
    except Exception as e:
        print(f"Error: {e}")
