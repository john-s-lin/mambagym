import numpy as np
import argparse
import os

def compute_results(data_dict):
    psnrs, ssims, rmses = data_dict['psnrs'], data_dict['ssims'], data_dict['rmses']
    avg_psnr, avg_ssim, avg_rmse = np.mean(psnrs), np.mean(ssims), np.mean(rmses)
    return avg_psnr, avg_ssim, avg_rmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average results of reconstruction performance')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing .npy files')
    
    args = parser.parse_args()

    data_dict = {}
    npy_count = 0
    for filename in os.listdir(args.folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(args.folder_path, filename)
            
            if filename.startswith('psnrs'):
                data_dict['psnrs'] = np.load(file_path)
            elif filename.startswith('ssims'):
                data_dict['ssims'] = np.load(file_path)
            elif filename.startswith('rmses'):
                data_dict['rmses'] = np.load(file_path)

            npy_count += 1

    assert(npy_count == 3)
    
    try:
        avg_psnr, avg_ssim, avg_rmse = compute_results(data_dict)
        
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.2f}")
        print(f"Average RMSE: {avg_rmse:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
