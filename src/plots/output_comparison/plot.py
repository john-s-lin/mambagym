import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = {
    "Fixed-Point + RED-CNN": {
        "path": "fixed_pt_red_cnn_recon.png",
        "metrics": "PSNR: 29.76 dB\nSSIM: 0.6699\nRMSE: 0.1331"
    },
    "ADMM + RED-CNN": {
        "path": "admm_red_cnn_recon.png",
        "metrics": "PSNR: 35.77 dB\nSSIM: 0.8346\nRMSE: 0.0667"
    },
    "SART": {
        "path": "sart_recon.png",
        "metrics": "PSNR: 33.56 dB\nSSIM: 0.4776\nRMSE: 0.0860"
    },
    "Fixed-Point + DenoMamba": {
        "path": "fixed_pt_deno_recon.png",
        "metrics": "PSNR: 32.80 dB\nSSIM: 0.5751\nRMSE: 0.0939"
    },
    "FBP": {
        "path": "fbp_recon.png",
        "metrics": "PSNR: 21.76 dB\nSSIM: 0.0897\nRMSE: 0.3344"
    },
    "P3 + DenoMamba": {
        "path": "p3_deno_recon.png",
        "metrics": "PSNR: 30.28 dB\nSSIM: 0.4646\nRMSE: 0.1254"
    },
    "ADMM + DenoMamba": {
        "path": "admm_deno_recon.png",
        "metrics": "PSNR: 30.14 dB\nSSIM: 0.4457\nRMSE: 0.1275"
    },
    "ADMM TV": {
        "path": "admm_tv_recon.png",
        "metrics": "PSNR: 28.75 dB\nSSIM: 0.2693\nRMSE: 0.1495"
    }
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, (name, details) in enumerate(images.items()):
    img = mpimg.imread(details["path"])
    ax = axes[idx // 4, idx % 4]
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(name, fontsize=10, pad=5, weight='bold')
    ax.text(0.5, -0.15, details["metrics"], fontsize=8, ha='center', transform=ax.transAxes)

plt.tight_layout(h_pad=3, rect=[0, 0, 1, 1])
plt.savefig('algo_comp_diagram.png')
