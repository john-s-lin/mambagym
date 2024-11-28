import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from .measure import compute_measure
from .networks import RED_CNN
from .prep import printProgressBar


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.save_fig_path = args.save_fig_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_epochs = args.test_epochs
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.REDCNN = RED_CNN()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print("Use {} GPUs".format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.functional.l1_loss
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)

    def save_model(self, epoch: int):
        f = os.path.join(self.save_path, f"REDCNN_{epoch}_epoch.ckpt")
        torch.save(self.REDCNN.state_dict(), f)

    def load_model(self, epoch: int):
        f = os.path.join(self.save_path, f"REDCNN_{epoch}_epoch.ckpt")
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        if not os.path.exists(self.save_fig_path):
            os.makedirs(self.save_fig_path)
            print(f"Create path : {self.save_fig_path}")

        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title("Quarter-dose", fontsize=30)
        ax[0].set_xlabel(
            "PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
                original_result[0], original_result[1], original_result[2]
            ),
            fontsize=20,
        )
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title("Result", fontsize=30)
        ax[1].set_xlabel(
            "PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0], pred_result[1], pred_result[2]),
            fontsize=20,
        )
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title("Full-dose", fontsize=30)

        f.savefig(os.path.join(self.save_fig_path, "result_{}.png".format(fig_name)))
        plt.close()

    def train(self, last_epoch: int | None = None):
        train_losses = []
        total_iter = 0
        start_epoch = 0 if not last_epoch else last_epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iter += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size:  # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iter % self.print_iters == 0:
                    print(
                        "STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(
                            total_iter,
                            epoch,
                            self.num_epochs,
                            iter_ + 1,
                            len(self.data_loader),
                            loss.item(),
                            time.time() - start_time,
                        )
                    )
                # learning rate decay
                if total_iter % self.decay_iters == 0:
                    self.lr_decay()

            # save model every epoch or when you reach the end
            self.save_model(epoch)
            np.save(os.path.join(self.save_path, "loss_{}_epoch.npy".format(epoch)), np.array(train_losses))

    def test(self):
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_epochs)
        self.REDCNN.eval()

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (data, target) in enumerate(self.data_loader):
                condition = data.to(self.device)
                raw_prediction = self.REDCNN(condition)

                prediction = raw_prediction.data.squeeze().cpu().detach()
                img = target[0].data.squeeze().cpu().detach()

                # Convert to numpy
                # x = prediction.view(shape_, shape_).cpu().detach()
                # y = y.view(shape_, shape_).cpu().detach()
                # pred = pred.view(shape_, shape_).cpu().detach()

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(condition, img, prediction, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(condition, img, prediction, i, original_result, pred_result)

                printProgressBar(
                    i, len(self.data_loader), prefix="Compute measurements ..", suffix="Complete", length=25
                )
            print("\n")
            print(
                "Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}".format(
                    ori_psnr_avg / len(self.data_loader),
                    ori_ssim_avg / len(self.data_loader),
                    ori_rmse_avg / len(self.data_loader),
                )
            )
            print("\n")
            print(
                "Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}".format(
                    pred_psnr_avg / len(self.data_loader),
                    pred_ssim_avg / len(self.data_loader),
                    pred_rmse_avg / len(self.data_loader),
                )
            )
