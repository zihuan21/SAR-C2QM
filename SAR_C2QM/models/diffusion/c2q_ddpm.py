import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

from ldm.models.diffusion.ddpm import LatentDiffusion

from SAR_C2QM.data import data_trans_tools


class C2QLDM_RHV_bs(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=False, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        DDPM.register_schedule(self, given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = False

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """反向扩散：在时间步t下，结合条件cond，预测噪声(eps)或原始无噪声图像(x0)（潜在空间中）"""

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            # cond_cache = [cond["modal_1"], cond["modal_2"]]
            # key = (
            #     "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            # )
            # cond_k = {key: cond_cache}
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond_k = {key: cond}

        x_recon = self.model(x_noisy, t, **cond_k)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    
    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,  # 用于指定条件的键
        return_original_cond=False,
        bs=None,
    ):
        x = super(LatentDiffusion, self).get_input(batch, k)  # 调用祖父类的方法
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            xc = {}
            for ck in cond_key:
                if ck in batch:
                    xc[ck] = super(LatentDiffusion, self).get_input(batch, ck).to(self.device)
                else:
                    raise ValueError(f"Key {ck} not found in batch")
            # xc = super(LatentDiffusion, self).get_input(batch, cond_key).to(self.device)

            # 若条件编码器不可训练或强制编码条件
            if not self.cond_stage_trainable or force_c_encode:
                c = self.get_learned_conditioning(xc)
            else:
                c = xc
            if bs is not None:
                if isinstance(c, list):
                    c = [ck[:bs] for ck in c]
                elif isinstance(c, dict):
                    c = [c["modal_1"][:bs], c["modal_2"][:bs]]  # log_images方法中需要
                else:
                    c = c[:bs]

        else:
            c = None
            xc = None
        out = [z, c]  # 输入与条件的潜在表示
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample=False,
        ddim_steps=200,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=False,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        **kwargs,
    ):

        log = dict()
        z, c, x_QL, x_QL_rec, x_cond = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )
        N = min(x_QL.shape[0], N)
        n_row = min(x_QL.shape[0], n_row)
        if self.first_stage_key == "img_XQ":
            pass
        if self.first_stage_key == "img_XL":
            x_QL = data_trans_tools.trans_imgXL_9_to_imgXF_9_pt(x_QL)
            x_QL_rec = data_trans_tools.trans_imgXL_9_to_imgXF_9_pt(x_QL_rec)
        log["obj_quad_pol"] = self.XQ_9_to_rgb(x_QL)  # 目标图像
        log["obj_quad_pol_rec"] = self.XQ_9_to_rgb(x_QL_rec)  # 自编码器输出的目标图像
        log["cond_comp_pol"] = self.XC_4_to_rgb(x_cond["modal_1"])  # 条件图像
        if "img_geoInfo" in x_cond:
            log["cond_geo"] = self.Xgeo_to_rgb(x_cond["modal_2"])
        # log["cond_comp_pol"] = self.XC_to_rgb(x_cond)  # 条件图像

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    x_QL_noisy = self.decode_first_stage(z_noisy)
                    if self.first_stage_key == "img_XQ":
                        pass
                    if self.first_stage_key == "img_XL":
                        x_QL_noisy = data_trans_tools.trans_imgXL_9_to_imgXF_9_pt(x_QL_noisy)
                    diffusion_row.append(self.XQ_9_to_rgb(x_QL_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_process"] = (
                diffusion_grid  # 在扩散过程中不同时间步的图像噪声添加情况
            )

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(
                    c,
                    shape=(self.channels, self.image_size, self.image_size),
                    batch_size=N,
                )
            prog_row = self._get_denoise_row_from_list(
                progressives, desc="Progressive Generation"
            )
            log["reconstruction_process"] = (
                prog_row  # 展示了在扩散模型逐步去噪过程中，不同时间步生成的中间图像结果
            )

            x_QL_img = self.decode_first_stage(img)
            if self.first_stage_key == "img_XQ":
                pass
            if self.first_stage_key == "img_XL":
                x_QL_img = data_trans_tools.trans_imgXL_9_to_imgXF_9_pt(x_QL_img)
            log["reconstruction_result"] = self.XQ_9_to_rgb(
                x_QL_img
            )  # 通过反向扩散过程从纯噪声逐步去噪后得到的最终图像

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def _get_denoise_row_from_list(
        self, samples, desc="", force_no_decoder_quantization=False
    ):
        """展示一系列去噪过程的解码图像"""
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            x_QL_zd = self.decode_first_stage(
                zd.to(self.device), force_not_quantize=force_no_decoder_quantization
            )
            if self.first_stage_key == "img_XQ":
                pass
            if self.first_stage_key == "img_XL":
                x_QL_zd = data_trans_tools.trans_imgXL_9_to_imgXF_9_pt(x_QL_zd)
            denoise_row.append(self.XQ_9_to_rgb(x_QL_zd))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def XC_4_to_rgb(self, x):
        """
        :param x: img_XC, log(abs(CC11))、log(abs(CC22))、log(abs(CC12))、angle(CC12)  减缩极化协方差矩阵元素
        """
    
        x = x.detach().cpu().numpy()
        img_C11 = np.exp(x[:, 0, :, :])
        img_C22 = np.exp(x[:, 1, :, :])
        img_C12 = np.exp(x[:, 2, :, :]) * np.exp(1j * x[:, 3, :, :])

        img_r = img_C22
        img_g = abs(img_C11 - 2 * img_C12.real + img_C22)
        img_b = img_C11

        pauli_tensor = self._generate_rgb_img(img_r, img_g, img_b)

        return pauli_tensor

    def XQ_9_to_rgb(self, x):
        """
        :param x: img_XF，B,C,H,W，log(abs(CQ11))、log(abs(CQ22))、log(abs(CQ33))、log(abs(CQ12))、log(abs(CQ13))、log(abs(CQ23))、
                                    angle(CQ12)、angle(CQ13)、angle(CQ23)
        """

        x = x.detach().cpu().numpy()
        img_C11 = np.exp(x[:, 0, :, :])
        img_C22 = np.exp(x[:, 1, :, :])
        img_C33 = np.exp(x[:, 2, :, :])
        img_C13 = np.exp(x[:, 4, :, :]) * np.exp(1j * x[:, 7, :, :])

        img_T11 = 0.5 * (img_C11 + img_C33 + 2 * img_C13.real)
        img_T22 = 0.5 * (img_C11 + img_C33 - 2 * img_C13.real)
        img_T33 = img_C22

        pauli_r = img_T22
        pauli_g = img_T33
        pauli_b = img_T11

        pauli_tensor = self._generate_rgb_img(pauli_r, pauli_g, pauli_b)

        return pauli_tensor
    
    def Xgeo_to_rgb(self, x):
        """
        :param x: img_geoInfo, [dem, 坡度, 坡向, 局部入射角]
        """
    
        x = x.detach().cpu().numpy()
        img_r = x[:, 0, :, :]
        img_g = x[:, 2, :, :]
        img_b = x[:, 3, :, :]

        pauli_tensor = self._generate_rgb_img(img_r, img_g, img_b)

        return pauli_tensor
    
    def _generate_rgb_img(self, img_r, img_g, img_b):

        pauli_r = img_r
        pauli_g = img_g
        pauli_b = img_b

        b, h, w = pauli_r.shape

        pauli_r = np.sqrt(abs(pauli_r))
        pauli_g = np.sqrt(abs(pauli_g))
        pauli_b = np.sqrt(abs(pauli_b))

        pauli_r = 20 * np.log10(pauli_r + 1)
        pauli_g = 20 * np.log10(pauli_g + 1)
        pauli_b = 20 * np.log10(pauli_b + 1)

        # 对每个通道的图像进行处理
        for i in range(b):
            pauli_r[i, :, :] = pauli_r[i, :, :] / np.max(pauli_r[i, :, :]) * 255
            pauli_g[i, :, :] = pauli_g[i, :, :] / np.max(pauli_g[i, :, :]) * 255
            pauli_b[i, :, :] = pauli_b[i, :, :] / np.max(pauli_b[i, :, :]) * 255

        pauli_r = np.floor(pauli_r).astype(np.uint8)
        pauli_g = np.floor(pauli_g).astype(np.uint8)
        pauli_b = np.floor(pauli_b).astype(np.uint8)

        for i in range(b):
            pauli_r[i, :, :] = cv2.equalizeHist(pauli_r[i, :, :])
            pauli_g[i, :, :] = cv2.equalizeHist(pauli_g[i, :, :])
            pauli_b[i, :, :] = cv2.equalizeHist(pauli_b[i, :, :])

        # 将 numpy 数组转换为 PyTorch 张量
        pauli_r_tensor = torch.tensor(pauli_r, dtype=torch.float32)
        pauli_g_tensor = torch.tensor(pauli_g, dtype=torch.float32)
        pauli_b_tensor = torch.tensor(pauli_b, dtype=torch.float32)
        # 使用 torch.stack 将张量在新的维度上堆叠
        pauli_tensor = torch.stack(
            [pauli_r_tensor, pauli_g_tensor, pauli_b_tensor], dim=1
        )
        pauli_tensor = pauli_tensor / 255
        pauli_tensor = pauli_tensor.to(self.device)

        return pauli_tensor
    
class C2QLDM_RHV_imp(C2QLDM_RHV_bs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample=False,
        ddim_steps=200,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=False,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        **kwargs,
    ):

        log = dict()
        z, c, x_QL, x_QL_rec, x_cond = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )
        N = min(x_QL.shape[0], N)
        n_row = min(x_QL.shape[0], n_row)
        if self.first_stage_key == "img_XQ":
            pass
        if self.first_stage_key == "img_XL":
            x_QL = data_trans_tools.trans_imgXL_12_to_imgXF_12_pt(x_QL)
            x_QL_rec = data_trans_tools.trans_imgXL_12_to_imgXF_12_pt(x_QL_rec)
        log["obj_quad_pol"] = self.XQ_12_to_rgb(x_QL)  # 目标图像
        log["obj_quad_pol_rec"] = self.XQ_12_to_rgb(x_QL_rec)  # 自编码器输出的目标图像
        log["cond_comp_pol"] = self.XC_5_to_rgb(x_cond["modal_1"])  # 条件图像
        if "img_geoInfo" in x_cond:
            log["cond_geo"] = self.Xgeo_to_rgb(x_cond["modal_2"])
        # log["cond_comp_pol"] = self.XC_to_rgb(x_cond)  # 条件图像

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    x_QL_noisy = self.decode_first_stage(z_noisy)
                    if self.first_stage_key == "img_XQ":
                        pass
                    if self.first_stage_key == "img_XL":
                        x_QL_noisy = data_trans_tools.trans_imgXL_12_to_imgXF_12_pt(x_QL_noisy)
                    diffusion_row.append(self.XQ_12_to_rgb(x_QL_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_process"] = (
                diffusion_grid  # 在扩散过程中不同时间步的图像噪声添加情况
            )

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(
                    c,
                    shape=(self.channels, self.image_size, self.image_size),
                    batch_size=N,
                )
            prog_row = self._get_denoise_row_from_list(
                progressives, desc="Progressive Generation"
            )
            log["reconstruction_process"] = (
                prog_row  # 展示了在扩散模型逐步去噪过程中，不同时间步生成的中间图像结果
            )

            x_QL_img = self.decode_first_stage(img)
            if self.first_stage_key == "img_XQ":
                pass
            if self.first_stage_key == "img_XL":
                x_QL_img = data_trans_tools.trans_imgXL_12_to_imgXF_12_pt(x_QL_img)
            log["reconstruction_result"] = self.XQ_12_to_rgb(
                x_QL_img
            )  # 通过反向扩散过程从纯噪声逐步去噪后得到的最终图像

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def _get_denoise_row_from_list(
        self, samples, desc="", force_no_decoder_quantization=False
    ):
        """展示一系列去噪过程的解码图像"""
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            x_QL_zd = self.decode_first_stage(
                zd.to(self.device), force_not_quantize=force_no_decoder_quantization
            )
            if self.first_stage_key == "img_XQ":
                pass
            if self.first_stage_key == "img_XL":
                x_QL_zd = data_trans_tools.trans_imgXL_12_to_imgXF_12_pt(x_QL_zd)
            denoise_row.append(self.XQ_12_to_rgb(x_QL_zd))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def XC_5_to_rgb(self, x):
        """
        :param x: img_XC, log(abs(CC11))、log(abs(CC22))、log(abs(CC12))、cos(angle(CC12))、sin(angle(CC12))  减缩极化协方差矩阵元素
        """
    
        x = x.detach().cpu().numpy()
        img_C11 = np.exp(x[:, 0, :, :])
        img_C22 = np.exp(x[:, 1, :, :])
        img_C12 = np.exp(x[:, 2, :, :]) * np.exp(1j * np.arctan2(x[:, 4, :, :], x[:, 3, :, :]))

        img_r = img_C22
        img_g = abs(img_C11 - 2 * img_C12.real + img_C22)
        img_b = img_C11

        pauli_tensor = self._generate_rgb_img(img_r, img_g, img_b)

        return pauli_tensor
    
    def XQ_12_to_rgb(self, x):
        """
        :param x: img_XF，B,C,H,W，log(abs(CQ11))、log(abs(CQ22))、log(abs(CQ33))、log(abs(CQ12))、log(abs(CQ13))、log(abs(CQ23))、
                                    cos(angle(CQ12))、sin(angle(CQ12))、cos(angle(CQ13))、sin(angle(CQ13))、cos(angle(CQ23))、sin(angle(CQ23))
        """

        x = x.detach().cpu().numpy()
        img_C11 = np.exp(x[:, 0, :, :])
        img_C22 = np.exp(x[:, 1, :, :])
        img_C33 = np.exp(x[:, 2, :, :])
        img_C13 = np.exp(x[:, 4, :, :]) * np.exp(1j * np.arctan2(x[:, 9, :, :], x[:, 8, :, :]))

        img_T11 = 0.5 * (img_C11 + img_C33 + 2 * img_C13.real)
        img_T22 = 0.5 * (img_C11 + img_C33 - 2 * img_C13.real)
        img_T33 = img_C22

        pauli_r = img_T22
        pauli_g = img_T33
        pauli_b = img_T11

        pauli_tensor = self._generate_rgb_img(pauli_r, pauli_g, pauli_b)

        return pauli_tensor
