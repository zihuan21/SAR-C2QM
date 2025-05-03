import torch
import numpy as np
import cv2

from ldm.models.autoencoder import AutoencoderKL

from SAR_C2QM.data import data_trans_tools

class AutoencoderKL_RHV_bs(AutoencoderKL):
    def __init__(self, data_name=None, *args, **kwargs):
        self.data_name = data_name
        super().__init__(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return opt_ae

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            # assert x.shape[1] == 9
            # assert xrec.shape[1] == 9
            if self.data_name == "img_XQ":
                XF_x = x
                XF_xrec = xrec
                x_show = self.XQ_9_to_rgb(XF_x)
                xrec_show = self.XQ_9_to_rgb(XF_xrec)

            # log["samples"] = self.imgXF_to_rgb(self.decode(torch.randn_like(posterior.sample())))
            log["reconstructions"] = xrec_show
            log["inputs"] = x_show
        return log

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
    

class AutoencoderKL_RHV_imp(AutoencoderKL_RHV_bs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            # assert x.shape[1] == 9
            # assert xrec.shape[1] == 9
            if self.data_name == "img_XL":
                XF_x = data_trans_tools.trans_imgXL_12_to_imgC3_3x3_pt(x)
                XF_x = data_trans_tools.trans_imgC3_3x3_to_imgXF_12_pt(XF_x)
                XF_xrec = data_trans_tools.trans_imgXL_12_to_imgC3_3x3_pt(xrec)
                XF_xrec = data_trans_tools.trans_imgC3_3x3_to_imgXF_12_pt(XF_xrec)
                x_show = self.XQ_12_to_rgb(XF_x)
                xrec_show = self.XQ_12_to_rgb(XF_xrec)

            # log["samples"] = self.imgXF_to_rgb(self.decode(torch.randn_like(posterior.sample())))
            log["reconstructions"] = xrec_show
            log["inputs"] = x_show
        return log
    
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