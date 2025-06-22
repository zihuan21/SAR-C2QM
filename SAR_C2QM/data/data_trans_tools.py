import numpy as np
import torch
from tqdm import tqdm


def trans_imgC3_6_to_imgC3_3x3_npy(img_in):
    """
    :param img_in: 输入张量，形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23
    :return: 重塑后的张量，形状为 (H, W, 3, 3)
    """
    C, H, W = img_in.shape
    assert C == 6, "输入张量的通道数必须为 6"

    img_out = np.zeros((9, H, W), dtype=np.complex64)
    img_out[0, :, :] = img_in[0, :, :]  # C11
    img_out[1, :, :] = img_in[3, :, :]  # C12
    img_out[2, :, :] = img_in[4, :, :]  # C13
    img_out[3, :, :] = np.conj(img_out[1, :, :])  # C21
    img_out[4, :, :] = img_in[1, :, :]  # C22
    img_out[5, :, :] = img_in[5, :, :]  # C23
    img_out[6, :, :] = np.conj(img_out[2, :, :])  # C31
    img_out[7, :, :] = np.conj(img_out[5, :, :])  # C32
    img_out[8, :, :] = img_in[2, :, :]  # C33

    # 重塑为 (3, 3, H, W)
    img_out_reshaped = img_out.reshape(3, 3, H, W)
    # 维度调整为 (H, W, 3, 3)
    img_out_final = img_out_reshaped.transpose(2, 3, 0, 1)

    return img_out_final


def trans_pixelC3_6_to_pixelC3_3x3_pt(img_in):
    """
    :param img_in: 输入张量，形状为 (B, 6)，通道顺序为 C11, C22, C33, C12, C13, C23
    :return: 重塑后的张量，形状为 (B, 3, 3)
    """
    B, C = img_in.shape
    assert C == 6, "输入张量的通道数必须为 6"

    img_out = torch.zeros((B, 9), dtype=torch.complex64, device=img_in.device)
    img_out[:, 0] = img_in[:, 0]  # C11
    img_out[:, 1] = img_in[:, 3]  # C12
    img_out[:, 2] = img_in[:, 4]  # C13
    img_out[:, 3] = torch.conj(img_out[:, 1])  # C21
    img_out[:, 4] = img_in[:, 1]  # C22
    img_out[:, 5] = img_in[:, 5]  # C23
    img_out[:, 6] = torch.conj(img_out[:, 2])  # C31
    img_out[:, 7] = torch.conj(img_out[:, 5])  # C32
    img_out[:, 8] = img_in[:, 2]  # C33

    # 将张量重塑为 (B, 3, 3)
    img_out = img_out.view(B, 3, 3)

    return img_out


def trans_imgXF_9_to_imgC3_6_pt(img_in):
    """
    :param img_in: 形状为 (B, 9, H, W), log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、angle(C12)、angle(C13)、angle(C23)
    :return: 形状为 (B, 6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23
    """
    B, C, H, W = img_in.shape

    img_out = torch.zeros((B, 6, H, W), dtype=torch.complex64, device=img_in.device)
    img_out[:, 0, :, :] = torch.exp(img_in[:, 0, :, :])
    img_out[:, 1, :, :] = torch.exp(img_in[:, 1, :, :])
    img_out[:, 2, :, :] = torch.exp(img_in[:, 2, :, :])
    img_out[:, 3, :, :] = torch.exp(img_in[:, 3, :, :]) * torch.exp(
        1j * img_in[:, 6, :, :]
    )
    img_out[:, 4, :, :] = torch.exp(img_in[:, 4, :, :]) * torch.exp(
        1j * img_in[:, 7, :, :]
    )
    img_out[:, 5, :, :] = torch.exp(img_in[:, 5, :, :]) * torch.exp(
        1j * img_in[:, 8, :, :]
    )

    return img_out

def trans_imgXF_12_to_imgC3_6_pt(img_in):
    """
    :param img_in: 形状为 (B, 12, H, W)，
                    log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、
                    cos(angle(C12))、sin(angle(C12))、cos(angle(C13))、sin(angle(C13))、cos(angle(C23)、sin(angle(C23))
    :return: 形状为 (B, 6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23
    """
    B, C, H, W = img_in.shape

    img_out = torch.zeros((B, 6, H, W), dtype=torch.complex64, device=img_in.device)
    img_out[:, 0, :, :] = torch.exp(img_in[:, 0, :, :])
    img_out[:, 1, :, :] = torch.exp(img_in[:, 1, :, :])
    img_out[:, 2, :, :] = torch.exp(img_in[:, 2, :, :])
    img_out[:, 3, :, :] = torch.exp(img_in[:, 3, :, :]) * torch.exp(
        1j * torch.atan2(img_in[:, 7, :, :], img_in[:, 6, :, :])
    )
    img_out[:, 4, :, :] = torch.exp(img_in[:, 4, :, :]) * torch.exp(
        1j * torch.atan2(img_in[:, 9, :, :], img_in[:, 8, :, :])
    )
    img_out[:, 5, :, :] = torch.exp(img_in[:, 5, :, :]) * torch.exp(
        1j * torch.atan2(img_in[:, 11, :, :], img_in[:, 10, :, :])
    )

    return img_out


def trans_imgC3_3x3_to_imgC3_6_pt(img_in):
    """
    :param img_in: 形状为 (B, H, W, 3, 3)
    :return: 形状为 (B, 6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23
    """
    B, H, W, _, _ = img_in.shape
    assert img_in.shape[-2:] == (3, 3), "输入张量的最后两个维度必须为 (3, 3)"

    img_out = torch.zeros((B, 6, H, W), dtype=torch.complex64, device=img_in.device)

    # 提取对角线元素
    img_out[:, 0, :, :] = img_in[:, :, :, 0, 0]  # C11
    img_out[:, 1, :, :] = img_in[:, :, :, 1, 1]  # C22
    img_out[:, 2, :, :] = img_in[:, :, :, 2, 2]  # C33

    # 提取上三角部分
    img_out[:, 3, :, :] = img_in[:, :, :, 0, 1]  # C12
    img_out[:, 4, :, :] = img_in[:, :, :, 0, 2]  # C13
    img_out[:, 5, :, :] = img_in[:, :, :, 1, 2]  # C23

    return img_out


def trans_imgC3_3x3_to_imgXF_9_pt(img_C3):
    """
    :param img_C3: 形状为 (B, H, W, 3, 3)
    :return: 形状为 (B, 9, H, W)，log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、angle(C12)、angle(C13)、angle(C23)
    """
    B, H, W, _, _ = img_C3.shape

    img_out = torch.zeros((B, 9, H, W), dtype=torch.float32, device=img_C3.device)

    # 对角元素的对数幅度
    img_out[:, 0, :, :] = torch.log(torch.abs(img_C3[:, :, :, 0, 0]) + 1e-10)  # log(abs(C11))
    img_out[:, 1, :, :] = torch.log(torch.abs(img_C3[:, :, :, 1, 1]) + 1e-10)  # log(abs(C22))
    img_out[:, 2, :, :] = torch.log(torch.abs(img_C3[:, :, :, 2, 2]) + 1e-10)  # log(abs(C33))
    img_out[:, 3, :, :] = torch.log(torch.abs(img_C3[:, :, :, 0, 1]) + 1e-10)  # log(abs(C12))
    img_out[:, 4, :, :] = torch.log(torch.abs(img_C3[:, :, :, 0, 2]) + 1e-10)  # log(abs(C13))
    img_out[:, 5, :, :] = torch.log(torch.abs(img_C3[:, :, :, 1, 2]) + 1e-10)  # log(abs(C23))
    img_out[:, 6, :, :] = torch.angle(img_C3[:, :, :, 0, 1])  # angle(C12)
    img_out[:, 7, :, :] = torch.angle(img_C3[:, :, :, 0, 2])  # angle(C13)
    img_out[:, 8, :, :] = torch.angle(img_C3[:, :, :, 1, 2])  # angle(C23)

    return img_out


def trans_imgC3_3x3_to_imgXF_12_pt(img_C3):
    """
    :param img_C3: 形状为 (B, H, W, 3, 3)
    :return: 形状为 (B, 12, H, W)，
                log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、
                cos(angle(C12))、sin(angle(C12))、cos(angle(C13))、sin(angle(C13))、cos(angle(C23))、sin(angle(C23))
    """
    B, H, W, _, _ = img_C3.shape

    img_out = torch.zeros((B, 12, H, W), dtype=torch.float32, device=img_C3.device)

    # 对角元素的对数幅度
    img_out[:, 0, :, :] = torch.log(torch.abs(img_C3[:, :, :, 0, 0]) + 1e-10)  # log(abs(C11))
    img_out[:, 1, :, :] = torch.log(torch.abs(img_C3[:, :, :, 1, 1]) + 1e-10)  # log(abs(C22))
    img_out[:, 2, :, :] = torch.log(torch.abs(img_C3[:, :, :, 2, 2]) + 1e-10)  # log(abs(C33))
    img_out[:, 3, :, :] = torch.log(torch.abs(img_C3[:, :, :, 0, 1]) + 1e-10)  # log(abs(C12))
    img_out[:, 4, :, :] = torch.log(torch.abs(img_C3[:, :, :, 0, 2]) + 1e-10)  # log(abs(C13))
    img_out[:, 5, :, :] = torch.log(torch.abs(img_C3[:, :, :, 1, 2]) + 1e-10)  # log(abs(C23))

    img_out[:, 6, :, :] = torch.cos(torch.angle(img_C3[:, :, :, 0, 1]))  # angle(C12)
    img_out[:, 7, :, :] = torch.sin(torch.angle(img_C3[:, :, :, 0, 1]))  # angle(C12)
    img_out[:, 8, :, :] = torch.cos(torch.angle(img_C3[:, :, :, 0, 2]))  # angle(C13)
    img_out[:, 9, :, :] = torch.sin(torch.angle(img_C3[:, :, :, 0, 2]))  # angle(C13)
    img_out[:, 10, :, :] = torch.cos(torch.angle(img_C3[:, :, :, 1, 2]))  # angle(C23)
    img_out[:, 11, :, :] = torch.sin(torch.angle(img_C3[:, :, :, 1, 2]))  # angle(C23)

    return img_out


def trans_imgXL_9_to_imgC3_3x3_pt(imgCholesky):
    """
    将Cholesky分解下三角矩阵转换为C3
    :param imgCholesky: 形状为 (B, 9, H, W)，通道顺序为   
                log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
                angle(L[1,0])、angle(L[2,0])、angle(L[2,1])
    :return: 形状为 (B, H, W, 3, 3)
    """

    B, C, H, W = imgCholesky.shape

    img_L = torch.zeros((B, 9, H, W), dtype=torch.complex64, device=imgCholesky.device)
    img_L[:, 0, :, :] = torch.exp(imgCholesky[:, 0, :, :])
    img_L[:, 3, :, :] = torch.exp(imgCholesky[:, 1, :, :]) * torch.exp(
        1j * imgCholesky[:, 6, :, :]
    )
    img_L[:, 4, :, :] = torch.exp(imgCholesky[:, 2, :, :])
    img_L[:, 6, :, :] = torch.exp(imgCholesky[:, 3, :, :]) * torch.exp(
        1j * imgCholesky[:, 7, :, :]
    )
    img_L[:, 7, :, :] = torch.exp(imgCholesky[:, 4, :, :]) * torch.exp(
        1j * imgCholesky[:, 8, :, :]
    )
    img_L[:, 8, :, :] = torch.exp(imgCholesky[:, 5, :, :])

    # 将张量重塑为 (B, 3, 3, H, W)
    img_L = img_L.view(B, 3, 3, H, W)
    # 维度调整为 (B, H, W, 3, 3)
    img_L = img_L.permute(0, 3, 4, 1, 2)

    img_C3_3x3 = torch.matmul(img_L, img_L.conj().transpose(-2, -1))

    return img_C3_3x3


def trans_imgXL_12_to_imgC3_3x3_pt(imgCholesky):
    """
    将Cholesky分解下三角矩阵转换为C3
    :param imgCholesky: 形状为 (B, 12, H, W)，通道顺序为   
                log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
                cos(angle(L[1,0]))、sin(angle(L[1,0]))、cos(angle(L[2,0]))、sin(angle(L[2,0]))、cos(angle(L[2,1]))、sin(angle(L[2,1]))
    :return: 形状为 (B, H, W, 3, 3)
    """

    B, C, H, W = imgCholesky.shape

    img_L = torch.zeros((B, 9, H, W), dtype=torch.complex64, device=imgCholesky.device)
    img_L[:, 0, :, :] = torch.exp(imgCholesky[:, 0, :, :])
    img_L[:, 3, :, :] = torch.exp(imgCholesky[:, 1, :, :]) * torch.exp(
        1j * torch.atan2(imgCholesky[:, 7, :, :], imgCholesky[:, 6, :, :]))
    img_L[:, 4, :, :] = torch.exp(imgCholesky[:, 2, :, :])
    img_L[:, 6, :, :] = torch.exp(imgCholesky[:, 3, :, :]) * torch.exp(
        1j * torch.atan2(imgCholesky[:, 9, :, :], imgCholesky[:, 8, :, :]))
    img_L[:, 7, :, :] = torch.exp(imgCholesky[:, 4, :, :]) * torch.exp(
        1j * torch.atan2(imgCholesky[:, 11, :, :], imgCholesky[:, 10, :, :]))
    img_L[:, 8, :, :] = torch.exp(imgCholesky[:, 5, :, :])

    # 将张量重塑为 (B, 3, 3, H, W)
    img_L = img_L.view(B, 3, 3, H, W)
    # 维度调整为 (B, H, W, 3, 3)
    img_L = img_L.permute(0, 3, 4, 1, 2)

    img_C3_3x3 = torch.matmul(img_L, img_L.conj().transpose(-2, -1))

    return img_C3_3x3


def trans_imgXL_9_to_imgXF_9_pt(imgXL):
    """
    将Cholesky分解下三角矩阵元素转换为C3矩阵元素
    :param imgXL: 形状为 (B, 9, H, W)，通道顺序为   
            log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
            angle(L[1,0])、angle(L[2,0])、angle(L[2,1])
    :return: 形状为 (B, 9, H, W)，
            log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、
            angle(C12)、angle(C13)、angle(C23)
    """

    B, C, H, W = imgXL.shape

    img_L = torch.zeros((B, 9, H, W), dtype=torch.complex64, device=imgXL.device)
    img_L[:, 0, :, :] = torch.exp(imgXL[:, 0, :, :])
    img_L[:, 3, :, :] = torch.exp(imgXL[:, 1, :, :]) * torch.exp(1j * imgXL[:, 6, :, :])
    img_L[:, 4, :, :] = torch.exp(imgXL[:, 2, :, :])
    img_L[:, 6, :, :] = torch.exp(imgXL[:, 3, :, :]) * torch.exp(1j * imgXL[:, 7, :, :])
    img_L[:, 7, :, :] = torch.exp(imgXL[:, 4, :, :]) * torch.exp(1j * imgXL[:, 8, :, :])
    img_L[:, 8, :, :] = torch.exp(imgXL[:, 5, :, :])

    # 将张量重塑为 (B, 3, 3, H, W)
    img_L = img_L.view(B, 3, 3, H, W)
    # 维度调整为 (B, H, W, 3, 3)
    img_L = img_L.permute(0, 3, 4, 1, 2)

    img_C3_3x3 = torch.matmul(img_L, img_L.conj().transpose(-2, -1))
    imgXF = trans_imgC3_3x3_to_imgXF_9_pt(img_C3_3x3)

    return imgXF


def trans_imgXL_12_to_imgXF_12_pt(imgXL):
    """
    将Cholesky分解下三角矩阵元素转换为C3矩阵元素
    :param imgXL: 形状为 (B, 12, H, W)，通道顺序为   
            log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
            cos(angle(L[1,0]))、sin(angle(L[1,0]))、cos(angle(L[2,0]))、sin(angle(L[2,0]))、cos(angle(L[2,1]))、sin(angle(L[2,1]))
    :return: 形状为 (B, 12, H, W)，
            log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、
            cos(angle(C12))、sin(angle(C12))、cos(angle(C13))、sin(angle(C13))、cos(angle(C23))、sin(angle(C23))
    """

    B, C, H, W = imgXL.shape

    img_L = torch.zeros((B, 9, H, W), dtype=torch.complex64, device=imgXL.device)
    img_L[:, 0, :, :] = torch.exp(imgXL[:, 0, :, :])
    img_L[:, 3, :, :] = torch.exp(imgXL[:, 1, :, :]) * torch.exp(
        1j * torch.atan2(imgXL[:, 7, :, :], imgXL[:, 6, :, :]))
    img_L[:, 4, :, :] = torch.exp(imgXL[:, 2, :, :])
    img_L[:, 6, :, :] = torch.exp(imgXL[:, 3, :, :]) * torch.exp(
        1j * torch.atan2(imgXL[:, 9, :, :], imgXL[:, 8, :, :]))
    img_L[:, 7, :, :] = torch.exp(imgXL[:, 4, :, :]) * torch.exp(
        1j * torch.atan2(imgXL[:, 11, :, :], imgXL[:, 10, :, :]))
    img_L[:, 8, :, :] = torch.exp(imgXL[:, 5, :, :])

    # 将张量重塑为 (B, 3, 3, H, W)
    img_L = img_L.view(B, 3, 3, H, W)
    # 维度调整为 (B, H, W, 3, 3)
    img_L = img_L.permute(0, 3, 4, 1, 2)

    img_C3_3x3 = torch.matmul(img_L, img_L.conj().transpose(-2, -1))
    imgXF = trans_imgC3_3x3_to_imgXF_12_pt(img_C3_3x3)

    return imgXF


def trans_imgXF_9_to_imgC3_6_npy(img_XQ):
    """
    :param img_XQ: 形状为 (9, H, W)，
            log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、
            angle(C12)、angle(C13)、angle(C23)
    :return: 形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23，复数numpy数组
    """
    C, H, W = img_XQ.shape

    img_out = np.zeros([6, H, W], dtype=np.complex64)
    img_out[0, ...] = np.exp(img_XQ[0, ...])
    img_out[1, ...] = np.exp(img_XQ[1, ...])
    img_out[2, ...] = np.exp(img_XQ[2, ...])
    img_out[3, ...] = np.exp(img_XQ[3, ...]) * np.exp(1j * img_XQ[6, ...])
    img_out[4, ...] = np.exp(img_XQ[4, ...]) * np.exp(1j * img_XQ[7, ...])
    img_out[5, ...] = np.exp(img_XQ[5, ...]) * np.exp(1j * img_XQ[8, ...])

    return img_out


def trans_imgXF_12_to_imgC3_6_npy(img_XQ):
    """
    :param img_XQ: 形状为 (12, H, W)，
            log(abs(C11))、log(abs(C22))、log(abs(C33))、log(abs(C12))、log(abs(C13))、log(abs(C23))、
            cos(angle(C12))、sin(angle(C12))、cos(angle(C13))、sin(angle(C13))、cos(angle(C23))、sin(angle(C23))
    :return: 形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23，复数numpy数组
    """
    C, H, W = img_XQ.shape

    img_out = np.zeros([6, H, W], dtype=np.complex64)
    img_out[0, ...] = np.exp(img_XQ[0, ...])
    img_out[1, ...] = np.exp(img_XQ[1, ...])
    img_out[2, ...] = np.exp(img_XQ[2, ...])
    img_out[3, ...] = np.exp(img_XQ[3, ...]) * np.exp(
        1j * np.arctan2(img_XQ[7, ...], img_XQ[6, ...]))
    img_out[4, ...] = np.exp(img_XQ[4, ...]) * np.exp(
        1j * np.arctan2(img_XQ[9, ...], img_XQ[8, ...]))
    img_out[5, ...] = np.exp(img_XQ[5, ...]) * np.exp(
        1j * np.arctan2(img_XQ[11, ...], img_XQ[10, ...]))

    return img_out


def trans_imgC3_6_to_imgXL_9_npy(img_in, device="cpu"):
    """
    :param img_in: 形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23，复数numpy数组
    :param device: cuda, cpu
    :return: 形状为 (9, H, W)，C3的Cholesky分解下三角矩阵元素，通道顺序为：
                log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
                angle(L[1,0])、angle(L[2,0])、angle(L[2,1])
    """
    img_C3_3x3 = trans_imgC3_6_to_imgC3_3x3_npy(img_in)
    H = img_in.shape[1]
    W = img_in.shape[2]

    img_C3_3x3_pt = torch.from_numpy(img_C3_3x3)
    img_C3_3x3_pt = img_C3_3x3_pt.contiguous().view(H * W, 3, 3)
    img_C3_3x3_pt = img_C3_3x3_pt.to(device)
    img_L_flat = torch.zeros_like(img_C3_3x3_pt, dtype=torch.complex64)
    batch_size = 200
    inter_list = tqdm(
        range(0, img_C3_3x3_pt.shape[0], batch_size), desc="C3->Cholesky..."
    )
    for i in inter_list:
        # 获取当前批次的结束索引，确保不会超出数组大小
        end_index = min(i + batch_size, img_C3_3x3_pt.shape[0])

        batch_C3 = img_C3_3x3_pt[i:end_index, :, :]

        epsilon = 1e-8
        eye = torch.eye(3, device=device).unsqueeze(0).expand(batch_C3.shape[0], -1, -1)
        batch_C3 = batch_C3 + epsilon * eye
        
        batch_L = torch.linalg.cholesky(batch_C3)  # 下三角矩阵
        img_L_flat[i:end_index, :, :] = batch_L
    img_L = img_L_flat.view(H, W, 3, 3)

    if img_L.is_cuda:
        img_L = img_L.cpu()
    img_L_npy = img_L.numpy()

    img_C3_Cholesky = np.zeros([9, H, W], dtype=np.float32)  # cholesky分解下三角元素
    img_C3_Cholesky[0, :, :] = np.log(img_L_npy[:, :, 0, 0].real)
    img_C3_Cholesky[1, :, :] = np.log(abs(img_L_npy[:, :, 1, 0]))
    img_C3_Cholesky[2, :, :] = np.log(img_L_npy[:, :, 1, 1].real)
    img_C3_Cholesky[3, :, :] = np.log(abs(img_L_npy[:, :, 2, 0]))
    img_C3_Cholesky[4, :, :] = np.log(abs(img_L_npy[:, :, 2, 1]))
    img_C3_Cholesky[5, :, :] = np.log(img_L_npy[:, :, 2, 2].real)
    img_C3_Cholesky[6, :, :] = np.arctan2(
        img_L_npy[:, :, 1, 0].imag, img_L_npy[:, :, 1, 0].real
    )
    img_C3_Cholesky[7, :, :] = np.arctan2(
        img_L_npy[:, :, 2, 0].imag, img_L_npy[:, :, 2, 0].real
    )
    img_C3_Cholesky[8, :, :] = np.arctan2(
        img_L_npy[:, :, 2, 1].imag, img_L_npy[:, :, 2, 1].real
    )

    return img_C3_Cholesky


def trans_imgC3_6_to_imgXL_12_npy(img_in, device="cpu"):
    """
    :param img_in: 形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23，复数numpy数组
    :param device: cuda, cpu
    :return: 形状为 (12, H, W)，C3的Cholesky分解下三角矩阵元素，通道顺序为：
                log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
                cos(angle(L[1,0]))、sin(angle(L[1,0]))、cos(angle(L[2,0]))、sin(angle(L[2,0]))、cos(angle(L[2,1]))、sin(angle(L[2,1]))
    """
    img_C3_3x3 = trans_imgC3_6_to_imgC3_3x3_npy(img_in)
    H = img_in.shape[1]
    W = img_in.shape[2]

    img_C3_3x3_pt = torch.from_numpy(img_C3_3x3)
    img_C3_3x3_pt = img_C3_3x3_pt.contiguous().view(H * W, 3, 3)
    img_C3_3x3_pt = img_C3_3x3_pt.to(device)
    img_L_flat = torch.zeros_like(img_C3_3x3_pt, dtype=torch.complex64)
    batch_size = 200
    inter_list = tqdm(
        range(0, img_C3_3x3_pt.shape[0], batch_size), desc="C3->Cholesky..."
    )
    for i in inter_list:
        # 获取当前批次的结束索引，确保不会超出数组大小
        end_index = min(i + batch_size, img_C3_3x3_pt.shape[0])

        batch_C3 = img_C3_3x3_pt[i:end_index, :, :]

        epsilon = 1e-8
        eye = torch.eye(3, device=device).unsqueeze(0).expand(batch_C3.shape[0], -1, -1)
        batch_C3 = batch_C3 + epsilon * eye
        
        batch_L = torch.linalg.cholesky(batch_C3)  # 下三角矩阵
        img_L_flat[i:end_index, :, :] = batch_L
    img_L = img_L_flat.view(H, W, 3, 3)

    if img_L.is_cuda:
        img_L = img_L.cpu()
    img_L_npy = img_L.numpy()

    img_C3_Cholesky = np.zeros([12, H, W], dtype=np.float32)  # cholesky分解下三角元素
    img_C3_Cholesky[0, :, :] = np.log(img_L_npy[:, :, 0, 0].real)
    img_C3_Cholesky[1, :, :] = np.log(abs(img_L_npy[:, :, 1, 0]))
    img_C3_Cholesky[2, :, :] = np.log(img_L_npy[:, :, 1, 1].real)
    img_C3_Cholesky[3, :, :] = np.log(abs(img_L_npy[:, :, 2, 0]))
    img_C3_Cholesky[4, :, :] = np.log(abs(img_L_npy[:, :, 2, 1]))
    img_C3_Cholesky[5, :, :] = np.log(img_L_npy[:, :, 2, 2].real)
    cache = np.arctan2(img_L_npy[:, :, 1, 0].imag, img_L_npy[:, :, 1, 0].real)
    cache_x = np.cos(cache)
    cache_y = np.sin(cache)
    img_C3_Cholesky[6, :, :] = cache_x
    img_C3_Cholesky[7, :, :] = cache_y
    cache = np.arctan2(img_L_npy[:, :, 2, 0].imag, img_L_npy[:, :, 2, 0].real)
    cache_x = np.cos(cache)
    cache_y = np.sin(cache)
    img_C3_Cholesky[8, :, :] = cache_x
    img_C3_Cholesky[9, :, :] = cache_y
    cache = np.arctan2(img_L_npy[:, :, 2, 1].imag, img_L_npy[:, :, 2, 1].real)
    cache_x = np.cos(cache)
    cache_y = np.sin(cache)
    img_C3_Cholesky[10, :, :] = cache_x
    img_C3_Cholesky[11, :, :] = cache_y

    return img_C3_Cholesky


def trans_imgXL_9_to_imgC3_6_npy(img_in, device="cpu"):
    """
    将Cholesky分解下三角矩阵转换为C3
    :param imgCholesky: 形状为 (9, H, W)，通道顺序为：
            log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
            angle(L[1,0])、angle(L[2,0])、angle(L[2,1])
    :param device: cuda, cpu
    :return: 形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23，复数numpy数组
    """
    # 获取输入形状
    _, H, W = img_in.shape

    # 初始化下三角矩阵 L，形状为 (H, W, 3, 3)
    img_L = torch.zeros((H, W, 3, 3), dtype=torch.complex64)

    # 从 img_in 中重建 L
    img_L[:, :, 0, 0] = torch.exp(torch.from_numpy(img_in[0, :, :]))  # L[0, 0]
    img_L[:, :, 1, 0] = torch.exp(torch.from_numpy(img_in[1, :, :])) * torch.exp(
        1j * torch.from_numpy(img_in[6, :, :])
    )  # L[1, 0]
    img_L[:, :, 1, 1] = torch.exp(torch.from_numpy(img_in[2, :, :]))  # L[1, 1]
    img_L[:, :, 2, 0] = torch.exp(torch.from_numpy(img_in[3, :, :])) * torch.exp(
        1j * torch.from_numpy(img_in[7, :, :])
    )  # L[2, 0]
    img_L[:, :, 2, 1] = torch.exp(torch.from_numpy(img_in[4, :, :])) * torch.exp(
        1j * torch.from_numpy(img_in[8, :, :])
    )  # L[2, 1]
    img_L[:, :, 2, 2] = torch.exp(torch.from_numpy(img_in[5, :, :]))  # L[2, 2]
    img_L = img_L.contiguous().view(H * W, 3, 3)

    # 将 img_L 转移到 GPU 上（如果有 GPU 可用）
    img_L = img_L.to(device)

    # 初始化输出数组 img_C3_6，形状为 (6, H, W)
    img_C3_3x3 = torch.zeros_like(img_L)
    batch_size = 200
    inter_list = tqdm(range(0, img_C3_3x3.shape[0], batch_size), desc="Cholesky->C3...")
    for i in inter_list:
        # 获取当前批次的结束索引，确保不会超出数组大小
        end_index = min(i + batch_size, img_C3_3x3.shape[0])

        batch_L = img_L[i:end_index, ...]
        batch_C3 = torch.matmul(
            batch_L, batch_L.conj().transpose(-2, -1)
        )  # 矩阵乘法及共轭转置
        img_C3_3x3[i:end_index, ...] = batch_C3
    img_C3_3x3 = img_C3_3x3.view(H, W, 3, 3)

    if img_C3_3x3.is_cuda:
        img_C3_3x3 = img_C3_3x3.cpu()
    img_C3_3x3 = img_C3_3x3.numpy()

    # 提取 C11, C22, C33, C12, C13, C23 并将它们存储为输出
    img_C3_out = np.zeros((6, H, W), dtype=np.complex64)
    img_C3_out[0, :, :] = img_C3_3x3[:, :, 0, 0]  # C11
    img_C3_out[1, :, :] = img_C3_3x3[:, :, 1, 1]  # C22
    img_C3_out[2, :, :] = img_C3_3x3[:, :, 2, 2]  # C33
    img_C3_out[3, :, :] = img_C3_3x3[:, :, 0, 1]  # C12
    img_C3_out[4, :, :] = img_C3_3x3[:, :, 0, 2]  # C13
    img_C3_out[5, :, :] = img_C3_3x3[:, :, 1, 2]  # C23

    return img_C3_out


def trans_imgXL_12_to_imgC3_6_npy(img_in, device="cpu"):
    """
    将Cholesky分解下三角矩阵转换为C3
    :param imgCholesky: 形状为 (12, H, W)，通道顺序为：
            log(abs(L[0,0]))、log(abs(L[1,0]))、log(abs(L[1,1]))、log(abs(L[2,0]))、log(abs(L[2,1]))、log(abs(L[2,2]))
            cos(angle(L[1,0]))、sin(angle(L[1,0]))、cos(angle(L[2,0]))、sin(angle(L[2,0]))、cos(angle(L[2,1]))、sin(angle(L[2,1]))
    :param device: cuda, cpu
    :return: 形状为 (6, H, W)，通道顺序为 C11, C22, C33, C12, C13, C23，复数numpy数组
    """
    # 获取输入形状
    _, H, W = img_in.shape

    # 初始化下三角矩阵 L，形状为 (H, W, 3, 3)
    img_L = torch.zeros((H, W, 3, 3), dtype=torch.complex64)

    # 从 img_in 中重建 L
    img_L[:, :, 0, 0] = torch.exp(torch.from_numpy(img_in[0, :, :]))  # L[0, 0]
    img_L[:, :, 1, 0] = torch.exp(torch.from_numpy(img_in[1, :, :])) * torch.exp(
        1j * torch.from_numpy(np.arctan2(img_in[7, :, :], img_in[6, :, :]))
    )  # L[1, 0]
    img_L[:, :, 1, 1] = torch.exp(torch.from_numpy(img_in[2, :, :]))  # L[1, 1]
    img_L[:, :, 2, 0] = torch.exp(torch.from_numpy(img_in[3, :, :])) * torch.exp(
        1j * torch.from_numpy(np.arctan2(img_in[9, :, :], img_in[8, :, :]))
    )  # L[2, 0]
    img_L[:, :, 2, 1] = torch.exp(torch.from_numpy(img_in[4, :, :])) * torch.exp(
        1j * torch.from_numpy(np.arctan2(img_in[11, :, :], img_in[10, :, :]))
    )  # L[2, 1]
    img_L[:, :, 2, 2] = torch.exp(torch.from_numpy(img_in[5, :, :]))  # L[2, 2]
    img_L = img_L.contiguous().view(H * W, 3, 3)

    # 将 img_L 转移到 GPU 上（如果有 GPU 可用）
    img_L = img_L.to(device)

    # 初始化输出数组 img_C3_6，形状为 (6, H, W)
    img_C3_3x3 = torch.zeros_like(img_L)
    batch_size = 200
    inter_list = tqdm(range(0, img_C3_3x3.shape[0], batch_size), desc="Cholesky->C3...")
    for i in inter_list:
        # 获取当前批次的结束索引，确保不会超出数组大小
        end_index = min(i + batch_size, img_C3_3x3.shape[0])

        batch_L = img_L[i:end_index, ...]
        batch_C3 = torch.matmul(
            batch_L, batch_L.conj().transpose(-2, -1)
        )  # 矩阵乘法及共轭转置
        img_C3_3x3[i:end_index, ...] = batch_C3
    img_C3_3x3 = img_C3_3x3.view(H, W, 3, 3)

    if img_C3_3x3.is_cuda:
        img_C3_3x3 = img_C3_3x3.cpu()
    img_C3_3x3 = img_C3_3x3.numpy()

    # 提取 C11, C22, C33, C12, C13, C23 并将它们存储为输出
    img_C3_out = np.zeros((6, H, W), dtype=np.complex64)
    img_C3_out[0, :, :] = img_C3_3x3[:, :, 0, 0]  # C11
    img_C3_out[1, :, :] = img_C3_3x3[:, :, 1, 1]  # C22
    img_C3_out[2, :, :] = img_C3_3x3[:, :, 2, 2]  # C33
    img_C3_out[3, :, :] = img_C3_3x3[:, :, 0, 1]  # C12
    img_C3_out[4, :, :] = img_C3_3x3[:, :, 0, 2]  # C13
    img_C3_out[5, :, :] = img_C3_3x3[:, :, 1, 2]  # C23

    return img_C3_out


def trans_dictT3_to_dictC3(img_T3):
    """
    T3图像转为C3图像
    """
    coeff = 1 / 2**0.5
    img_C3 = {}
    img_C3['C11'] = 0.5 * (img_T3['T11'] + img_T3['T22'] + 2 * img_T3['T12'].real)
    img_C3['C12'] = coeff * (img_T3['T13'] + img_T3['T23'])
    img_C3['C13'] = 0.5 * (img_T3['T11'] - img_T3['T22']) - 1j * img_T3['T12'].imag
    img_C3['C22'] = img_T3['T33']
    img_C3['C23'] = coeff * ((img_T3['T13'].real - img_T3['T23'].real) + 1j * (img_T3['T23'].imag - img_T3['T13'].imag))
    img_C3['C33'] = 0.5 * (img_T3['T11'] + img_T3['T22'] - 2 * img_T3['T12'].real)

    return img_C3


def trans_dictC3_to_dictT3(img_C3):
    """
    C3图像转为T3图像
    """
    img_T3 = {}
    img_T3['T11'] = 0.5 * (img_C3['C11'] + img_C3['C33'] + 2 * img_C3['C13'].real)
    img_T3['T12'] = 0.5 * (img_C3['C11'] - img_C3['C33']) - 1j * img_C3['C13'].imag
    img_T3['T13'] = 2**0.5 * 0.5 * ((img_C3['C12'].real + img_C3['C23'].real) + 1j * (img_C3['C12'].imag - img_C3['C23'].imag))
    img_T3['T22'] = 0.5 * (img_C3['C11'] + img_C3['C33'] - 2 * img_C3['C13'].real)
    img_T3['T23'] = 2**0.5 * 0.5 * ((img_C3['C12'].real - img_C3['C23'].real) + 1j * (img_C3['C12'].imag + img_C3['C23'].imag))
    img_T3['T33'] = img_C3['C22']

    return img_T3
    

def trans_imgC3_6_npy_to_dictC3(img_C3 : np.ndarray):
    """
    :param img_C3: C11, C22, C33, C12, C13, C23
    """

    out_C3 = {}
    out_C3["C11"] = img_C3[0, ...].real
    out_C3["C22"] = img_C3[1, ...].real
    out_C3["C33"] = img_C3[2, ...].real
    out_C3["C12"] = img_C3[3, ...]
    out_C3["C13"] = img_C3[4, ...]
    out_C3["C23"] = img_C3[5, ...]

    return out_C3


def trans_dictC3_to_imgC3_6_npy(img_C3):
    """
    :param img_C3: C11, C22, C33, C12, C13, C23
    """
    out_C3 = []
    out_C3.append(img_C3["C11"])
    out_C3.append(img_C3["C22"])
    out_C3.append(img_C3["C33"])
    out_C3.append(img_C3["C12"])
    out_C3.append(img_C3["C13"])
    out_C3.append(img_C3["C23"])
    out_C3 = np.array(out_C3, dtype=np.complex64)

    return out_C3


def trans_dictT3_to_imgT3_6_npy(img_T3):
    """
    :param img_T3: T11, T22, T33, T12, T13, T23
    """
    out_T3 = []
    out_T3.append(img_T3["T11"])
    out_T3.append(img_T3["T22"])
    out_T3.append(img_T3["T33"])
    out_T3.append(img_T3["T12"])
    out_T3.append(img_T3["T13"])
    out_T3.append(img_T3["T23"])
    out_T3 = np.array(out_T3, dtype=np.complex64)

    return out_T3


def trans_dictC2_to_imgXC_4_npy(img_C2):

    sig_1 = np.log(abs(img_C2["C11"]) + 1e-10)
    sig_2 = np.log(abs(img_C2["C22"]) + 1e-10)
    sig_3 = np.log(abs(img_C2["C12"]) + 1e-10)
    phi_4 = np.arctan2(img_C2["C12"].imag, img_C2["C12"].real)
    img_XC = [sig_1, sig_2, sig_3, phi_4]

    img_XC = np.array(img_XC)
    img_XC = img_XC.astype(np.float32)

    return img_XC

def trans_dictC2_to_imgXC_5_npy(img_C2):

    sig_1 = np.log(abs(img_C2["C11"]) + 1e-10)
    sig_2 = np.log(abs(img_C2["C22"]) + 1e-10)
    sig_3 = np.log(abs(img_C2["C12"]) + 1e-10)
    phi_3 = np.arctan2(img_C2["C12"].imag, img_C2["C12"].real)
    phi_3_x = np.cos(phi_3)
    phi_3_y = np.sin(phi_3)
    
    img_XC = [sig_1, sig_2, sig_3, phi_3_x, phi_3_y]

    img_XC = np.array(img_XC)
    img_XC = img_XC.astype(np.float32)

    return img_XC


def trans_dictC3_to_imgXF_9_npy(img_C3):

    sig_1 = np.log(abs(img_C3["C11"]) + 1e-10)
    sig_2 = np.log(abs(img_C3["C22"]) + 1e-10)
    sig_3 = np.log(abs(img_C3["C33"]) + 1e-10)
    sig_4 = np.log(abs(img_C3["C12"]) + 1e-10)
    sig_5 = np.log(abs(img_C3["C13"]) + 1e-10)
    sig_6 = np.log(abs(img_C3["C23"]) + 1e-10)
    phi_7 = np.arctan2(img_C3["C12"].imag, img_C3["C12"].real)
    phi_8 = np.arctan2(img_C3["C13"].imag, img_C3["C13"].real)
    phi_9 = np.arctan2(img_C3["C23"].imag, img_C3["C23"].real)
    img_XQ = [sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, phi_7, phi_8, phi_9]

    img_XQ = np.array(img_XQ)
    img_XQ = img_XQ.astype(np.float32)

    return img_XQ


def trans_dictC3_to_imgXF_ori_9_npy(img_C3):

    sig_1 = img_C3["C11"]
    sig_2 = img_C3["C22"]
    sig_3 = img_C3["C33"]
    sig_4 = abs(img_C3["C12"])
    sig_5 = abs(img_C3["C13"])
    sig_6 = abs(img_C3["C23"])
    phi_7 = np.arctan2(img_C3["C12"].imag, img_C3["C12"].real)
    phi_8 = np.arctan2(img_C3["C13"].imag, img_C3["C13"].real)
    phi_9 = np.arctan2(img_C3["C23"].imag, img_C3["C23"].real)
    img_XQ = [sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, phi_7, phi_8, phi_9]

    img_XQ = np.array(img_XQ)
    img_XQ = img_XQ.astype(np.float32)

    return img_XQ


def trans_dictC3_to_imgXF_12_npy(img_C3):

    sig_1 = np.log(abs(img_C3["C11"]) + 1e-10)
    sig_2 = np.log(abs(img_C3["C22"]) + 1e-10)
    sig_3 = np.log(abs(img_C3["C33"]) + 1e-10)
    sig_4 = np.log(abs(img_C3["C12"]) + 1e-10)
    sig_5 = np.log(abs(img_C3["C13"]) + 1e-10)
    sig_6 = np.log(abs(img_C3["C23"]) + 1e-10)
    phi_4 = np.arctan2(img_C3["C12"].imag, img_C3["C12"].real)
    phi_5 = np.arctan2(img_C3["C13"].imag, img_C3["C13"].real)
    phi_6 = np.arctan2(img_C3["C23"].imag, img_C3["C23"].real)
    phi_4_x = np.cos(phi_4)
    phi_4_y = np.sin(phi_4)
    phi_5_x = np.cos(phi_5)
    phi_5_y = np.sin(phi_5)
    phi_6_x = np.cos(phi_6)
    phi_6_y = np.sin(phi_6)
    img_XQ = [sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, phi_4_x, phi_4_y, phi_5_x, phi_5_y, phi_6_x, phi_6_y]

    img_XQ = np.array(img_XQ)
    img_XQ = img_XQ.astype(np.float32)

    return img_XQ

