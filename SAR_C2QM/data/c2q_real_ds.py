import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


_DATA_AUG_TIMES = 4  # 数据增强次数
def data_augmentation(image, mode, shape_flag=0):
    """
    数据增强
    :param image: 形状为 (c, h, w) 或 (h, w, c) 的图像
    :param mode: 增强方法
    :param shape_flag: 0, (c, h, w); 1, (h, w, c)
    :return: _description_
    """
    if shape_flag == 0:
        image = np.transpose(image, (1, 2, 0))  # 转换为 (h, w, c)

    if mode == 0:
        # 原图
        res = image.copy()
    elif mode == 1:
        # 上下翻转
        res = np.flipud(image).copy()
    elif mode == 2:
        # 左右翻转
        res = np.fliplr(image).copy()
    elif mode == 3:
        # 上下翻转后左右翻转
        res = np.flipud(np.fliplr(image)).copy()
    
    if shape_flag == 0:
        res = np.transpose(res, (2, 0, 1)).copy()  # 转换为 (c, h, w)

    return res


class RealNumPyDataset_Atec(Dataset):
    def __init__(self, ds_dir, flag_ds, flag_data, flag_aug, transform=None):
        """
        自定义数据集，自编码器训练，用于加载NumPy格式的图像。
        对应 generate_dataset_C2QLDM_RHV_list, generate_dataset_info_C2QLDM_RHV_list 生成数据集
        :param ds_dir: 样本集文件夹，需包含"/images_xx" "/info"。
        :param flag_ds: 0，训练样本；1，验证样本；2，测试样本
        :param flag_data: img_XC (训练条件编码器); img_XQ (训练主自编码器); img_XL (训练主自编码器)
        :param flag_aug: 数据增强标志，False, 不增强; True, 增强
        :param transform: 一个可选的变换列表，应用于加载的数据。
        """

        info_dir = ds_dir + r"/info"  # 样本集信息文件夹
        info_train_path = info_dir + r"/train_files.txt"
        info_val_path = info_dir + r"/val_files.txt"
        info_test_path = info_dir + r"/test_files.txt"

        self.flag_aug = flag_aug
        self.transform = transform
        self.img_paths = []
        self.image_size = None  # 初始化图像尺寸属性

        # -------------------------------#
        #   从./info中读取样本相对路径信息
        # -------------------------------#
        if flag_ds == 0:
            with open(info_train_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]
        if flag_ds == 1:
            with open(info_val_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]
        if flag_ds == 2:
            with open(info_test_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]

        # -------------------------------#
        #   获取每个.npy文件的路径
        # -------------------------------#
        for name in sample_fileNames:
            self.img_paths.append("{}/{}/{}".format(ds_dir, flag_data, name))

    def __len__(self):
        return len(self.img_paths) if self.flag_aug is False else _DATA_AUG_TIMES * len(self.img_paths)

    def __getitem__(self, idx):

        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#

        if self.flag_aug is False:
            img = np.load(self.img_paths[int(idx)])
            img = img.astype(np.float32)
            img = np.transpose(img, (1, 2, 0))  # 转换为H,W,C

        if self.flag_aug is True:
            original_idx = idx // _DATA_AUG_TIMES
            mode_aug = idx % _DATA_AUG_TIMES

            img = np.load(self.img_paths[original_idx])
            img = img.astype(np.float32)
            img = np.transpose(img, (1, 2, 0))  # 转换为H,W,C
            img = data_augmentation(img, mode=mode_aug, shape_flag=1)  # shape_flag=1表示输入为H,W,C格式

        example = {}
        example["image_in"] = img
        
        return example


class RealNumPyDataset_C2QLDM_bs(Dataset):
    def __init__(self, ds_dir, flag_use, flag_obj_data, flag_cond_data, flag_aug, transform=None):
        """
        自定义数据集，C2QLDM_RHV_bs 训练，用于加载NumPy格式的图像
        对应 generate_dataset_C2QLDM_RHV_list, generate_dataset_info_C2QLDM_RHV_list 生成数据集
        :param ds_dir: 样本集文件夹，需包含"/img_xx" "/info"。
        :param flag_use: 0，训练样本；1，验证样本；2，测试样本
        :param flag_obj_data: 数据 img_XQ; 数据 img_XL
        :param flag_cond_data: 单模态 [img_XC]; 多模态 [img_XC, img_Xgeo]
        :param flag_aug: 数据增强标志，False, 不增强; True, 增强
        :param transform: 一个可选的变换列表，应用于加载的数据。
        """

        info_dir = ds_dir + r"/info"  # 样本集信息文件夹
        info_train_path = info_dir + r"/train_files.txt"
        info_val_path = info_dir + r"/val_files.txt"
        info_test_path = info_dir + r"/test_files.txt"

        self.flag_obj_data = flag_obj_data
        self.flag_cond_data = flag_cond_data
        self.flag_aug = flag_aug
        self.transform = transform

        self.img_paths_XC = []
        self.img_paths_Xgeo = []
        self.img_paths_XQL = []
        
        # -------------------------------#
        #   从./info中读取样本相对路径信息
        # -------------------------------#
        if flag_use == 0:
            with open(info_train_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]
        if flag_use == 1:
            with open(info_val_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]
        if flag_use == 2:
            with open(info_test_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]

        # -------------------------------#
        #   获取每个.npy文件的路径
        # -------------------------------#
        for name in sample_fileNames:
            self.img_paths_XC.append("{}/{}/{}".format(ds_dir, flag_cond_data[0], name))
            if len(flag_cond_data) == 2:
                self.img_paths_Xgeo.append("{}/{}/{}".format(ds_dir, flag_cond_data[1], name))
            self.img_paths_XQL.append("{}/{}/{}".format(ds_dir, flag_obj_data, name))

    def __len__(self):
        return len(self.img_paths_XC) if self.flag_aug is False else _DATA_AUG_TIMES * len(self.img_paths_XC)

    def __getitem__(self, idx):

        if self.flag_aug is False:
            img_XC = np.load(self.img_paths_XC[idx])
            img_XC = img_XC.astype(np.float32)
            img_XC = np.transpose(img_XC, (1, 2, 0))  # 转换为H,W,C

            if len(self.flag_cond_data) == 2:
                img_Xgeo = np.load(self.img_paths_Xgeo[idx])
                img_Xgeo = img_Xgeo.astype(np.float32)
                img_Xgeo = np.transpose(img_Xgeo, (1, 2, 0))

            img_XQL = np.load(self.img_paths_XQL[idx])
            img_XQL = img_XQL.astype(np.float32)
            img_XQL = np.transpose(img_XQL, (1, 2, 0))  # 转换为H,W,C

        if self.flag_aug is True:
            original_idx = idx // _DATA_AUG_TIMES
            mode_aug = idx % _DATA_AUG_TIMES

            img_XC = np.load(self.img_paths_XC[original_idx])
            img_XC = img_XC.astype(np.float32)
            img_XC = np.transpose(img_XC, (1, 2, 0))  # 转换为H,W,C
            img_XC = data_augmentation(img_XC, mode=mode_aug, shape_flag=1)

            if len(self.flag_cond_data) == 2:
                img_Xgeo = np.load(self.img_paths_Xgeo[original_idx])
                img_Xgeo = img_Xgeo.astype(np.float32)
                img_Xgeo = np.transpose(img_Xgeo, (1, 2, 0))
                img_Xgeo = data_augmentation(img_Xgeo, mode=mode_aug, shape_flag=1)

            img_XQL = np.load(self.img_paths_XQL[original_idx])
            img_XQL = img_XQL.astype(np.float32)
            img_XQL = np.transpose(img_XQL, (1, 2, 0))
            img_XQL = data_augmentation(img_XQL, mode=mode_aug, shape_flag=1)

        example = {}
        example[self.flag_cond_data[0]] = img_XC
        if len(self.flag_cond_data) == 2:
            example[self.flag_cond_data[1]] = img_Xgeo
        example[self.flag_obj_data] = img_XQL

        return example


class RealNumPyDataset_C2QLDM(Dataset):
    def __init__(self, ds_dir, flag_use, flag_obj_data, flag_cond_data, flag_aug, transform=None):
        """
        自定义数据集，C2QLDM_RHV_bs 训练，用于加载NumPy格式的图像
        对应 generate_dataset_C2QLDM_RHV_list, generate_dataset_info_C2QLDM_RHV_list 生成数据集
        :param ds_dir: 样本集文件夹，需包含"/img_xx" "/info"。
        :param flag_use: 0，训练样本；1，验证样本；2，测试样本
        :param flag_obj_data: 数据 img_XQ; 数据 img_XL
        :param flag_cond_data: 单模态 [img_XC]; 多模态 [img_XC, img_Xgeo]
        :param flag_aug: 数据增强标志，False, 不增强; True, 增强
        :param transform: 一个可选的变换列表，应用于加载的数据。
        """

        info_dir = ds_dir + r"/info"  # 样本集信息文件夹
        info_train_path = info_dir + r"/train_files.txt"
        info_val_path = info_dir + r"/val_files.txt"
        info_test_path = info_dir + r"/test_files.txt"

        self.flag_obj_data = flag_obj_data
        self.flag_cond_data = flag_cond_data
        self.flag_aug = flag_aug
        self.transform = transform

        self.img_paths_X_m1_1 = []
        self.img_paths_X_m1_2 = []
        self.img_paths_X_m2 = []
        self.img_paths_X_obj = []
        
        # -------------------------------#
        #   从./info中读取样本相对路径信息
        # -------------------------------#
        if flag_use == 0:
            with open(info_train_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]
        if flag_use == 1:
            with open(info_val_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]
        if flag_use == 2:
            with open(info_test_path, "r") as f:
                sample_fileNames = [line.strip() for line in f.readlines()]

        # -------------------------------#
        #   获取每个.npy文件的路径
        # -------------------------------#
        for name in sample_fileNames:
            if len(flag_cond_data['modal_1']) == 1:
                self.img_paths_X_m1_1.append("{}/{}/{}".format(ds_dir, flag_cond_data['modal_1'][0], name))
            if len(flag_cond_data['modal_1']) == 2:
                self.img_paths_X_m1_1.append("{}/{}/{}".format(ds_dir, flag_cond_data['modal_1'][0], name))
                self.img_paths_X_m1_2.append("{}/{}/{}".format(ds_dir, flag_cond_data['modal_1'][1], name))
            if len(flag_cond_data['modal_2']) == 1:
                self.img_paths_X_m2.append("{}/{}/{}".format(ds_dir, flag_cond_data['modal_2'][0], name))
            self.img_paths_X_obj.append("{}/{}/{}".format(ds_dir, flag_obj_data, name))

    def __len__(self):
        return len(self.img_paths_X_m1_1) if self.flag_aug is False else _DATA_AUG_TIMES * len(self.img_paths_X_m1_1)

    def __getitem__(self, idx):

        if self.flag_aug is False:

            if len(self.flag_cond_data['modal_1']) == 1:
                img_X_m1_1 = np.load(self.img_paths_X_m1_1[idx])
                img_X_m1_1 = img_X_m1_1.astype(np.float32)
                img_X_m1 = np.transpose(img_X_m1_1, (1, 2, 0))  # 转换为H,W,C
            
            if len(self.flag_cond_data['modal_1']) == 2:
                img_X_m1_1 = np.load(self.img_paths_X_m1_1[idx])
                img_X_m1_1 = img_X_m1_1.astype(np.float32)
                img_X_m1_2 = np.load(self.img_paths_X_m1_2[idx])
                img_X_m1_2 = img_X_m1_2.astype(np.float32)
                img_X_m1 = np.concatenate([img_X_m1_1, img_X_m1_2], axis=0)
                img_X_m1 = np.transpose(img_X_m1, (1, 2, 0))

            if len(self.flag_cond_data['modal_2']) == 1:
                img_X_m2 = np.load(self.img_paths_X_m2[idx])
                img_X_m2 = img_X_m2.astype(np.float32)
                img_X_m2 = np.transpose(img_X_m2, (1, 2, 0))

            img_X_obj = np.load(self.img_paths_X_obj[idx])
            img_X_obj = img_X_obj.astype(np.float32)
            img_X_obj = np.transpose(img_X_obj, (1, 2, 0))  # 转换为H,W,C

        if self.flag_aug is True:
            original_idx = idx // _DATA_AUG_TIMES
            mode_aug = idx % _DATA_AUG_TIMES

            if len(self.flag_cond_data['modal_1']) == 1:
                img_X_m1_1 = np.load(self.img_paths_X_m1_1[idx])
                img_X_m1_1 = img_X_m1_1.astype(np.float32)
                img_X_m1 = np.transpose(img_X_m1_1, (1, 2, 0))  # 转换为H,W,C
            
            if len(self.flag_cond_data['modal_1']) == 2:
                img_X_m1_1 = np.load(self.img_paths_X_m1_1[idx])
                img_X_m1_1 = img_X_m1_1.astype(np.float32)
                img_X_m1_2 = np.load(self.img_paths_X_m1_2[idx])
                img_X_m1_2 = img_X_m1_2.astype(np.float32)

                img_X_m1 = np.concatenate([img_X_m1_1, img_X_m1_2], axis=0)
                img_X_m1 = np.transpose(img_X_m1, (1, 2, 0))
            img_X_m1 = data_augmentation(img_X_m1, mode=mode_aug, shape_flag=1)

            if len(self.flag_cond_data['modal_2']) == 1:
                img_X_m2 = np.load(self.img_paths_X_m2[idx])
                img_X_m2 = img_X_m2.astype(np.float32)
                img_X_m2 = np.transpose(img_X_m2, (1, 2, 0))
                img_X_m2 = data_augmentation(img_X_m2, mode=mode_aug, shape_flag=1)

            img_X_obj = np.load(self.img_paths_X_obj[original_idx])
            img_X_obj = img_X_obj.astype(np.float32)
            img_X_obj = np.transpose(img_X_obj, (1, 2, 0))
            img_X_obj = data_augmentation(img_X_obj, mode=mode_aug, shape_flag=1)

        example = {}
        example['modal_1'] = img_X_m1
        if len(self.flag_cond_data['modal_2']) == 1:
            example['modal_2'] = img_X_m2
        example[self.flag_obj_data] = img_X_obj

        return example

