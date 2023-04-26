from torch.utils.data import Dataset
import os
import nibabel as nib
import torch


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # path目录下的文件名列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # path目录下的文件名列表中的第idx个文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)

        img = nib.load(img_item_path)
        img = img.get_fdata()

        img = process_img(img)

        label = self.label_dir
        label = ["AD", "CN", "MCI", "EMCI", "LMCI"].index(label)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.img_path)  # 返回path目录下的文件名列表长度


def get_datasets():
    myroot_dir = "./data"
    AD_dir = "AD"
    CN_dir = "CN"
    MCI_dir = "MCI"
    EMCI_dir = "EMCI"
    LMCI_dir = "LMCI"
    ad_dataset = MyData(myroot_dir, AD_dir)
    cn_dataset = MyData(myroot_dir, CN_dir)
    mci_dataset = MyData(myroot_dir, MCI_dir)
    emci_dataset = MyData(myroot_dir, EMCI_dir)
    lmci_dataset = MyData(myroot_dir, LMCI_dir)
    myroot_dir+="\\test"
    ad_dataset_test = MyData(myroot_dir, AD_dir)
    cn_dataset_test = MyData(myroot_dir, CN_dir)
    mci_dataset_test= MyData(myroot_dir, MCI_dir)
    emci_dataset_test = MyData(myroot_dir, EMCI_dir)
    lmci_dataset_test = MyData(myroot_dir, LMCI_dir)

    # print(ad_dataset[0][0].shape)  # "[x]"调用__getitem__(x)函数
    # print(cn_dataset[0][1])
    # print(len(ad_dataset))  # len()调用__len__函数

    train_set = ad_dataset+cn_dataset+lmci_dataset+mci_dataset+emci_dataset
    val_set = ad_dataset_test + cn_dataset_test+lmci_dataset_test+mci_dataset_test+emci_dataset_test

    # print(train_set[0])
    # print(ad_dataset)  # __main__.MyData object
    # print(train_set)  # torch.utils.data.dataset.ConcatDataset object

    return train_set, val_set


def process_img(img):
    img = torch.from_numpy(img)
    img = img.squeeze()
    img = img.reshape(1, -1, 256, 256)
    img = img[:, 0:160, :, :].float()
    return img
