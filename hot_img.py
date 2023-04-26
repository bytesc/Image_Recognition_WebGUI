import torch

import model
import innvestigator
import interpretation

import random

import nibabel as nib
import matplotlib.pyplot as plt

from PIL import Image


def generate_random_str(target_length=16):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(target_length):
        random_str += base_str[random.randint(0, length)]
    return random_str


def process_img(img):
    img = torch.from_numpy(img)
    img = img.squeeze()
    img = img.reshape(1, -1, 256, 256)
    img = img[:, 0:160, :, :].float()
    return img


def hot_img(nii_path, layer, hot_type="LRP"):
    img_name = generate_random_str()
    img = nib.load(nii_path)
    img = img.get_fdata()

    img_path1 = "./imgs/img_raw/" + img_name + ".png"
    plt.imshow(img[:, :, layer], cmap='gray')
    plt.axis('off')
    plt.savefig(img_path1, bbox_inches='tight', pad_inches=0)

    img = process_img(img)
    img = img.reshape((1, 1, -1, 256, 256))
    # img.shape

    net = model.ClassificationModel3D()
    # net.cuda(device)
    net.load_state_dict(torch.load("./data/model_save/myModel_state_dict_130.pth",
                                   map_location='cpu'), strict=True)
    net.eval()
    net = torch.nn.Sequential(net, torch.nn.Softmax(dim=1))
    inn_model = innvestigator.InnvestigateModel(net, lrp_exponent=1,
                                                method="b-rule",
                                                beta=0, epsilon=1e-6)
    # inn_model = innvestigator.InnvestigateModel(net, lrp_exponent=1,
    #                                   method="b-rule",
    #                                   beta=0, epsilon=1e-6).cuda(device)
    inn_model.eval()

    def run_guided_backprop(net, image_tensor):
        return interpretation.guided_backprop(net, image_tensor,
                                              cuda=True if torch.cuda.is_available() else False,
                                              verbose=False, apply_softmax=False)

    def run_LRP(net, image_tensor):
        return inn_model.innvestigate(in_tensor=image_tensor, rel_for_class=1)

    image_tensor = img

    LRP_map = None
    GB_map=None
    img2 = None

    if hot_type == "LRP":
        AD_score, LRP_map = run_LRP(inn_model, image_tensor)
        LRP_map = LRP_map.detach().numpy().squeeze()
        img2 = LRP_map[:, :, layer]
        # print(LRP_map.shape)

    if hot_type == "GB":
        GB_map = run_guided_backprop(inn_model, image_tensor)
        GB_map = GB_map.squeeze()
        img2 = GB_map[:, :, layer]
        # print(GB_map.shape)

    img_path2 = "./imgs/img_hot/" + img_name + ".png"

    plt.imshow(img2, cmap='hot')
    plt.axis('off')
    plt.savefig(img_path2, bbox_inches='tight', pad_inches=0)
    # plt.show()

    def blend_two_images(path1, path2):
        img1 = Image.open(path1)
        img1 = img1.resize((512, 360))
        img1 = img1.convert('RGBA')

        img2 = Image.open(path2)
        img2 = img2.resize((512, 360))
        img2 = img2.convert('RGBA')

        img = Image.blend(img1, img2, 0.6)
        return img

    img_final = blend_two_images(img_path1, img_path2)
    # img_final.show()
    img_path3 = "./imgs/img_merge/" + img_name + ".png"
    img_final.save(img_path3)

    return img_path1, img_path2, img_path3  # 原图，纯热力图，热力图


if __name__ == "__main__":
    img = "D:\\ECUST\\DACHUANG\\IDAdata\\data\\CN\\ADNI_021_S_0984_MR_MP-RAGE__br_raw_20091218112557500_1_S77336_I161155.nii"
    hot_img(img, 145)

