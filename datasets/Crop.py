import os
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# 定义目标倍数
TARGET_MULTIPLE = 32


def crop_to_multiple(image):
    """将图片裁剪为32的倍数"""
    width, height = image.size
    new_width = (width // TARGET_MULTIPLE) * TARGET_MULTIPLE
    new_height = (height // TARGET_MULTIPLE) * TARGET_MULTIPLE
    return image.crop((0, 0, new_width, new_height))


def process_images_in_folder(folder_path):
    """遍历文件夹中的所有图片并处理"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if width % TARGET_MULTIPLE != 0 or height % TARGET_MULTIPLE != 0:
                    print(f"Processing {filename}...")
                    cropped_img = crop_to_multiple(img)
                    cropped_img.save(file_path)
                    print(f"Saved cropped image: {filename}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")






if __name__ == "__main__":
    # current_folder = os.getcwd()
    # process_images_in_folder(current_folder)
    file_path = r"E:\Dev\Code\Python\SR\HCFlow\datasets\set5\HR_mod32\baby.png"
    img = Image.open(file_path)
    img = TF.to_tensor(img)
    lr_img = F.interpolate(
        img.unsqueeze(0),  # 添加批次维度
        scale_factor=0.5,
        mode='bicubic',
        align_corners=False
    ).squeeze(0)  # 移除批次维度

    # 保存为图片
    lr_img_pil = TF.to_pil_image(lr_img)
    lr_img_pil.save("baby_lr.png")



