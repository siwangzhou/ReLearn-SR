import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

def load_images(img1_path, img2_path):
    """加载两张图像进行比较"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像文件，请检查路径是否正确")
    
    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        print(f"警告: 图像尺寸不同 {img1.shape} vs {img2.shape}")
        # 可以选择调整大小或抛出错误
    
    return img1, img2

def compute_difference(img1, img2, amplify_factor=50, save_path=None):
    """计算并可视化两图像之间的差异"""
    diff = cv2.absdiff(img1, img2)
    enhanced_diff = diff * amplify_factor
    gray_diff = cv2.cvtColor(enhanced_diff, cv2.COLOR_BGR2GRAY)
    
    if save_path:
        cv2.imwrite(save_path, gray_diff)
        
    return diff, gray_diff

def calculate_metrics(img1, img2):
    """计算图像质量评估指标: MSE, PSNR, SSIM"""
    # 计算MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    # 计算PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else float('inf')

    # 计算SSIM - 解决bug
    # 转换为灰度计算SSIM以避免多通道问题
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 确定win_size (必须是奇数且小于或等于图像的最小边长)
    min_side = min(gray1.shape[0], gray1.shape[1])
    win_size = min(7, min_side) if min_side < 7 else 7
    # 确保win_size是奇数
    win_size = win_size if win_size % 2 == 1 else win_size - 1

    ssim_value = ssim(gray1, gray2, win_size=win_size, data_range=255)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_value
    }

def blockwise_psnr_analysis(img1, img2, block_size=8, save_path=None):
    # 转换为YUV颜色空间（分离亮度信息）
    yuv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

    # 仅取亮度通道
    y1 = yuv1[:, :, 0].astype(np.float32)
    y2 = yuv2[:, :, 0].astype(np.float32)

    # 分块计算
    height, width = y1.shape
    total_pixels = height * width
    contribution_map = np.zeros_like(y1)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 提取当前区块
            block1 = y1[i:i + block_size, j:j + block_size]
            block2 = y2[i:i + block_size, j:j + block_size]
            # 计算区块MSE
            mse_block = np.mean((block1 - block2) ** 2)
            # 计算区块对全局PSNR的贡献权重
            block_area = block_size ** 2
            contribution = (mse_block * block_area) / (total_pixels * np.mean((y1 - y2) ** 2))

            # 记录贡献度
            contribution_map[i:i + block_size, j:j + block_size] = contribution


    # 可视化并保存
    plt.figure(figsize=(10, 5))
    plt.imshow(contribution_map, cmap='viridis')
    plt.colorbar(label='贡献度')
    plt.title('块状PSNR贡献度图')
    plt.axis('on')
    plt.tight_layout()  # 调整布局以消除空白
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 裁剪空白边缘,保留0.1英寸边距
    plt.close()
    return contribution_map



def main():
    # 图像路径
    img1_path = r'E:\Dev\Code\Python\SR\ReLearn\results\2X\001_test_P2P_HCD_EDSR_conv_K32O16_2X\Set14\monarch\monarch_best.png'
    img2_path = r'E:\Dev\Code\Python\SR\ReLearn\results\2X\001_test_P2P_HCD_EDSR_conv_K32O16_2X\Set14\monarch\monarch_first.png'
    # 创建输出目录
    output_dir = 'image_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    # 加载图像
    img1, img2 = load_images(img1_path, img2_path)
    # 计算差异
    diff, gray_diff = compute_difference(
        img1, img2, 
        amplify_factor=30,
        save_path=os.path.join(output_dir, 'difference.jpg')
    )

    blockwise_psnr_analysis(img1, img2,
                            block_size=8,
                            save_path=os.path.join(output_dir, 'blockwise_psnr_analysis.jpg') )
    
    # 计算质量指标
    metrics = calculate_metrics(img1, img2)
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")


if __name__ == "__main__":
    main()