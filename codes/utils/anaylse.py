import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def blockwise_psnr_analysis(img1, img2, block_size=32):
    # 转换为YUV颜色空间（分离亮度信息）
    yuv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

    # 仅取亮度通道
    y1 = yuv1[:, :, 0].astype(np.float32)
    y2 = yuv2[:, :, 0].astype(np.float32)

    # 分块计算
    height, width = y1.shape
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
            total_pixels = height * width
            contribution = (mse_block * block_area) / (total_pixels * np.mean((y1 - y2) ** 2))

            # 记录贡献度
            contribution_map[i:i + block_size, j:j + block_size] = contribution

    return contribution_map


def pixel_level_sensitivity(img1, img2):
    diff = cv2.absdiff(img1, img2).mean(axis=2)
    original = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 计算梯度敏感度
    sobelx = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 构建敏感度模型
    sensitivity_map = diff * (1 + gradient_mag / 255)

    return sensitivity_map


img1_path = r'E:\Dev\Code\Python\SR\HCFlow\results\001_test_P2P_HCD_CARN_conv_K16O8_2X\Set5\baby\baby_best.png'
img2_path = r'E:\Dev\Code\Python\SR\HCFlow\results\001_test_P2P_HCD_CARN_conv_K16O8_2X\Set5\baby\baby_first.png'
img1, img2 = load_images(img1_path, img2_path)

# 使用示例
contribution_map = blockwise_psnr_analysis(img1, img2, 2)

# 可视化
plt.figure(figsize=(10, 8))
plt.imshow(contribution_map, cmap='jet')
plt.colorbar(label='Contribution Weight')
plt.title('Local Contribution to PSNR Degradation')
plt.show()
#
#
# sensitivity_map = pixel_level_sensitivity(img1, img2)
#
# # 可视化
# plt.figure(figsize=(10,8))
# plt.imshow(sensitivity_map, cmap='jet')
# plt.colorbar(label='Sensitivity Index')
# plt.title('Pixel-level PSNR Sensitivity Analysis')
# plt.show()
#
#
# def noise_gradient_correlation(img1, img2):
#     # 计算噪声场
#     noise = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#     # 计算原图梯度
#     original = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gx = cv2.Sobel(original, cv2.CV_64F, 1, 0)
#     gy = cv2.Sobel(original, cv2.CV_64F, 0, 1)
#     gradient = np.sqrt(gx ** 2 + gy ** 2)
#
#     # 计算相关系数
#     correlation = np.corrcoef(noise.flatten(), gradient.flatten())[0, 1]
#
#     # 可视化联合分布
#     plt.figure(figsize=(10, 6))
#     plt.hexbin(gradient.flatten(), noise.flatten(), gridsize=50, cmap='jet')
#     plt.colorbar(label='Density')
#     plt.xlabel('Gradient Magnitude')
#     plt.ylabel('Noise Value')
#     plt.title(f'Noise-Gradient Correlation (r={correlation:.3f})')
#     plt.show()
#
#
# # 使用示例
# noise_gradient_correlation(img1, img2)
#
#
# def frequency_band_analysis(img1, img2):
#     # 转换为灰度
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#     # 计算差异图的频域
#     diff = gray1.astype(np.float32) - gray2.astype(np.float32)
#     fft_diff = np.fft.fft2(diff)
#     fft_shift = np.fft.fftshift(fft_diff)
#     magnitude = np.abs(fft_shift)
#
#     # 创建环形蒙版分析频带
#     height, width = gray1.shape
#     cy, cx = height // 2, width // 2
#     radius_step = 50  # 频率带宽度（像素）
#
#     band_contributions = []
#     for r in range(0, min(height, width) // 2, radius_step):
#         mask = np.zeros_like(gray1)
#         cv2.circle(mask, (cx, cy), r + radius_step, 1, -1)
#         cv2.circle(mask, (cx, cy), r, 0, -1)
#
#         band_energy = np.sum(magnitude * mask)
#         band_contributions.append(band_energy)
#
#     # 可视化
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.arange(len(band_contributions)) * radius_step,
#              band_contributions / np.sum(band_contributions))
#     plt.xlabel('Frequency Band Radius (pixels)')
#     plt.ylabel('Energy Contribution Ratio')
#     plt.title('Frequency Band Contribution to PSNR Degradation')
#     plt.grid(True)
#     plt.show()
#
#
# # 使用示例
# frequency_band_analysis(img1, img2)