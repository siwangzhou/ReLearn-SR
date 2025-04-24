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

def analyze_frequency_spectrum(img1, img2, show=True, save_path=None):
    """使用傅里叶变换分析图像频谱差异"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    f1 = np.fft.fft2(gray1)
    fshift1 = np.fft.fftshift(f1)
    magnitude_spectrum1 = 20*np.log(np.abs(fshift1) + 1)  # 添加1避免log(0)

    f2 = np.fft.fft2(gray2)
    fshift2 = np.fft.fftshift(f2)
    magnitude_spectrum2 = 20*np.log(np.abs(fshift2) + 1)

    if show:
        plt.figure(figsize=(12, 5))
        plt.subplot(121), plt.imshow(magnitude_spectrum1, cmap='gray')
        plt.title('原始图像频谱'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum2, cmap='gray')
        plt.title('对比图像频谱'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    return magnitude_spectrum1, magnitude_spectrum2

def analyze_histograms(img1, img2, show=True, save_path=None):
    """对比两图像的颜色直方图差异"""
    color = ('b','g','r')
    histograms = []
    
    if show:
        plt.figure(figsize=(14, 5))
    
    for i, col in enumerate(color):
        hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
        histograms.append((hist1, hist2))
        
        if show:
            plt.subplot(1, 3, i+1)
            plt.plot(hist1, color=col, label='原始图像')
            plt.plot(hist2, color='k', linestyle='--', label='对比图像')
            plt.fill_between(range(256), hist1.flatten(), hist2.flatten(), 
                            color=col, alpha=0.3)
            plt.xlim([0, 256])
            plt.title(f'{col.upper()} 通道直方图')
            if i == 0:
                plt.legend()
    
    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    return histograms

def analyze_residuals(img1, img2, kernel_size=(25, 25), amplify_factor=10, show=True, save_path=None):
    """使用高斯模糊分析残差差异"""
    # 使用高斯模糊消除原始内容
    blur1 = cv2.GaussianBlur(img1, kernel_size, 0)
    blur2 = cv2.GaussianBlur(img2, kernel_size, 0)

    # 计算残差
    residual1 = img1 - blur1
    residual2 = img2 - blur2

    # 计算并放大残差差异
    residual_diff = cv2.absdiff(residual1, residual2) * amplify_factor
    
    # 创建彩色热图以更好地显示残差
    residual_heat = cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.cvtColor(residual_diff, cv2.COLOR_BGR2GRAY)), 
        cv2.COLORMAP_JET
    )

    if show:
        # 改用matplotlib代替cv2.imshow
        plt.figure(figsize=(10, 8))
        # OpenCV使用BGR，matplotlib使用RGB，需要转换颜色通道
        residual_heat_rgb = cv2.cvtColor(residual_heat, cv2.COLOR_BGR2RGB)
        plt.imshow(residual_heat_rgb)
        plt.title('残差差异 (放大)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    return residual_diff

def main():
    # 图像路径
    img1_path = r'E:\Dev\Code\Python\SR\HCFlow\results\001_test_P2P_HCD_CARN_conv_K16O8_2X\Set5\bird\bird_best.png'
    img2_path = r'E:\Dev\Code\Python\SR\HCFlow\results\001_test_P2P_HCD_CARN_conv_K16O8_2X\Set5\bird\bird_first.png'
    # 创建输出目录
    output_dir = 'image_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    try:
        img1, img2 = load_images(img1_path, img2_path)
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return
    
    # 计算差异
    diff, gray_diff = compute_difference(
        img1, img2, 
        amplify_factor=50,
        save_path=os.path.join(output_dir, 'difference.jpg')
    )
    
    # 计算质量指标
    metrics = calculate_metrics(img1, img2)
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # 频谱分析
    analyze_frequency_spectrum(
        img1, img2, 
        save_path=os.path.join(output_dir, 'frequency_spectrum.png')
    )
    
    # 直方图分析
    analyze_histograms(
        img1, img2,
        save_path=os.path.join(output_dir, 'histograms.png')
    )
    
    # 残差分析
    analyze_residuals(
        img1, img2,
        save_path=os.path.join(output_dir, 'residuals.png')
    )

if __name__ == "__main__":
    main()