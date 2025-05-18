import cv2
import numpy as np


def draw_pixel_grid_with_alpha(image_path, cell_size=8, alpha=0.7):
    """
    在图片上绘制可调节透明度的像素级网格

    参数:
        image_path: 图片文件路径
        cell_size: 每个网格单元的大小（像素）
        alpha: 原图透明度 (0.0-1.0，1为完全不透明)
    """
    # 读取图片
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("错误: 无法读取图片，请检查文件路径")
        return None

    # 创建纯白背景和网格图层
    height, width = original_img.shape[:2]
    grid_layer = np.ones_like(original_img) * 255  # 白色背景

    # 计算网格线颜色（黑色）
    grid_color = (0, 0, 0)

    # 绘制垂直线
    for x in range(0, width, cell_size):
        cv2.line(grid_layer, (x, 0), (x, height), grid_color, 1, lineType=cv2.LINE_AA)

    # 绘制水平线
    for y in range(0, height, cell_size):
        cv2.line(grid_layer, (0, y), (width, y), grid_color, 1, lineType=cv2.LINE_AA)

    # 混合原图和网格层
    result = cv2.addWeighted(original_img, alpha, grid_layer, 1 - alpha, 0)

    return result


if __name__ == "__main__":
    print("=== 图片网格化工具 ===")
    image_path = r"E:\Dev\Code\Python\SR\ReLearn\datasets\set14\HR_mod32\monarch.png"

    cell_size = 16
    # 获取透明度
    alpha = 0.5

    # 处理图片
    result_img = draw_pixel_grid_with_alpha(image_path, cell_size, alpha)

    if result_img is not None:


        # 自动保存
        save_path = f'_grid_{cell_size}px_alpha{alpha:.1f}.jpg'
        cv2.imwrite(save_path, result_img)
        print(f"\n图片已自动保存为: {save_path}")




image_path = r"E:\Dev\Code\Python\SR\ReLearn\datasets\set14\HR_mod32\monarch.png"