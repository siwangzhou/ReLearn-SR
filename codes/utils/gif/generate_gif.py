from PIL import Image
import os

# 设置参数
output_gif = "output.gif"  # 输出GIF文件名
duration = 1000  # 每帧显示时间（毫秒）
loop = 0  # 循环次数（0表示无限循环）

# 获取当前目录下所有图片文件（按文件名排序）
image_files = sorted([f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

if not image_files:
    print("未找到图片文件！请确保目录下有.png/.jpg等格式的图片。")
else:
    print(f"正在处理图片：{image_files}")

    frames = []
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                # 强制转换为 RGB/RGBA，并确保数据独立
                if img.mode == 'P':
                    img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                # 创建一个全新的空白图像，并粘贴当前帧
                new_frame = Image.new(img.mode, img.size)
                new_frame.paste(img, (0, 0))
                frames.append(new_frame)
        except Exception as e:
            print(f"跳过文件 {img_file}（错误：{e}）")

    if frames:
        # 保存为GIF
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=True,
            disposal=1  # 关键：确保前一帧被清除
        )
        print(f"GIF生成成功：{output_gif}（共 {len(frames)} 帧）")
    else:
        print("无有效图片可处理！")