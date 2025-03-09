from gradio_client import Client, handle_file
from PIL import Image, ImageDraw, ImageFont
import ast
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# 定义带重试机制的API调用函数
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_image_with_gradio(client, img_path):
    """调用Gradio API处理图片，带重试机制"""
    return client.predict(
        image_input=handle_file(img_path),
        box_threshold=0.05,
        iou_threshold=0.1,
        use_paddleocr=True,
        imgsz=640,
        api_name="/process"
    )

# 初始化 Gradio 客户端
try:
    client = Client("https://2e1b4ade4618c88fb7.gradio.live/")
    print("Gradio服务连接成功！")
except Exception as e:
    print(f"无法连接Gradio服务: {e}")
    exit()

# 定义输入和输出文件夹
input_dir = 'unparsed'
output_dir = 'parsed'

# 获取 unparsed 文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("未找到图片文件！")
    exit()

# 遍历并处理每张图片
for img_file in image_files:
    # 构造输入和输出路径
    img_path = os.path.join(input_dir, img_file)
    output_file = f"{os.path.splitext(img_file)[0]}_parsed.jpg"  # 添加 _parsed 后缀
    output_path = os.path.join(output_dir, output_file)

    try:
        # API 调用（带重试）
        result = process_image_with_gradio(client, img_path)
        if not isinstance(result, tuple) or len(result) < 2:
            raise ValueError("Expected a tuple with file path and OCR string")

        # 解析 OCR 数据
        ocr_data = [ast.literal_eval(line.split(': ', 1)[1])
                    for line in result[1].split('\n')
                    if line.startswith('icon') and 'bbox' in line and 'content' in line]

        # 加载图片并准备绘制
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        font = ImageFont.load_default()

        # 绘制检测框和标签
        for item in ocr_data:
            x1, y1, x2, y2 = [float(coord) * dim for coord, dim in zip(item['bbox'], [width, height, width, height])]
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            text = str(item['content'])
            draw.text((x1, y2 + 2 if y2 + 10 <= height else y1 - 10), text, fill='red', font=font)

        # 保存处理后的图片
        img.save(output_path)
        print(f"已保存解析后的图片到: {output_path}")

    except Exception as e:
        print(f"处理 {img_file} 时出错: {e}")

print("所有图片解析完成！")