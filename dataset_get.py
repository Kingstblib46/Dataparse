from datasets import load_dataset
from huggingface_hub import login
import os

def main():
    login(token="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")
    
    # 使用流式加载数据集
    dataset = load_dataset("leonardPKU/os_track_mac_full", split="train", streaming=True)

    num = 20
    iterator = iter(dataset)
    front_records = [next(iterator) for _ in range(num)]

    output_dir = "unparsed"
    os.makedirs(output_dir, exist_ok=True)

    for i, record in enumerate(front_records):
        image = record['image']  # 获取 PIL 图像对象
        file_name = f"image_{i+1}.jpg"  # 文件名，例如 image_1.jpg
        file_path = os.path.join(output_dir, file_name)  # 完整文件路径
        image.save(file_path, "JPEG")  # 保存为 JPEG 格式
        print(f"已保存第 {i+1} 张图片到: {file_path}")

if __name__ == "__main__":
    main()
    # 强制退出以避免 Python 正常退出时进行垃圾回收导致的线程状态问题
    import os
    os._exit(0)