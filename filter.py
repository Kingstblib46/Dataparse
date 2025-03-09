import os
import numpy as np
import torch
from PIL import Image
import time
import sys

# 检查GPU可用性，若不可用则退出
if not torch.cuda.is_available():
    print("GPU不可用！请确保安装了支持CUDA的PyTorch并有可用GPU。")
    sys.exit(1)


# 定义MSE计算函数（仅GPU）
def mse(images_tensor):
    """批量计算图片对的MSE（GPU加速）"""
    N = images_tensor.size(0)
    diff = images_tensor.unsqueeze(1) - images_tensor.unsqueeze(0)  # [N, N, C, H, W]
    mse_matrix = torch.mean(diff ** 2, dim=[2, 3, 4])  # [N, N]
    return mse_matrix


# 分组筛选重复图片（改进逻辑）
def filter_group(images_tensor, threshold, group_indices):
    """在图片组内筛选重复图片，使用连通分量去重"""
    mse_matrix = mse(images_tensor)  # [N, N]
    N = len(group_indices)

    # 构建相似性图（MSE < threshold 表示相似）
    similar_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if mse_matrix[i, j] < threshold:
                similar_pairs.append((i, j))

    # 找到连通分量
    from collections import defaultdict
    graph = defaultdict(list)
    for i, j in similar_pairs:
        graph[i].append(j)
        graph[j].append(i)

    # DFS 找到所有连通分量
    visited = set()
    components = []
    for i in range(N):
        if i not in visited:
            component = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(group_indices[node])
                    stack.extend(n for n in graph[node] if n not in visited)
            components.append(component)

    # 每个连通分量保留一张图片（第一个）
    kept_indices = [comp[0] for comp in components]
    removed_indices = [idx for comp in components for idx in comp[1:]]  # 其余删除

    # 输出组内MSE范围
    mse_min, mse_max = mse_matrix.min().item(), mse_matrix.max().item()
    print(f"组内MSE范围: {mse_min:.2f} - {mse_max:.2f}")

    return kept_indices, removed_indices


# 主函数：分组使用GPU筛选并删除重复图片
def filter_similar_images(input_dir="unparsed", threshold=20, resize_size=(64, 64), group_size=5):
    """
    分组使用MSE筛选相似图片并删除重复的（GPU加速，精准去重）
    :param input_dir: 输入文件夹
    :param threshold: MSE阈值，降低到20以提高精度
    :param resize_size: 调整图片尺寸
    :param group_size: 每组图片数量
    """
    start_time = time.time()

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("未找到图片文件！")
        return

    # 加载并预处理图片
    print("加载并预处理图片...")
    images = []
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(resize_size, Image.LANCZOS)  # 高质量插值
            img_array = np.array(img, dtype=np.float32)  # [H, W, C]
            img_array = np.transpose(img_array, (2, 0, 1))  # [C, H, W]
            images.append(img_array)
        except Exception as e:
            print(f"加载 {img_file} 失败: {e}")

    if len(images) < 2:
        print("图片数量不足，无法比较！")
        return

    # 动态分组：少于10张全局比较
    num_images = len(images)
    if num_images <= 10:
        group_size = num_images
        print("图片数量较少，进行全局比较...")
    else:
        print(f"使用GPU分组筛选（每组 {group_size} 张）...")

    all_kept_indices = []
    all_removed_indices = []

    for start_idx in range(0, num_images, group_size):
        end_idx = min(start_idx + group_size, num_images)
        group_indices = list(range(start_idx, end_idx))
        if len(group_indices) < 2:
            all_kept_indices.extend(group_indices)
            continue

        # 转换为torch张量并移到GPU
        group_images = torch.tensor(np.stack(images[start_idx:end_idx])).cuda()

        # 组内筛选
        kept_indices, removed_indices = filter_group(group_images, threshold, group_indices)
        all_kept_indices.extend(kept_indices)
        all_removed_indices.extend(removed_indices)

    # 删除重复的图片
    print("删除重复的图片...")
    for idx in sorted(all_removed_indices, reverse=True):
        file_to_remove = image_files[idx]
        file_path = os.path.join(input_dir, file_to_remove)
        try:
            os.remove(file_path)
            print(f"已删除重复图片: {file_path}")
        except Exception as e:
            print(f"删除 {file_path} 失败: {e}")

    # 统计信息
    print(f"总图片数: {len(image_files)}")
    print(f"保留图片数: {len(all_kept_indices)}")
    print(f"删除图片数: {len(all_removed_indices)}")
    print(f"耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    filter_similar_images(
        input_dir="unparsed",
        threshold=20,  # 更严格的阈值
        resize_size=(64, 64),
        group_size=5
    )