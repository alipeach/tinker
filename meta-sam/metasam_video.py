import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from datetime import datetime,timedelta

# 用于存储每个目标的颜色
target_colors = {}

def generate_and_save_masks(sam, input_video_path, output_video_path):
    """
    生成并保存视频的分割掩码和渲染后的视频
    :param sam: 加载的SAM模型
    :param input_video_path: 输入视频的路径
    :param output_video_path: 输出视频的路径
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频 '{input_video_path}'，跳过...")
        return

    # 获取视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化掩码生成器
    amg_kwargs = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100,  # 需要opencv进行后处理
    }
    generator = SamAutomaticMaskGenerator(sam, **amg_kwargs)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        frame_num = frame_num + 1
        if not ret:
            break
        # if frame_num % 100 == 0:
        print(f"当前检测进度：已检测 {frame_num} 帧，共 {total_frames} 帧。")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 生成掩码
        masks = generator.generate(frame)

        # 渲染并保存渲染后的帧
        rendered_frame = render_masks_on_image(frame, masks)
        rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
        out.write(rendered_frame)

    # 释放资源
    cap.release()
    out.release()

def render_masks_on_image(image, masks):
    """
    将掩码渲染到原始图像上
    :param image: 原始图像
    :param masks: 生成的掩码列表
    :return: 渲染后的图像
    """
    global target_colors
    overlay = image.copy()
    for mask in masks:
        mask_id = hash(tuple(mask["segmentation"].flatten()))  # 为每个目标生成唯一的ID
        if mask_id not in target_colors:
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            target_colors[mask_id] = color.tolist()
        else:
            color = target_colors[mask_id]
        binary_mask = mask["segmentation"]
        overlay[binary_mask] = color
    alpha = 0.5
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return output

def main():
    # 指定模型类型和检查点路径
    model_type = "vit_b"
    checkpoint = "/workspace/model/sam_vit_b_01ec64.pth"

    # model_type = "vit_l"
    # checkpoint = "/workspace/model/sam_vit_l_0b3195.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("正在加载模型...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    # 指定输入文件
    input_video_path = "/workspace/fpv追船.mp4"
    
    # 获取输入文件名和扩展名
    input_basename = os.path.basename(input_video_path)
    input_name, input_ext = os.path.splitext(input_basename)
    
    # 生成当前时间戳（精确到分钟）
    local_time = datetime.now()
    beijing_time = local_time + timedelta(hours=8)
    timestamp = beijing_time.strftime("%Y%m%d_%H%M")
    
    # 构建输出文件名
    output_name = f"{input_name}_{timestamp}.mp4"
    output_dir = "/workspace/outputs/metasam"
    output_video_path = os.path.join(output_dir, output_name)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入视频: {input_video_path}")
    print(f"输出视频: {output_video_path}")

    # 生成并保存掩码
    generate_and_save_masks(sam, input_video_path, output_video_path)

if __name__ == "__main__":
    main()