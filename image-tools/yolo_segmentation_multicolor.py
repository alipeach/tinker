# Ultralytics 🚀 AGPL - 3.0 License - https://ultralytics.com/license

import json
from pathlib import Path
from typing import List, Optional, Union, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta


def get_beijing_time():
    """获取本地时间加八小时"""
    local_time = datetime.now()
    beijing_time = local_time + timedelta(hours=0)
    return beijing_time


def segment_image(
    img_path: Union[str, Path],
    ann_path: Optional[Union[str, Path]] = None,
    yolo_model: str = "yolov8n-seg.pt",
    output_dir: Path = Path("segment_outputs"),
    device: str = "cuda:0",
    filter_colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = [(0, 255, 0)],  # 支持单颜色或颜色数组
    alpha: float = 0.3,
    conf_threshold: float = 0.5
) -> None:
    """
    对单张图片使用YOLO分割模型进行分割处理，为每种颜色生成独立结果
    
    Args:
        img_path: 图片路径
        ann_path: 可选，YOLO格式标注文件路径（用于过滤特定目标）
        yolo_model: YOLO分割模型路径或名称
        output_dir: 输出目录
        device: 运行设备
        filter_colors: 分割区域滤镜颜色 (B, G, R)，支持单颜色或颜色数组
        alpha: 透明度，范围0-1
        conf_threshold: 置信度阈值
    """
    # 确保filter_colors是列表格式
    if isinstance(filter_colors, tuple):
        filter_colors = [filter_colors]
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    
    # 加载YOLO分割模型
    model = YOLO(yolo_model)
    
    # 执行分割预测
    results = model(img, device=device, conf=conf_threshold, save=False, verbose=False)
    
    # 提取分割结果
    result = results[0]
    masks = result.masks  # 分割掩码
    if masks is None:
        print("未检测到可分割的目标")
        # 为每种颜色生成包含原始图像的结果
        for filter_color in filter_colors:
            r, g, b = filter_color
            color_suffix = f"R{r}G{g}B{b}"
            output_path = output_dir / f"{img_path.stem}_segmented_{color_suffix}.jpg"
            cv2.imwrite(str(output_path), img)
        return
    
    # 如果提供了标注文件，仅保留标注中指定类别的分割结果
    selected_indices = None
    if ann_path and Path(ann_path).exists():
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        # 提取标注中的类别ID
        annotated_classes = set()
        for line in lines:
            parts = line.strip().split()
            if parts:  # 确保行不为空
                annotated_classes.add(int(parts[0]))
        
        # 找到与标注类别匹配的索引
        selected_indices = [i for i, cls in enumerate(result.boxes.cls.tolist()) 
                          if int(cls) in annotated_classes]
        
        if not selected_indices:
            print("未找到与标注匹配的目标")
            # 为每种颜色生成包含原始图像的结果
            for filter_color in filter_colors:
                r, g, b = filter_color
                color_suffix = f"R{r}G{g}B{b}"
                output_path = output_dir / f"{img_path.stem}_segmented_{color_suffix}.jpg"
                cv2.imwrite(str(output_path), img)
            return
    
    # 为每种颜色生成独立的分割结果
    for filter_color in filter_colors:
        # 复制原始图像用于当前颜色的渲染
        rendered_img = img.copy()
        overlay = rendered_img.copy()
        
        # 对所有选中的分割区域应用当前颜色
        for i, mask in enumerate(masks.xy):
            # 如果有选中的索引且当前索引不在其中，则跳过
            if selected_indices is not None and i not in selected_indices:
                continue
                
            # 填充分割区域，使用当前颜色
            cv2.fillPoly(overlay, [mask.astype(np.int32)], color=filter_color)
        
        # 混合原图和分割层
        cv2.addWeighted(overlay, alpha, rendered_img, 1 - alpha, 0, rendered_img)
        
        # 生成包含当前颜色信息的输出文件名
        b, g, r = filter_color
        color_suffix = f"R{r}G{g}B{b}"
        output_path = output_dir / f"{img_path.stem}_segmented_{color_suffix}.jpg"
        cv2.imwrite(str(output_path), rendered_img)
        print(f"分割结果已保存至: {output_path}")
        


def process_single_image(
    img_path: Union[str, Path],
    ann_path: Optional[Union[str, Path]] = None,
    yolo_model: str = "yolov8n-seg.pt",
    device: str = "cuda:0",
    output_dir: Optional[Union[str, Path]] = None,
    filter_colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = [(0, 255, 0)],  # 支持单颜色或颜色数组
    alpha: float = 0.3,
    conf_threshold: float = 0.5
) -> None:
    """
    处理单张图片的主函数，使用YOLO分割模型为每种颜色生成独立结果
    
    Args:
        img_path: 图片路径
        ann_path: 可选，标注文件路径（用于过滤特定目标）
        yolo_model: YOLO分割模型路径或名称
        device: 运行设备
        output_dir: 输出目录
        filter_colors: 分割区域颜色 (B, G, R)，支持单颜色或颜色数组
        alpha: 透明度
        conf_threshold: 置信度阈值
    """
    start_time = datetime.now()
    img_path = Path(img_path)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = img_path.parent / f"{img_path.stem}_segment_results"
    output_dir = Path(output_dir)
    
    # 执行分割
    segment_image(
        img_path=img_path,
        ann_path=ann_path,
        yolo_model=yolo_model,
        output_dir=output_dir,
        device=device,
        filter_colors=filter_colors,
        alpha=alpha,
        conf_threshold=conf_threshold
    )
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"所有颜色处理完成，耗时 {elapsed_time.total_seconds():.2f} 秒")


if __name__ == "__main__":
    # 示例参数
    # 示例参数
    img_path = "/Users/apple/Blibli/2025-09/20250915-01-在你身后的巨猫/000001.jpg"  # 输入图片路径
    ann_path = "/Users/apple/Blibli/2025-09/20250915-01-在你身后的巨猫/000001.txt"  # 默认为同路径下同名.txt文件
    yolo_model_path = "/Users/apple/work/models-yolo11/yolo11m-seg.pt"  # seg模型路径
    output_dir = "/Users/apple/Blibli/2025-09/20250915-01-在你身后的巨猫"  # 输出目录
    device = "mps"  # 运行设备，CPU可设置为"cpu"
    
    # 颜色数组 - 每种颜色将生成一张独立的分割结果图片
    filter_colors = [
        (0, 0, 255),    # 红色
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (255, 255, 0)   # 黄色
    ]

    # YOLO分割模型默认使用的颜色（RGB格式）
    default_seg_colors = [
        (0, 0, 0),        # 黑色
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 洋红色
        (0, 255, 255),    # 青色
        (128, 0, 0),      # 深红色
        (0, 128, 0),      # 深绿色
        (0, 0, 128),      # 深蓝色
        (128, 128, 0),    # 深黄色
        (128, 0, 128),    # 深洋红色
        (0, 128, 128),    # 深青色
        (192, 192, 192),  # 银色
        (128, 128, 128),  # 灰色
        (64, 0, 0),       # 暗红色
        (0, 64, 0),       # 暗绿色
        (0, 0, 64),       # 暗蓝色
        (64, 64, 0),      # 暗黄色
        (64, 0, 64)       # 暗洋红色
    ]
    
    alpha = 0.4  # 透明度
    conf_threshold = 0.5  # 置信度阈值
    
    # 执行处理
    process_single_image(
        img_path=img_path,
        ann_path=ann_path,
        yolo_model=yolo_model_path,
        device=device,
        output_dir=output_dir,
        filter_colors=default_seg_colors,
        alpha=alpha,
        conf_threshold=conf_threshold
    )
    