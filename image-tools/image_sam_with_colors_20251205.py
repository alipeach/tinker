# Ultralytics 🚀 AGPL - 3.0 License - https://ultralytics.com/license

import json
from pathlib import Path
from typing import List, Optional, Union, Tuple
import cv2
import numpy as np
from ultralytics import SAM  # 移除YOLO导入（仅保留SAM）
from datetime import datetime, timedelta


def get_beijing_time():
    """获取本地时间加八小时"""
    local_time = datetime.now()
    beijing_time = local_time + timedelta(hours=8)
    return beijing_time


def convert_yolo_to_boxes(yolo_lines, img_width, img_height):
    """将YOLO标注格式转换为边界框坐标（从detect_and_sam-1024移植）"""
    boxes, class_ids = [], []
    for line in yolo_lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        x_center, y_center = float(parts[1]), float(parts[2])
        width, height = float(parts[3]), float(parts[4])

        x1 = (x_center - width / 2) * img_width
        y1 = (y_center - height / 2) * img_height
        x2 = (x_center + width / 2) * img_width
        y2 = (y_center + height / 2) * img_height

        boxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
    return boxes, class_ids


def segment_image(
        img_path: Union[str, Path],
        ann_path: Union[str, Path],  # 改为必传参数（必须提供标注文件）
        sam_model: str = "sam2_l.pt",
        output_dir: Path = Path("segment_outputs"),
        device: str = "cuda:0",
        filter_colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = [(0, 255, 0)],
        alpha: float = 0.3
) -> None:
    """
    直接基于标注文件的边界框调用SAM进行分割，为每种颜色生成独立结果

    核心逻辑变更：
    1. 移除YOLO检测逻辑，完全依赖标注文件生成边界框
    2. 标注文件为必传参数，无标注文件则直接报错
    3. 基于标注边界框调用SAM进行精确分割
    4. 保留多颜色渲染功能
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
    img_height, img_width = img.shape[:2]

    # 1. 从标注文件解析边界框（核心：替代YOLO检测逻辑）
    if not Path(ann_path).exists():
        raise FileNotFoundError(f"标注文件不存在: {ann_path}")

    with open(ann_path, 'r') as f:
        yolo_lines = f.readlines()

    boxes, class_ids = convert_yolo_to_boxes(yolo_lines, img_width, img_height)
    if not boxes:
        raise ValueError(f"标注文件中未解析到有效边界框: {ann_path}")

    # 转换为numpy数组（适配SAM输入格式）
    boxes_np = np.array(boxes)

    # 2. 调用SAM进行分割（基于标注文件的边界框）
    sam = SAM(sam_model)
    sam_results = sam(img, bboxes=boxes_np, verbose=False, save=False, device=device)
    segments = sam_results[0].masks.xyn  # 获取归一化的分割掩码

    # 处理无分割结果的情况
    if not segments:
        print("基于标注边界框未检测到可分割的目标")
        for filter_color in filter_colors:
            r, g, b = filter_color
            color_suffix = f"({r},{g},{b})"
            output_path = output_dir / f"{img_path.stem}_segmented_{color_suffix}.jpg"
            cv2.imwrite(str(output_path), img)
        return

    # 3. 多颜色渲染（保留原多颜色功能）
    for index, filter_color in enumerate(filter_colors):
        rendered_img = img.copy()
        overlay = rendered_img.copy()

        # 对每个分割区域应用当前颜色
        for segment in segments:
            # 将归一化坐标转换为图像实际坐标
            segment_coords = (segment * np.array([img_width, img_height])).astype(np.int32)
            # RGB转BGR（OpenCV颜色通道格式）
            bgr_color = (filter_color[2], filter_color[1], filter_color[0])
            cv2.fillPoly(overlay, [segment_coords], color=bgr_color)

        # 混合原图和分割层
        cv2.addWeighted(overlay, alpha, rendered_img, 1 - alpha, 0, rendered_img)

        # 保存结果
        r, g, b = filter_color
        color_suffix = f"({r},{g},{b})"
        output_path = output_dir / f"{img_path.stem}-{index}-{color_suffix}.jpg"
        cv2.imwrite(str(output_path), rendered_img)
        print(f"分割结果已保存至: {output_path}")


def process_single_image(
        img_path: Union[str, Path],
        ann_path: Union[str, Path],  # 改为必传参数
        sam_model: str = "sam2_l.pt",
        device: str = "cuda:0",
        output_dir: Optional[Union[str, Path]] = None,
        filter_colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = [(0, 255, 0)],
        alpha: float = 0.3
) -> None:
    """处理单张图片的主函数，直接基于标注文件边界框+SAM分割+多颜色渲染"""
    start_time = datetime.now()
    img_path = Path(img_path)
    ann_path = Path(ann_path)

    if output_dir is None:
        output_dir = img_path.parent / f"{img_path.stem}_segment_results"
    output_dir = Path(output_dir)

    # 执行分割
    segment_image(
        img_path=img_path,
        ann_path=ann_path,
        sam_model=sam_model,
        output_dir=output_dir,
        device=device,
        filter_colors=filter_colors,
        alpha=alpha
    )

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"所有颜色处理完成，耗时 {elapsed_time.total_seconds():.2f} 秒")


if __name__ == "__main__":

    # 示例参数
    # 输入图片路径
    img_path = "xxx"
    # 默认为同路径下同名.txt文件  
    ann_path = "xxx" 

    #分割模型路径
    sam_model_path = "xxx"
    # 输出目录
    output_dir = "xxx"  
    device = "cuda:0"  

    filter_colors = [
        (220, 20, 60),  # 鲜明的红色，用于第一类目标
        (119, 11, 32),  # 深红色，与红色有区分，可用于第二类
        (0, 0, 255),  # 蓝色，适合第三类
        (0, 255, 0),  # 绿色，代表第四类
        (255, 255, 0),  # 黄色，用于第五类
        (255, 165, 0),  # 橙色，第六类
        (128, 0, 128),  # 紫色，第七类
        (255, 0, 255),  # 品红色，第八类
        (0, 255, 255),  # 青色，第九类
        (139, 69, 19),  # 棕色，第十类
        (127, 255, 212),  # 浅蓝绿色，第十一类
        (144, 238, 144),  # 淡绿色，第十二类
        (255, 105, 180),  # 浅粉红色，第十三类
        (240, 230, 140),  # 米色，第十四类
        (255, 228, 181),  # 浅黄色，第十五类
        (173, 255, 47),  # 亮绿色偏黄，第十六类
        (100, 149, 237),  # 淡蓝色，第十七类
        (218, 112, 214),  # 浅紫色，第十八类
        (199, 21, 133),  # 深粉色，第十九类
        (255, 248, 220)  # 象牙色，第二十类
    ]

    alpha = 0.4

    # 执行处理
    process_single_image(
        img_path=img_path,
        ann_path=ann_path,
        sam_model=sam_model_path,
        device=device,
        output_dir=output_dir,
        filter_colors=filter_colors,
        alpha=alpha
    )