# Ultralytics 🚀 AGPL - 3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime, timedelta
from ultralytics import SAM, YOLO


def get_beijing_time():
    """获取本地时间加八小时（北京时间）"""
    return datetime.now() + timedelta(hours=8)


def convert_to_yolo_format(boxes, img_width, img_height, class_ids):
    """将检测结果转换为YOLO标注格式"""
    yolo_lines = []
    for i, box in enumerate(boxes):
        x_center = (box[0] + box[2]) / (2 * img_width)
        y_center = (box[1] + box[3]) / (2 * img_height)
        width = (box[2] - box[0]) / img_width
        height = (box[3] - box[1]) / img_height
        yolo_lines.append(f"{class_ids[i]} {x_center} {y_center} {width} {height}")
    return yolo_lines


def convert_yolo_to_boxes(yolo_lines, img_width, img_height):
    """将YOLO标注格式转换为边界框坐标"""
    boxes, class_ids = [], []
    for line in yolo_lines:
        parts = line.strip().split()
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


def split_and_detect(
    video_path: Union[str, Path],
    det_model: YOLO,
    output_dir: Path,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    classes: Optional[List[int]],
    max_split_frames: int
):
    """拆帧并进行目标检测，返回处理的帧数"""
    # 创建保存目录
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # 视频拆帧
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total_frames if max_split_frames == -1 else min(max_split_frames, total_frames)

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        if frame_count % 100 == 0:
            print(f"拆帧进度：{frame_count}/{max_frames}")
            
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 保存原始帧
        cv2.imwrite(str(images_dir / f"{frame_count:06d}.jpg"), frame)

    cap.release()
    print(f"拆帧完成，共{max_frames}帧")

    # 目标检测
    for frame_num in range(1, frame_count + 1):
        if frame_num % 100 == 0:
            print(f"检测进度：{frame_num}/{max_frames}")

        frame = cv2.imread(str(images_dir / f"{frame_num:06d}.jpg"))
        img_height, img_width = frame.shape[:2]

        # 模型检测
        result = det_model(frame, device=device, conf=conf, iou=iou, 
                          imgsz=imgsz, max_det=max_det, classes=classes, verbose=False)[0]
        
        # 处理检测结果
        class_ids = result.boxes.cls.int().tolist() if result.boxes.cls.numel() > 0 else []
        yolo_lines = convert_to_yolo_format(result.boxes.xyxy.cpu().numpy().tolist(), 
                                           img_width, img_height, class_ids) if class_ids else []

        # 保存标注结果
        with open(labels_dir / f"{frame_num:06d}.txt", 'w') as f:
            f.write('\n'.join(yolo_lines))

    return frame_count


def fill_missing_detections(labels_dir: Path, max_frames: int):
    """
    处理无检测结果的帧：逐帧检查，仅当空帧的前后五帧均存在有效检测结果时才补充
    在检测完成后、分割开始前执行
    """
    success_count = 0
    fail_count = 0

    # 逐帧检查（从1到最大帧数）
    for frame_num in range(1, max_frames + 1):
        label_path = labels_dir / f"{frame_num:06d}.txt"
        
        # 检查当前帧是否为空标注（不存在或大小为0）
        if label_path.exists() and label_path.stat().st_size > 0:
            continue  # 非空帧直接跳过
        
        # 查找最近的有效前帧（前5帧内）
        closest_prev = None
        for offset in range(1, 6):
            prev_num = frame_num - offset
            if prev_num < 1:
                break  # 超出起始范围
            prev_path = labels_dir / f"{prev_num:06d}.txt"
            if prev_path.exists() and prev_path.stat().st_size > 0:
                closest_prev = prev_num
                break  # 找到最近的有效前帧即停止

        # 查找最近的有效后帧（后5帧内）
        closest_next = None
        for offset in range(1, 6):
            next_num = frame_num + offset
            if next_num > max_frames:
                break  # 超出最大范围
            next_path = labels_dir / f"{next_num:06d}.txt"
            if next_path.exists() and next_path.stat().st_size > 0:
                closest_next = next_num
                break  # 找到最近的有效后帧即停止

        # 仅当前后均有有效帧时才补充标注（优先使用前帧）
        if closest_prev is not None and closest_next is not None:
            with open(labels_dir / f"{closest_prev:06d}.txt", 'r') as f_prev, \
                 open(label_path, 'w') as f_curr:
                f_curr.write(f_prev.read())
            success_count += 1
        else:
            fail_count += 1
            # 控制警告输出频率，避免刷屏
            if fail_count <= 10 or fail_count % 100 == 0:
                print(f"警告：帧{frame_num}前后5帧未同时存在有效标注，无法补充（累计{fail_count}个）")

    print(f"缺失标注处理完成：成功补充{success_count}帧，无法补充{fail_count}帧")


def segment_frames(
    video_path: Union[str, Path],
    sam_model: SAM,
    input_dir: Path,
    output_path: Path,
    device: str,
    num_sam_frames: int,
    max_split_frames: int,
    color: tuple,
    alpha: float
):
    """基于检测结果进行分割并生成带掩码的视频（使用预处理后的标注）"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 确定需要处理的帧数
    process_frames = max_split_frames if num_sam_frames == -1 else min(num_sam_frames, max_split_frames)
    images_dir, labels_dir = input_dir / "images", input_dir / "labels"

    for frame_num in range(1, process_frames + 1):
        if frame_num % 100 == 0:
            print(f"分割进度：{frame_num}/{process_frames}")

        # 读取帧和预处理后的检测结果
        frame = cv2.imread(str(images_dir / f"{frame_num:06d}.jpg"))
        img_height, img_width = frame.shape[:2]
        
        with open(labels_dir / f"{frame_num:06d}.txt", 'r') as f:
            yolo_lines = f.readlines()

        boxes, class_ids = convert_yolo_to_boxes(yolo_lines, img_width, img_height)

        # 执行分割并渲染
        rendered_frame = frame.copy()
        if class_ids:
            # SAM分割
            sam_results = sam_model(frame, bboxes=np.array(boxes), verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            # 渲染掩码
            overlay = rendered_frame.copy()
            bgr_color = (color[2], color[1], color[0])  # RGB转BGR
            for segment in segments:
                segment = (segment * np.array([img_width, img_height])).astype(np.int32)
                try:
                    cv2.fillPoly(overlay, [segment], bgr_color)
                except cv2.error as e:
                    print(f"第{frame_num}帧绘制异常: {e}")

            # 混合图层
            cv2.addWeighted(overlay, alpha, rendered_frame, 1 - alpha, 0, rendered_frame)
        else:
            print(f"第{frame_num}帧无有效标注，直接使用原始帧")

        out.write(rendered_frame)

    out.release()
    cap.release()


def process_video(
    data: Union[str, Path],
    det_model: str,
    sam_model: str,
    device: str = "cuda:0",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 1,
    classes: Optional[List[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    num_sam_frames: int = -1,
    max_split_frames: int = -1,
    mask_color: tuple = (255, 0, 255),  # 掩码颜色(RGB)
    mask_alpha: float = 0.35  # 掩码透明度
) -> None:
    """处理视频的主函数：拆帧->检测->补充缺失标注->分割"""
    start_time = datetime.now()
    data = Path(data)
    
    # 初始化模型
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    # 配置输出目录
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_processed"
    output_dir = Path(output_dir)
    batch_dir = output_dir / get_beijing_time().strftime("%Y-%m-%d_%H-%M")
    batch_dir.mkdir(exist_ok=True, parents=True)

    # 1. 拆帧与检测
    frame_count = split_and_detect(
        video_path=data,
        det_model=det_model,
        output_dir=batch_dir,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        classes=classes,
        max_split_frames=max_split_frames
    )

    # 2. 补充缺失的检测结果（检测后、分割前执行）
    labels_dir = batch_dir / "labels"
    fill_missing_detections(labels_dir, frame_count)

    # 3. 执行分割并生成结果视频
    output_video = batch_dir / f"{data.stem}_{get_beijing_time().strftime('%Y%m%d%H%M%S')}_segmented.mp4"
    segment_frames(
        video_path=data,
        sam_model=sam_model,
        input_dir=batch_dir,
        output_path=output_video,
        device=device,
        num_sam_frames=num_sam_frames,
        max_split_frames=frame_count,
        color=mask_color,
        alpha=mask_alpha
    )

    # 输出执行信息
    elapsed_time = datetime.now() - start_time
    print(f"处理完成！共处理{frame_count}帧，耗时{elapsed_time}，结果保存至：{batch_dir}")


if __name__ == "__main__":
    # 配置参数
    data_path = r"C:\Users\liuzhuo\Workspace\project\1017-chaojia-cat\海边吵架的猫.mp4"
    output_dir = r"C:\Users\liuzhuo\Workspace\project\1017-chaojia-cat"
    
    det_model_name = r"C:\Users\liuzhuo\Workspace\model\yolo11x.pt"
    sam_model_name = r"C:\Users\liuzhuo\Workspace\model\sam2_l.pt"
    
    # 检测参数
    conf = 0.45
    classes = [14, 15, 16]
    max_det = 2
    max_split_frames = -1  # 拆帧数量，-1表示全部
    num_sam_frames = -1    # 分割数量，-1表示全部
    
    # 渲染参数
    mask_color = (255, 0, 255)  # RGB格式
    mask_alpha = 0.35

    # 执行处理
    process_video(
        data=data_path,
        det_model=det_model_name,
        sam_model=sam_model_name,
        output_dir=output_dir,
        conf=conf,
        max_det=max_det,
        classes=classes,
        num_sam_frames=num_sam_frames,
        max_split_frames=max_split_frames,
        mask_color=mask_color,
        mask_alpha=mask_alpha
    )