# Ultralytics 🚀 AGPL - 3.0 License - https://ultralytics.com/license

import json
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from ultralytics import SAM, YOLO, RTDETR
from datetime import datetime, timedelta


def get_beijing_time():
    """
    获取本地时间加八小时
    """
    local_time = datetime.now()
    beijing_time = local_time + timedelta(hours=8)
    return beijing_time


def convert_to_yolo_format(boxes, img_width, img_height, class_ids):
    """
    将检测结果转换为YOLO标注格式
    """
    yolo_lines = []
    for i, box in enumerate(boxes):
        x_center = (box[0] + box[2]) / (2 * img_width)
        y_center = (box[1] + box[3]) / (2 * img_height)
        width = (box[2] - box[0]) / img_width
        height = (box[3] - box[1]) / img_height
        yolo_lines.append(f"{class_ids[i]} {x_center} {y_center} {width} {height}")
    return yolo_lines


def split_and_detect(
    data: Union[str, Path],
    det_model: RTDETR,
    batch_dir: Path,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    classes: Optional[List[int]],
    max_split_frames: int
):
    """
    拆帧并进行目标检测
    """
    # 创建不同的子目录
    images_dir = batch_dir / "images"
    images_dir.mkdir(exist_ok=True)
    labels_dir = batch_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # 处理视频文件，拆帧保存原始文件
    cap = cv2.VideoCapture(str(data))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_split_frames == -1:
        max_split_frames = total_frames
    else:
        max_split_frames = min(max_split_frames, total_frames)

    frame_count = 0
    while cap.isOpened() and frame_count < max_split_frames:
        if frame_count % 100 == 0:
            print(f"当前拆帧进度：已拆分 {frame_count} 帧，共 {max_split_frames} 帧。")
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 保存原始帧
        raw_frame_path = images_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(str(raw_frame_path), frame)

    cap.release()
    print(f"拆帧完成，共拆了 {max_split_frames} 帧。即将开始检测。")

    for frame_num in range(1, frame_count + 1):
        if frame_num % 100 == 0:
            print(f"当前检测进度：已检测 {frame_num} 帧，共 {max_split_frames} 帧。")

        raw_frame_path = images_dir / f"{frame_num:06d}.jpg"
        frame = cv2.imread(str(raw_frame_path))
        img_height, img_width = frame.shape[:2]

        # 使用 detect 方法进行目标检测，关闭日志输出
        result = det_model(frame, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes, verbose=False)[0]
        class_ids = result.boxes.cls.int().tolist() if result.boxes.cls.numel() > 0 else []
        confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf.numel() > 0 else []

        if class_ids:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()

            # 保存检测结果到YOLO标注格式
            yolo_lines = convert_to_yolo_format(boxes, img_width, img_height, class_ids)
            label_path = labels_dir / f"{frame_num:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        else:
            # 没有检测结果，保存为空文件
            label_path = labels_dir / f"{frame_num:06d}.txt"
            with open(label_path, 'w') as f:
                pass

    return frame_count


def convert_yolo_to_boxes(yolo_lines, img_width, img_height):
    """
    将YOLO标注格式转换为边界框坐标
    """
    boxes = []
    class_ids = []
    for line in yolo_lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x1 = (x_center - width / 2) * img_width
        y1 = (y_center - height / 2) * img_height
        x2 = (x_center + width / 2) * img_width
        y2 = (y_center + height / 2) * img_height

        boxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
    return boxes, class_ids


def segment_frames(
    data: Union[str, Path],
    sam_model: SAM,
    batch_dir: Path,
    device: str,
    num_sam_frames: int,
    max_split_frames: int
):
    """
    基于拆帧和检测结果进行分割
    """
    cap = cv2.VideoCapture(str(data))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    timestamp = get_beijing_time().strftime("%Y%m%d%H%M%S")
    output_video_path = batch_dir / f"{data.stem}_{timestamp}_segmented.mp4"
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if num_sam_frames == -1:
        num_sam_frames = max_split_frames
    else:
        num_sam_frames = min(num_sam_frames, max_split_frames)

    images_dir = batch_dir / "images"
    labels_dir = batch_dir / "labels"

    empty_frames = []
    for frame_num in range(1, num_sam_frames + 1):
        if frame_num % 100 == 0:
            print(f"当前分割进度：已处理 {frame_num} 帧，共 {num_sam_frames} 帧。")

        raw_frame_path = images_dir / f"{frame_num:06d}.jpg"
        frame = cv2.imread(str(raw_frame_path))
        img_height, img_width = frame.shape[:2]

        # 从本地加载检测结果
        label_path = labels_dir / f"{frame_num:06d}.txt"
        with open(label_path, 'r') as f:
            yolo_lines = f.readlines()

        boxes, class_ids = convert_yolo_to_boxes(yolo_lines, img_width, img_height)
        confidences = [1.0] * len(class_ids)  # 假设置信度为1.0

        # 如果当前帧无检测结果，取前后5帧中有结果的一帧
        if not class_ids:
            for offset in range(1, 30):
                prev_frame_num = frame_num - offset
                next_frame_num = frame_num + offset
                prev_result = None
                next_result = None

                if prev_frame_num > 0:
                    prev_label_path = labels_dir / f"{prev_frame_num:06d}.txt"
                    with open(prev_label_path, 'r') as f:
                        prev_yolo_lines = f.readlines()
                        prev_boxes, prev_class_ids = convert_yolo_to_boxes(prev_yolo_lines, img_width, img_height)
                        if prev_class_ids:
                            prev_result = {"boxes": prev_boxes, "class_ids": prev_class_ids}
                if next_frame_num <= max_split_frames:
                    next_label_path = labels_dir / f"{next_frame_num:06d}.txt"
                    with open(next_label_path, 'r') as f:
                        next_yolo_lines = f.readlines()
                        next_boxes, next_class_ids = convert_yolo_to_boxes(next_yolo_lines, img_width, img_height)
                        if next_class_ids:
                            next_result = {"boxes": next_boxes, "class_ids": next_class_ids}

                if prev_result and prev_result["class_ids"]:
                    boxes = prev_result["boxes"]
                    class_ids = prev_result["class_ids"]
                    break
                elif next_result and next_result["class_ids"]:
                    boxes = next_result["boxes"]
                    class_ids = next_result["class_ids"]
                    break

        rendered_frame = frame
        if class_ids:
            if empty_frames:
                print(f"帧 {', '.join(map(str, empty_frames))} 无检测结果")
                empty_frames = []
            boxes = np.array(boxes)
            sam_results = sam_model(frame, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            # Render the masks on a new image with transparency
            rendered_frame = frame.copy()
            h, w = rendered_frame.shape[:2]
            overlay = rendered_frame.copy()
            alpha = 0.4  # 透明度，范围从 0 到 1，值越小越透明


            #紫色
            # color = (128, 0, 128)

             #紫色-2
            color=(204,0,153)
            #绿色
            # color = (0, 255, 0)
            #草绿色
            # color = (124, 252, 0)

            for segment in segments:
                segment = (segment * np.array([w, h])).astype(np.int32)
                try:
                    cv2.fillPoly(overlay, [segment], color = color)
                except cv2.error as e:
                    print(f"异常信息：第 {frame_num} 帧，fillPoly执行异常: {e}")

            # 将覆盖层与原图混合
            cv2.addWeighted(overlay, alpha, rendered_frame, 1 - alpha, 0, rendered_frame)

        else:
            empty_frames.append(frame_num)
            if len(empty_frames) == 10:
                print(f"帧 {', '.join(map(str, empty_frames))} 无检测结果")
                empty_frames = []

        out.write(rendered_frame)

    if empty_frames:
        print(f"帧 {', '.join(map(str, empty_frames))} 无检测结果")

    out.release()
    cap.release()


def auto_annotate(
    data: Union[str, Path],
    det_model: str = "/workspace/yolo11x.pt",
    sam_model: str = "/workspace/sam_b.pt",
    device: str = "cuda:0",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 3,
    classes: Optional[List[int]] = [15],
    output_dir: Optional[Union[str, Path]] = "/workspace/outputs",
    num_sam_frames: int = -1,
    max_split_frames: int = -1,
    mode: str = "detect_and_segment"
) -> None:
    """
    Automatically annotate a video using a YOLO object detection model and a SAM segmentation model.

    This function processes frames of a video, detects objects using a YOLO model,
    and then generates segmentation masks using a SAM model. The resulting masks are rendered on a new video.

    Args:
        data (str | Path): Path to a video file to be annotated.
        det_model (str): Path or name of the pre - trained YOLO detection model.
        sam_model (str): Path or name of the pre - trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        conf (float): Confidence threshold for detection model.
        iou (float): IoU threshold for filtering overlapping boxes in detection results.
        imgsz (int): Input image resize dimension.
        max_det (int): Maximum number of detections per image.
        classes (List[int] | None): Filter predictions to specified class IDs, returning only relevant detections.
        output_dir (str | Path | None): Directory to save the annotated results. If None, a default directory is created.
        num_sam_frames (int): Number of frames to process for SAM segmentation after detection. -1 means process all detected frames.
        max_split_frames (int): Maximum number of frames to split. -1 means split all frames.
        mode (str): Mode of operation. Can be 'detect', 'segment', or 'detect_and_segment'.
    """
    start_time = datetime.now()
    det_model = RTDETR(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    output_dir = Path(output_dir)

    # 获取本地时间加八小时并生成批次号目录，精确到分钟
    beijing_time = get_beijing_time()
    batch_dir_name = beijing_time.strftime("%Y-%m-%d_%H-%M")
    batch_dir = output_dir / batch_dir_name
    batch_dir.mkdir(exist_ok=True, parents=True)

    frame_count = 0
    if mode in ["detect", "detect_and_segment"]:
        # 拆帧并检测
        frame_count = split_and_detect(
            data, det_model, batch_dir, device, conf, iou, imgsz, max_det, classes, max_split_frames
        )

    if mode in ["segment", "detect_and_segment"]:
        if frame_count == 0:
            # 如果只进行分割，需要计算最大帧数
            cap = cv2.VideoCapture(str(data))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_split_frames == -1:
                frame_count = total_frames
            else:
                frame_count = min(max_split_frames, total_frames)
            cap.release()
        # 分割
        segment_frames(data, sam_model, batch_dir, device, num_sam_frames, frame_count)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    log_message = f"程序执行完成，处理了 {frame_count} 帧，耗时 {elapsed_time}。"
    print(log_message)


# 执行逻辑
# 1、原视频逐帧拆解，并保存在本地
# 2、使用检测模型，取本地帧进行检测，检测结果文本、渲染后的图像保存在本地
# 3、判断是否存在连续帧里面有漏检的情况，对漏检帧进行检测结果融合，并更新本地检测结果
# 4、取本地检测结果，进行分割，并保存为视频文件
# 5、支持调试，可以设置只拆部分帧和全量帧
if __name__ == "__main__":
    # 请根据实际情况修改这些参数
    # data_path = "/workspace/猫岛的猫-5.mp4"
    # output_dir = "/workspace/outputs/猫岛的猫-5"
    
    # data_path = "/workspace/猫岛的猫-6.mp4"
    # output_dir = "/workspace/outputs/猫岛的猫-6"

    # data_path = "/workspace/西湖的松鼠-01.mp4"
    # output_dir = "/workspace/outputs/西湖的松鼠-01"

    # data_path = "/workspace/cat-flight.mp4"
    # output_dir = "/workspace/outputs/cat-flight"

    data_path = "/workspace/white-fox.mp4"
    output_dir = "/workspace/outputs/white-fox"
    
    det_model_name = "/workspace/model/rtdetr-x.pt"
    sam_model_name = "/workspace/model/sam2_b.pt"
    # conf = 0.20
    # classes = [15,16,21]

    conf = 0.25
    classes = [16]
    # 拆帧并检测后，需要进行SAM处理的最大帧数
    num_sam_frames = -1
    # 直接指定最多拆多少帧
    max_split_frames = -1
    # 选择模式：'detect', 'segment', 'detect_and_segment'
    mode = 'detect_and_segment'

    auto_annotate(data=data_path, det_model=det_model_name, sam_model=sam_model_name, output_dir=output_dir,
                  conf=conf, classes=classes, num_sam_frames=num_sam_frames, max_split_frames=max_split_frames, mode=mode)
    