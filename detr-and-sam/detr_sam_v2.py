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


def save_detection_result(frame_count, result, batch_dir):
    """
    保存每一帧的检测结果到本地
    """
    frame_result = {
        "frame_count": frame_count,
        "class_ids": result.boxes.cls.int().tolist() if result.boxes.cls.numel() > 0 else [],
        "boxes": result.boxes.xyxy.cpu().numpy().tolist() if result.boxes.xyxy.numel() > 0 else [],
        "confidences": result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf.numel() > 0 else []
    }
    frame_result_path = batch_dir / "detection_results" / f"{frame_count:06d}_detection_result.json"
    with open(frame_result_path, 'w') as f:
        json.dump(frame_result, f)


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
    raw_frames_dir = batch_dir / "raw_frames"
    raw_frames_dir.mkdir(exist_ok=True)
    detection_results_dir = batch_dir / "detection_results"
    detection_results_dir.mkdir(exist_ok=True)
    detection_rendered_dir = batch_dir / "detection_rendered"
    detection_rendered_dir.mkdir(exist_ok=True)

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

        # 注释掉图像清晰度提升的代码
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # sharpened_frame = cv2.filter2D(frame, -1, kernel)

        # 保存原始帧
        raw_frame_path = raw_frames_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(str(raw_frame_path), frame)

    cap.release()
    print(f"拆帧完成，共拆了 {max_split_frames} 帧。即将开始检测。")

    for frame_num in range(1, frame_count + 1):
        if frame_num % 100 == 0:
            print(f"当前检测进度：已检测 {frame_num} 帧，共 {max_split_frames} 帧。")

        raw_frame_path = raw_frames_dir / f"{frame_num:06d}.jpg"
        frame = cv2.imread(str(raw_frame_path))

        # 使用 detect 方法进行目标检测，关闭日志输出
        result = det_model(frame, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes, verbose=False)[0]
        class_ids = result.boxes.cls.int().tolist() if result.boxes.cls.numel() > 0 else []
        confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf.numel() > 0 else []

        if class_ids:
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            # 这里可以添加简单的画框逻辑，仅为了保存检测结果可视化
            for i, box in enumerate(boxes):
                box = box.cpu().numpy().astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # 加上分类ID以及置信度
                text = f"Class: {class_ids[i]}, Conf: {confidences[i]:.2f}"
                cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存检测结果渲染后的文件
        detection_rendered_path = detection_rendered_dir / f"{frame_num:06d}.jpg"
        cv2.imwrite(str(detection_rendered_path), frame)

        # 保存每一帧的检测结果文本
        frame_text_output_path = detection_results_dir / f"{frame_num:06d}_detection.txt"
        with open(frame_text_output_path, 'w') as f:
            if class_ids:
                for i in range(len(class_ids)):
                    box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    conf_score = result.boxes.conf[i].cpu().numpy()
                    f.write(f"Class: {class_ids[i]}, Box: {box}, Confidence: {conf_score}\n")
            else:
                f.write("No detections\n")

        # 保存检测结果到本地
        save_detection_result(frame_num, result, batch_dir)

    return frame_count

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

    raw_frames_dir = batch_dir / "raw_frames"
    detection_results_dir = batch_dir / "detection_results"

    empty_frames = []
    for frame_num in range(1, num_sam_frames + 1):
        if frame_num % 100 == 0:
            print(f"当前分割进度：已处理 {frame_num} 帧，共 {num_sam_frames} 帧。")

        raw_frame_path = raw_frames_dir / f"{frame_num:06d}.jpg"
        frame = cv2.imread(str(raw_frame_path))

        # 从本地加载检测结果
        frame_result_path = detection_results_dir / f"{frame_num:06d}_detection_result.json"
        with open(frame_result_path, 'r') as f:
            frame_result = json.load(f)
        class_ids = frame_result["class_ids"]
        confidences = frame_result["confidences"]


        # 如果当前帧无检测结果，取前后帧中有结果的一帧
        if not class_ids:
            prev_frame_num = frame_num - 1
            next_frame_num = frame_num + 1
            prev_result = None
            next_result = None

            if prev_frame_num > 0:
                prev_result_path = detection_results_dir / f"{prev_frame_num:06d}_detection_result.json"
                with open(prev_result_path, 'r') as f:
                    prev_result = json.load(f)
            if next_frame_num <= max_split_frames:
                next_result_path = detection_results_dir / f"{next_frame_num:06d}_detection_result.json"
                with open(next_result_path, 'r') as f:
                    next_result = json.load(f)

            if prev_result and prev_result["class_ids"]:
                frame_result = prev_result
            elif next_result and next_result["class_ids"]:
                frame_result = next_result

            class_ids = frame_result["class_ids"]
            confidences = frame_result["confidences"]

        
        rendered_frame = frame
        if class_ids:
            if empty_frames:
                print(f"帧 {', '.join(map(str, empty_frames))} 无检测结果")
                empty_frames = []
            boxes = np.array(frame_result["boxes"])
            sam_results = sam_model(frame, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            # Render the masks on a new image with transparency
            rendered_frame = frame.copy()
            h, w = rendered_frame.shape[:2]
            overlay = rendered_frame.copy()
            alpha = 0.3  # 透明度，范围从 0 到 1，值越小越透明

            for segment in segments:
                segment = (segment * np.array([w, h])).astype(np.int32)
                cv2.fillPoly(overlay, [segment], color=(0, 255, 0))

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
    max_det: int = 10,
    classes: Optional[List[int]] = [15],
    output_dir: Optional[Union[str, Path]] = "/workspace/outputs",
    num_sam_frames: int = -1,
    max_split_frames: int = -1
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

    # 拆帧并检测
    frame_count = split_and_detect(
        data, det_model, batch_dir, device, conf, iou, imgsz, max_det, classes, max_split_frames
    )

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
    data_path = "/workspace/猫岛的猫-6.mp4"
    output_dir = "/workspace/outputs/猫岛的猫-6"
    det_model_name = "/workspace/model/rtdetr-x.pt"
    sam_model_name = "/workspace/model/sam2_l.pt"
    conf = 0.15
    classes = [15,16,21]
    # 拆帧并检测后，需要进行SAM处理的最大帧数
    num_sam_frames = -1
    # 直接指定最多拆多少帧
    max_split_frames = -1

    auto_annotate(data=data_path, det_model=det_model_name, sam_model=sam_model_name, output_dir=output_dir,
                  conf=conf, classes=classes, num_sam_frames=num_sam_frames, max_split_frames=max_split_frames)

    