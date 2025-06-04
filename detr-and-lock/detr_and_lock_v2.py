# Ultralytics 🚀 AGPL - 3.0 License - https://ultralytics.com/license

#yolo track model=/Users/apple/Work/models - yolo11/yolo11x.pt source=猫岛的猫 - 5.mp4 device=mps show=true

from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from ultralytics import RTDETR
import datetime


def is_close(box1, box2, threshold=2.5):
    """
    判断两个bbox是否位置相近
    """
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    avg_width = (box1[2] - box1[0] + box2[2] - box2[0]) / 2
    return distance / avg_width < threshold


def interpolate_box(box1, box2, num_frames, frame_index):
    """
    对漏检的帧进行预测
    """
    x1 = box1[0] + (box2[0] - box1[0]) * frame_index / (num_frames + 1)
    y1 = box1[1] + (box2[1] - box1[1]) * frame_index / (num_frames + 1)
    x2 = box1[2] + (box2[2] - box1[2]) * frame_index / (num_frames + 1)
    y2 = box1[3] + (box2[3] - box1[3]) * frame_index / (num_frames + 1)
    return [x1, y1, x2, y2]


def auto_annotate(
    data: Union[str, Path],
    det_model: str = "/workspace/yolo11x.pt",
    device: str = "cuda:0",
    conf: float = 0.25,
    iou: float = 0.45,
    # iou: float = 0.75,
    imgsz: int = 640,
    max_det: int = 1,
    classes: Optional[List[int]] = [15],
    output_dir: Optional[Union[str, Path]] = "/workspace/outputs",
) -> None:
    """
    Automatically annotate a video using a YOLO object detection model.

    This function processes frames of a video, detects objects using a YOLO model,
    and then generates a new video with crosshair lock boxes on detected objects.

    Args:
        data (str | Path): Path to a video file to be annotated.
        det_model (str): Path or name of the pre - trained YOLO detection model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        conf (float): Confidence threshold for detection model.
        iou (float): IoU threshold for filtering overlapping boxes in detection results.
        imgsz (int): Input image resize dimension.
        max_det (int): Maximum number of detections per image.
        classes (List[int] | None): Filter predictions to specified class IDs, returning only relevant detections.
        output_dir (str | Path | None): Directory to save the annotated results. If None, a default directory is created.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="video.mp4", det_model="yolo11n.pt")
    """
    det_model = RTDETR(det_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 创建label目录
    label_dir = output_dir / "label"
    label_dir.mkdir(exist_ok=True, parents=True)

    # 处理视频文件
    cap = cv2.VideoCapture(str(data))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 根据时间戳生成新的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_video_path = output_dir / f"{data.stem}_{timestamp}_processed.mp4"
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # 保存每一帧的检测结果
    all_results = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"正在处理第 {frame_count} 帧，总共 {total_frames} 帧")
        # 使用 detect 方法
        result = det_model(frame, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes,
                           verbose=False)[0]
        class_ids = result.boxes.cls.int().tolist()  # noqa
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) if len(result.boxes) > 0 else []
        all_results.append((class_ids, boxes))

    # 处理漏检的帧
    for i in range(len(all_results)):
        if len(all_results[i][1]) == 0:
            # 向前查找最近的检测结果
            prev_index = None
            for j in range(i - 1, max(-1, i - 15), -1):
                if len(all_results[j][1]) > 0:
                    prev_index = j
                    break
            # 向后查找最近的检测结果
            next_index = None
            for j in range(i + 1, min(len(all_results), i + 15)):
                if len(all_results[j][1]) > 0:
                    next_index = j
                    break
            if prev_index is not None and next_index is not None:
                prev_box = all_results[prev_index][1][0]
                next_box = all_results[next_index][1][0]
                if is_close(prev_box, next_box):
                    num_frames = next_index - prev_index - 1
                    frame_index = i - prev_index
                    interpolated_box = interpolate_box(prev_box, next_box, num_frames, frame_index)
                    # 考虑前后帧边界问题
                    interpolated_box = [max(0, coord) for coord in interpolated_box]
                    interpolated_box[2] = min(width, interpolated_box[2])
                    interpolated_box[3] = min(height, interpolated_box[3])
                    all_results[i] = (all_results[prev_index][0], [np.array(interpolated_box).astype(int)])
                    print(f"第 {i + 1} 帧漏检，已根据第 {prev_index + 1} 帧和第 {next_index + 1} 帧预测检测结果")

    # 渲染矩形框和军事瞄准线
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"正在渲染第 {frame_count} 帧的矩形框和军事瞄准线，总共 {total_frames} 帧")
        class_ids, boxes = all_results[frame_count - 1]

        # 保存检测结果到label目录
        label_file_path = label_dir / f"{frame_count:06d}.txt"
        with open(label_file_path, 'w') as f:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                class_id = class_ids[i]
                f.write(f"{class_id} {x1} {y1} {x2} {y2}\n")

        # 渲染矩形框和军事瞄准线
        rendered_frame = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 控制四个直角大小比例的因子
            corner_ratio = 0.3  # 可以根据需要调整这个比例
            corner_length_x = int((x2 - x1) * corner_ratio)
            corner_length_y = int((y2 - y1) * corner_ratio)

            # 绘制矩形框（只展示四个直角）
            rectangle_color = (0, 255, 0)  # 绿色
            cv2.line(rendered_frame, (x1, y1), (x1 + corner_length_x, y1), rectangle_color, 2)
            cv2.line(rendered_frame, (x1, y1), (x1, y1 + corner_length_y), rectangle_color, 2)
            cv2.line(rendered_frame, (x2, y1), (x2 - corner_length_x, y1), rectangle_color, 2)
            cv2.line(rendered_frame, (x2, y1), (x2, y1 + corner_length_y), rectangle_color, 2)
            cv2.line(rendered_frame, (x1, y2), (x1 + corner_length_x, y2), rectangle_color, 2)
            cv2.line(rendered_frame, (x1, y2), (x1, y2 - corner_length_y), rectangle_color, 2)
            cv2.line(rendered_frame, (x2, y2), (x2 - corner_length_x, y2), rectangle_color, 2)
            cv2.line(rendered_frame, (x2, y2), (x2, y2 - corner_length_y), rectangle_color, 2)

            # 绘制十字线（虚线）
            crosshair_color = (128, 128, 128)  # 比矩形框颜色浅的黑色
            dash_length = 10
            gap_length = 10
            # 缩小十字线的范围，例如缩小为原来的0.8倍
            shrink_factor = 0.5
            new_x1 = int(center_x - (center_x - x1) * shrink_factor)
            new_x2 = int(center_x + (x2 - center_x) * shrink_factor)
            new_y1 = int(center_y - (center_y - y1) * shrink_factor)
            new_y2 = int(center_y + (y2 - center_y) * shrink_factor)

            for i in range(new_y1, new_y2, dash_length + gap_length):
                end_y = min(i + dash_length, new_y2)
                cv2.line(rendered_frame, (center_x, i), (center_x, end_y), crosshair_color, 2)
            for i in range(new_x1, new_x2, dash_length + gap_length):
                end_x = min(i + dash_length, new_x2)
                cv2.line(rendered_frame, (i, center_y), (end_x, center_y), crosshair_color, 2)

        out.write(rendered_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    # 请根据实际情况修改这些参数
    data_path = "/workspace/fpv追船.mp4"
    output_dir = "/workspace/outputs/fpv追船"
    det_model_name = "/workspace/model/rtdetr-x.pt"
    # det_model_name = "/workspace/model/yolo11x.pt"
    classes = [8]
    conf = 0.75
    auto_annotate(data=data_path, det_model=det_model_name, output_dir=output_dir, classes=classes, conf=conf)