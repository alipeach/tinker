import cv2
import numpy as np
from ultralytics import SAM
from pathlib import Path
from datetime import datetime, timedelta

def get_beijing_time():
    """
    获取本地时间加八小时
    """
    local_time = datetime.now()
    beijing_time = local_time + timedelta(hours=8)
    return beijing_time

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
    data: str,
    sam_model: SAM,
    parent_dir: Path,
    device: str,
    num_sam_frames: int
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
    output_video_path = parent_dir / f"{Path(data).stem}_{timestamp}_segmented.mp4"
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_sam_frames == -1:
        num_sam_frames = total_frames
    else:
        num_sam_frames = min(num_sam_frames, total_frames)

    images_dir = parent_dir / "images"
    labels_dir = parent_dir / "labels"

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

        # 如果当前帧无检测结果，取前后15帧中有结果的一帧
        if not class_ids:
            for offset in range(1, 15):
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
                if next_frame_num <= total_frames:
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
            alpha = 0.3  # 透明度，范围从 0 到 1，值越小越透明
            color=(204,0,153)
            
            for segment in segments:
                segment = (segment * np.array([w, h])).astype(np.int32)
                try:
                    cv2.fillPoly(overlay, [segment], color=color)
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


if __name__ == "__main__":
    # 请根据实际情况修改这些参数
    data_path = "/workspace/猫和羊的友谊.mp4"
    parent_dir = Path("/workspace/outputs/猫和羊的友谊/2025-07-21_14-14")
    sam_model_name = "/workspace/model/sam2_l.pt"
    device = "cuda:0"
    # 拆帧并检测后，需要进行SAM处理的最大帧数
    num_sam_frames = -1

    sam_model = SAM(sam_model_name)
    segment_frames(data_path, sam_model, parent_dir, device, num_sam_frames)
    