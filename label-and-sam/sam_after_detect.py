import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from segment_anything import SamPredictor, sam_model_registry

def get_beijing_time():
    """获取本地时间加八小时"""
    local_time = datetime.now()
    beijing_time = local_time + timedelta(hours=0)
    return beijing_time

def convert_yolo_to_boxes(yolo_lines, img_width, img_height):
    """将YOLO标注格式转换为边界框坐标"""
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
    sam_predictor: SamPredictor,
    parent_dir: Path,
    device: str,
    num_sam_frames: int,
    seg_dir:Path
):
    """基于拆帧和检测结果进行分割"""
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
        if frame is None:
            print(f"无法读取帧 {frame_num} 的图像文件")
            continue
            
        img_height, img_width = frame.shape[:2]

        # 从本地加载检测结果
        label_path = labels_dir / f"{frame_num:06d}.txt"
        try:
            with open(label_path, 'r') as f:
                yolo_lines = f.readlines()
        except FileNotFoundError:
            print(f"帧 {frame_num} 的标签文件不存在")
            yolo_lines = []

        boxes, class_ids = convert_yolo_to_boxes(yolo_lines, img_width, img_height)

        # 如果当前帧无检测结果，在前后几帧都有结果的前提下，取最近的检测结果
        prev_result = None
        next_result = None
        if not class_ids:
            for offset in range(1, 5):
                prev_frame_num = frame_num - offset
                next_frame_num = frame_num + offset

                if prev_frame_num > 0:
                    prev_label_path = labels_dir / f"{prev_frame_num:06d}.txt"
                    if prev_label_path.exists():
                        with open(prev_label_path, 'r') as f:
                            prev_yolo_lines = f.readlines()
                            prev_boxes, prev_class_ids = convert_yolo_to_boxes(prev_yolo_lines, img_width, img_height)
                            if prev_class_ids is not None:
                                prev_result = {"boxes": prev_boxes, "class_ids": prev_class_ids}
                if next_frame_num <= total_frames:
                    next_label_path = labels_dir / f"{next_frame_num:06d}.txt"
                    if next_label_path.exists():
                        with open(next_label_path, 'r') as f:
                            next_yolo_lines = f.readlines()
                            next_boxes, next_class_ids = convert_yolo_to_boxes(next_yolo_lines, img_width, img_height)
                            if next_class_ids is not None:
                                next_result = {"boxes": next_boxes, "class_ids": next_class_ids}

                if prev_result is not None and next_result is not None:
                    boxes = prev_result["boxes"]
                    class_ids = prev_result["class_ids"]
                    break


        rendered_frame = frame
        mask_list = []
        all_masks=[]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_rgb)
        if class_ids:
            if empty_frames:
                print(f"帧 {', '.join(map(str, empty_frames))} 无检测结果")
                empty_frames = []

            # 对每个检测框进行分割
            for box in boxes:
                input_box = np.array(box)
                try: 
                    # 预测掩码
                    masks, _, _ = sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    # print("after predict")
                    # 保存当前检测框的分割结果
                    for i, mask in enumerate(masks):
                        # 将PyTorch张量转换为NumPy数组，再转换数据类型
                        mask_image = mask.astype(np.uint8) * 255
                        all_masks.append({"segmentation": mask})
                        mask_list.append(mask)
                except Exception as e:
                    # 异常处理逻辑（捕获所有异常，不推荐直接使用，建议捕获具体异常）
                    print(f"发生错误: {str(e)}")
            
            # 渲染掩码
            rendered_frame = frame.copy()
            h, w = rendered_frame.shape[:2]
            overlay = rendered_frame.copy()
            alpha = 0.4  # 透明度
            color = (204, 0, 153)  # 紫色

            # 处理每个掩码
            for mask in mask_list:
                binary_mask = mask
                overlay[binary_mask] = color
            
            # 混合覆盖层和原图
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
    # 配置参数
    data_path = r"C:\Users\xxx\Workspace\project\0811-cat\8月11日.mp4"
    parent_dir = Path(r"C:\Users\xxx\Workspace\project\0811-cat\2025-08-12_10-12")
    seg_dir = parent_dir / "seg"
    # seg_dir.mkdir(exist_ok=True)
    # 替换为你的模型路径
    # sam_checkpoint = r"C:\Users\xxx\Workspace\model\sam_vit_b_01ec64.pth"  
    # 加载Meta的SAM模型
    # sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)


    # 替换为你的模型路径
    sam_checkpoint = r"C:\Users\xxx\Workspace\model\sam_vit_h_4b8939.pth"  
    # 加载Meta的SAM模型
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    print(f"模型是否成功加载：{sam is not None}")


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)


    # 处理所有帧
    num_sam_frames = -1  

    # 执行分割
    segment_frames(data_path, sam_predictor, parent_dir, device, num_sam_frames,seg_dir)