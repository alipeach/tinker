import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def generate_and_save_masks(sam, input_image_path, output_dir):
    """
    生成并保存图像的分割掩码和渲染后的图像
    :param sam: 加载的SAM模型
    :param input_image_path: 输入图像的路径
    :param output_dir: 输出结果的目录
    """
    # 加载图像
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"无法加载图像 '{input_image_path}'，跳过...")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    # 生成掩码
    masks = generator.generate(image)

    # 获取输入文件名
    base = os.path.basename(input_image_path)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output_dir, base)
    os.makedirs(save_base, exist_ok=True)

    # 保存分割结果
    for i, mask in enumerate(masks):
        mask_image = mask["segmentation"].astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_base, f"mask_{i}.png"), mask_image)

    # 渲染并保存渲染后的图片
    rendered_image = render_masks_on_image(image, masks)
    rendered_image_path = os.path.join(output_dir, f"{base}_rendered.png")
    cv2.imwrite(rendered_image_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))


def render_masks_on_image(image, masks):
    """
    将掩码渲染到原始图像上
    :param image: 原始图像
    :param masks: 生成的掩码列表
    :return: 渲染后的图像
    """
    overlay = image.copy()
    for mask in masks:
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        color = color.tolist()
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

    # 指定输入文件和输出目录
    input_image_path = "/workspace/000001.jpg"
    output_dir = "/workspace/outputs/metasam/"
    os.makedirs(output_dir, exist_ok=True)

    # 生成并保存掩码
    generate_and_save_masks(sam, input_image_path, output_dir)


if __name__ == "__main__":
    main()