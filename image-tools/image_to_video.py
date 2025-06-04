import os
from moviepy.editor import ImageClip, concatenate_videoclips

def images_to_video(image_paths, output_path, total_duration=3, fps=24):
    """
    将多张照片合成指定时长的视频
    
    参数:
        image_paths: 照片路径列表（需3张）
        output_path: 输出视频路径（如'output.mp4'）
        total_duration: 视频总时长（秒），默认3秒
        fps: 视频帧率，默认24
    """
    if len(image_paths) != 3:
        raise ValueError("需要提供3张照片路径")
    
    # 单张照片时长 = 总时长 / 照片数量（3秒/3张=1秒/张）
    clip_duration = total_duration / len(image_paths)
    
    clips = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"文件未找到: {img_path}")
        
        # 加载图片并设置时长，自动统一为第一张图片的尺寸（避免尺寸不一致问题）
        clip = ImageClip(img_path).set_duration(clip_duration)
        clips.append(clip)
    
    # 合并所有片段
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # 写入视频文件（H.264编码，MP4格式）
    final_clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",  # 保留可能的音频（如果图片含EXIF音频）
        bitrate="2000k"  # 视频码率（可调整，越高画质越好）
    )

if __name__ == "__main__":
    # 替换为你的照片路径（需3张）
    image_list = [
        "photo1.jpg",
        "photo2.jpg",
        "photo3.jpg"
    ]
    
    try:
        images_to_video(
            image_paths=image_list,
            output_path="merged_video.mp4",
            total_duration=3,
            fps=24
        )
        print("视频生成成功，路径: merged_video.mp4")
    except Exception as e:
        print(f"生成失败: {str(e)}")
    