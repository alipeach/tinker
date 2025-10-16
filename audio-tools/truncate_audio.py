import os
from pydub import AudioSegment


def truncate_audio(audio_path, target_dir):
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio)
    segment_length = 6 * 1000
    start = 0
    segment_num = 1
    segments = []
    while start < total_duration:
        end = start + segment_length
        if end > total_duration:
            if segments:
                last_segment = segments[-1]
                segments[-1] = last_segment + audio[start:total_duration]
            else:
                segments.append(audio[start:total_duration])
            break
        segments.append(audio[start:end])
        start = end
    file_name, file_ext = os.path.splitext(os.path.basename(audio_path))
    for i, segment in enumerate(segments):
        output_file_name = f"{file_name}_{i + 1}{file_ext}"
        output_path = os.path.join(target_dir, output_file_name)
        segment.export(output_path, format=file_ext[1:])


def truncate_audio_files(source, target_dir):
    if os.path.isfile(source):
        if source.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.WAV')):
            truncate_audio(source, target_dir)
    elif os.path.isdir(source):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for root, dirs, files in os.walk(source):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.WAV')):
                    audio_path = os.path.join(root, file)
                    truncate_audio(audio_path, target_dir)


# 定义源文件/目录和目标目录
source = "/Users/apple/Blibli/语音素材/李云龙-亮剑视频语音切片"
target_directory = "/Users/apple/Blibli/语音素材/李云龙-亮剑语音-6秒"
truncate_audio_files(source, target_directory)
