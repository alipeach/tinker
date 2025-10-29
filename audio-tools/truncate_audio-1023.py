import os
from pydub import AudioSegment


def truncate_audio(audio_path, target_dir, global_counter):
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio)
    segment_length = 6 * 1000
    start = 0
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
    # file_ext = os.path.splitext(os.path.basename(audio_path))[1]
    file_ext = '.wav'
    for segment in segments:
        output_file_name = f"{global_counter[0]}{file_ext}"
        output_path = os.path.join(target_dir, output_file_name)
        segment.export(output_path, format=file_ext[1:])
        global_counter[0] += 1


def truncate_audio_files(source, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 使用列表存储计数器以实现引用传递
    global_counter = [1]
    
    if os.path.isfile(source):
        if source.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.WAV')):
            truncate_audio(source, target_dir, global_counter)
    elif os.path.isdir(source):
        for root, dirs, files in os.walk(source):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.WAV')):
                    audio_path = os.path.join(root, file)
                    truncate_audio(audio_path, target_dir, global_counter)


# 定义源文件/目录和目标目录
source = "/Users/apple/Blibli/语音素材/李云龙/李云龙-亮剑-语音"
target_directory = "/Users/apple/Blibli/语音素材/李云龙/李云龙-亮剑语音-6秒-1023"
truncate_audio_files(source, target_directory)