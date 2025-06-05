import os
import subprocess
from tqdm import tqdm

input_dir = './audio'
output_dir = './audio_wav'
os.makedirs(output_dir, exist_ok=True)

# 遍历 mp3 文件
mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]

for filename in tqdm(mp3_files, desc="Converting MP3 to WAV"):
    mp3_path = os.path.join(input_dir, filename)
    wav_name = os.path.splitext(filename)[0] + '.wav'
    wav_path = os.path.join(output_dir, wav_name)

    try:
        subprocess.run([
            'ffmpeg',
            '-y',                 # 自动覆盖
            '-i', mp3_path,       # 输入文件
            '-ac', '1',           # 单声道
            '-ar', '16000',       # 采样率 16kHz
            wav_path              # 输出文件
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {filename}")
