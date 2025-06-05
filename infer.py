import os
import torch
import torchaudio
import subprocess
import argparse
from train import DashengAdClassifier  # 假设训练脚本文件名为 train.py，其中定义了 DashengAdClassifier

# 设置 torchaudio 使用 sox_io 后端，以支持 mp3 格式
torchaudio.set_audio_backend("sox_io")


def convert_mp3_to_wav(mp3_path):
    """
    如果输入是 .mp3 文件，则调用 ffmpeg 转为 .wav 并返回新的路径。
    """
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):
        print(f"⚙️ 发现 mp3 文件，正在转换为 wav: {wav_path}")
        subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)
    return wav_path


def segment_audio(waveform, sample_rate, segment_duration=5.0):
    """
    将 waveform 按照固定时长 segment_duration（秒）切成若干不重叠的片段，
    返回列表 segments（每项 Tensor[1, segment_samples]）和对应的时间戳列表 timestamps（(start_sec, end_sec)）。
    """
    segment_samples = int(segment_duration * sample_rate)
    total_samples = waveform.shape[1]
    segments = []
    timestamps = []

    for start in range(0, total_samples - segment_samples + 1, segment_samples):
        end = start + segment_samples
        segment = waveform[:, start:end]
        if segment.shape[1] == segment_samples:
            segments.append(segment)
            timestamps.append((start / sample_rate, end / sample_rate))
    return segments, timestamps


def format_timestamp(seconds):
    """
    将秒数转换为 HH:MM:SS 格式
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def predict_ad_segments(audio_path, model_path, threshold=0.5, segment_duration=5.0):
    """
    对整段音频做广告片段检测，返回广告时间戳列表。
    只打印 Sigmoid 后的概率值，不再打印 Logit。
    """
    sample_rate = 16000

    # 如果是 mp3，先转成 wav
    if audio_path.lower().endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    print(f"✅ 音频路径: {audio_path}")

    # 1. 读取音频并重采样、转换单声道
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)  # 转为单声道
    total_duration = waveform.shape[1] / sample_rate  # 总时长（秒）

    # 2. 切分成若干个固定时长的 segment
    segments, timestamps = segment_audio(waveform, sample_rate, segment_duration)

    # 3. 加载模型，并设置与训练时相同的 segment_duration
    model = DashengAdClassifier(freeze_dasheng=False,
                                segment_duration=segment_duration).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ad_timestamps = []
    with torch.no_grad():
        for segment, (start_sec, end_sec) in zip(segments, timestamps):
            # segment: Tensor[1, segment_samples]，先在前面加 batch 维度
            input_tensor = segment.unsqueeze(0).to(device)  # [1, 1, seg_len_samples]

            # 构造 seg_start 和 total_duration 两个 Tensor
            seg_start_tensor = torch.tensor([start_sec], dtype=torch.float32, device=device)       # [1]
            total_d_tensor = torch.tensor([total_duration], dtype=torch.float32, device=device)    # [1]

            # 前向计算得到 logit，再用 sigmoid 转成概率
            logits = model(input_tensor, seg_start_tensor, total_d_tensor)  # [1]
            prob = torch.sigmoid(logits).item()

            # 打印概率
            if prob >= threshold:
                ad_timestamps.append((start_sec, end_sec))
                print(f"✅ 广告片段 {format_timestamp(start_sec)} ({start_sec:.2f}s) ~ {format_timestamp(end_sec)} ({end_sec:.2f}s)，概率={prob:.4f}")
            else:
                print(f"{format_timestamp(start_sec)} ({start_sec:.2f}s) ~ {format_timestamp(end_sec)} ({end_sec:.2f}s)，概率={prob:.4f}")

    return ad_timestamps


def merge_continuous_segments(segments, gap=0.0):
    """
    合并连续或接近的广告片段。例如 [(0,5), (5,10)] 以及 gap=0 会合并成 [(0,10)]。
    """
    if not segments:
        return []

    merged = [segments[0]]
    for curr_start, curr_end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if curr_start - prev_end <= gap:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append((curr_start, curr_end))
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="广告检测推理脚本（只打印概率）")
    parser.add_argument('--audio', type=str, required=True, help='输入音频文件路径（支持 mp3 或 wav）')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型权重文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='判断阈值（Sigmoid 后概率 ≥ 阈值即视为广告）')
    parser.add_argument('--merge_gap', type=float, default=0.0,
                        help='合并连续广告片段的最大间隔（秒），默认为 0，即严格相连才合并')
    parser.add_argument('--segment_duration', type=float, default=4.0,
                        help='切片时长（秒），要与训练时保持一致，默认 4.0s')
    args = parser.parse_args()

    # 预测并合并
    raw_results = predict_ad_segments(
        audio_path=args.audio,
        model_path=args.model,
        threshold=args.threshold,
        segment_duration=args.segment_duration
    )
    merged_results = merge_continuous_segments(raw_results, gap=args.merge_gap)

    print("\n🚨 识别出的广告时间段（合并后）：")
    for start, end in merged_results:
        print(f"广告：{format_timestamp(start)} ({start:.2f}s) ~ {format_timestamp(end)} ({end:.2f}s)")
