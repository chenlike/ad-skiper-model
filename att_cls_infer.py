import os
import json
import argparse
import subprocess
import tempfile

import torch
import torchaudio
import dasheng
import numpy as np
from torch import nn

# ===================================================================
# 模型类定义 (与之前相同)
# ===================================================================
class ContextualAdClassifier(nn.Module):
    def __init__(self, backbone, freeze_backbone=True, d_model=768, nhead=8, num_layers=3):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.d_model = self.backbone.embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1)
        )

    def forward(self, x: torch.Tensor):
        B, S, C, L = x.shape
        x = x.view(B * S, C, L).squeeze(1)
        
        with torch.set_grad_enabled(not self.freeze_backbone):
            sequence_features = self.backbone(x) 
            segment_embeddings = sequence_features.mean(dim=1)

        segment_embeddings = segment_embeddings.view(B, S, -1)
        contextual_embeddings = self.transformer_encoder(segment_embeddings)
        logits = self.classifier(contextual_embeddings).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs

# ===================================================================
# 推理主函数
# ===================================================================
def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"======= 使用设备: {device} =======")

    # 1. 模型加载 (与之前相同)
    print("🛠️  加载模型...")
    backbone = dasheng.dasheng_base()
    model = ContextualAdClassifier(backbone=backbone, freeze_backbone=False, num_layers=3, nhead=8).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载成功: {args.model_path}")

    # 2. 音频加载与转换 (与之前相同)
    audio_path_to_load = args.audio_path
    temp_wav_path = None
    if audio_path_to_load.lower().endswith('.mp3'):
        tmp_f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = tmp_f.name
        tmp_f.close()
        print(f"🎞️  检测到 MP3，正在调用 ffmpeg 转为临时 WAV...")
        ffmpeg_cmd = ['ffmpeg', '-y', '-i', audio_path_to_load, '-ar', str(args.sample_rate), '-ac', '1', temp_wav_path]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ 错误：ffmpeg 转码失败。请确保已安装 ffmpeg 并将其添加至系统路径。")
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            return
        audio_path_to_load = temp_wav_path
    print(f"🎧 正在处理音频文件: {args.audio_path} ...")
    try:
        waveform, sr = torchaudio.load(audio_path_to_load)
    except Exception as e:
        print(f"❌ 错误: 无法加载音频文件. {e}")
        if temp_wav_path and os.path.exists(temp_wav_path): os.remove(temp_wav_path)
        return
    if temp_wav_path and os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    # 3. 预处理 (与之前相同)
    if sr != args.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 4 & 5. 切片与打包序列 (与之前相同)
    seg_len_samples = int(args.segment_duration * args.sample_rate)
    step_samples = seg_len_samples // 2
    total_samples = waveform.shape[1]
    all_segments = [waveform[:, s:s + seg_len_samples] for s in range(0, total_samples - seg_len_samples + 1, step_samples)]
    if not all_segments:
        print("❌ 音频太短，无法切分出任何片段。")
        return
    all_sequences = []
    if len(all_segments) >= args.sequence_length:
        for i in range(len(all_segments) - args.sequence_length + 1):
            all_sequences.append(torch.stack(all_segments[i:i + args.sequence_length]))
    else:
        print(f"⚠️ 警告: 音频片段数不足 {args.sequence_length}，无法进行推理。")
        return
        
    # 6. 模型推理 (与之前相同)
    print(f"🚀 开始推理...")
    segment_predictions = np.zeros(len(all_segments))
    segment_counts = np.zeros(len(all_segments))
    with torch.no_grad():
        for i in range(0, len(all_sequences), args.batch_size):
            batch_tensor = torch.stack(all_sequences[i:i + args.batch_size]).to(device)
            probs = model(batch_tensor).cpu().numpy()
            for j, seq_probs in enumerate(probs):
                original_sequence_index = i + j
                for k, prob in enumerate(seq_probs):
                    original_segment_index = original_sequence_index + k
                    segment_predictions[original_segment_index] += prob
                    segment_counts[original_segment_index] += 1
    segment_counts[segment_counts == 0] = 1
    final_segment_probs = segment_predictions / segment_counts

    # 输出每个片段的时间和概率
    print("📊 片段检测结果详情:")
    print(f"{'序号':<4} {'开始时间':<12} {'结束时间':<12} {'概率':<8} {'判断'}")
    print("-" * 50)
    for i, prob in enumerate(final_segment_probs):
        segment_start_sec = (i * step_samples) / args.sample_rate
        segment_end_sec = segment_start_sec + args.segment_duration
        is_ad = "广告" if prob > args.threshold else "非广告"
        print(f"{i+1:<4} {format_time(segment_start_sec):<12} {format_time(segment_end_sec):<12} {prob:8.4f} {is_ad}")
    print("-" * 50)

    # 7. 后处理
    print("📈 后处理结果...")
    final_segment_labels = (final_segment_probs > args.threshold).astype(int)
    ad_timestamps = []
    is_in_ad = False
    ad_start_time = 0
    for i, label in enumerate(final_segment_labels):
        segment_start_sec = (i * step_samples) / args.sample_rate
        if label == 1 and not is_in_ad:
            is_in_ad = True
            ad_start_time = segment_start_sec
        elif label == 0 and is_in_ad:
            is_in_ad = False
            ad_end_time = segment_start_sec + (args.segment_duration / 2)
            ad_timestamps.append({"startTime": round(ad_start_time, 2), "endTime": round(ad_end_time, 2)})
    if is_in_ad:
        last_segment_start = ((len(final_segment_labels) - 1) * step_samples) / args.sample_rate
        ad_end_time = last_segment_start + args.segment_duration
        ad_timestamps.append({"startTime": round(ad_start_time, 2), "endTime": round(ad_end_time, 2)})

    # --- 关键后处理步骤 ---

    # 步骤 7a: 首先，根据最小广告时长，过滤掉所有不合格的短广告
    if args.min_ad_duration > 0 and ad_timestamps:
        print(f"🗑️  步骤1: 过滤掉时长小于 {args.min_ad_duration}s 的广告...")
        original_count = len(ad_timestamps)
        ad_timestamps = [ad for ad in ad_timestamps if (ad['endTime'] - ad['startTime']) >= args.min_ad_duration]
        print(f"   - 过滤前: {original_count} 条, 过滤后: {len(ad_timestamps)} 条")

    # 步骤 7b: 然后，在过滤后的合格广告列表基础上，合并间距较近的广告
    if args.merge_gap_duration > 0 and len(ad_timestamps) > 1:
        print(f"🔄 步骤2: 在合格广告之间，合并小于 {args.merge_gap_duration}s 的间距...")
        original_count = len(ad_timestamps)
        merged_ads = [ad_timestamps[0]]
        for current_ad in ad_timestamps[1:]:
            last_merged_ad = merged_ads[-1]
            gap = current_ad['startTime'] - last_merged_ad['endTime']
            if gap <= args.merge_gap_duration:
                last_merged_ad['endTime'] = current_ad['endTime']
            else:
                merged_ads.append(current_ad)
        ad_timestamps = merged_ads
        print(f"   - 合并前: {original_count} 条, 合并后: {len(ad_timestamps)} 条")

    # 8. 保存结果
    segment_results = []
    for i, prob in enumerate(final_segment_probs):
        segment_start_sec = (i * step_samples) / args.sample_rate
        segment_end_sec = segment_start_sec + args.segment_duration
        segment_results.append({
            "startTime": round(segment_start_sec, 2),
            "endTime": round(segment_end_sec, 2),
            "probability": round(float(prob), 4),
            "isAd": bool(prob > args.threshold)
        })

    output_data = {
        "audioPath": os.path.basename(args.audio_path),
        "detectedAds": ad_timestamps,
        "segmentResults": segment_results
    }
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ 推理完成！结果已保存至: {args.output_json}")
    if ad_timestamps:
        print("检测到的最终广告时间戳:")
        for ad in ad_timestamps:
            duration = ad['endTime'] - ad['startTime']
            print(f"  - 开始: {format_time(ad['startTime'])}, "
                  f"结束: {format_time(ad['endTime'])} "
                  f"(时长: {format_time(duration)})")
    else:
        print("未检测到满足条件的广告。")

def format_time(seconds):
    """将秒数转换为 'MM:SS.xx (seconds)' 格式的字符串
    
    Args:
        seconds (float): 需要转换的秒数
    
    Returns:
        str: 格式化后的时间字符串
    """
    minutes = int(seconds) // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f} ({seconds:.2f}s)"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="上下文感知广告检测模型推理脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 参数定义...
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重文件路径 (.pth)')
    parser.add_argument('--audio_path', type=str, required=True, help='需要检测的单个音频文件路径 (支持 .wav, .mp3)')
    parser.add_argument('--output_json', type=str, default='detected_ads.json', help='输出结果的 JSON 文件路径')
    parser.add_argument('--segment_duration', type=float, default=3.0, help='每个片段的时长（秒）')
    parser.add_argument('--sequence_length', type=int, default=8, help='Transformer 的上下文窗口大小 (片段数量)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    parser.add_argument('--batch_size', type=int, default=16, help='推理时使用的批处理大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='判断为广告的概率阈值')
    
    # --- 后处理参数 (附带更清晰的说明) ---
    parser.add_argument(
        '--min_ad_duration', 
        type=float, 
        default=0, 
        help='过滤掉时长小于此值(秒)的广告。\n'
             '例如，设为5，则所有时长小于5秒的广告会被首先丢弃。\n'
             '默认为0，不过滤。'
    )
    parser.add_argument(
        '--merge_gap_duration', 
        type=float, 
        default=0, 
        help='在满足min_ad_duration的广告之间，\n'
             '如果非广告间距小于此值(秒)，则将它们合并。\n'
             '例如，设为60，[合格广告A][60秒非广告][合格广告B]会被合并。\n'
             '默认为0，不合并。'
    )
    
    args = parser.parse_args()
    run_inference(args)