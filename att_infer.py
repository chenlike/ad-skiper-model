import os
import json
import argparse
import torch
import torchaudio
import dasheng
import numpy as np
from torch import nn

# ===================================================================
# 重要：你需要从你的训练脚本中复制模型类的定义
# Python需要知道类的结构才能加载模型权重
# ===================================================================
class ContextualAdClassifier(nn.Module):
    def __init__(self, backbone, freeze_backbone=True, d_model=768, nhead=8, num_layers=3):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            # 在推理时，冻结与否不影响结果，但为了保持一致性而保留
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
    """
    对单个音频文件进行广告检测推理。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"======= 使用设备: {device} =======")

    # 1. 初始化模型结构并加载训练好的权重
    print("🛠️  加载模型...")
    backbone = dasheng.dasheng_base()
    model = ContextualAdClassifier(
        backbone=backbone,
        freeze_backbone=True, # 推理时此参数不重要
        num_layers=3, # 确保这些超参数与你训练时使用的模型一致
        nhead=8
    ).to(device)
    
    # 加载状态字典
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    # 切换到评估模式（这会关闭dropout等层）
    model.eval()
    print(f"✅ 模型加载成功: {args.model_path}")

    # 2. 加载并预处理音频 (与训练时的Dataset逻辑一致)
    print(f"🎧 正在处理音频文件: {args.audio_path}...")
    try:
        waveform, sr = torchaudio.load(args.audio_path)
    except Exception as e:
        print(f"❌ 错误: 无法加载音频文件. {e}")
        return

    # 重采样和转单声道
    if sr != args.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 3. 将整个音频切分为重叠的片段 (Segments)
    seg_len_samples = int(args.segment_duration * args.sample_rate)
    step_samples = seg_len_samples // 2 # 50% 重叠
    total_samples = waveform.shape[1]
    
    all_segments = []
    for start_sample in range(0, total_samples - seg_len_samples + 1, step_samples):
        end_sample = start_sample + seg_len_samples
        segment_waveform = waveform[:, start_sample:end_sample]
        all_segments.append(segment_waveform)

    if not all_segments:
        print("❌ 音频太短，无法切分出任何片段。")
        return

    # 4. 将片段打包成序列 (Sequences)，与模型输入匹配
    all_sequences = []
    if len(all_segments) >= args.sequence_length:
        for i in range(len(all_segments) - args.sequence_length + 1):
            sequence = all_segments[i:i + args.sequence_length]
            # 堆叠成 [Seq_Len, 1, Samples]
            all_sequences.append(torch.stack(sequence))
    else:
        print(f"⚠️ 警告: 音频片段数 ({len(all_segments)}) 小于序列长度 ({args.sequence_length}). 无法进行推理。")
        return
        
    # 5. 模型推理
    print(f"🚀 开始推理... 共 {len(all_sequences)} 个序列")
    
    # 用于存储每个独立片段的预测结果（处理重叠问题）
    # 数组的长度是总片段数
    segment_predictions = np.zeros(len(all_segments))
    segment_counts = np.zeros(len(all_segments))

    with torch.no_grad():
        # 为了防止显存溢出，可以分批处理序列
        for i in range(0, len(all_sequences), args.batch_size):
            batch_sequences_list = all_sequences[i:i + args.batch_size]
            
            # 整理成一个批次 [B, Seq_Len, 1, Samples]
            batch_tensor = torch.stack(batch_sequences_list).to(device)
            
            # 模型预测，输出 [B, Seq_Len]
            probs = model(batch_tensor).cpu().numpy()
            
            # 核心步骤：将重叠的预测结果累加到对应的片段上
            for j, seq_probs in enumerate(probs):
                # 当前序列在 `all_sequences` 中的起始索引
                original_sequence_index = i + j
                for k, prob in enumerate(seq_probs):
                    # 这个概率对应于 `all_segments` 中的哪个片段
                    original_segment_index = original_sequence_index + k
                    segment_predictions[original_segment_index] += prob
                    segment_counts[original_segment_index] += 1
    
    # 对重叠预测进行平均
    # 避免除以0
    segment_counts[segment_counts == 0] = 1 
    final_segment_probs = segment_predictions / segment_counts

    # 6. 后处理并输出结果
    print("📈 后处理结果...")
    
    # 将概率转换为二进制预测
    final_segment_labels = (final_segment_probs > args.threshold).astype(int)

    # 合并连续的广告片段
    ad_timestamps = []
    is_in_ad = False
    ad_start_time = 0

    for i, label in enumerate(final_segment_labels):
        segment_start_sec = (i * step_samples) / args.sample_rate
        segment_end_sec = segment_start_sec + args.segment_duration

        if label == 1 and not is_in_ad:
            is_in_ad = True
            ad_start_time = segment_start_sec
        elif label == 0 and is_in_ad:
            is_in_ad = False
            ad_timestamps.append({"startTime": round(ad_start_time, 2), "endTime": round(segment_start_sec + (args.segment_duration / 2), 2)}) # 取上一个片段的中间作为结束

    # 如果音频以广告结束
    if is_in_ad:
        last_segment_start = ((len(final_segment_labels) - 1) * step_samples) / args.sample_rate
        ad_timestamps.append({"startTime": round(ad_start_time, 2), "endTime": round(last_segment_start + args.segment_duration, 2)})

    # 7. 保存结果
    output_data = {
        "audioPath": os.path.basename(args.audio_path),
        "detectedAds": ad_timestamps
    }

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ 推理完成！结果已保存至: {args.output_json}")
    print("检测到的广告时间戳:")
    for ad in ad_timestamps:
        print(f"  - 开始: {ad['startTime']:.2f}s, 结束: {ad['endTime']:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="上下文感知广告检测模型推理脚本")
    # --- 路径参数 ---
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重文件路径 (.pth)')
    parser.add_argument('--audio_path', type=str, required=True, help='需要检测的单个音频文件路径')
    parser.add_argument('--output_json', type=str, default='detected_ads.json', help='输出结果的JSON文件路径')
    
    # --- 模型和数据参数 (必须与训练时保持一致) ---
    parser.add_argument('--segment_duration', type=float, default=3.0, help='每个片段的时长（秒）')
    parser.add_argument('--sequence_length', type=int, default=8, help='Transformer的上下文窗口大小 (片段数量)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    
    # --- 推理控制参数 ---
    parser.add_argument('--batch_size', type=int, default=16, help='推理时使用的批处理大小，以防显存不足')
    parser.add_argument('--threshold', type=float, default=0.5, help='判断为广告的概率阈值')
    
    args = parser.parse_args()
    
    run_inference(args)