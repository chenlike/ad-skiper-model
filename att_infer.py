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
# 模型类定义 (与训练代码对齐)
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
        
        # <--- 修改: 返回值是预测的比例 (Prediction/Ratio), 而非分类概率 (Probability)
        predictions = torch.sigmoid(logits)
        return predictions

# ===================================================================
# 广告检测器类 (已更新以适应回归任务)
# ===================================================================
class AdDetector:
    def __init__(self, model_path, device=None, segment_duration=3.0, sequence_length=8, 
                 sample_rate=16000, batch_size=16, threshold=0.5, min_ad_duration=0, 
                 merge_gap_duration=0, overlap_ratio=0.5):
        """初始化广告检测器
        
        Args:
            model_path (str): 模型权重文件路径
            ...
            threshold (float): 用于判断一个片段是否为广告的预测比例阈值
            merge_gap_duration (float): 合并广告的最大间隔（秒）
            overlap_ratio (float): 音频切片的重叠率, 范围 [0.0, 1.0)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.segment_duration = segment_duration
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.threshold = threshold
        self.min_ad_duration = min_ad_duration
        self.merge_gap_duration = merge_gap_duration
        self.overlap_ratio = overlap_ratio
        
        # 加载模型
        self.backbone = dasheng.dasheng_base()
        self.model = ContextualAdClassifier(
            backbone=self.backbone, 
            freeze_backbone=False, 
            num_layers=3, 
            nhead=8
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _load_audio(self, audio_path):
        """加载并预处理音频文件"""
        temp_wav_path = None
        # 如果是mp3，先用ffmpeg转码为wav
        if audio_path.lower().endswith('.mp3'):
            tmp_f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = tmp_f.name
            tmp_f.close()
            
            ffmpeg_cmd = ['ffmpeg', '-y', '-i', audio_path, '-ar', str(self.sample_rate), '-ac', '1', temp_wav_path]
            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if os.path.exists(temp_wav_path): 
                    os.remove(temp_wav_path)
                raise RuntimeError("ffmpeg转码失败，请确保已安装ffmpeg并添加到系统路径") from e
            
            audio_path = temp_wav_path
            
        try:
            waveform, sr = torchaudio.load(audio_path)
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        return waveform, self.sample_rate

    def _process_segments(self, waveform):
        """处理音频片段"""
        seg_len_samples = int(self.segment_duration * self.sample_rate)
        
        # 根据重叠率计算步长
        step_samples = int(seg_len_samples * (1 - self.overlap_ratio))
        if step_samples < 1:
            step_samples = 1

        total_samples = waveform.shape[1]
        
        all_segments = [
            waveform[:, s:s + seg_len_samples] 
            for s in range(0, total_samples - seg_len_samples + 1, step_samples)
        ]
        
        if not all_segments:
            raise ValueError("音频太短，无法切分出任何片段")
            
        all_sequences = []
        if len(all_segments) >= self.sequence_length:
            for i in range(len(all_segments) - self.sequence_length + 1):
                all_sequences.append(torch.stack(all_segments[i:i + self.sequence_length]))
        else:
            # 对于推理，我们可以用padding来处理短音频，但这里为保持与训练一致，先抛出错误
            raise ValueError(f"音频片段数 ({len(all_segments)}) 不足 {self.sequence_length}，无法进行推理")
            
        return all_segments, all_sequences, step_samples

    def _run_inference(self, all_sequences, all_segments):
        """运行模型推理"""
        # <--- 修改: 变量名反映其内容是预测值的累加
        segment_prediction_sums = np.zeros(len(all_segments))
        segment_counts = np.zeros(len(all_segments))
        
        with torch.no_grad():
            for i in range(0, len(all_sequences), self.batch_size):
                batch_tensor = torch.stack(all_sequences[i:i + self.batch_size]).to(self.device)
                # <--- 修改: model返回的是 predictions
                predictions = self.model(batch_tensor).cpu().numpy()
                
                for j, seq_preds in enumerate(predictions):
                    original_sequence_index = i + j
                    for k, pred in enumerate(seq_preds):
                        original_segment_index = original_sequence_index + k
                        segment_prediction_sums[original_segment_index] += pred
                        segment_counts[original_segment_index] += 1
                        
        segment_counts[segment_counts == 0] = 1
        # <--- 修改: 返回平均后的最终预测比例
        final_segment_predictions = segment_prediction_sums / segment_counts
        return final_segment_predictions

    def _post_process(self, final_segment_predictions, step_samples):
        """后处理推理结果"""
        # 基于阈值将连续的预测比例转换为二进制标签
        final_segment_labels = (final_segment_predictions > self.threshold).astype(int)
        ad_timestamps = []
        is_in_ad = False
        ad_start_time = 0
        
        # 生成时间戳
        for i, label in enumerate(final_segment_labels):
            segment_start_sec = (i * step_samples) / self.sample_rate
            if label == 1 and not is_in_ad:
                is_in_ad = True
                ad_start_time = segment_start_sec
            elif label == 0 and is_in_ad:
                is_in_ad = False
                # 结束时间应该基于片段的中心或开始，这里用开始+片段时长作为近似
                ad_end_time = segment_start_sec + (self.segment_duration - (step_samples / self.sample_rate))
                ad_timestamps.append({
                    "startTime": round(ad_start_time, 2),
                    "endTime": round(ad_end_time, 2)
                })
                
        if is_in_ad:
            last_segment_start = ((len(final_segment_labels) - 1) * step_samples) / self.sample_rate
            ad_end_time = last_segment_start + self.segment_duration
            ad_timestamps.append({
                "startTime": round(ad_start_time, 2),
                "endTime": round(ad_end_time, 2)
            })
            
        # 过滤短广告
        if self.min_ad_duration > 0 and ad_timestamps:
            ad_timestamps = [
                ad for ad in ad_timestamps 
                if (ad['endTime'] - ad['startTime']) >= self.min_ad_duration
            ]
            
        # 合并相近广告
        if self.merge_gap_duration > 0 and len(ad_timestamps) > 1:
            merged_ads = [ad_timestamps[0]]
            for current_ad in ad_timestamps[1:]:
                last_merged_ad = merged_ads[-1]
                gap = current_ad['startTime'] - last_merged_ad['endTime']
                if gap <= self.merge_gap_duration:
                    last_merged_ad['endTime'] = current_ad['endTime']
                else:
                    merged_ads.append(current_ad)
            ad_timestamps = merged_ads
            
        # 生成每个片段的详细结果
        segment_results = []
        # <--- 修改: 迭代的是 final_segment_predictions
        for i, pred in enumerate(final_segment_predictions):
            segment_start_sec = (i * step_samples) / self.sample_rate
            segment_end_sec = segment_start_sec + self.segment_duration
            segment_results.append({
                "startTime": round(segment_start_sec, 2),
                "endTime": round(segment_end_sec, 2),
                # <--- 修改: 键名为 "ad_ratio"，值来自 pred
                "ad_ratio": round(float(pred), 4),
                "isAd": bool(pred > self.threshold)
            })
            
        return ad_timestamps, segment_results

    def detect(self, audio_path, output_json=None):
        """检测音频文件中的广告"""
        # 1. 加载音频
        waveform, _ = self._load_audio(audio_path)
        
        # 2. 处理片段
        all_segments, all_sequences, step_samples = self._process_segments(waveform)
        
        # 3. 运行推理
        # <--- 修改: 变量名对齐
        final_segment_predictions = self._run_inference(all_sequences, all_segments)
        
        # 4. 后处理
        # <--- 修改: 传入变量名对齐
        ad_timestamps, segment_results = self._post_process(final_segment_predictions, step_samples)
        
        # 5. 整理结果
        output_data = {
            "audioPath": os.path.basename(audio_path),
            "detectedAds": ad_timestamps,
            "segmentResults": segment_results
        }
        
        # 6. 保存结果（如果需要）
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
                
        return output_data

def format_time(seconds):
    """将秒数转换为 'MM:SS.xx (seconds)' 格式的字符串"""
    minutes = int(seconds) // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f} ({seconds:.2f}s)"

def run_inference(args):
    """命令行接口的推理函数"""
    print(f"🔧 初始化广告检测器...")
    print(f"   - 模型路径: {args.model_path}")
    print(f"   - 音频切片重叠率: {args.overlap_ratio * 100:.0f}%")
    
    detector = AdDetector(
        model_path=args.model_path,
        segment_duration=args.segment_duration,
        sequence_length=args.sequence_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_ad_duration=args.min_ad_duration,
        merge_gap_duration=args.merge_gap_duration,
        overlap_ratio=args.overlap_ratio
    )
    
    print(f"\n🎧 开始检测音频文件: {args.audio_path}")
    try:
        result = detector.detect(args.audio_path, args.output_json)
        
        print("\n" + "="*50)
        if result['detectedAds']:
            print("✅ 检测完成! 最终广告时间戳:")
            for ad in result['detectedAds']:
                duration = ad['endTime'] - ad['startTime']
                print(f"   - 开始: {format_time(ad['startTime'])}, "
                      f"结束: {format_time(ad['endTime'])} "
                      f"(时长: {format_time(duration)})")
        else:
            print("✅ 检测完成! 未检测到满足条件的广告。")
        print("="*50)

        if args.output_json:
            print(f"\n💾 详细结果已保存至: {args.output_json}")
            
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        # <--- 修改: 描述更新为回归任务
        description="上下文感知广告比例预测模型推理脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重文件路径 (.pth)')
    parser.add_argument('--audio_path', type=str, required=True, help='需要检测的单个音频文件路径 (支持 .wav, .mp3)')
    parser.add_argument('--output_json', type=str, default=None, help='输出结果的 JSON 文件路径 (可选, 不指定则不保存)')
    parser.add_argument('--segment_duration', type=float, default=3.0, help='(应与训练时一致) 每个片段的时长（秒）')
    parser.add_argument('--sequence_length', type=int, default=8, help='(应与训练时一致) Transformer 的上下文窗口大小')
    
    parser.add_argument('--overlap_ratio', type=float, default=0.5, 
                        help='(应与训练时一致) 音频切片的重叠率 (例如 0.5 表示 50%% 重叠)')

    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    parser.add_argument('--batch_size', type=int, default=16, help='推理时使用的批处理大小')
    # <--- 修改: 更新threshold的帮助文本
    parser.add_argument('--threshold', type=float, default=0.5, help='判断为广告的预测比例阈值')
    parser.add_argument('--min_ad_duration', type=float, default=0, 
                        help='(后处理) 过滤掉时长小于此值(秒)的广告块。\n'
                             '默认为0，不过滤。')
    parser.add_argument('--merge_gap_duration', type=float, default=0, 
                        help='(后处理) 合并间隔小于此值(秒)的广告块。\n'
                             '默认为0，不合并。')
    
    args = parser.parse_args()
    
    run_inference(args)