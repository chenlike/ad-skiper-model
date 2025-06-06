import os
import json
import argparse
import torch
import torchaudio
import dasheng
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

# ----------------------------
# 设置随机种子以保证可复现
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ----------------------------
# 自定义数据集 (修改后): 提供连续的片段序列以捕获上下文
# ----------------------------
class ContextualAdDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 audio_dir: str,
                 sequence_length: int = 8,
                 segment_duration: float = 3.0,
                 sample_rate: int = 16000,
                 ad_ratio_threshold: float = 0.5,
                 max_files: int = -1):
        """
        json_path: 标注 JSON 文件路径
        audio_dir: 音频文件所在目录
        sequence_length: 每个样本包含的连续片段数量 (用于Transformer上下文)
        segment_duration: 每个片段的时长（秒）
        sample_rate: 目标采样率
        ad_ratio_threshold: 广告占比超过此阈值则标记为广告 (1), 否则为非广告 (0)
        max_files: 若 >0, 则只加载前 max_files 个音频文件, 用于快速测试
        """
        self.sequence_length = sequence_length
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.seg_len_samples = int(segment_duration * sample_rate)
        self.ad_ratio_threshold = ad_ratio_threshold
        self.sequences = []

        # 1. 按音频文件分组处理
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        files_to_process = data
        if max_files > 0:
            files_to_process = data[:max_files]

        for item in tqdm(files_to_process, desc="加载并切分数据", unit="文件"):
            audio_path = os.path.join(audio_dir, item['audioPath'])
            if not os.path.exists(audio_path):
                continue

            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            # 重采样和转单声道
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0, keepdim=True)
            total_duration_samples = waveform.shape[1]
            
            ad_intervals = sorted([(ad['startTime'], ad['endTime']) for ad in item.get('ads', [])], key=lambda x: x[0])

            # 2. 生成该文件的所有连续片段及其标签
            file_segments = []
            step_samples = self.seg_len_samples // 2  # 50% 重叠
            for start_sample in range(0, total_duration_samples - self.seg_len_samples + 1, step_samples):
                end_sample = start_sample + self.seg_len_samples
                segment_waveform = waveform[:, start_sample:end_sample]

                # 计算广告重叠率
                seg_start_sec = start_sample / self.sample_rate
                seg_end_sec = end_sample / self.sample_rate
                ad_overlap_sec = 0.0
                for ad_start, ad_end in ad_intervals:
                    overlap_start = max(seg_start_sec, ad_start)
                    overlap_end = min(seg_end_sec, ad_end)
                    if overlap_end > overlap_start:
                        ad_overlap_sec += (overlap_end - overlap_start)
                
                ad_ratio = ad_overlap_sec / self.segment_duration
                # 根据阈值生成二分类标签
                label = 1 if ad_ratio >= self.ad_ratio_threshold else 0
                
                file_segments.append({'waveform': segment_waveform, 'label': torch.tensor(label, dtype=torch.float32)})
            
            # 3. 从该文件的片段列表中构建序列
            if len(file_segments) >= self.sequence_length:
                for i in range(len(file_segments) - self.sequence_length + 1):
                    self.sequences.append(file_segments[i:i + self.sequence_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence_data = self.sequences[idx]
        
        waveforms = torch.stack([s['waveform'] for s in sequence_data]) # [Seq_Len, 1, Samples]
        labels = torch.stack([s['label'] for s in sequence_data])       # [Seq_Len]
        
        return waveforms, labels

# ----------------------------
# 上下文感知广告分类模型 (修正后)
# ----------------------------
class ContextualAdClassifier(nn.Module):
    def __init__(self, backbone, freeze_backbone=True, d_model=768, nhead=8, num_layers=3):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            print("🚫 冻结 dasheng 骨干所有参数")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Dasheng 输出的特征维度
        self.d_model = self.backbone.embed_dim

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True  # 重要: 输入格式为 [B, Seq_Len, Dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, Seq_Len, 1, Samples] -> 一批音频片段序列
        返回: [B, Seq_Len] -> 每个片段是广告的概率
        """
        B, S, C, L = x.shape
        
        # 1. 将批次和序列维度合并, 以便批量通过backbone
        x = x.view(B * S, C, L).squeeze(1) # -> [B*S, Samples]
        
        # 2. 使用 dasheng 提取每个片段的时序特征
        with torch.set_grad_enabled(not self.freeze_backbone):
            # ----------- 【错误的代码 - 已注释】 -----------
            # segment_embeddings = self.backbone.forward_cls_token(x) # -> [B*S, D]
            
            # ----------- 【修正后的代码】 -----------
            # self.backbone(x) 返回时序特征 [B*S, TimeSteps, Dim]
            sequence_features = self.backbone(x) 
            # 通过在时间维度上进行平均池化, 得到每个片段的单一特征向量
            segment_embeddings = sequence_features.mean(dim=1) # -> [B*S, Dim]
            # ----------------------------------------

        # 3. 恢复序列维度
        segment_embeddings = segment_embeddings.view(B, S, -1) # -> [B, S, D]
        
        # 4. 通过 Transformer Encoder 融合上下文信息
        contextual_embeddings = self.transformer_encoder(segment_embeddings) # -> [B, S, D]
        
        # 5. 通过分类头得到每个片段的 logits
        logits = self.classifier(contextual_embeddings).squeeze(-1) # -> [B, S]
        
        # 6. 使用 Sigmoid 获得概率
        probs = torch.sigmoid(logits)
        
        return probs
# ----------------------------
# 训练与验证函数 (更新后)
# ----------------------------
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    
    bar = tqdm(dataloader, desc="🚀 Training", leave=False)
    for waveforms, labels in bar:
        # waveforms: [B, S, 1, L], labels: [B, S]
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 获得模型输出的概率 [B, S]
        predictions = model(waveforms)
        
        # 将预测和标签展平, 以便计算损失
        loss = criterion(predictions.view(-1), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)
    with torch.no_grad():
        for waveforms, labels in bar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            predictions = model(waveforms)
            
            loss = criterion(predictions.view(-1), labels.view(-1))
            total_loss += loss.item()
            
            # 收集预测和标签用于计算指标
            # 将概率转换为二进制预测 (0或1)
            binary_preds = (predictions > 0.5).float()
            
            all_preds.append(binary_preds.view(-1).cpu())
            all_labels.append(labels.view(-1).cpu())

    # 合并所有批次的结果
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # 计算分类指标
    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0)
    }
    
    return metrics

# ----------------------------
# 主函数 (更新后)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="上下文感知广告检测模型训练脚本")
    parser.add_argument('--json_path', type=str, default='./audio_ads.json', help='标注 JSON 文件路径')
    parser.add_argument('--audio_dir', type=str, default='./audio/', help='音频目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小 (序列较长, 建议减小)')
    parser.add_argument('--num_epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--freeze_dasheng', action='store_true', help='是否冻结 dasheng 骨干')
    parser.add_argument('--max_files', type=int, default=50, help='最大音频文件数, 用于快速测试 (-1 表示所有文件)')
    parser.add_argument('--segment_duration', type=float, default=3.0, help='每个片段的时长（秒）')
    parser.add_argument('--sequence_length', type=int, default=8, help='Transformer的上下文窗口大小 (片段数量)')
    parser.add_argument('--output_dir', type=str, default='./output_contextual', help='模型输出目录')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"======= 使用设备: {device} =======")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据集
    print("📂 加载并构建序列数据集...")
    full_dataset = ContextualAdDataset(
        json_path=args.json_path,
        audio_dir=args.audio_dir,
        sequence_length=args.sequence_length,
        segment_duration=args.segment_duration,
        max_files=args.max_files
    )
    print(f"✅ 数据集加载完成 - 总序列数: {len(full_dataset)}")

    # 2. 划分训练/验证集
    dataset_size = len(full_dataset)
    val_size = int(np.floor(args.val_split * dataset_size))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"   训练集序列数: {len(train_dataset)} | 验证集序列数: {len(val_dataset)}")

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. 初始化模型、损失和优化器
    print("🛠️ 初始化上下文感知模型...")
    backbone = dasheng.dasheng_base() # 使用 base 版本以平衡性能和效率
    model = ContextualAdClassifier(
        backbone=backbone,
        freeze_backbone=args.freeze_dasheng
    ).to(device)
    
    criterion = nn.BCELoss() # 二分类交叉熵
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True)

    # 5. 训练循环
    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        print(f"\n======== Epoch {epoch+1}/{args.num_epochs} ========")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        current_f1 = val_metrics['f1']
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_metrics['loss']:.4f}")
        print(f"  {'验证准确率:':<15} {val_metrics['accuracy']:.4f}")
        print(f"  {'验证精确率:':<15} {val_metrics['precision']:.4f}")
        print(f"  {'验证召回率:':<15} {val_metrics['recall']:.4f}")
        print(f"  {'验证 F1 分数:':<15} {val_metrics['f1']:.4f}")
        
        scheduler.step(current_f1)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"\n💾 新的最佳 F1 分数！模型已保存至: {best_model_path}")

    print(f"\n🎉 训练完成! 最佳 F1 分数为: {best_f1:.4f}")

if __name__ == '__main__':
    main()