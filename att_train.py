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
from sklearn.model_selection import train_test_split

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
                 file_items: list,  # 改为直接传入文件项列表
                 audio_dir: str,
                 sequence_length: int = 8,
                 segment_duration: float = 3.0,
                 sample_rate: int = 16000,
                 ad_ratio_threshold: float = 0.5):
        """
        file_items: 标注JSON文件中的项列表
        audio_dir: 音频文件所在目录
        sequence_length: 每个样本包含的连续片段数量 (用于Transformer上下文)
        segment_duration: 每个片段的时长（秒）
        sample_rate: 目标采样率
        ad_ratio_threshold: 广告占比超过此阈值则标记为广告 (1), 否则为非广告 (0)
        """
        self.sequence_length = sequence_length
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.seg_len_samples = int(segment_duration * sample_rate)
        self.ad_ratio_threshold = ad_ratio_threshold
        self.sequences = []
        self.audio_dir = audio_dir

        # 按音频文件分组处理
        for item in tqdm(file_items, desc="加载并切分数据", unit="文件"):
            audio_path = os.path.join(self.audio_dir, item['audioPath'])
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

            # 生成该文件的所有连续片段及其标签
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
            
            # 从该文件的片段列表中构建序列
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
# 上下文感知广告分类模型
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
            sequence_features = self.backbone(x) 
            segment_embeddings = sequence_features.mean(dim=1) # -> [B*S, Dim]

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
# 训练与验证函数
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
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(waveforms)
        
        loss = criterion(predictions.view(-1), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            
            binary_preds = (predictions > 0.5).float()
            
            all_preds.append(binary_preds.view(-1).cpu())
            all_labels.append(labels.view(-1).cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0)
    }
    
    return metrics

# ----------------------------
# 主函数 (完整修改)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="上下文感知广告检测模型训练脚本")
    parser.add_argument('--json_path', type=str, default='./audio_ads.json', help='标注 JSON 文件路径')
    parser.add_argument('--audio_dir', type=str, default='./audio/', help='音频目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--freeze_dasheng', action='store_true', help='是否冻结 dasheng 骨干')
    parser.add_argument('--max_files', type=int, default=-1, help='最大音频文件数 (-1 表示所有文件)')
    parser.add_argument('--segment_duration', type=float, default=3.0, help='每个片段的时长（秒）')
    parser.add_argument('--sequence_length', type=int, default=8, help='上下文窗口大小 (片段数量)')
    parser.add_argument('--output_dir', type=str, default='./output_contextual', help='模型输出目录')
    
    # 新增断点恢复参数
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')
    parser.add_argument('--resume_epoch', type=int, default=0, help='恢复训练的起始epoch')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"======= 使用设备: {device} =======")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载并分割数据集（按音频文件级别）
    print("📂 加载数据集元信息并划分训练/验证集...")
    with open(args.json_path, 'r') as f:
        all_items = json.load(f)
    
    # 如果限制了最大文件数
    if args.max_files > 0:
        all_items = all_items[:args.max_files]
    
    # 按文件划分训练集和验证集
    train_items, val_items = train_test_split(
        all_items, 
        test_size=args.val_split, 
        random_state=42  # 固定随机种子保证可复现
    )
    
    print(f"✅ 数据集划分完成 - 总文件: {len(all_items)} | 训练文件: {len(train_items)} | 验证文件: {len(val_items)}")
    
    # 2. 创建训练集和验证集数据集
    print("🛠️ 构建训练数据集...")
    train_dataset = ContextualAdDataset(
        file_items=train_items,
        audio_dir=args.audio_dir,
        sequence_length=args.sequence_length,
        segment_duration=args.segment_duration
    )
    
    print("🛠️ 构建验证数据集...")
    val_dataset = ContextualAdDataset(
        file_items=val_items,
        audio_dir=args.audio_dir,
        sequence_length=args.sequence_length,
        segment_duration=args.segment_duration
    )
    
    print(f"   训练集序列数: {len(train_dataset)} | 验证集序列数: {len(val_dataset)}")

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. 初始化模型、损失和优化器
    print("🛠️ 初始化上下文感知模型...")
    backbone = dasheng.dasheng_base()
    model = ContextualAdClassifier(
        backbone=backbone,
        freeze_backbone=args.freeze_dasheng
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True)
    
    start_epoch = 0
    best_f1 = 0.0
    
    # 5. 恢复训练检查点（如果提供了）
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 从检查点恢复训练: {args.resume}")
        
        # 加载完整检查点
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        
        # 使用命令行参数覆盖检查点中的epoch设置
        if args.resume_epoch > 0:
            start_epoch = args.resume_epoch
        
        print(f"   恢复训练状态 - 起始Epoch: {start_epoch}, 最佳F1: {best_f1:.4f}")

    # 6. 训练循环 (支持断点恢复)
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n======== Epoch {epoch+1}/{args.num_epochs} ({(epoch+1)/args.num_epochs*100:.1f}%) ========")
        
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
        
        # 7. 创建检查点信息
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'current_f1': current_f1
        }
        
        # 8. 保存最新检查点
        latest_checkpoint = os.path.join(args.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_checkpoint)
        print(f"\n💾 保存最新检查点到: {latest_checkpoint}")
        
        # 9. 保存最佳模型检查点
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_checkpoint = os.path.join(args.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_checkpoint)
            
            # 单独保存模型用于部署
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            
            print(f"\n🏆 新的最佳 F1 分数！模型已保存至: {best_checkpoint}")

    print(f"\n🎉 训练完成! 最终最佳 F1 分数为: {best_f1:.4f}")

if __name__ == '__main__':
    main()