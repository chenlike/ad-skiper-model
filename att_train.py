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
# <--- 修改: 从 sklearn.metrics 中引入 r2_score
from sklearn.metrics import mean_absolute_error, r2_score
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
# 自定义数据集
# ----------------------------
class ContextualAdDataset(Dataset):
    def __init__(self,
                 file_items: list,
                 audio_dir: str,
                 sequence_length: int = 8,
                 segment_duration: float = 3.0,
                 sample_rate: int = 16000,
                 overlap_ratio: float = 0.5):
        self.sequence_length = sequence_length
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.seg_len_samples = int(segment_duration * sample_rate)
        self.overlap_ratio = overlap_ratio
        self.sequences = []
        self.audio_dir = audio_dir

        for item in tqdm(file_items, desc="加载并切分数据", unit="文件"):
            audio_path = os.path.join(self.audio_dir, item['audioPath'])
            if not os.path.exists(audio_path):
                continue

            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0, keepdim=True)
            total_duration_samples = waveform.shape[1]
            
            ad_intervals = sorted([(ad['startTime'], ad['endTime']) for ad in item.get('ads', [])], key=lambda x: x[0])

            file_segments = []
            step_samples = int(self.seg_len_samples * (1 - self.overlap_ratio))
            if step_samples < 1:
                step_samples = 1
            
            for start_sample in range(0, total_duration_samples - self.seg_len_samples + 1, step_samples):
                end_sample = start_sample + self.seg_len_samples
                segment_waveform = waveform[:, start_sample:end_sample]

                seg_start_sec = start_sample / self.sample_rate
                seg_end_sec = end_sample / self.sample_rate
                ad_overlap_sec = 0.0
                for ad_start, ad_end in ad_intervals:
                    overlap_start = max(seg_start_sec, ad_start)
                    overlap_end = min(seg_end_sec, ad_end)
                    if overlap_end > overlap_start:
                        ad_overlap_sec += (overlap_end - overlap_start)
                
                ad_ratio = ad_overlap_sec / self.segment_duration
                
                label = ad_ratio
                
                file_segments.append({'waveform': segment_waveform, 'label': torch.tensor(label, dtype=torch.float32)})
            
            if len(file_segments) >= self.sequence_length:
                for i in range(len(file_segments) - self.sequence_length + 1):
                    self.sequences.append(file_segments[i:i + self.sequence_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence_data = self.sequences[idx]
        waveforms = torch.stack([s['waveform'] for s in sequence_data])
        labels = torch.stack([s['label'] for s in sequence_data])
        return waveforms, labels

# ----------------------------
# 上下文感知模型
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
        predictions = torch.sigmoid(logits)
        
        return predictions

# ----------------------------
# 训练函数
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

# ----------------------------
# 验证函数
# ----------------------------
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
            
            all_preds.append(predictions.view(-1).cpu())
            all_labels.append(labels.view(-1).cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # <--- 修改: 添加 r2_score 计算
    metrics = {
        "loss": total_loss / len(dataloader),
        "mae": mean_absolute_error(all_labels, all_preds),
        "r2": r2_score(all_labels, all_preds)
    }
    
    return metrics

# ----------------------------
# 主函数
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="上下文感知广告比例预测模型训练脚本 (回归任务)")
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
    parser.add_argument('--output_dir', type=str, default='./output_contextual_regression', help='模型输出目录')
    parser.add_argument('--overlap_ratio', type=float, default=0.5, help='音频片段的重叠率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')
    parser.add_argument('--resume_epoch', type=int, default=0, help='恢复训练的起始epoch')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"======= 使用设备: {device} =======")
    print(f"======= 任务类型: 回归 (预测广告占比) =======")
    print(f"======= 片段重叠率: {args.overlap_ratio * 100:.0f}% =======")
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("📂 加载数据集元信息并划分训练/验证集...")
    with open(args.json_path, 'r') as f:
        all_items = json.load(f)
    
    if args.max_files > 0:
        all_items = all_items[:args.max_files]
    
    train_items, val_items = train_test_split(all_items, test_size=args.val_split, random_state=42)
    
    print(f"✅ 数据集划分完成 - 总文件: {len(all_items)} | 训练文件: {len(train_items)} | 验证文件: {len(val_items)}")
    
    print("🛠️ 构建训练数据集...")
    train_dataset = ContextualAdDataset(
        file_items=train_items,
        audio_dir=args.audio_dir,
        sequence_length=args.sequence_length,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio
    )
    
    print("🛠️ 构建验证数据集...")
    val_dataset = ContextualAdDataset(
        file_items=val_items,
        audio_dir=args.audio_dir,
        sequence_length=args.sequence_length,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio
    )
    
    print(f"   训练集序列数: {len(train_dataset)} | 验证集序列数: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("🛠️ 初始化上下文感知模型...")
    backbone = dasheng.dasheng_base()
    model = ContextualAdClassifier(backbone=backbone, freeze_backbone=args.freeze_dasheng).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf')) 
        if args.resume_epoch > 0:
            start_epoch = args.resume_epoch
        print(f"   恢复训练状态 - 起始Epoch: {start_epoch}, 最佳验证损失: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n======== Epoch {epoch+1}/{args.num_epochs} ({(epoch+1)/args.num_epochs*100:.1f}%) ========")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        current_val_loss = val_metrics['loss']
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"   训练损失: {train_loss:.4f}")
        print(f"   验证损失: {val_metrics['loss']:.4f}")
        # <--- 修改: 更新打印语句以包含 R² 分数
        print(f"   {'验证集 MAE:':<15} {val_metrics['mae']:.4f}  |  {'验证集 R²:':<12} {val_metrics['r2']:.4f}")
        
        scheduler.step(current_val_loss)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'current_val_loss': current_val_loss
        }
        
        latest_checkpoint = os.path.join(args.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_checkpoint)
        print(f"\n💾 保存最新检查点到: {latest_checkpoint}")
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"\n🏆 新的最佳验证损失！模型已保存至: {best_checkpoint_path}")

    print(f"\n🎉 训练完成! 最终最佳验证损失为: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()