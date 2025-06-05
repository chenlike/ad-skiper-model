import os
import json
import argparse
import torch
import torchaudio
import dasheng
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

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
# 自定义数据集：按固定长度 segment_duration 切分音频，
# 如果一个 segment 部分是广告部分是非广告，则抛弃。
# 每个样本返回 waveform, label, seg_start, total_duration
# ----------------------------
class AdSegmentDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 audio_dir: str,
                 segment_duration: float = 5.0,
                 sample_rate: int = 16000,
                 max_items: int = -1):
        """
        json_path: 标注 JSON 文件路径
        audio_dir: 音频文件所在目录
        segment_duration: 每个片段的时长（秒），默认 5.0
        sample_rate: 目标采样率，默认 16000
        max_items: 若 >0，则只加载前 max_items 条记录，用于快速测试
        """
        self.samples = []
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.seg_len_samples = int(segment_duration * sample_rate)

        # 读取 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)
        if max_items > 0:
            data = data[:max_items]

        # 遍历每条标注
        for item in tqdm(data, desc="加载数据", unit="条"):
            audio_path = os.path.join(audio_dir, item['audioPath'])
            if not os.path.exists(audio_path):
                continue  # 文件不存在则跳过

            # 读取音频
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            # 重采样到目标采样率
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)

            # 转为单声道
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, T_raw]
            total_duration = waveform.shape[1] / sample_rate  # 以秒为单位

            # 提取广告区间，并按起始时间排序
            ad_intervals = sorted([(ad['startTime'], ad['endTime']) for ad in item.get('ads', [])],
                                  key=lambda x: x[0])

            # 判断整个 [seg_start, seg_end] 是否完全在某个广告区间内
            def is_fully_in_ad(seg_start: float, seg_end: float, intervals: list):
                for a, b in intervals:
                    if seg_start >= a and seg_end <= b:
                        return True
                return False

            # 判断 [seg_start, seg_end] 是否与任意广告区间有部分重叠
            def has_overlap(seg_start: float, seg_end: float, intervals: list):
                for a, b in intervals:
                    if seg_start < b and seg_end > a:
                        return True
                return False

            # 按 segment_duration 进行非重叠切分
            step = segment_duration
            current = 0.0
            while current + segment_duration <= total_duration + 1e-6:
                seg_start = current
                seg_end = current + segment_duration

                # 检查是否与广告区间重叠
                overlap = has_overlap(seg_start, seg_end, ad_intervals)
                if overlap:
                    # 如果完全包含在某个广告区间内，划为广告
                    if is_fully_in_ad(seg_start, seg_end, ad_intervals):
                        start_sample = int(seg_start * sample_rate)
                        end_sample = start_sample + self.seg_len_samples
                        if end_sample <= waveform.shape[1]:
                            segment = waveform[:, start_sample:end_sample]
                            if segment.shape[1] == self.seg_len_samples:
                                self.samples.append((segment, 1, seg_start, total_duration))
                    # 部分重叠则丢弃
                    # else: do nothing
                else:
                    # 完全不与任何广告区间重叠，标为非广告
                    start_sample = int(seg_start * sample_rate)
                    end_sample = start_sample + self.seg_len_samples
                    if end_sample <= waveform.shape[1]:
                        segment = waveform[:, start_sample:end_sample]
                        if segment.shape[1] == self.seg_len_samples:
                            self.samples.append((segment, 0, seg_start, total_duration))

                current += step

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        waveform, label, seg_start, total_duration = self.samples[idx]
        return (
            waveform,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(seg_start, dtype=torch.float32),
            torch.tensor(total_duration, dtype=torch.float32)
        )


# ----------------------------
# 生成基于“播放比例” 的正余弦位置编码
# ----------------------------
def get_ratio_positional_encoding(num_steps: int, d_model: int, segment_duration: float, seg_start: torch.Tensor, total_duration: torch.Tensor, device):
    """
    num_steps: 序列长度 T'
    d_model: 特征维度 D
    segment_duration: 片段时长（秒），这里固定 5.0
    seg_start: Tensor [B]，该片段在原音频的起始秒数
    total_duration: Tensor [B]，该音频总时长（秒）
    device: 设备
    返回: [B, T', D] 的位置编码
    """
    B = seg_start.size(0)
    T_prime = num_steps

    # 计算每帧对应的真实时间百分比
    ratio_start = seg_start / (total_duration + 1e-6)  # 防止除零

    # 每帧对应的秒数步长
    frame_sec = segment_duration / T_prime  # 标量

    # 构造 idx = [0,1,...,T'-1]
    idx = torch.arange(0, T_prime, dtype=torch.float32, device=device)  # [T']
    idx_frame = idx.unsqueeze(0).expand(B, T_prime)  # [B, T']

    # 扩展 ratio_start 与 total_duration 到 [B, T']
    ratio_start_expand = ratio_start.unsqueeze(1).expand(B, T_prime)       # [B, T']
    total_d_expand = total_duration.unsqueeze(1).expand(B, T_prime)        # [B, T']

    # 计算每个时间步的全局百分比位置：
    pe_ratio = ratio_start_expand + idx_frame * (frame_sec / total_d_expand)  # [B, T']

    # 为 pe_ratio 编码成 [B, T', D] 的正余弦
    pe = torch.zeros((B, T_prime, d_model), device=device)
    k = torch.arange(0, d_model // 2, device=device, dtype=torch.float32)
    denom = torch.pow(10000, (2 * k) / d_model)  # [D/2]
    angle = pe_ratio.unsqueeze(2) / denom.unsqueeze(0).unsqueeze(0)  # [B, T', D/2]

    pe[:, :, 0::2] = torch.sin(angle)
    pe[:, :, 1::2] = torch.cos(angle)
    return pe  # [B, T', D]


# ----------------------------
# 模型定义：在 dasheng 基础上加一层 Transformer 注意力层，位置编码使用“播放比例”
# ----------------------------
class DashengAdClassifier(nn.Module):
    def __init__(self,
                 freeze_dasheng: bool = True,
                 nhead: int = 8,
                 dim_feedforward: int = None,
                 num_transformer_layers: int = 1,
                 segment_duration: float = 5.0
                 ):
        super().__init__()
        # 使用 dasheng 作为 backbone
        self.dashengmodel = dasheng.dasheng_12B()
        self.freeze_dasheng = freeze_dasheng

        if self.freeze_dasheng:
            print("🚫 冻结 dasheng 骨干所有参数")
            for param in self.dashengmodel.parameters():
                param.requires_grad = False

        D = self.dashengmodel.embed_dim
        if dim_feedforward is None:
            dim_feedforward = D * 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=D,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 1)
        )

        self.embed_dim = D
        self.segment_duration = segment_duration 

    def forward(self, x: torch.Tensor, seg_start: torch.Tensor, total_duration: torch.Tensor):
        """
        x: [B, 1, seg_len_samples]
        seg_start: [B]，该片段在原音频中的起始秒数
        total_duration: [B]，该音频总时长（秒）
        """
        B = x.size(0)
        device = x.device

        # 先去掉频道维度 -> [B, seg_len_samples]
        x = x.squeeze(1)

        # 1) dasheng 提取时序特征，输出 [B, T', D]
        with torch.set_grad_enabled(not self.freeze_dasheng):
            seq_feats = self.dashengmodel(x)  # [B, T', D]

        T_prime = seq_feats.size(1)
        D = self.embed_dim

        # 2) 生成“播放比例”位置编码并加到 seq_feats
        pe = get_ratio_positional_encoding(num_steps=T_prime,
                                           d_model=D,
                                           segment_duration=self.segment_duration,
                                           seg_start=seg_start,
                                           total_duration=total_duration,
                                           device=device)  # [B, T', D]
        seq_feats = seq_feats + pe

        # 3) Transformer Encoder
        transformer_out = self.transformer_encoder(seq_feats)  # [B, T', D]

        # 4) 对时间维度做平均 Pooling -> [B, D]
        pooled = transformer_out.mean(dim=1)  # [B, D]

        # 5) 分类并返回 logit [B]
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        return logits


# ----------------------------
# 训练一个 epoch
# ----------------------------
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    bar = tqdm(dataloader, desc="Training", leave=False)
    for waveform, labels, seg_start, total_duration in bar:
        waveform = waveform.to(device)         # [B, 1, seg_len_samples]
        labels = labels.to(device)             # [B]
        seg_start = seg_start.to(device)       # [B]
        total_duration = total_duration.to(device)  # [B]

        optimizer.zero_grad()
        logits = model(waveform, seg_start, total_duration)  # [B]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * waveform.size(0)
        bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader.dataset)


# ----------------------------
# 验证函数（带准确率、Precision、Recall、F1）
# ----------------------------
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    correct = 0

    bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for waveform, labels, seg_start, total_duration in bar:
            waveform = waveform.to(device)         # [B, 1, seg_len_samples]
            labels = labels.to(device)             # [B]
            seg_start = seg_start.to(device)       # [B]
            total_duration = total_duration.to(device)  # [B]

            logits = model(waveform, seg_start, total_duration)  # [B]
            loss = criterion(logits, labels)
            total_loss += loss.item() * waveform.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = correct / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(dataloader.dataset), accuracy, precision, recall, f1


# ----------------------------
# 主函数
# ----------------------------
def main():
    # ----------------------------
    # 通过 argparse 添加“续训”参数
    # ----------------------------
    parser = argparse.ArgumentParser(description="广告片段分类模型训练脚本")
    parser.add_argument('--json_path', type=str, default='./audio_ads.json', help='标注 JSON 文件路径')
    parser.add_argument('--audio_dir', type=str, default='./audio/', help='音频目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--freeze_dasheng', action='store_true', help='是否冻结 dasheng 骨干')
    parser.add_argument('--max_items', type=int, default=-1, help='最大样本数，用于快速测试')
    parser.add_argument('--segment_duration', type=float, default=4.0, help='每个片段的时长（秒）')
    parser.add_argument('--resume_pth', type=str, default=None, help='已训练好的 .pth 模型文件路径，用于继续训练')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------
    # 加载数据集
    # ----------------------------
    dataset = AdSegmentDataset(json_path=args.json_path,
                               audio_dir=args.audio_dir,
                               segment_duration=args.segment_duration,
                               sample_rate=16000,
                               max_items=args.max_items)  # max_items 可设 50 进行快速测试

    # 提取所有切片的标签，用于划分
    labels_all = np.array([label for _, label, _, _ in dataset.samples])
    ad_indices = np.where(labels_all == 1)[0]
    non_ad_indices = np.where(labels_all == 0)[0]

    # 打乱索引
    np.random.shuffle(ad_indices)
    np.random.shuffle(non_ad_indices)

    # 划分验证集数量
    ad_val_size = int(len(ad_indices) * args.val_ratio)
    non_ad_val_size = int(len(non_ad_indices) * args.val_ratio)

    ad_val_idx = ad_indices[:ad_val_size]
    ad_train_idx = ad_indices[ad_val_size:]
    non_ad_val_idx = non_ad_indices[:non_ad_val_size]
    non_ad_train_idx = non_ad_indices[non_ad_val_size:]

    # 下采样非广告训练样本（只保留 50%）
    non_ad_keep_ratio = 0.5
    keep_non_ad_train_idx = non_ad_train_idx[:int(len(non_ad_train_idx) * non_ad_keep_ratio)]

    # 合并训练/验证索引
    train_idx = np.concatenate([ad_train_idx, keep_non_ad_train_idx])
    val_idx = np.concatenate([ad_val_idx, non_ad_val_idx])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # ----------------------------
    # 计算训练集正负样本权重，用于加权 Loss 和采样
    # ----------------------------
    train_labels = np.array([dataset.samples[i][1] for i in train_idx])
    num_pos = int(train_labels.sum())
    num_neg = len(train_labels) - num_pos

    pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    weight_pos = 1.0 / (num_pos + 1e-6)
    weight_neg = 1.0 / (num_neg + 1e-6)
    sample_weights = [weight_pos if label == 1 else weight_neg for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # ----------------------------
    # 创建 DataLoader
    # ----------------------------
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    # ----------------------------
    # 初始化模型、优化器
    # ----------------------------
    model = DashengAdClassifier(freeze_dasheng=args.freeze_dasheng,
                                nhead=8,
                                dim_feedforward=None,
                                num_transformer_layers=1,
                                segment_duration=args.segment_duration).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 如果用户指定了 --resume_pth，就加载已有的模型权重
    start_epoch = 0
    best_f1 = 0.0
    if args.resume_pth is not None and os.path.isfile(args.resume_pth):
        print(f"🔄 从已有模型 {args.resume_pth} 加载权重，并继续训练")
        checkpoint = torch.load(args.resume_pth, map_location=device)
        # 如果保存的是 state_dict，则直接加载
        model.load_state_dict(checkpoint)
        # 如果想恢复优化器状态，需要在保存时一并保存 optimizer.state_dict()
        # 例如：torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, 'xxx.pth')
        # 然后这里加载并恢复：optimizer.load_state_dict(checkpoint['optimizer']); start_epoch = checkpoint['epoch'] + 1
        # 若 checkpoint 只是模型权重，则 start_epoch 依然从 0 开始
        print("✅ 模型权重加载完成\n")
    else:
        if args.resume_pth is not None:
            print(f"⚠️ 指定的 resume_pth 文件不存在：{args.resume_pth}，将从头开始训练\n")

    # ----------------------------
    # 训练与验证循环
    # ----------------------------
    best_model_path = 'best_model.pth'
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n======== Epoch {epoch+1}/{args.num_epochs} ========")
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion, device)
        print(f"Val   Loss: {val_loss:.4f} | Acc: {accuracy:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # 保存最优模型（以 F1 为准）
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ 保存新最佳模型，F1 提升到 {best_f1:.4f}")

    print(f"\n训练结束，最佳 F1 为 {best_f1:.4f}")
    print(f"最佳模型已保存到: {best_model_path}")


if __name__ == '__main__':
    main()
