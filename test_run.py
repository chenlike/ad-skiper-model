import os
import json
import numpy as np
from att_infer import AdDetector
from tqdm import tqdm

def load_test_data(json_path):
    """加载测试数据
    
    Args:
        json_path (str): 测试数据JSON文件路径
        
    Returns:
        dict: 测试数据字典，格式为 {audio_file: [ad_segments]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 将列表格式转换为字典格式
        return {item['audioPath']: item['ads'] for item in data}

def calculate_iou(segment1, segment2):
    """计算两个时间段的IoU（交并比）
    
    Args:
        segment1 (tuple): (start_time, end_time)
        segment2 (tuple): (start_time, end_time)
        
    Returns:
        float: IoU值
    """
    start1, end1 = segment1
    start2, end2 = segment2
    
    # 计算交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # 计算并集
    union = (end1 - start1) + (end2 - start2) - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_detection(true_ads, detected_ads, iou_threshold=0.5):
    """评估检测结果
    
    Args:
        true_ads (list): 真实广告时间段列表
        detected_ads (list): 检测到的广告时间段列表
        iou_threshold (float): IoU阈值
        
    Returns:
        tuple: (precision, recall, f1_score, false_positives, false_negatives)
    """
    if not true_ads and not detected_ads:
        return 1.0, 1.0, 1.0, 0, 0
    if not true_ads or not detected_ads:
        return 0.0, 0.0, 0.0, len(detected_ads), len(true_ads)
    
    # 将时间段转换为元组列表
    true_segments = [(ad['startTime'], ad['endTime']) for ad in true_ads]
    detected_segments = [(ad['startTime'], ad['endTime']) for ad in detected_ads]
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(true_segments), len(detected_segments)))
    for i, true_seg in enumerate(true_segments):
        for j, det_seg in enumerate(detected_segments):
            iou_matrix[i, j] = calculate_iou(true_seg, det_seg)
    
    # 找到匹配的检测结果
    matched_true = set()
    matched_det = set()
    
    # 对每个真实广告，找到最佳匹配的检测结果
    for i in range(len(true_segments)):
        best_j = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_j] >= iou_threshold and best_j not in matched_det:
            matched_true.add(i)
            matched_det.add(best_j)
    
    # 计算指标
    true_positives = len(matched_true)
    false_positives = len(detected_segments) - len(matched_det)
    false_negatives = len(true_segments) - len(matched_true)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score, false_positives, false_negatives

def format_time(seconds):
    """将秒数转换为 'MM:SS.xx' 格式的字符串"""
    minutes = int(seconds) // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"

def main():
    # 配置参数
    model_path = "./best_model1200_9587.pth"  # 请替换为实际的模型路径
    test_data_path = "audio_ads_test.json"
    audio_dir = "./audio_test"
    
    # 加载测试数据
    print("📂 加载测试数据...")
    test_data = load_test_data(test_data_path)
    
    # 创建检测器
    print("🛠️  初始化检测器...")
    detector = AdDetector(
        model_path=model_path,
        segment_duration=3.0,
        sequence_length=8,
        sample_rate=16000,
        batch_size=16,
        threshold=0.5,
        min_ad_duration=5.0,    # 过滤掉小于5秒的广告
        merge_gap_duration=60  # 合并间隔小于3秒的广告
    )
    
    # 评估结果
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_fp = 0
    total_fn = 0
    total_files = len(test_data)
    
    print("\n🚀 开始评估...")
    for audio_path, true_ads in tqdm(test_data.items(), desc="处理音频文件"):
        if not os.path.exists(audio_path):
            print(f"⚠️  警告: 找不到音频文件 {audio_path}")
            continue
            
        try:
            # 运行检测
            result = detector.detect(audio_path)
            detected_ads = result['detectedAds']
            
            # 评估结果
            precision, recall, f1, fp, fn = evaluate_detection(true_ads, detected_ads)
            
            # 累加指标
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_fp += fp
            total_fn += fn
            
            # 打印详细结果
            print(f"\n📊 文件: {audio_path}")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  误报数: {fp}")
            print(f"  漏报数: {fn}")
            
            # 打印时间戳对比
            print("\n  真实广告:")
            for ad in true_ads:
                print(f"    {format_time(ad['startTime'])} - {format_time(ad['endTime'])}")
            
            print("\n  检测结果:")
            for ad in detected_ads:
                print(f"    {format_time(ad['startTime'])} - {format_time(ad['endTime'])}")
                
        except Exception as e:
            print(f"❌ 处理文件 {audio_path} 时出错: {str(e)}")
            continue
    
    # 计算平均指标
    avg_precision = total_precision / total_files
    avg_recall = total_recall / total_files
    avg_f1 = total_f1 / total_files
    
    # 打印总体评估结果
    print("\n📈 总体评估结果:")
    print(f"处理文件数: {total_files}")
    print(f"平均精确率: {avg_precision:.4f}")
    print(f"平均召回率: {avg_recall:.4f}")
    print(f"平均F1分数: {avg_f1:.4f}")
    print(f"总误报数: {total_fp}")
    print(f"总漏报数: {total_fn}")
    
    # 保存评估结果
    results = {
        "total_files": total_files,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("\n✅ 评估完成！结果已保存至 evaluation_results.json")

if __name__ == "__main__":
    main()



