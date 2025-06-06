# ad-skiper-model
一个B站内嵌广告识别模型   
基于 <a href="https://github.com/XiaoMi/dasheng" target="__blank">Dasheng (大声)</a>  
数据集来源: <a href="https://github.com/hanydd/BilibiliSponsorBlock" target="__blank">B站空降助手</a>

# WIP🚧
> 施工中


```bash
python3 .\att_infer.py --model_path .\att_best_model_epoch1_9530.pth --min_ad_duration 5 --merge_gap_duration 60  --audio_path .\audio_test\BV1NMTMzvEkP.mp3
```