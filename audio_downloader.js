const { AudioDownloader, AudioQualityEnums } = require("./utils/audio.js");
const fs = require('fs');
const { db } = require('./utils/db.js');

async function downloadAudio(videoID) {
    try {
        let res = await new AudioDownloader(`https://www.bilibili.com/video/${videoID}`, AudioQualityEnums.Low).run();
        fs.writeFileSync(`./audio_test/${videoID}.mp3`, res.buffer);
        db.prepare(`UPDATE sponsor_spider SET download_status = 'finish' WHERE videoID = ?`).run(videoID);
    } catch (e) {
        console.error(e);
    }
}

async function main() {
    const rows = db.prepare(`SELECT * FROM sponsor_spider where download_status !='finish'  or download_status is null `).all();
    console.log(rows.length);
    let finished = 0;
    let total = rows.length;
    
    const batchSize = 10; // 限制并发数为10
    let currentIndex = 0;

    while (currentIndex < rows.length) {
        const batch = rows.slice(currentIndex, currentIndex + batchSize); // 获取当前批次
        const promises = batch.map(row => {
            console.log(`[音频下载器 ${finished + 1}/${total}] 开始下载视频: ${row.videoID}`);
            finished++;
            return downloadAudio(row.videoID);
        });

        await Promise.allSettled(promises); // 等待当前批次的下载任务完成
        currentIndex += batchSize; // 移动到下一个批次
    }
}

downloadAudio("BV139jMzTEKf")
// main().catch(console.error);
