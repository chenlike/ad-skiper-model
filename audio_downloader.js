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

async function downloadAll() {
    const rows = db.prepare(`SELECT * FROM sponsor_spider where download_status !='finish'  or download_status is null `).all();
    console.log(`总共需要下载 ${rows.length} 个视频`);
    let finished = 0;
    let total = rows.length;
    
    const batchSize = 10; // 限制并发数为10
    let currentIndex = 0;

    while (currentIndex < rows.length) {
        const batch = rows.slice(currentIndex, currentIndex + batchSize);
        const promises = batch.map(row => {
            console.log(`[音频下载器 ${finished + 1}/${total}] 开始下载视频: ${row.videoID}`);
            finished++;
            return downloadAudio(row.videoID);
        });

        await Promise.allSettled(promises);
        currentIndex += batchSize;
    }
}

function printUsage() {
    console.log(`
使用方法:
    下载单个视频:  node audio_downloader.js -v BV号
    下载所有视频:  node audio_downloader.js --all
    
示例:
    node audio_downloader.js -v BV1BHJtz6Ey9
    node audio_downloader.js --all
`);
}

async function main() {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        printUsage();
        return;
    }

    if (args[0] === '--all') {
        await downloadAll();
    } else if (args[0] === '-v' && args[1]) {
        console.log(`开始下载视频: ${args[1]}`);
        await downloadAudio(args[1]);
    } else {
        printUsage();
    }
}

main().catch(console.error);
