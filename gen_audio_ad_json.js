const fs = require('fs');
const { db } = require('./utils/db.js');
const path = require('path');

function convertSecondsToTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    return `${hours}:${minutes}:${remainingSeconds}`;
}

async function main() {
    // 读取 audio 目录下的所有 MP3 文件
    const audioDir = './audio';
    const files = fs.readdirSync(audioDir);
    const mp3Files = files.filter(file => file.endsWith('.mp3'));

    let arr = [];

    for (const mp3File of mp3Files) {
        // 从文件名中提取 videoID
        const videoID = path.parse(mp3File).name;
        
        // 查询数据库中该 videoID 的所有记录
        const rows = db.prepare(`
            SELECT * FROM sponsor_spider 
            WHERE videoID = ? AND download_status = 'finish'
        `).all(videoID);

        let video = {
            videoID,
            audioPath: `./audio/${videoID}.mp3`,
            ads: []
        }
        // 处理每条记录
        for (const row of rows) {
            let startTime = parseFloat(row.startTime);
            let endTime = parseFloat(row.endTime);

            let start = convertSecondsToTime(startTime);
            let end = convertSecondsToTime(endTime);
            
            video.ads.push({
                startTime,
                endTime,
                start,
                end,
            });
        }
        arr.push(video);
        console.log(`处理完成，共找到 ${video.ads.length} 条记录`);
    }

    // 将结果写入 JSON 文件
    fs.writeFileSync('audio_ads.json', JSON.stringify(arr, null, 2));
    console.log(`处理完成，共找到 ${arr.length} 条记录`);
}

main().catch(console.error);
