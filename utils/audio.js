// 代码来源: https://github.com/lxw15337674/bilibili-audio-downloader

const axios = require('axios');


const AudioQualityEnums = {
    Low: 64,
    Medium: 132,
    High: 192,
    Highest: 320,
};

class AudioDownloader {
    constructor(baseUrl, audioQuality = AudioQualityEnums.High) {
        this.baseUrl = baseUrl;
        this.audioQuality = audioQuality;
        this.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com",
            "Origin": "https://www.bilibili.com",
        };
        this.bv = '';
        this.cid = '';
        this.title = '';
        this.audioUrl = '';
        this.maxRetries = 1;
        this.retryDelay = 1; // 3 seconds
        this.downloadStartTime = 0;
        this.axiosInstance = axios.create({
            timeout: 60000,
            maxRedirects: 5,
            headers: this.headers,
        });
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    }

    formatSpeed(bytesPerSecond) {
        return `${this.formatBytes(bytesPerSecond)}/s`;
    }

    async retryOperation(operation, retryCount = 0) {
        try {
            return await operation();
        } catch (error) {
            if (retryCount >= this.maxRetries) {
                throw error;
            }

            console.log(`[音频下载器] 第${retryCount + 1}次尝试失败，${this.retryDelay / 1000}秒后重试...`);
            await this.sleep(this.retryDelay);
            return this.retryOperation(operation, retryCount + 1);
        }
    }

    async run() {
        console.log("[音频下载器] 开始下载音频...");
        await this.retryOperation(() => this.getCid());
        await this.retryOperation(() => this.getAudioUrl());
        const buffer = await this.retryOperation(() => this.downloadAudio());
        return {
            buffer,
            filename: `${this.title}.mp3`
        };
    }

    async getCid() {
        const pattern = /(BV[a-zA-Z0-9]+)/;
        const match = this.baseUrl.match(pattern);
        if (!match) throw new Error("Invalid BiliBili URL");
        this.bv = match[1];

        const response = await this.axiosInstance.get("https://api.bilibili.com/x/web-interface/view", {
            params: { bvid: this.bv },
            headers: this.headers,
        });

        if (!response.data.data) {
            throw new Error("Failed to get video information");
        }

        this.cid = response.data.data.cid;
        this.title = response.data.data.title.replace(/[<>:"/\\|?*]/g, '_'); // Remove invalid filename characters
        console.log(`[音频下载器] 获取到CID: ${this.cid}`);
        console.log(`[音频下载器] 视频标题: ${this.title}`);
    }

    async getAudioUrl() {
        const response = await this.axiosInstance.get("https://api.bilibili.com/x/player/wbi/playurl", {
            params: {
                bvid: this.bv,
                cid: this.cid,
                qn: this.audioQuality,
                fnver: 0,
                fnval: 4048,
                fourk: 1,
            },
            headers: this.headers,
        });

        if (!response.data.data?.dash?.audio?.length) {
            throw new Error("No audio stream found");
        }

        const audioStreams = response.data.data.dash.audio;
        let selectedStream = audioStreams.find(stream => stream.id === this.audioQuality);
        if (!selectedStream) {
            selectedStream = audioStreams[0];
        }

        this.audioUrl = selectedStream.baseUrl;
        console.log(`[音频下载器] 获取到音频URL，质量: ${selectedStream.id}kbps`);
    }

    async downloadAudio() {
        try {
            this.downloadStartTime = Date.now();
            const response = await this.axiosInstance.get(this.audioUrl, {
                headers: {
                    ...this.headers,
                    referer: this.baseUrl,
                },
                responseType: 'arraybuffer',
                decompress: true,
                maxRedirects: 10,
                onDownloadProgress: (progressEvent) => {
                    const elapsedTime = (Date.now() - this.downloadStartTime) / 1000;
                    const speed = progressEvent.loaded / elapsedTime;
                    const percent = Math.round((progressEvent.loaded * 100) / (progressEvent.total || progressEvent.loaded));
                    const downloadedSize = this.formatBytes(progressEvent.loaded);
                    const totalSize = progressEvent.total ? this.formatBytes(progressEvent.total) : 'Unknown';
                    const downloadSpeed = this.formatSpeed(speed);

                    process.stdout.write(`\r[音频下载器] 下载进度: ${percent}% | ${downloadedSize}/${totalSize} | ${downloadSpeed}`);

                    if (progressEvent.loaded === progressEvent.total) {
                        process.stdout.write('\n');
                    }
                },
            });

            console.log(`[音频下载器] 音频下载成功`);
            return Buffer.from(response.data);
        } catch (error) {
            console.error(`[音频下载器] 下载音频失败:`, error);
            throw error;
        }
    }
}

// 示例用法
/*
const bvid = 'BV1ZM4m1y7aD'
new AudioDownloader(`https://www.bilibili.com/video/${bvid}`,AudioQualityEnums.Low).run()
.then(({ buffer, filename }) => { 
    // 存为文件
    fs.writeFileSync(filename, buffer);
})
.catch(console.error);
*/

module.exports = {
    AudioDownloader,
    AudioQualityEnums
}