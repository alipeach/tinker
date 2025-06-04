function checkForAds() {
    const adElements = document.querySelectorAll('.ad-showing');
    const sponsorAdButton = document.querySelector('.ytp-ad-overlay-close-button');
    const skipButton = document.querySelector('.ytp-ad-skip-button');
    const adPlayerOverlay = document.querySelector('.ytp-ad-player-overlay-layout');
    const video = document.querySelector('video');

    if (video) {
        // 保存播放进度到 chrome.storage
        const currentTime = Math.floor(video.currentTime)
        chrome.storage.local.set({ 'videoPlaybackTime': currentTime });
        console.log(`保存播放进度: ${currentTime}s`);
    }

    if (adElements.length > 0 || sponsorAdButton || skipButton || adPlayerOverlay) {
        console.log("刷新页面:" + location.href);
        location.replace(location.href);
    }



}

// 每 2.5 秒检查一次是否有广告
setInterval(checkForAds, 2500);

// 页面加载后恢复播放进度
document.addEventListener('DOMContentLoaded', () => {
    const video = document.querySelector('video');
    if (!video) return;

    chrome.storage.local.get(['videoPlaybackTime'], (result) => {
        const playbackTime = result.videoPlaybackTime;
        if (playbackTime) {
            video.currentTime = playbackTime;
            chrome.storage.local.remove('videoPlaybackTime');
            console.log(`已恢复播放进度: ${playbackTime}s`);
        }
    });
});    
    