const STORAGE_KEY = 'videoPlaybackTime';
const CHECK_INTERVAL_MS = 2500;
const SAVE_THROTTLE_MS = 2000;

let lastKnownTime = 0;
let lastSaveAt = 0;
let lastReloadAt = 0;

function isVisible(el) {
    if (!el) return false;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
        return false;
    }
    return el.getClientRects().length > 0;
}

function isAdShowing() {
    if (document.querySelector('.ad-showing')) return true;

    const skipButton = document.querySelector('.ytp-ad-skip-button');
    if (isVisible(skipButton)) return true;

    const sponsorAdButton = document.querySelector('.ytp-ad-overlay-close-button');
    if (isVisible(sponsorAdButton)) return true;

    const adPlayerOverlay = document.querySelector('.ytp-ad-player-overlay-layout');
    if (isVisible(adPlayerOverlay)) return true;

    return false;
}

function saveProgress(video) {
    const time = Math.floor(video.currentTime);
    if (!Number.isFinite(time)) return;

    const now = Date.now();
    if (time === lastKnownTime && now - lastSaveAt < SAVE_THROTTLE_MS) return;

    lastKnownTime = time;
    lastSaveAt = now;

    chrome.storage.local.set({ [STORAGE_KEY]: time });
    console.log(`保存播放进度: ${time}s`);
}

function checkForAds() {
    const video = document.querySelector('video');
    const adShowing = isAdShowing();

    if (video && !adShowing) {
        saveProgress(video);
    }

    if (adShowing) {
        const now = Date.now();
        if (now - lastReloadAt < 3000) return;

        lastReloadAt = now;

        if (Number.isFinite(lastKnownTime)) {
            chrome.storage.local.set({ [STORAGE_KEY]: lastKnownTime });
        }

        console.log("刷新页面:" + location.href);
        location.replace(location.href);
    }
}

function restoreProgress(video) {
    chrome.storage.local.get([STORAGE_KEY], (result) => {
        const playbackTime = result[STORAGE_KEY];
        if (!Number.isFinite(playbackTime)) return;

        const apply = () => {
            video.currentTime = playbackTime;
            chrome.storage.local.remove([STORAGE_KEY]);
            console.log(`已恢复播放进度: ${playbackTime}s`);
        };

        if (video.readyState >= 1) {
            apply();
        } else {
            video.addEventListener('loadedmetadata', apply, { once: true });
        }
    });
}

function restoreWhenVideoAvailable() {
    const video = document.querySelector('video');
    if (video) {
        restoreProgress(video);
        return;
    }

    const observer = new MutationObserver(() => {
        const v = document.querySelector('video');
        if (v) {
            observer.disconnect();
            restoreProgress(v);
        }
    });

    observer.observe(document.documentElement, { childList: true, subtree: true });
    setTimeout(() => observer.disconnect(), 10000);
}

// 每 2.5 秒检查一次是否有广告
setInterval(checkForAds, CHECK_INTERVAL_MS);

// 页面加载后恢复播放进度
document.addEventListener('DOMContentLoaded', restoreWhenVideoAvailable);
// YouTube SPA 导航后恢复播放进度
document.addEventListener('yt-navigate-finish', restoreWhenVideoAvailable);
