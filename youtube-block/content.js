function checkForAds() {
    const adElements = document.querySelectorAll('.ad-showing');
    if (adElements.length > 0) {
        location.replace(location.href);
        return;
    }

    const sponsorAdButton = document.querySelector('.ytp-ad-overlay-close-button');
    const skipButton = document.querySelector('.ytp-ad-skip-button');
    if (sponsorAdButton || skipButton) {
        location.replace(location.href);
    }
}

// 每 0.1 秒（100 毫秒）检查一次是否有广告
setInterval(checkForAds, 100);
    