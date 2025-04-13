function checkForAds() {
    const adElements = document.querySelectorAll('.ad-showing');
    if (adElements.length > 0) {
        location.reload();
    }
}

// 每 0.1 秒检查一次是否有广告
setInterval(checkForAds, 100);
    