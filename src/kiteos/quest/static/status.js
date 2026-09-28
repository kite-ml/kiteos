// Copyright 2026 Kite ML.
// KiteOS status pill for the Quest teleop page. It polls the proxy's /kiteos/api/status
// and only touches its own #kiteos-status element, so nothing here can affect the stock page's
// teleop.js (which owns #status, the buttons and the WebXR session).
(function () {
    'use strict';

    const STATUS_URL = '/kiteos/api/status';
    const POLL_MS = 3000;
    const TIMEOUT_MS = 2500;

    function render(state, text, detail) {
        const pill = document.getElementById('kiteos-status');
        if (!pill) return;
        if (pill.dataset.state !== state) pill.dataset.state = state;
        if (pill.textContent !== text) pill.textContent = text;
        pill.title = detail;
    }

    function describe(status) {
        if (status.upstream !== 'up') return ['down', 'Teleop offline'];
        if (status.adb === 'unauthorized') return ['warn', 'Allow USB debugging'];
        return ['ok', 'Teleop online'];
    }

    async function refresh() {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
        try {
            const response = await fetch(STATUS_URL, { cache: 'no-store', signal: controller.signal });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const status = await response.json();
            const [state, text] = describe(status);
            const adb = status.adb_detail ? `${status.adb} (${status.adb_detail})` : status.adb;
            render(state, text, `KiteOS ${status.version} · teleop server ${status.upstream} · adb ${adb}`);
        } catch (error) {
            render('down', 'Proxy offline', 'Cannot reach the KiteOS Quest proxy');
        } finally {
            clearTimeout(timer);
        }
    }

    async function loop() {
        // Skip network work while the 2D page is hidden (e.g. during an immersive session).
        if (!document.hidden) await refresh();
        setTimeout(loop, POLL_MS);
    }

    async function start() {
        await refresh(); // always paint once, even if the page starts hidden
        setTimeout(loop, POLL_MS);
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) refresh();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start, { once: true });
    } else {
        start();
    }
})();
