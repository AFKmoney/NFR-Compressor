
let currentMode = 'compress';
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const filenameSpan = document.getElementById('filename');
const filesizeSpan = document.getElementById('filesize');
const actionBtn = document.getElementById('action-btn');
const terminal = document.getElementById('terminal');
const logs = document.getElementById('logs');
const resultOverlay = document.getElementById('result-overlay');
let selectedFile = null;

// Tab Switching
function switchTab(mode) {
    currentMode = mode;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    actionBtn.textContent = mode === 'compress' ? 'INITIALIZE NEURAL LINK' : 'DECODE FRACTAL STREAM';
    resetUI();
}

function resetUI() {
    selectedFile = null;
    fileInfo.classList.add('hidden');
    actionBtn.disabled = true;
    terminal.classList.add('hidden');
    logs.innerHTML = '<div class="log-line">Waiting for input stream...</div>';
}

// Drag & Drop Logic
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        updateFileInfo(selectedFile);
        actionBtn.disabled = false;

        // Auto-detect mode if obvious
        if (selectedFile.name.endsWith('.dmn')) {
            // If in compress mode, maybe suggest decompress?
            // Optional UX improvement
        }
    }
}

function updateFileInfo(file) {
    fileInfo.classList.remove('hidden');
    filenameSpan.textContent = file.name;
    filesizeSpan.textContent = formatBytes(file.size);
}

function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

// Action Logic
actionBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    terminal.classList.remove('hidden');
    log(">> DAEMON: Establishing Neural Link...");
    log(">> SYSTEM: Uploading to Secure Core...");

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Simulate some logs for effect
    await sleep(800);
    log(`>> MODE: ${currentMode.toUpperCase()}`);
    log(">> CORE: Initializing NFR Engine v1.0");

    const endpoint = currentMode === 'compress' ? '/compress' : '/decompress';

    try {
        log(">> NETWORK: Transmitting Payload...");
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Unknown Error');
        }

        const data = await response.json();

        log(">> SUCCESS: Operation Complete.");

        if (currentMode === 'compress') {
            log(`>> METRICS: ${data.original_size} -> ${data.compressed_size} bytes`);
            log(`>> RATIO: ${data.ratio}x`);
            showResult(data);
        } else {
            log(">> SYSTEM: File Restored.");
            showResultDecomp(data);
        }

    } catch (error) {
        log(`>> ERROR: ${error.message}`, 'red');
    }
});

function log(text, color = 'inherit') {
    const div = document.createElement('div');
    div.className = 'log-line';
    div.textContent = text;
    if (color === 'red') div.style.color = '#ff5f56';
    logs.appendChild(div);
    logs.scrollTop = logs.scrollHeight;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Result Overlay
function showResult(data) {
    document.getElementById('res-original').textContent = formatBytes(data.original_size);
    document.getElementById('res-compressed').textContent = formatBytes(data.compressed_size);
    document.getElementById('res-ratio').textContent = data.ratio + 'x';

    const dlBtn = document.getElementById('download-btn');
    dlBtn.onclick = () => window.location.href = data.download_url;

    resultOverlay.classList.remove('hidden');
}

function showResultDecomp(data) {
    // Simplified result for decompression
    document.getElementById('res-original').textContent = '-';
    document.getElementById('res-compressed').textContent = '-';
    document.getElementById('res-ratio').textContent = 'Restored';

    const dlBtn = document.getElementById('download-btn');
    dlBtn.onclick = () => window.location.href = data.download_url;

    resultOverlay.classList.remove('hidden');
}

function closeResult() {
    resultOverlay.classList.add('hidden');
}
