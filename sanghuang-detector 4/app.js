class SanghuangDetector {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.currentImage = null;
        this.currentImageSrc = null;
        this.currentImgSize = 640;
        this.resultCanvas = null;
        this.padX = 0;
        this.padY = 0;
        this.scale = 1;
        this.init();
    }

    async init() {
        this.bindEvents();
        await this.loadModel();
    }

    // ============================================
    // 模型加载（带进度条 + IndexedDB 缓存）
    // ============================================
    async loadModel() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = document.querySelector('.loading-text');
        const loadingHint = document.querySelector('.loading-hint');

        loadingOverlay.style.display = 'flex';
        this.injectProgressBar(loadingOverlay);

        try {
            // 清理旧缓存
            await this.deleteFromCache('sanghuang_model_v1');
            await this.deleteFromCache('sanghuang_model_v2');

            loadingText.textContent = '检查缓存...';
            let modelBuffer = await this.loadFromCache('sanghuang_model_v3');

            if (modelBuffer) {
                console.log('✅ 从缓存加载模型');
                loadingText.textContent = '从缓存加载中...';
                this.updateProgress(100, '从缓存加载...');
            } else {
                console.log('⏬ 首次下载模型...');
                loadingText.textContent = '首次加载，正在下载模型...';
                modelBuffer = await this.downloadWithProgress('model/model.onnx');
                loadingText.textContent = '正在缓存模型...';
                await this.saveToCache('sanghuang_model_v3', modelBuffer);
            }

            loadingText.textContent = '初始化推理引擎...';
            this.updateProgress(100, '初始化推理引擎...');

            this.model = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.isModelLoaded = true;
            console.log('✅ 模型加载成功');
            console.log('   输入名:', this.model.inputNames);
            console.log('   输出名:', this.model.outputNames);

        } catch (error) {
            console.error('❌ 模型加载失败:', error);
            loadingText.textContent = '加载失败';
            loadingHint.textContent = error.message;
            loadingHint.style.color = '#ff6b6b';
            await this.deleteFromCache('sanghuang_model_v3');
            setTimeout(() => { loadingOverlay.style.display = 'none'; }, 5000);
            return;
        }

        loadingOverlay.style.display = 'none';
    }

    async downloadWithProgress(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`模型文件不存在 (HTTP ${response.status})，路径: ${url}`);
        }

        const contentLength = response.headers.get('content-length');
        const total = parseInt(contentLength, 10);
        let loaded = 0;

        const reader = response.body.getReader();
        const chunks = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            if (total) {
                const progress = Math.round((loaded / total) * 100);
                this.updateProgress(progress, `下载中 ${progress}%`);
            }
        }

        const blob = new Blob(chunks);
        return await blob.arrayBuffer();
    }

    injectProgressBar(container) {
        if (document.getElementById('progressBarContainer')) return;
        const progressContainer = document.createElement('div');
        progressContainer.id = 'progressBarContainer';
        progressContainer.className = 'progress-container';
        const progressBar = document.createElement('div');
        progressBar.id = 'progressBar';
        progressBar.className = 'progress-bar';
        const progressText = document.createElement('div');
        progressText.id = 'progressText';
        progressText.className = 'progress-text';
        progressContainer.appendChild(progressBar);
        progressContainer.appendChild(progressText);
        container.appendChild(progressContainer);
    }

    updateProgress(percent, text) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        if (progressBar && progressText) {
            progressBar.style.width = percent + '%';
            progressText.textContent = text;
        }
    }

    // ============================================
    // IndexedDB 缓存
    // ============================================
    async loadFromCache(key) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('SanghuangDetector', 1);
            request.onerror = () => reject('IndexedDB 打开失败');
            request.onsuccess = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    resolve(null);
                    return;
                }
                const transaction = db.transaction(['models'], 'readonly');
                const store = transaction.objectStore('models');
                const getRequest = store.get(key);
                getRequest.onsuccess = () => resolve(getRequest.result);
                getRequest.onerror = () => resolve(null);
            };
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    db.createObjectStore('models');
                }
            };
        });
    }

    async saveToCache(key, buffer) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('SanghuangDetector', 1);
            request.onerror = () => reject('IndexedDB 打开失败');
            request.onsuccess = (event) => {
                const db = event.target.result;
                const transaction = db.transaction(['models'], 'readwrite');
                const store = transaction.objectStore('models');
                const putRequest = store.put(buffer, key);
                putRequest.onsuccess = () => resolve();
                putRequest.onerror = () => reject('缓存失败');
            };
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    db.createObjectStore('models');
                }
            };
        });
    }

    async deleteFromCache(key) {
        return new Promise((resolve) => {
            const request = indexedDB.open('SanghuangDetector', 1);
            request.onsuccess = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    resolve();
                    return;
                }
                const transaction = db.transaction(['models'], 'readwrite');
                const store = transaction.objectStore('models');
                store.delete(key);
                resolve();
            };
            request.onerror = () => resolve();
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    db.createObjectStore('models');
                }
            };
        });
    }

    // ============================================
    // 事件绑定
    // ============================================
    bindEvents() {
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const detectBtn = document.getElementById('detectBtn');
        const exportBtn = document.getElementById('exportBtn');
        const confidenceSlider = document.getElementById('confidence');
        const confidenceValue = document.getElementById('confidenceValue');
        const imageSizeSelect = document.getElementById('imageSize');

        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', (e) => e.preventDefault());
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) this.handleFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) this.handleFile(e.target.files[0]);
        });

        detectBtn.addEventListener('click', () => this.detect());
        exportBtn.addEventListener('click', () => this.exportResult());

        confidenceSlider.addEventListener('input', (e) => {
            const v = (parseInt(e.target.value) / 100).toFixed(2);
            confidenceValue.textContent = v;
            document.getElementById('confidenceDisplay').textContent = v;
        });

        imageSizeSelect.addEventListener('change', (e) => {
            if (e.target.value !== 'auto') this.currentImgSize = parseInt(e.target.value);
        });

        // 尺寸转换
        const resizeUploadZone = document.getElementById('resizeUploadZone');
        const resizeFileInput = document.getElementById('resizeFileInput');
        const resizeBtn = document.getElementById('resizeBtn');
        const downloadResizedBtn = document.getElementById('downloadResizedBtn');

        if (resizeUploadZone) {
            resizeUploadZone.addEventListener('click', () => resizeFileInput.click());
            resizeUploadZone.addEventListener('dragover', (e) => e.preventDefault());
            resizeUploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                if (e.dataTransfer.files.length > 0) this.handleResizeFile(e.dataTransfer.files[0]);
            });
            resizeFileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) this.handleResizeFile(e.target.files[0]);
            });
            if (resizeBtn) resizeBtn.addEventListener('click', () => this.resizeImage());
            if (downloadResizedBtn) downloadResizedBtn.addEventListener('click', () => this.downloadResizedImage());
        }
    }

    async handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('请上传图片文件！');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.currentImageSrc = e.target.result;
                this.displayImage(img, 'originalImage');
                document.getElementById('detectBtn').disabled = false;
                document.getElementById('exportBtn').disabled = true;
                document.getElementById('imageSizeValue').textContent = `${img.width}x${img.height}`;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    async handleResizeFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('请上传图片文件！');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.resizeImageElement = img;
                this.resizeImageSrc = e.target.result;
                this.displayImage(img, 'resizePreview');
                document.getElementById('resizeBtn').disabled = false;
                document.getElementById('downloadResizedBtn').disabled = true;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    displayImage(img, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        const imgElement = document.createElement('img');
        imgElement.src = img.src || this.currentImageSrc;
        imgElement.className = 'uploaded-image';
        imgElement.style.maxWidth = '100%';
        imgElement.style.maxHeight = '300px';
        container.appendChild(imgElement);
    }

    // ============================================
    // 检测主流程
    // ============================================
    async detect() {
        if (!this.isModelLoaded || !this.currentImage) {
            alert('请先上传图片！');
            return;
        }

        const startTime = performance.now();

        try {
            // 🔪 统一走切片推理（小图会自动跳过切片）
            const detections = await this.detectWithSlicing(this.currentImage, 640, 0.25);

            this.displayResults(detections, this.currentImage);

            const endTime = performance.now();
            const detectionTime = Math.round(endTime - startTime);
            document.getElementById('detectionTime').textContent = `${detectionTime}ms`;
            document.getElementById('detectionCount').textContent = detections.length;
            document.getElementById('foundCount').textContent = detections.length;
            document.getElementById('exportBtn').disabled = false;

        } catch (error) {
            console.error('检测失败:', error);
            alert('检测失败: ' + error.message);
        }
    }

    // ============================================
    // 切片推理：用于高分辨率 + 小目标密集场景
    // ============================================
    async detectWithSlicing(image, sliceSize = 640, overlap = 0.25) {
        console.log(`🔪 启动切片推理: 原图 ${image.width}x${image.height}, 切片 ${sliceSize}, 重叠 ${overlap}`);

        // 小图直接走原流程
        if (image.width <= sliceSize && image.height <= sliceSize) {
            console.log('  图片较小，跳过切片');
            const inputTensor = await this.preprocessImage(image, sliceSize);
            const results = await this.runInference(inputTensor);
            return this.postprocessResults(results);
        }

        const step = Math.round(sliceSize * (1 - overlap));
        const slices = [];

        for (let y = 0; y < image.height; y += step) {
            for (let x = 0; x < image.width; x += step) {
                const sx = Math.max(0, Math.min(x, image.width - sliceSize));
                const sy = Math.max(0, Math.min(y, image.height - sliceSize));
                const sw = Math.min(sliceSize, image.width - sx);
                const sh = Math.min(sliceSize, image.height - sy);
                slices.push({ x: sx, y: sy, w: sw, h: sh });
                if (x + sliceSize >= image.width) break;
            }
            if (y + sliceSize >= image.height) break;
        }

        // 去重（边界切片可能重复）
        const uniq = [];
        const seen = new Set();
        for (const s of slices) {
            const k = `${s.x},${s.y},${s.w},${s.h}`;
            if (!seen.has(k)) { seen.add(k); uniq.push(s); }
        }

        console.log(`🔪 切成 ${uniq.length} 片`);

        const allDetections = [];
        for (let i = 0; i < uniq.length; i++) {
            const s = uniq[i];

            // 裁出这一片
            const cvs = document.createElement('canvas');
            cvs.width = s.w;
            cvs.height = s.h;
            cvs.getContext('2d').drawImage(image, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);

            // 转成 Image（因为 preprocessImage 期待 Image 对象）
            const sliceImg = await new Promise(resolve => {
                const im = new Image();
                im.onload = () => resolve(im);
                im.src = cvs.toDataURL();
            });

            const inputTensor = await this.preprocessImage(sliceImg, sliceSize);
            const results = await this.runInference(inputTensor);
            const dets = this.postprocessResults(results);

            console.log(`  片 ${i+1}/${uniq.length} @(${s.x},${s.y}) ${s.w}x${s.h} → ${dets.length} 个`);

            // 坐标映射回全图
            dets.forEach(d => {
                allDetections.push({
                    ...d,
                    x1: d.x1 + s.x,
                    y1: d.y1 + s.y,
                    x2: d.x2 + s.x,
                    y2: d.y2 + s.y,
                });
            });
        }

        console.log(`📦 合并前总框数: ${allDetections.length}`);
        const merged = this.applyNMS(allDetections, 0.45);
        console.log(`📦 全局 NMS 后: ${merged.length}`);
        return merged;
    }

    // ============================================
    // 预处理（letterbox 填充 114 灰边，和 Ultralytics 一致）
    // ============================================
    preprocessImage(image, img_size) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img_size;
        canvas.height = img_size;

        const scale = Math.min(img_size / image.width, img_size / image.height);
        const newWidth = Math.round(image.width * scale);
        const newHeight = Math.round(image.height * scale);
        const padX = (img_size - newWidth) / 2;
        const padY = (img_size - newHeight) / 2;

        this.padX = padX;
        this.padY = padY;
        this.scale = scale;

        // YOLO 默认 letterbox 填充 RGB(114,114,114)
        ctx.fillStyle = 'rgb(114,114,114)';
        ctx.fillRect(0, 0, img_size, img_size);
        ctx.drawImage(image, padX, padY, newWidth, newHeight);

        const imageData = ctx.getImageData(0, 0, img_size, img_size);
        const data = imageData.data;
        const inputData = new Float32Array(3 * img_size * img_size);
        const pixelCount = img_size * img_size;

        // RGB 顺序，归一化 /255，CHW 排列
        for (let i = 0, p = 0; i < data.length; i += 4, p++) {
            inputData[p] = data[i] / 255.0;
            inputData[p + pixelCount] = data[i + 1] / 255.0;
            inputData[p + 2 * pixelCount] = data[i + 2] / 255.0;
        }

        return new ort.Tensor('float32', inputData, [1, 3, img_size, img_size]);
    }

    async runInference(inputTensor) {
        const inputName = this.model.inputNames[0] || 'images';
        const feeds = {};
        feeds[inputName] = inputTensor;
        return await this.model.run(feeds);
    }

    // ============================================
    // 后处理（坐标保持在 640 像素空间，不归一化）
    // ============================================
    postprocessResults(results) {
        const outName = this.model.outputNames[0] || 'output0';
        const outputTensor = results[outName] || results[Object.keys(results)[0]];
        const output = outputTensor.data;
        const outputShape = outputTensor.dims;

        console.log('输出 shape:', outputShape);

        const confidenceThreshold = parseFloat(document.getElementById('confidence').value) / 100;
        const detections = [];

        const numChannels = outputShape[1];
        const numDetections = outputShape[2];
        const numClasses = numChannels - 4;

        console.log(`通道=${numChannels}, 框=${numDetections}, 类别=${numClasses}, 阈值=${confidenceThreshold}`);

        for (let i = 0; i < numDetections; i++) {
            let maxScore = 0;
            let bestClass = 0;
            for (let c = 0; c < numClasses; c++) {
                const score = output[(4 + c) * numDetections + i];
                if (score > maxScore) {
                    maxScore = score;
                    bestClass = c;
                }
            }
            if (maxScore < confidenceThreshold) continue;

            const cx = output[0 * numDetections + i];
            const cy = output[1 * numDetections + i];
            const w  = output[2 * numDetections + i];
            const h  = output[3 * numDetections + i];

            // xywh → xyxy (仍在 640 输入空间)
            const x1_640 = cx - w / 2;
            const y1_640 = cy - h / 2;
            const x2_640 = cx + w / 2;
            const y2_640 = cy + h / 2;

            // ✅ 反 letterbox：减灰边偏移，再除缩放比例
            // 这里用的是最近一次 preprocessImage 设置的 padX/padY/scale
            // 在切片模式下，就是该切片对应的值（正好是本切片的局部坐标）
            const x1 = (x1_640 - this.padX) / this.scale;
            const y1 = (y1_640 - this.padY) / this.scale;
            const x2 = (x2_640 - this.padX) / this.scale;
            const y2 = (y2_640 - this.padY) / this.scale;

            detections.push({
                x1, y1, x2, y2,
                confidence: maxScore,
                classId: bestClass
            });
        }

        console.log(`阈值过滤后: ${detections.length} 个`);
        const nms = this.applyNMS(detections, 0.45);
        console.log(`NMS 后: ${nms.length} 个`);
        return nms;
    }

    applyNMS(detections, iouThreshold) {
        detections.sort((a, b) => b.confidence - a.confidence);
        const keep = [];
        const suppressed = new Array(detections.length).fill(false);
        for (let i = 0; i < detections.length; i++) {
            if (suppressed[i]) continue;
            keep.push(detections[i]);
            for (let j = i + 1; j < detections.length; j++) {
                if (!suppressed[j] && this.computeIOU(detections[i], detections[j]) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return keep;
    }

    computeIOU(a, b) {
        const x1 = Math.max(a.x1, b.x1);
        const y1 = Math.max(a.y1, b.y1);
        const x2 = Math.min(a.x2, b.x2);
        const y2 = Math.min(a.y2, b.y2);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (areaA + areaB - inter);
    }

    // ============================================
    // 绘制结果
    // ============================================
    displayResults(detections, image) {
        console.log(`🎯 绘制 ${detections.length} 个框，原图 ${image.width}x${image.height}`);

        const container = document.getElementById('resultImage');
        container.innerHTML = '';

        // 画布尺寸限制，避免超大图 toDataURL 失败
        const MAX_SIDE = 1600;
        const dispScale = Math.min(1, MAX_SIDE / Math.max(image.width, image.height));
        const cw = Math.round(image.width * dispScale);
        const ch = Math.round(image.height * dispScale);

        const canvas = document.createElement('canvas');
        canvas.width = cw;
        canvas.height = ch;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, cw, ch);

        const lw = Math.max(4, Math.round(cw / 250));
        const fontSize = Math.max(18, Math.round(cw / 45));

        if (detections.length === 0) {
            ctx.fillStyle = 'rgba(255,0,0,0.9)';
            ctx.font = `bold ${fontSize}px Arial`;
            ctx.fillText('未检测到桑黄', 20, fontSize + 10);
        } else {
            detections.forEach((det, idx) => {
                // ★★★ 关键：det 坐标已经是原图空间，只需乘 dispScale 适配画布
                // ★★★ 不要再减 padX / 除 scale / 乘 img_size！
                const x1 = det.x1 * dispScale;
                const y1 = det.y1 * dispScale;
                const x2 = det.x2 * dispScale;
                const y2 = det.y2 * dispScale;

                console.log(`画框${idx}: (${x1.toFixed(0)},${y1.toFixed(0)}) -> (${x2.toFixed(0)},${y2.toFixed(0)})`);

                // 双色描边，超醒目
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = lw;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                ctx.strokeStyle = '#FF0000';
                ctx.lineWidth = Math.max(2, lw / 2);
                ctx.strokeRect(x1 + lw/2, y1 + lw/2, x2 - x1 - lw, y2 - y1 - lw);

                // 标签
                const label = `桑黄 ${(det.confidence * 100).toFixed(1)}%`;
                ctx.font = `bold ${fontSize}px Arial`;
                const textW = ctx.measureText(label).width + 12;
                const textH = fontSize + 8;
                const ly = Math.max(textH, y1);

                ctx.fillStyle = '#00FF00';
                ctx.fillRect(x1, ly - textH, textW, textH);
                ctx.fillStyle = '#000';
                ctx.fillText(label, x1 + 6, ly - 6);
            });
        }

        this.resultCanvas = canvas;

        // 直接把 canvas 加入 DOM，不用 toDataURL
        canvas.style.maxWidth = '100%';
        canvas.style.height = 'auto';
        canvas.style.display = 'block';
        canvas.style.border = '3px solid blue'; // 加边框便于调试
        container.appendChild(canvas);

        // 保险措施，确保容器可见
        container.style.display = 'block';
        container.style.minHeight = '100px';

        console.log('✅ 结果已渲染');
    }

    exportResult() {
        if (!this.resultCanvas) return;
        const link = document.createElement('a');
        link.download = `sanghuang_detection_${Date.now()}.jpg`;
        link.href = this.resultCanvas.toDataURL('image/jpeg', 0.95);
        link.click();
    }

    // ============================================
    // 尺寸转换工具
    // ============================================
    async resizeImage() {
        if (!this.resizeImageElement) {
            alert('请先上传要转换的图片！');
            return;
        }
        const targetSize = parseInt(document.getElementById('resizeTargetSize').value);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = targetSize;
        canvas.height = targetSize;

        const scale = Math.min(targetSize / this.resizeImageElement.width, targetSize / this.resizeImageElement.height);
        const newWidth = Math.round(this.resizeImageElement.width * scale);
        const newHeight = Math.round(this.resizeImageElement.height * scale);
        const padX = (targetSize - newWidth) / 2;
        const padY = (targetSize - newHeight) / 2;

        ctx.fillStyle = 'rgb(114,114,114)';
        ctx.fillRect(0, 0, targetSize, targetSize);
        ctx.drawImage(this.resizeImageElement, padX, padY, newWidth, newHeight);

        this.resizedCanvas = canvas;
        const imgElement = document.createElement('img');
        imgElement.src = canvas.toDataURL('image/jpeg', 0.95);
        imgElement.className = 'resized-image';
        imgElement.style.maxWidth = '100%';
        imgElement.style.maxHeight = '300px';

        const preview = document.getElementById('resizePreview');
        preview.innerHTML = '';
        preview.appendChild(imgElement);

        document.getElementById('downloadResizedBtn').disabled = false;
        alert(`图片已调整为 ${targetSize}x${targetSize} 尺寸`);
    }

    downloadResizedImage() {
        if (!this.resizedCanvas) return;
        const link = document.createElement('a');
        link.download = `resized_${Date.now()}.jpg`;
        link.href = this.resizedCanvas.toDataURL('image/jpeg', 0.95);
        link.click();
    }
}

// ============================================
// 初始化应用
// ============================================
window.addEventListener('DOMContentLoaded', () => {
    new SanghuangDetector();
});
