// ============================================================================
// Real-Time YOLO Segmentation with Mask Overlay
// ============================================================================

class RealtimeSegmentation {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.overlayCanvas = null;
        this.overlayCtx = null;
        this.stream = null;
        this.isProcessing = false;
        this.frameInterval = 500; // Process every 500ms (2 FPS)
        this.lastFrameTime = 0;
        this.currentCamera = 'environment'; // Default to back camera
        this.detectionActive = false;
        this.animationFrameId = null;

        // Detection results cache
        this.latestResult = null;

        // Brightness detection
        this.frameCount = 0;
        this.brightnessCheckInterval = 8; // Check every 8 frames (~4 FPS at 2 FPS processing)
        this.brightnessThreshold = 50; // Below this = too dark (0-255 scale)
        this.brightnessWarningShown = false;

        // Guidance text fade timer
        this.guidanceTextTimer = null;
        this.guidanceShowInterval = null;

        // Color mapping for burn severity classes (updated for 4 classes)
        this.classColors = {
            0: { r: 255, g: 200, b: 100, name: 'First Degree' },    // Light orange
            1: { r: 255, g: 150, b: 50, name: 'Second Degree' },    // Orange
            2: { r: 255, g: 50, b: 50, name: 'Third Degree' },      // Red
            3: { r: 139, g: 0, b: 0, name: 'Fourth Degree' }        // Dark red
        };
    }

    async init() {
        // Create UI container
        const container = document.createElement('div');
        container.id = 'realtimeSegmentationContainer';
        container.className = 'realtime-container';
        container.innerHTML = `
            <div class="realtime-header">
                <button id="closeRealtimeBtn" class="realtime-btn close-btn">
                    <i class="fas fa-times"></i>
                </button>
                <h6 class="realtime-title">
                    <i class="fas fa-video me-2"></i>
                    Real-Time Burn Detection
                </h6>
                <button id="switchRealtimeCameraBtn" class="realtime-btn switch-btn">
                    <i class="fas fa-sync-alt"></i>
                </button>
            </div>

            <div class="realtime-video-container">
                <video id="realtimeVideo" autoplay muted playsinline></video>
                <canvas id="realtimeCanvas"></canvas>
                <canvas id="realtimeOverlay"></canvas>

                <!-- Guidance Overlay (like QR code scanner) -->
                <div class="guidance-overlay" id="guidanceOverlay">
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                        <!-- Corner brackets -->
                        <path d="M 30 25 L 25 25 L 25 30" stroke="#4CAF50" stroke-width="0.5" fill="none" />
                        <path d="M 70 25 L 75 25 L 75 30" stroke="#4CAF50" stroke-width="0.5" fill="none" />
                        <path d="M 30 75 L 25 75 L 25 70" stroke="#4CAF50" stroke-width="0.5" fill="none" />
                        <path d="M 70 75 L 75 75 L 75 70" stroke="#4CAF50" stroke-width="0.5" fill="none" />
                        <!-- Center rectangle -->
                        <rect x="25" y="25" width="50" height="50" stroke="#4CAF50" stroke-width="0.3" fill="none" stroke-dasharray="2,2" opacity="0.6" />
                    </svg>
                    <div class="guidance-text">Please frame the burn within this area</div>
                </div>

                <!-- Brightness Warning Toast -->
                <div class="brightness-warning d-none" id="brightnessWarning">
                    <span>Lighting too low! Pleasemove to a brighter area</span>
                </div>

                <div class="realtime-stats">
                    <div class="stat-item" id="detectionsItem">
                        <i class="fas fa-bullseye"></i>
                        <span id="realtimeDetections">0 detections</span>
                    </div>
                    <div class="stat-item" id="fpsItem">
                        <i class="fas fa-tachometer-alt"></i>
                        <span id="realtimeFPS">-- FPS</span>
                    </div>
                </div>
            </div>

            <div class="realtime-results">
                <div id="realtimeResultsContainer" class="results-list">
                    <div class="no-detection">
                        <i class="fas fa-search"></i>
                        <p>Scanning for burn injuries...</p>
                    </div>
                </div>
            </div>

            <div class="realtime-controls">
                <button id="pauseRealtimeBtn" class="btn-custom btn-warning">
                    <i class="fas fa-pause me-2"></i>
                    Pause Detection
                </button>
                <button id="captureRealtimeBtn" class="btn-custom btn-primary">
                    <i class="fas fa-camera me-2"></i>
                    Capture & Analyze
                </button>
            </div>
        `;

        document.body.appendChild(container);

        // Get elements
        this.video = document.getElementById('realtimeVideo');
        this.canvas = document.getElementById('realtimeCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlayCanvas = document.getElementById('realtimeOverlay');
        this.overlayCtx = this.overlayCanvas.getContext('2d');

        // Bind events
        document.getElementById('closeRealtimeBtn').addEventListener('click', () => this.stop());
        document.getElementById('switchRealtimeCameraBtn').addEventListener('click', () => this.switchCamera());
        document.getElementById('pauseRealtimeBtn').addEventListener('click', () => this.togglePause());
        document.getElementById('captureRealtimeBtn').addEventListener('click', () => this.captureAndAnalyze());

        // Start camera
        await this.startCamera();
    }

    async startCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: this.currentCamera,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    // Set canvas sizes to match video
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    this.overlayCanvas.width = this.video.videoWidth;
                    this.overlayCanvas.height = this.video.videoHeight;
                    resolve();
                };
            });

            // Start detection loop
            this.detectionActive = true;
            this.processFrame();

            // Setup guidance text fade behavior
            this.setupGuidanceTextFade();

        } catch (error) {
            console.error('Error starting camera:', error);
            alert('Could not access camera. Please check permissions.');
            this.stop();
        }
    }

    async switchCamera() {
        this.currentCamera = this.currentCamera === 'environment' ? 'user' : 'environment';

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        await this.startCamera();
    }

    setupGuidanceTextFade() {
        const guidanceText = document.querySelector('.guidance-text');
        if (!guidanceText) return;

        // Show initially for 3 seconds, then fade out
        this.guidanceTextTimer = setTimeout(() => {
            guidanceText.style.opacity = '0';
            guidanceText.style.transition = 'opacity 0.5s ease';
        }, 3000);

        // Show again every 10 seconds
        this.guidanceShowInterval = setInterval(() => {
            guidanceText.style.opacity = '1';
            setTimeout(() => {
                guidanceText.style.opacity = '0';
            }, 3000);
        }, 10000);
    }

    processFrame() {
        if (!this.detectionActive) return;

        const now = Date.now();

        // Draw video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Check brightness every N frames (avoid performance hit)
        this.frameCount++;
        if (this.frameCount % this.brightnessCheckInterval === 0) {
            this.checkBrightness();
        }

        // Process frame at specified interval
        if (now - this.lastFrameTime >= this.frameInterval && !this.isProcessing) {
            this.lastFrameTime = now;
            this.sendFrameForDetection();
        }

        this.animationFrameId = requestAnimationFrame(() => this.processFrame());
    }

    checkBrightness() {
        // Get image data from a sample region (center 50% of frame)
        const width = this.canvas.width;
        const height = this.canvas.height;
        const sampleX = width * 0.25;
        const sampleY = height * 0.25;
        const sampleWidth = width * 0.5;
        const sampleHeight = height * 0.5;

        const imageData = this.ctx.getImageData(sampleX, sampleY, sampleWidth, sampleHeight);
        const data = imageData.data;

        // Calculate mean brightness (convert to grayscale using luminosity formula)
        let totalBrightness = 0;
        for (let i = 0; i < data.length; i += 4) {
            // Luminosity formula: 0.299*R + 0.587*G + 0.114*B
            const brightness = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
            totalBrightness += brightness;
        }
        const meanBrightness = totalBrightness / (data.length / 4);

        // Show/hide warning based on threshold
        const warningEl = document.getElementById('brightnessWarning');
        if (warningEl) {
            if (meanBrightness < this.brightnessThreshold) {
                warningEl.classList.remove('d-none');
                this.brightnessWarningShown = true;
            } else {
                if (this.brightnessWarningShown) {
                    // Fade out after brightness improves
                    setTimeout(() => {
                        warningEl.classList.add('d-none');
                    }, 2000);
                    this.brightnessWarningShown = false;
                }
            }
        }
    }

    async sendFrameForDetection() {
        if (this.isProcessing) return;

        this.isProcessing = true;

        try {
            // Convert canvas to blob
            const blob = await new Promise(resolve => this.canvas.toBlob(resolve, 'image/jpeg', 0.8));

            // Create form data
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');

            // Send to server
            const response = await fetch('/api/realtime_detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // Update results
            this.latestResult = result;
            this.drawDetections(result);
            this.updateResultsUI(result);

        } catch (error) {
            console.error('Detection error:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    drawDetections(result) {
        // Clear overlay
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        if (!result.detections || result.detections.length === 0) {
            return;
        }

        result.detections.forEach(detection => {
            const color = this.classColors[detection.class_id];

            // Draw bounding box
            this.overlayCtx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.9)`;
            this.overlayCtx.lineWidth = 3;
            this.overlayCtx.strokeRect(
                detection.box[0],
                detection.box[1],
                detection.box[2] - detection.box[0],
                detection.box[3] - detection.box[1]
            );

            // Draw mask if available
            if (detection.mask) {
                this.drawMask(detection.mask, color, detection.box);
            }

            // Draw label
            const label = `${color.name} ${(detection.confidence * 100).toFixed(1)}%`;
            this.drawLabel(label, detection.box[0], detection.box[1], color);
        });

        // Update FPS
        const fps = (1000 / this.frameInterval).toFixed(1);
        document.getElementById('realtimeFPS').textContent = `${fps} FPS`;
        document.getElementById('realtimeDetections').textContent = `${result.detections.length} detection(s)`;
    }

    drawMask(maskData, color, box) {
        // maskData is a base64 encoded mask or array of polygon points
        // Draw semi-transparent colored overlay on detected region
        this.overlayCtx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`;

        if (Array.isArray(maskData)) {
            // Polygon points
            this.overlayCtx.beginPath();
            for (let i = 0; i < maskData.length; i += 2) {
                if (i === 0) {
                    this.overlayCtx.moveTo(maskData[i], maskData[i + 1]);
                } else {
                    this.overlayCtx.lineTo(maskData[i], maskData[i + 1]);
                }
            }
            this.overlayCtx.closePath();
            this.overlayCtx.fill();
        } else {
            // Fallback: fill box area
            this.overlayCtx.fillRect(
                box[0], box[1],
                box[2] - box[0], box[3] - box[1]
            );
        }
    }

    drawLabel(text, x, y, color) {
        const padding = 5;
        const fontSize = 16;
        this.overlayCtx.font = `bold ${fontSize}px Inter, sans-serif`;

        const textWidth = this.overlayCtx.measureText(text).width;
        const boxHeight = fontSize + padding * 2;

        // Background
        this.overlayCtx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.9)`;
        this.overlayCtx.fillRect(x, y - boxHeight, textWidth + padding * 2, boxHeight);

        // Text
        this.overlayCtx.fillStyle = '#ffffff';
        this.overlayCtx.fillText(text, x + padding, y - padding);
    }

    updateResultsUI(result) {
        const container = document.getElementById('realtimeResultsContainer');

        if (!result.detections || result.detections.length === 0) {
            container.innerHTML = `
                <div class="no-detection">
                    <i class="fas fa-check-circle"></i>
                    <p>Normal Skin or Background</p>
                </div>
            `;
            return;
        }

        container.innerHTML = result.detections.map(detection => {
            const color = this.classColors[detection.class_id];
            return `
                <div class="detection-card" style="border-left: 4px solid rgb(${color.r}, ${color.g}, ${color.b})">
                    <div class="detection-severity" style="background: rgba(${color.r}, ${color.g}, ${color.b}, 0.1)">
                        ${color.name}
                    </div>
                    <div class="detection-confidence">
                        Confidence: ${(detection.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `;
        }).join('');
    }

    togglePause() {
        const btn = document.getElementById('pauseRealtimeBtn');

        if (this.detectionActive) {
            this.detectionActive = false;
            btn.innerHTML = '<i class="fas fa-play me-2"></i>Resume Detection';
            btn.classList.remove('btn-warning');
            btn.classList.add('btn-success');
        } else {
            this.detectionActive = true;
            this.processFrame();
            btn.innerHTML = '<i class="fas fa-pause me-2"></i>Pause Detection';
            btn.classList.remove('btn-success');
            btn.classList.add('btn-warning');
        }
    }

    async captureAndAnalyze() {
        // Pause detection
        this.detectionActive = false;

        // Capture current frame
        this.canvas.toBlob(async (blob) => {
            // Create file
            const file = new File([blob], 'realtime-capture.jpg', { type: 'image/jpeg' });

            // Use existing upload mechanism
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);

            const fileInput = document.getElementById('file');
            fileInput.files = dataTransfer.files;

            // Close realtime view
            this.stop();

            // Show preview
            if (window.showPreview) {
                window.showPreview(file, 'camera');
            }
        }, 'image/jpeg', 0.95);
    }

    stop() {
        this.detectionActive = false;

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        if (this.guidanceTextTimer) {
            clearTimeout(this.guidanceTextTimer);
        }

        if (this.guidanceShowInterval) {
            clearInterval(this.guidanceShowInterval);
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        const container = document.getElementById('realtimeSegmentationContainer');
        if (container) {
            container.remove();
        }
    }
}

// Global instance
window.realtimeSegmentation = null;

// Initialize function
window.startRealtimeSegmentation = async function() {
    if (window.realtimeSegmentation) {
        window.realtimeSegmentation.stop();
    }

    window.realtimeSegmentation = new RealtimeSegmentation();
    await window.realtimeSegmentation.init();
};
