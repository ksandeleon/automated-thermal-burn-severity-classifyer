// Enhanced JavaScript for Professional Burn Classifier with Camera Support
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadArea = document.getElementById('uploadArea');
    const form = document.getElementById('uploadForm');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');

    console.log('DEBUG: Main upload functionality initialized');
    console.log('DEBUG: Preview elements found:', {
        previewImg: !!previewImg,
        fileName: !!fileName,
        fileSize: !!fileSize,
        imagePreviewSection: !!document.getElementById('imagePreviewSection')
    });

    // Initialize camera manager
    window.cameraManager = new CameraManager();

    // Enhanced file input handling with drag & drop
    if (fileInput && uploadArea) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', handleDrop, false);

        // Handle file selection via click
        fileInput.addEventListener('change', function(e) {
            handleFiles(e.target.files);
        });

        // Click to select file
        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        uploadArea.classList.add('dragover');
    }

    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];

            // Validate file
            if (validateFile(file)) {
                fileInput.files = files; // Set the file input
                showPreview(file);
                showUploadAnimation();
            }
        }
    }

    // Add visual feedback when image is selected
    function showImageSelectedFeedback() {
        console.log('DEBUG: showImageSelectedFeedback called');

        // Create a success notification
        showNotification('Image selected successfully! Review below and click "Analyze" when ready.', 'success');

        // Add a subtle pulse animation to the preview section
        const previewSection = document.getElementById('imagePreviewSection');
        if (previewSection) {
            console.log('DEBUG: Adding pulse animation to preview section');
            previewSection.style.animation = 'none';
            setTimeout(() => {
                previewSection.style.animation = 'slideInPreview 0.6s ease-out forwards';
            }, 10);
        } else {
            console.error('DEBUG: Preview section not found for animation');
        }
    }

    // Enhanced file validation with user feedback
    function validateFileWithFeedback(file) {
        // File size validation (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showAlert('File is too large. Maximum size is 16MB. Please choose a smaller image.', 'error');
            return false;
        }

        // File type validation
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid file type. Please select PNG, JPG, JPEG, GIF, or WEBP files only.', 'error');
            return false;
        }

        // Additional validation for image dimensions
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = function() {
                // Check minimum dimensions
                if (this.width < 50 || this.height < 50) {
                    showAlert('Image is too small. Please select an image at least 50x50 pixels.', 'error');
                    resolve(false);
                    return;
                }

                // Check maximum dimensions
                if (this.width > 4000 || this.height > 4000) {
                    showAlert('Image is too large. Please select an image smaller than 4000x4000 pixels.', 'error');
                    resolve(false);
                    return;
                }

                resolve(true);
            };
            img.onerror = function() {
                showAlert('Invalid image file. Please select a valid image.', 'error');
                resolve(false);
            };

            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    function validateFile(file) {
        // File size validation (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showAlert('File is too large. Maximum size is 16MB.', 'error');
            return false;
        }

        // File type validation
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid file type. Please select PNG, JPG, JPEG, GIF, or WEBP files.', 'error');
            return false;
        }

        return true;
    }

    function showPreview(file) {
        console.log('DEBUG: showPreview called with file:', file);

        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                console.log('DEBUG: Image loaded, dimensions:', this.width, 'x', this.height);

                // Update preview image
                const previewImg = document.getElementById('previewImg');
                if (previewImg) {
                    previewImg.src = e.target.result;
                    console.log('DEBUG: Preview image updated');
                } else {
                    console.error('DEBUG: previewImg element not found');
                }

                // Update metadata
                const fileName = document.getElementById('fileName');
                const fileSize = document.getElementById('fileSize');
                const imageDimensions = document.getElementById('imageDimensions');

                if (fileName) {
                    fileName.textContent = file.name;
                    console.log('DEBUG: File name updated:', file.name);
                }
                if (fileSize) {
                    fileSize.textContent = formatFileSize(file.size);
                    console.log('DEBUG: File size updated:', formatFileSize(file.size));
                }
                if (imageDimensions) {
                    imageDimensions.textContent = `${this.width} x ${this.height}`;
                    console.log('DEBUG: Dimensions updated:', `${this.width} x ${this.height}`);
                }

                // Handle UI element visibility
                const cameraSection = document.getElementById('cameraSection');
                const uploadBtnContainer = document.getElementById('uploadBtnContainer');
                const previewUploadContainer = document.getElementById('previewUploadContainer');
                const imagePreviewSection = document.getElementById('imagePreviewSection');

                console.log('DEBUG: UI elements found:', {
                    cameraSection: !!cameraSection,
                    uploadBtnContainer: !!uploadBtnContainer,
                    previewUploadContainer: !!previewUploadContainer,
                    imagePreviewSection: !!imagePreviewSection
                });

                // Hide camera section
                if (cameraSection) {
                    cameraSection.style.display = 'none';
                    console.log('DEBUG: Camera section hidden');
                }

                // Hide main upload button
                if (uploadBtnContainer) {
                    uploadBtnContainer.classList.add('d-none');
                    console.log('DEBUG: Upload button container hidden');
                }

                // Show preview upload button
                if (previewUploadContainer) {
                    previewUploadContainer.classList.remove('d-none');
                    console.log('DEBUG: Preview upload container shown');
                }

                // Show preview section with animation
                if (imagePreviewSection) {
                    console.log('DEBUG: Showing preview section');
                    imagePreviewSection.classList.remove('d-none');

                    // Force reflow to ensure the element is visible before animation
                    imagePreviewSection.offsetHeight;

                    setTimeout(() => {
                        imagePreviewSection.classList.add('fade-in');
                        console.log('DEBUG: Fade-in animation applied');

                        // Show success feedback
                        showImageSelectedFeedback();
                    }, 100);
                } else {
                    console.error('DEBUG: imagePreviewSection element not found');
                }
            };

            img.onerror = function() {
                console.error('DEBUG: Error loading image');
                showAlert('Error loading image. Please try a different file.', 'error');
            };

            img.src = e.target.result;
        };

        reader.onerror = function() {
            console.error('DEBUG: Error reading file');
            showAlert('Error reading file. Please try again.', 'error');
        };

        reader.readAsDataURL(file);
    }

    function showUploadAnimation() {
        // Add subtle animation to upload area
        uploadArea.style.transform = 'scale(0.98)';
        setTimeout(() => {
            uploadArea.style.transform = 'scale(1)';
        }, 150);
    }

    // Enhanced form submission with better UX
    if (form) {
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadBtnPreview = document.getElementById('uploadBtnPreview');

        // Handle main upload button
        if (uploadBtn) {
            uploadBtn.addEventListener('click', function(e) {
                const file = fileInput.files[0];
                if (!file) {
                    e.preventDefault();
                    showAlert('Please select a file first.', 'error');
                    return;
                }
                handleFormSubmission(e, uploadBtn);
            });
        }

        // Handle preview upload button
        if (uploadBtnPreview) {
            uploadBtnPreview.addEventListener('click', function(e) {
                const file = fileInput.files[0];
                if (!file) {
                    e.preventDefault();
                    showAlert('Please select a file first.', 'error');
                    return;
                }
                handleFormSubmission(e, uploadBtnPreview);
            });
        }

        // Handle form submission
        form.addEventListener('submit', function(e) {
            const file = fileInput.files[0];
            if (!file) {
                e.preventDefault();
                showAlert('Please select a file first.', 'error');
                return;
            }

            // Show enhanced loading state
            showLoadingState();

            // Add form loading class
            form.classList.add('loading');
        });
    }

    function handleFormSubmission(e, button) {
        // Show enhanced loading state for the specific button
        showLoadingStateForButton(button);

        // Add form loading class
        form.classList.add('loading');
    }

    function showLoadingStateForButton(button) {
        if (button) {
            button.classList.add('loading');
            const btnText = button.querySelector('.btn-text');
            const btnLoading = button.querySelector('.btn-loading');

            if (btnText) btnText.style.opacity = '0';
            if (btnLoading) btnLoading.classList.remove('d-none');

            button.disabled = true;
        }
    }

    function showLoadingState() {
        if (uploadBtn) {
            uploadBtn.classList.add('loading');
            const btnText = uploadBtn.querySelector('.btn-text');
            const btnLoading = uploadBtn.querySelector('.btn-loading');

            if (btnText) btnText.style.opacity = '0';
            if (btnLoading) btnLoading.classList.remove('d-none');

            uploadBtn.disabled = true;
        }
    }

    // Enhanced alert system
    function showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert-custom');
        existingAlerts.forEach(alert => alert.remove());

        // Create new alert
        const alert = document.createElement('div');
        alert.className = `alert-custom alert-${type === 'error' ? 'danger' : 'success'} mb-4`;
        alert.innerHTML = `
            <i class="fas ${type === 'error' ? 'fa-exclamation-triangle' : 'fa-check-circle'} me-2"></i>
            ${message}
        `;

        // Insert alert
        const cardBody = document.querySelector('.card-body-custom');
        if (cardBody) {
            cardBody.insertBefore(alert, cardBody.firstChild);
        }

        // Auto-hide after 5 seconds
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.remove();
            }, 300);
        }, 5000);
    }

    // Auto-hide existing alerts
    const alerts = document.querySelectorAll('.alert-custom');
    alerts.forEach(function(alert) {
        if (alert.classList.contains('alert-danger')) {
            setTimeout(function() {
                alert.style.opacity = '0';
                setTimeout(function() {
                    alert.remove();
                }, 300);
            }, 5000);
        }
    });

    // Enhanced hover effects for severity cards
    const severityCards = document.querySelectorAll('.severity-card');
    severityCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Smooth scrolling for any anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Progressive enhancement for better performance
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (prefersReducedMotion.matches) {
        // Disable animations for users who prefer reduced motion
        document.documentElement.style.setProperty('--animation-duration', '0s');
    }

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        // Space or Enter on upload area should trigger file selection
        if ((e.key === ' ' || e.key === 'Enter') && e.target === uploadArea) {
            e.preventDefault();
            fileInput.click();
        }
    });

    // Add accessibility attributes
    if (uploadArea) {
        uploadArea.setAttribute('tabindex', '0');
        uploadArea.setAttribute('role', 'button');
        uploadArea.setAttribute('aria-label', 'Click or drag to upload image file');
    }

    // Enhanced preview functionality
    const closePreviewBtn = document.getElementById('closePreviewBtn');
    const retryBtn = document.getElementById('retryBtn');

    // Close preview handler
    if (closePreviewBtn) {
        closePreviewBtn.addEventListener('click', function() {
            hidePreview();
        });
    }

    // Enhanced retry functionality
    if (retryBtn) {
        retryBtn.addEventListener('click', function() {
            hidePreview();
            // Reset file input
            if (fileInput) {
                fileInput.value = '';
            }
            // Reset other file inputs
            const galleryInput = document.getElementById('galleryInput');
            const cameraInput = document.getElementById('cameraInput');
            if (galleryInput) galleryInput.value = '';
            if (cameraInput) cameraInput.value = '';

            // Show camera section again
            const cameraSection = document.getElementById('cameraSection');
            if (cameraSection) {
                cameraSection.style.display = 'block';
            }
        });
    }

    // Function to hide preview and show camera section
    function hidePreview() {
        const imagePreview = document.getElementById('imagePreviewSection');
        const cameraSection = document.getElementById('cameraSection');
        const uploadBtnContainer = document.getElementById('uploadBtnContainer');
        const previewUploadContainer = document.getElementById('previewUploadContainer');

        console.log('DEBUG: Hiding preview section');

        if (imagePreview) {
            imagePreview.classList.add('d-none');
            imagePreview.classList.remove('fade-in');
            console.log('DEBUG: Preview section hidden');
        }

        if (cameraSection) {
            cameraSection.style.display = 'block';
            console.log('DEBUG: Camera section shown');
        }

        if (uploadBtnContainer) {
            uploadBtnContainer.classList.remove('d-none');
            console.log('DEBUG: Upload button container shown');
        }

        if (previewUploadContainer) {
            previewUploadContainer.classList.add('d-none');
            console.log('DEBUG: Preview upload container hidden');
        }

        // Clear preview data
        const previewImg = document.getElementById('previewImg');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const imageDimensions = document.getElementById('imageDimensions');

        if (previewImg) previewImg.src = '';
        if (fileName) fileName.textContent = 'No file selected';
        if (fileSize) fileSize.textContent = '0 MB';
        if (imageDimensions) imageDimensions.textContent = '-- x --';

        console.log('DEBUG: Preview data cleared');
    }

    // Debug: Test preview functionality
    console.log('DEBUG: Preview elements check');
    console.log('previewImg:', document.getElementById('previewImg'));
    console.log('fileName:', document.getElementById('fileName'));
    console.log('fileSize:', document.getElementById('fileSize'));
    console.log('imagePreviewSection:', document.getElementById('imagePreviewSection'));

    // Add debugging to showPreview function
    window.showPreviewDebug = function(file) {
        console.log('DEBUG: showPreview called with file:', file);
        showPreview(file);
    };

    // Test function to manually trigger preview (for debugging)
    window.testPreview = function() {
        console.log('Testing preview functionality...');
        const testFile = new File(['test'], 'test-image.jpg', { type: 'image/jpeg' });

        // Create a simple test image data URL
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 200;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(0, 0, 300, 200);
        ctx.fillStyle = 'white';
        ctx.font = '20px Arial';
        ctx.fillText('TEST IMAGE', 80, 110);

        const testDataUrl = canvas.toDataURL('image/jpeg');

        // Update preview elements directly
        const previewImg = document.getElementById('previewImg');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const imageDimensions = document.getElementById('imageDimensions');
        const imagePreviewSection = document.getElementById('imagePreviewSection');
        const cameraSection = document.getElementById('cameraSection');
        const uploadBtnContainer = document.getElementById('uploadBtnContainer');
        const previewUploadContainer = document.getElementById('previewUploadContainer');

        if (previewImg) previewImg.src = testDataUrl;
        if (fileName) fileName.textContent = 'test-image.jpg';
        if (fileSize) fileSize.textContent = '50 KB';
        if (imageDimensions) imageDimensions.textContent = '300 x 200';

        if (cameraSection) cameraSection.style.display = 'none';
        if (uploadBtnContainer) uploadBtnContainer.classList.add('d-none');
        if (previewUploadContainer) previewUploadContainer.classList.remove('d-none');

        if (imagePreviewSection) {
            imagePreviewSection.classList.remove('d-none');
            setTimeout(() => {
                imagePreviewSection.classList.add('fade-in');
            }, 100);
        }

        console.log('Test preview should now be visible');
    };

    // Enhanced clipboard functionality
    function copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(function() {
                showNotification('Copied to clipboard!', 'success');
            }).catch(function(err) {
                console.error('Failed to copy: ', err);
                fallbackCopyToClipboard(text);
            });
        } else {
            fallbackCopyToClipboard(text);
        }
    }

    function fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.top = '0';
        textArea.style.left = '0';
        textArea.style.position = 'fixed';

        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!', 'success');
        } catch (err) {
            console.error('Fallback: Unable to copy', err);
            showNotification('Unable to copy to clipboard', 'error');
        }

        document.body.removeChild(textArea);
    }

    // Enhanced notification system
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `position-fixed top-0 end-0 m-3 p-3 rounded-3 shadow-lg`;
        notification.style.zIndex = '9999';
        notification.style.minWidth = '300px';
        notification.style.transform = 'translateX(100%)';
        notification.style.transition = 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)';

        // Style based on type
        if (type === 'success') {
            notification.style.background = 'linear-gradient(135deg, var(--accent-1), var(--accent-2))';
            notification.style.color = 'white';
            notification.innerHTML = `<i class="fas fa-check-circle me-2"></i>${message}`;
        } else if (type === 'error') {
            notification.style.background = 'linear-gradient(135deg, #dc3545, #e83e8c)';
            notification.style.color = 'white';
            notification.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i>${message}`;
        } else {
            notification.style.background = 'linear-gradient(135deg, var(--primary-bg), var(--secondary-bg))';
            notification.style.color = 'var(--text-dark)';
            notification.innerHTML = `<i class="fas fa-info-circle me-2"></i>${message}`;
        }

        document.body.appendChild(notification);

        // Trigger animation
        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
        });

        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // Performance optimization: Lazy load heavy animations
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements that should animate when in view
    document.addEventListener('DOMContentLoaded', function() {
        const animateElements = document.querySelectorAll('.severity-card, .disclaimer-card');
        animateElements.forEach(el => observer.observe(el));
    });

    // Enhanced Camera and Upload Functionality
    class CameraManager {
        constructor() {
            this.stream = null;
            this.currentCamera = 'user'; // 'user' for front camera, 'environment' for back camera
            this.isInitialized = false;
            this.initializeElements();
            this.bindEvents();
        }

        initializeElements() {
            // Camera elements
            this.cameraChoiceContainer = document.getElementById('cameraChoiceContainer');
            this.cameraInterface = document.getElementById('cameraInterface');
            this.cameraError = document.getElementById('cameraError');
            this.cameraVideo = document.getElementById('cameraVideo');
            this.cameraCanvas = document.getElementById('cameraCanvas');

            // Choice buttons
            this.galleryChoice = document.getElementById('galleryChoice');
            this.cameraChoice = document.getElementById('cameraChoice');

            // Camera controls
            this.captureBtn = document.getElementById('captureBtn');
            this.cancelCameraBtn = document.getElementById('cancelCameraBtn');
            this.switchCameraBtn = document.getElementById('switchCameraBtn');
            this.tryAgainBtn = document.getElementById('tryAgainBtn');

            // File inputs
            this.galleryInput = document.getElementById('galleryInput');
            this.cameraInput = document.getElementById('cameraInput');
            this.mainFileInput = document.getElementById('file');

            // Error elements
            this.cameraErrorMessage = document.getElementById('cameraErrorMessage');
        }

        bindEvents() {
            // Choice buttons
            if (this.galleryChoice) {
                this.galleryChoice.addEventListener('click', () => this.openGallery());
            }

            if (this.cameraChoice) {
                this.cameraChoice.addEventListener('click', () => this.openCamera());
            }

            // Camera controls
            if (this.captureBtn) {
                this.captureBtn.addEventListener('click', () => this.capturePhoto());
            }

            if (this.cancelCameraBtn) {
                this.cancelCameraBtn.addEventListener('click', () => this.closeCamera());
            }

            if (this.switchCameraBtn) {
                this.switchCameraBtn.addEventListener('click', () => this.switchCamera());
            }

            if (this.tryAgainBtn) {
                this.tryAgainBtn.addEventListener('click', () => this.retryCamera());
            }

            // File input handlers
            if (this.galleryInput) {
                this.galleryInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
            }

            if (this.cameraInput) {
                this.cameraInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
            }
        }

        openGallery() {
            console.log('Opening gallery...');
            if (this.galleryInput) {
                this.galleryInput.click();
            }
        }

        async openCamera() {
            console.log('Opening camera...');
            this.hideError();

            // Check if camera is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                this.showError('Camera is not supported on this device or browser.');
                return;
            }

            try {
                this.showCameraLoading();

                const constraints = {
                    video: {
                        facingMode: this.currentCamera,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                };

                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
                this.cameraVideo.srcObject = this.stream;

                await new Promise((resolve) => {
                    this.cameraVideo.onloadedmetadata = resolve;
                });

                this.showCameraInterface();
                console.log('Camera opened successfully');

            } catch (error) {
                console.error('Error opening camera:', error);
                this.handleCameraError(error);
            }
        }

        showCameraLoading() {
            this.cameraChoiceContainer.classList.add('d-none');
            this.cameraInterface.classList.remove('d-none');
            this.cameraInterface.innerHTML = `
                <div class="camera-loading">
                    <i class="fas fa-spinner fa-spin me-2"></i>
                    <span>Opening camera...</span>
                </div>
            `;
        }

        showCameraInterface() {
            this.cameraInterface.innerHTML = `
                <div class="camera-container">
                    <video id="cameraVideo" autoplay muted playsinline></video>
                    <canvas id="cameraCanvas" style="display: none;"></canvas>
                    <div class="camera-overlay">
                        <div class="camera-grid">
                            <div class="grid-line"></div>
                            <div class="grid-line"></div>
                            <div class="grid-line"></div>
                            <div class="grid-line"></div>
                        </div>
                        <div class="camera-controls">
                            <button type="button" class="camera-btn cancel-btn" id="cancelCameraBtn">
                                <i class="fas fa-times"></i>
                            </button>
                            <button type="button" class="camera-btn capture-btn" id="captureBtn">
                                <i class="fas fa-camera"></i>
                            </button>
                            <button type="button" class="camera-btn switch-btn" id="switchCameraBtn">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;

            // Re-initialize elements and events
            this.cameraVideo = document.getElementById('cameraVideo');
            this.cameraCanvas = document.getElementById('cameraCanvas');
            this.captureBtn = document.getElementById('captureBtn');
            this.cancelCameraBtn = document.getElementById('cancelCameraBtn');
            this.switchCameraBtn = document.getElementById('switchCameraBtn');

            // Re-bind events
            this.captureBtn.addEventListener('click', () => this.capturePhoto());
            this.cancelCameraBtn.addEventListener('click', () => this.closeCamera());
            this.switchCameraBtn.addEventListener('click', () => this.switchCamera());

            // Set video source
            if (this.stream) {
                this.cameraVideo.srcObject = this.stream;
            }

            this.cameraInterface.classList.add('fade-in');
        }

        async switchCamera() {
            this.currentCamera = this.currentCamera === 'user' ? 'environment' : 'user';

            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }

            await this.openCamera();
        }

        capturePhoto() {
            if (!this.cameraVideo || !this.stream) {
                this.showError('Camera not available for capture.');
                return;
            }

            try {
                // Create canvas if it doesn't exist
                if (!this.cameraCanvas) {
                    this.cameraCanvas = document.createElement('canvas');
                }

                const canvas = this.cameraCanvas;
                const video = this.cameraVideo;

                // Set canvas dimensions to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Draw video frame to canvas
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Convert canvas to blob
                canvas.toBlob((blob) => {
                    if (blob) {
                        // Create a file from the blob
                        const file = new File([blob], `camera-capture-${Date.now()}.jpg`, {
                            type: 'image/jpeg',
                            lastModified: Date.now()
                        });

                        // Handle the captured file
                        this.handleFileSelection([file]);
                        this.closeCamera();
                    }
                }, 'image/jpeg', 0.9);

            } catch (error) {
                console.error('Error capturing photo:', error);
                this.showError('Failed to capture photo. Please try again.');
            }
        }

        closeCamera() {
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }

            this.cameraInterface.classList.add('d-none');
            this.cameraChoiceContainer.classList.remove('d-none');
            this.hideError();
        }

        retryCamera() {
            this.hideError();
            this.openCamera();
        }

        handleCameraError(error) {
            let errorMessage = 'Unable to access camera. ';

            switch (error.name) {
                case 'NotFoundError':
                    errorMessage += 'No camera found on this device.';
                    break;
                case 'NotAllowedError':
                    errorMessage += 'Camera access was denied. Please allow camera permissions and try again.';
                    break;
                case 'NotSupportedError':
                    errorMessage += 'Camera is not supported on this device.';
                    break;
                case 'OverconstrainedError':
                    errorMessage += 'Camera constraints could not be satisfied.';
                    break;
                default:
                    errorMessage += 'Please check your camera settings and try again.';
            }

            this.showError(errorMessage);
        }

        showError(message) {
            this.cameraInterface.classList.add('d-none');
            this.cameraChoiceContainer.classList.add('d-none');

            if (this.cameraErrorMessage) {
                this.cameraErrorMessage.textContent = message;
            }

            this.cameraError.classList.remove('d-none');
            this.cameraError.classList.add('fade-in');
        }

        hideError() {
            this.cameraError.classList.add('d-none');
            this.cameraError.classList.remove('fade-in');
            this.cameraChoiceContainer.classList.remove('d-none');
        }    handleFileSelection(files) {
            if (files && files.length > 0) {
                const file = files[0];

                // Validate file
                if (this.validateFile(file)) {
                    // Set the main file input
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    this.mainFileInput.files = dt.files;

                    // Use the existing showPreview function
                    showPreview(file);

                    console.log('File selected:', file.name, 'Size:', file.size);
                }
            }
        }

        validateFile(file) {
            // File size validation (16MB limit)
            if (file.size > 16 * 1024 * 1024) {
                this.showAlert('File is too large. Maximum size is 16MB.', 'error');
                return false;
            }

            // File type validation
            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
            if (!allowedTypes.includes(file.type)) {
                this.showAlert('Invalid file type. Please select PNG, JPG, JPEG, GIF, or WEBP files.', 'error');
                return false;
            }

            return true;
        }

        showAlert(message, type = 'info') {
            // Create or update alert
            let alertElement = document.querySelector('.camera-alert');
            if (!alertElement) {
                alertElement = document.createElement('div');
                alertElement.className = 'alert camera-alert';
                const cameraSection = document.getElementById('cameraSection');
                if (cameraSection) {
                    cameraSection.insertBefore(alertElement, cameraSection.firstChild);
                }
            }

            alertElement.className = `alert camera-alert alert-${type === 'error' ? 'danger' : 'info'}`;
            alertElement.innerHTML = `
                <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
                ${message}
            `;

            // Auto-hide after 5 seconds
            setTimeout(() => {
                if (alertElement && alertElement.parentNode) {
                    alertElement.remove();
                }
            }, 5000);
        }
    }

    // Initialize camera manager when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize camera manager
        window.cameraManager = new CameraManager();

        // ...existing code...
    });
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Enhanced clipboard functionality
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function() {
            showNotification('Copied to clipboard!', 'success');
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.top = '0';
    textArea.style.left = '0';
    textArea.style.position = 'fixed';

    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        document.execCommand('copy');
        showNotification('Copied to clipboard!', 'success');
    } catch (err) {
        console.error('Fallback: Unable to copy', err);
        showNotification('Unable to copy to clipboard', 'error');
    }

    document.body.removeChild(textArea);
}

// Enhanced notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `position-fixed top-0 end-0 m-3 p-3 rounded-3 shadow-lg`;
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    notification.style.transform = 'translateX(100%)';
    notification.style.transition = 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)';

    // Style based on type
    if (type === 'success') {
        notification.style.background = 'linear-gradient(135deg, var(--accent-1), var(--accent-2))';
        notification.style.color = 'white';
        notification.innerHTML = `<i class="fas fa-check-circle me-2"></i>${message}`;
    } else if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, #dc3545, #e83e8c)';
        notification.style.color = 'white';
        notification.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i>${message}`;
    } else {
        notification.style.background = 'linear-gradient(135deg, var(--primary-bg), var(--secondary-bg))';
        notification.style.color = 'var(--text-dark)';
        notification.innerHTML = `<i class="fas fa-info-circle me-2"></i>${message}`;
    }

    document.body.appendChild(notification);

    // Trigger animation
    requestAnimationFrame(() => {
        notification.style.transform = 'translateX(0)';
    });

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Performance optimization: Lazy load heavy animations
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
        }
    });
}, observerOptions);

// Observe elements that should animate when in view
document.addEventListener('DOMContentLoaded', function() {
    const animateElements = document.querySelectorAll('.severity-card, .disclaimer-card');
    animateElements.forEach(el => observer.observe(el));
});

// Enhanced Camera and Upload Functionality
class CameraManager {
    constructor() {
        this.stream = null;
        this.currentCamera = 'user'; // 'user' for front camera, 'environment' for back camera
        this.isInitialized = false;
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // Camera elements
        this.cameraChoiceContainer = document.getElementById('cameraChoiceContainer');
        this.cameraInterface = document.getElementById('cameraInterface');
        this.cameraError = document.getElementById('cameraError');
        this.cameraVideo = document.getElementById('cameraVideo');
        this.cameraCanvas = document.getElementById('cameraCanvas');

        // Choice buttons
        this.galleryChoice = document.getElementById('galleryChoice');
        this.cameraChoice = document.getElementById('cameraChoice');

        // Camera controls
        this.captureBtn = document.getElementById('captureBtn');
        this.cancelCameraBtn = document.getElementById('cancelCameraBtn');
        this.switchCameraBtn = document.getElementById('switchCameraBtn');
        this.tryAgainBtn = document.getElementById('tryAgainBtn');

        // File inputs
        this.galleryInput = document.getElementById('galleryInput');
        this.cameraInput = document.getElementById('cameraInput');
        this.mainFileInput = document.getElementById('file');

        // Error elements
        this.cameraErrorMessage = document.getElementById('cameraErrorMessage');
    }

    bindEvents() {
        // Choice buttons
        if (this.galleryChoice) {
            this.galleryChoice.addEventListener('click', () => this.openGallery());
        }

        if (this.cameraChoice) {
            this.cameraChoice.addEventListener('click', () => this.openCamera());
        }

        // Camera controls
        if (this.captureBtn) {
            this.captureBtn.addEventListener('click', () => this.capturePhoto());
        }

        if (this.cancelCameraBtn) {
            this.cancelCameraBtn.addEventListener('click', () => this.closeCamera());
        }

        if (this.switchCameraBtn) {
            this.switchCameraBtn.addEventListener('click', () => this.switchCamera());
        }

        if (this.tryAgainBtn) {
            this.tryAgainBtn.addEventListener('click', () => this.retryCamera());
        }

        // File input handlers
        if (this.galleryInput) {
            this.galleryInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
        }

        if (this.cameraInput) {
            this.cameraInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
        }
    }

    openGallery() {
        console.log('Opening gallery...');
        if (this.galleryInput) {
            this.galleryInput.click();
        }
    }

    async openCamera() {
        console.log('Opening camera...');
        this.hideError();

        // Check if camera is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Camera is not supported on this device or browser.');
            return;
        }

        try {
            this.showCameraLoading();

            const constraints = {
                video: {
                    facingMode: this.currentCamera,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.cameraVideo.srcObject = this.stream;

            await new Promise((resolve) => {
                this.cameraVideo.onloadedmetadata = resolve;
            });

            this.showCameraInterface();
            console.log('Camera opened successfully');

        } catch (error) {
            console.error('Error opening camera:', error);
            this.handleCameraError(error);
        }
    }

    showCameraLoading() {
        this.cameraChoiceContainer.classList.add('d-none');
        this.cameraInterface.classList.remove('d-none');
        this.cameraInterface.innerHTML = `
            <div class="camera-loading">
                <i class="fas fa-spinner fa-spin me-2"></i>
                <span>Opening camera...</span>
            </div>
        `;
    }

    showCameraInterface() {
        this.cameraInterface.innerHTML = `
            <div class="camera-container">
                <video id="cameraVideo" autoplay muted playsinline></video>
                <canvas id="cameraCanvas" style="display: none;"></canvas>
                <div class="camera-overlay">
                    <div class="camera-grid">
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                    </div>
                    <div class="camera-controls">
                        <button type="button" class="camera-btn cancel-btn" id="cancelCameraBtn">
                            <i class="fas fa-times"></i>
                        </button>
                        <button type="button" class="camera-btn capture-btn" id="captureBtn">
                            <i class="fas fa-camera"></i>
                        </button>
                        <button type="button" class="camera-btn switch-btn" id="switchCameraBtn">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Re-initialize elements and events
        this.cameraVideo = document.getElementById('cameraVideo');
        this.cameraCanvas = document.getElementById('cameraCanvas');
        this.captureBtn = document.getElementById('captureBtn');
        this.cancelCameraBtn = document.getElementById('cancelCameraBtn');
        this.switchCameraBtn = document.getElementById('switchCameraBtn');

        // Re-bind events
        this.captureBtn.addEventListener('click', () => this.capturePhoto());
        this.cancelCameraBtn.addEventListener('click', () => this.closeCamera());
        this.switchCameraBtn.addEventListener('click', () => this.switchCamera());

        // Set video source
        if (this.stream) {
            this.cameraVideo.srcObject = this.stream;
        }

        this.cameraInterface.classList.add('fade-in');
    }

    async switchCamera() {
        this.currentCamera = this.currentCamera === 'user' ? 'environment' : 'user';

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        await this.openCamera();
    }

    capturePhoto() {
        if (!this.cameraVideo || !this.stream) {
            this.showError('Camera not available for capture.');
            return;
        }

        try {
            // Create canvas if it doesn't exist
            if (!this.cameraCanvas) {
                this.cameraCanvas = document.createElement('canvas');
            }

            const canvas = this.cameraCanvas;
            const video = this.cameraVideo;

            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Draw video frame to canvas
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to blob
            canvas.toBlob((blob) => {
                if (blob) {
                    // Create a file from the blob
                    const file = new File([blob], `camera-capture-${Date.now()}.jpg`, {
                        type: 'image/jpeg',
                        lastModified: Date.now()
                    });

                    // Handle the captured file
                    this.handleFileSelection([file]);
                    this.closeCamera();
                }
            }, 'image/jpeg', 0.9);

        } catch (error) {
            console.error('Error capturing photo:', error);
            this.showError('Failed to capture photo. Please try again.');
        }
    }

    closeCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.cameraInterface.classList.add('d-none');
        this.cameraChoiceContainer.classList.remove('d-none');
        this.hideError();
    }

    retryCamera() {
        this.hideError();
        this.openCamera();
    }

    handleCameraError(error) {
        let errorMessage = 'Unable to access camera. ';

        switch (error.name) {
            case 'NotFoundError':
                errorMessage += 'No camera found on this device.';
                break;
            case 'NotAllowedError':
                errorMessage += 'Camera access was denied. Please allow camera permissions and try again.';
                break;
            case 'NotSupportedError':
                errorMessage += 'Camera is not supported on this device.';
                break;
            case 'OverconstrainedError':
                errorMessage += 'Camera constraints could not be satisfied.';
                break;
            default:
                errorMessage += 'Please check your camera settings and try again.';
        }

        this.showError(errorMessage);
    }

    showError(message) {
        this.cameraInterface.classList.add('d-none');
        this.cameraChoiceContainer.classList.add('d-none');

        if (this.cameraErrorMessage) {
            this.cameraErrorMessage.textContent = message;
        }

        this.cameraError.classList.remove('d-none');
        this.cameraError.classList.add('fade-in');
    }

    hideError() {
        this.cameraError.classList.add('d-none');
        this.cameraError.classList.remove('fade-in');
        this.cameraChoiceContainer.classList.remove('d-none');
    }    handleFileSelection(files) {
        if (files && files.length > 0) {
            const file = files[0];

            // Validate file
            if (this.validateFile(file)) {
                // Set the main file input
                const dt = new DataTransfer();
                dt.items.add(file);
                this.mainFileInput.files = dt.files;

                // Use the existing showPreview function
                showPreview(file);

                console.log('File selected:', file.name, 'Size:', file.size);
            }
        }
    }

    validateFile(file) {
        // File size validation (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            this.showAlert('File is too large. Maximum size is 16MB.', 'error');
            return false;
        }

        // File type validation
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            this.showAlert('Invalid file type. Please select PNG, JPG, JPEG, GIF, or WEBP files.', 'error');
            return false;
        }

        return true;
    }

    showAlert(message, type = 'info') {
        // Create or update alert
        let alertElement = document.querySelector('.camera-alert');
        if (!alertElement) {
            alertElement = document.createElement('div');
            alertElement.className = 'alert camera-alert';
            const cameraSection = document.getElementById('cameraSection');
            if (cameraSection) {
                cameraSection.insertBefore(alertElement, cameraSection.firstChild);
            }
        }

        alertElement.className = `alert camera-alert alert-${type === 'error' ? 'danger' : 'info'}`;
        alertElement.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
        `;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (alertElement && alertElement.parentNode) {
                alertElement.remove();
            }
        }, 5000);
    }
}

// Initialize camera manager when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize camera manager
    window.cameraManager = new CameraManager();

    // ...existing code...
});
