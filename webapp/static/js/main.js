// Enhanced JavaScript for Professional Burn Classifier
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadArea = document.getElementById('uploadArea');
    const form = document.getElementById('uploadForm');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');

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
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            if (fileName) fileName.textContent = file.name;
            if (fileSize) fileSize.textContent = formatFileSize(file.size);

            // Show preview with animation
            imagePreview.classList.remove('d-none');
            setTimeout(() => {
                imagePreview.classList.add('fade-in');
            }, 100);
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
    if (form && uploadBtn) {
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
