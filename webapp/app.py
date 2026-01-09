from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, session, send_from_directory
import os
from werkzeug.utils import secure_filename
# from model_utils import get_model_instance  # Old CNN model
from yolo_utils import get_yolo_instance as get_model_instance  # New YOLO model
import uuid
import json
import threading
import time
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Store progress for each analysis session
progress_store = {}
progress_lock = threading.Lock()

def convert_to_serializable(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')  # Use webapp/static/uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# PWA Routes
# ============================================================================
@app.route('/manifest.json')
def serve_manifest():
    """Serve PWA manifest with correct MIME type"""
    return send_from_directory('static', 'manifest.json', mimetype='application/manifest+json')

@app.route('/service-worker.js')
def serve_service_worker():
    """Serve service worker with correct MIME type"""
    return send_from_directory('static', 'service-worker.js', mimetype='application/javascript')

# ============================================================================
# Main Routes
# ============================================================================
@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload - returns session ID for progress tracking"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Generate unique filename and session ID
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        session_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Get XAI toggle state
        xai_enabled = request.form.get('xai_enabled', 'false') == 'true'
        print(f"DEBUG: XAI Enabled: {xai_enabled}")

        try:
            print("DEBUG: Received upload request")
            print("DEBUG: Session ID:", session_id)
            file.save(filepath)
            print("DEBUG: File saved")

            # Initialize progress
            with progress_lock:
                progress_store[session_id] = convert_to_serializable({'step': 'Starting analysis...', 'percent': 0})

            # Start analysis in background thread
            def run_analysis():
                try:
                    def progress_callback(step, percent):
                        with progress_lock:
                            progress_store[session_id] = convert_to_serializable({'step': step, 'percent': percent})
                        print(f"Progress: {step} - {percent}%")

                    model = get_model_instance()
                    result, error = model.predict(
                        filepath,
                        with_gradcam=xai_enabled,
                        with_computational_flow=xai_enabled,
                        detailed_analysis=xai_enabled,
                        show_concrete_math=xai_enabled,
                        trace_image=xai_enabled,
                        progress_callback=progress_callback
                    )

                    # Check for validation errors
                    if error or (result and result.get('error') == 'INVALID_BURN_IMAGE'):
                        error_message = error if error else result.get('message', 'Unknown error')
                        with progress_lock:
                            progress_store[session_id] = convert_to_serializable({
                                'step': 'Validation failed',
                                'percent': 100,
                                'error': True,
                                'error_type': 'INVALID_BURN_IMAGE',
                                'error_message': error_message,
                                'validation_details': result if result else None
                            })
                        return

                    # Save Grad-CAM overlay
                    gradcam_overlay = result.get('gradcam_overlay')
                    overlay_filename = 'overlay_' + filename
                    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)

                    if gradcam_overlay is not None:
                        import cv2
                        cv2.imwrite(overlay_path, gradcam_overlay)

                    # Store result in session
                    template_result = convert_to_serializable({
                        'predicted_class': result['predicted_class'],
                        'class_id': result['class_id'],
                        'confidence': result['confidence'],
                        'all_probabilities': result['all_probabilities']
                    })

                    computational_data = {}
                    if 'computational_flow' in result:
                        computational_data['flow'] = convert_to_serializable(result['computational_flow'])
                    if 'detailed_analysis' in result:
                        computational_data['detailed'] = convert_to_serializable(result['detailed_analysis'])
                    if 'concrete_computations' in result:
                        computational_data['concrete'] = convert_to_serializable(result['concrete_computations'])
                    if 'image_trace' in result:
                        computational_data['trace'] = convert_to_serializable(result['image_trace'])

                    # Store in session-specific cache
                    with progress_lock:
                        progress_store[session_id] = convert_to_serializable({
                            'step': 'Complete',
                            'percent': 100,
                            'result': template_result,
                            'computational_data': computational_data,
                            'image_filename': filename,
                            'overlay_filename': overlay_filename,
                            'session_id': session_id  # Store session_id so frontend can build URL
                        })

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    with progress_lock:
                        progress_store[session_id] = convert_to_serializable({'step': f'Error: {str(e)}', 'percent': 100, 'error': True})

            # Start analysis thread
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()

            # Return session ID for progress tracking
            return jsonify({'session_id': session_id})

        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WEBP files.')
        return redirect(url_for('index'))

@app.route('/result/<session_id>')
def show_result(session_id):
    """Display results for a completed analysis"""
    with progress_lock:
        session_data = progress_store.get(session_id)

    if not session_data or 'result' not in session_data:
        flash('Results not found or analysis not complete')
        return redirect(url_for('index'))

    # Check if overlay file exists
    overlay_filename = session_data.get("overlay_filename")
    has_overlay = False
    if overlay_filename:
        overlay_full_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        has_overlay = os.path.exists(overlay_full_path)

    return render_template('result.html',
                         result=session_data['result'],
                         computational_data=session_data.get('computational_data', {}),
                         image_path=url_for('static', filename=f'uploads/{session_data["image_filename"]}'),
                         overlay_path=url_for('static', filename=f'uploads/{overlay_filename}') if overlay_filename else None,
                         has_overlay=has_overlay)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    # Generate unique filename
    filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # Make prediction
        model = get_model_instance()
        result, error = model.predict(filepath)

        # Clean up uploaded file
        os.remove(filepath)

        if error:
            return jsonify({'error': error}), 500

        return jsonify(result)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime_detect', methods=['POST'])
def realtime_detect():
    """Real-time detection endpoint for video frames"""
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    frame = request.files['frame']

    # Generate temporary filename
    filename = 'temp_' + str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        frame.save(filepath)

        # Make prediction with low confidence threshold for real-time
        model = get_model_instance()
        result, error = model.predict(
            filepath,
            with_gradcam=False,
            with_computational_flow=False,
            detailed_analysis=False
        )

        # Clean up temporary file
        os.remove(filepath)

        if error:
            return jsonify({'detections': []})

        # Format response for real-time display
        detections = []

        # Check if result has detection data
        if result and 'boxes' in result and result['boxes'] is not None:
            boxes = result['boxes']
            masks = result.get('masks', [])
            confidences = result.get('confidences', [])
            class_ids = result.get('class_ids', [])

            for i in range(len(boxes)):
                detection = {
                    'box': boxes[i],
                    'confidence': float(confidences[i]) if i < len(confidences) else 0.0,
                    'class_id': int(class_ids[i]) if i < len(class_ids) else 0,
                    'mask': masks[i] if i < len(masks) else None
                }
                detections.append(detection)

        return jsonify({
            'detections': detections,
            'timestamp': time.time()
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'detections': [], 'error': str(e)})

@app.route('/debug/files')
def debug_files():
    """Debug route to check uploaded files"""
    upload_dir = app.config['UPLOAD_FOLDER']
    files = []
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, f)
            files.append({
                'name': f,
                'exists': os.path.exists(file_path),
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            })
    return jsonify({'upload_folder': upload_dir, 'files': files})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.route('/progress/<session_id>')
def progress_stream(session_id):
    """Server-Sent Events endpoint for real-time progress updates"""
    def generate():
        last_progress = -1
        timeout = 300  # 5 minutes timeout
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                yield f"data: {json.dumps({'step': 'timeout', 'percent': 100})}\n\n"
                break

            with progress_lock:
                progress = progress_store.get(session_id, {})

            current_percent = progress.get('percent', 0)

            # Only send if progress changed or it's complete
            if current_percent != last_progress or current_percent >= 100:
                last_progress = current_percent
                # Make sure to convert to serializable before sending
                safe_progress = convert_to_serializable(progress)
                yield f"data: {json.dumps(safe_progress)}\n\n"

            # Check if complete
            if current_percent >= 100:
                # Send explicit redirect message with session_id
                stored_session_id = progress.get('session_id', session_id)  # Fallback to URL session_id
                redirect_msg = convert_to_serializable({
                    'step': 'redirect',
                    'percent': 100,
                    'session_id': stored_session_id
                })
                yield f"data: {json.dumps(redirect_msg)}\n\n"
                break

            time.sleep(0.5)  # Check every 500ms

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize model (this might take a moment on first run)
    print("Initializing model...")
    get_model_instance()
    print("Model ready!")

    app.run(debug=True, host='0.0.0.0', port=5000)
