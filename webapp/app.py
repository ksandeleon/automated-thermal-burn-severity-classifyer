from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from model_utils import get_model_instance
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

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

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Generate unique filename to avoid conflicts
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            print("DEBUG: Received upload request")
            print("DEBUG: File name:", file.filename)
            print("DEBUG: File path:", filepath)
            file.save(filepath)
            print("DEBUG: File saved")

            # Get model instance and make prediction (with gradcam)
            model = get_model_instance()
            print("DEBUG: Model instance obtained")
            result, error = model.predict(filepath, with_gradcam=True)
            print("DEBUG: Prediction made:", result)
            print("DEBUG: Prediction error:", error)

            if error:
                flash(f'Prediction error: {error}')
                os.remove(filepath)  # Clean up uploaded file
                return redirect(url_for('index'))

            # Save Grad-CAM overlay image
            gradcam_overlay = result.get('gradcam_overlay')

            overlay_filename = 'overlay_' + filename
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            print("DEBUG: Overlay path:", overlay_path)
            print("DEBUG: gradcam_overlay type:", type(gradcam_overlay))
            print("DEBUG: gradcam_overlay is None:", gradcam_overlay is None)

            if gradcam_overlay is not None:
                import cv2
                print("DEBUG: Attempting to save overlay...")
                success = cv2.imwrite(overlay_path, gradcam_overlay)
                print("DEBUG: cv2.imwrite success:", success)
                print("DEBUG: Overlay file exists after save:", os.path.exists(overlay_path))
            else:
                print("DEBUG: No gradcam_overlay to save")

            # Final file existence check
            print("DEBUG: Final check - Image exists:", os.path.exists(filepath))
            print("DEBUG: Final check - Overlay exists:", os.path.exists(overlay_path))
            print("DEBUG: Image path for template:", url_for('static', filename=f'uploads/{filename}'))
            print("DEBUG: Overlay path for template:", url_for('static', filename=f'uploads/{overlay_filename}'))

            return render_template('result.html',
                                 result=result,
                                 image_path=url_for('static', filename=f'uploads/{filename}'),
                                 overlay_path=url_for('static', filename=f'uploads/{overlay_filename}'))

        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WEBP files.')
        return redirect(url_for('index'))

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

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize model (this might take a moment on first run)
    print("Initializing model...")
    get_model_instance()
    print("Model ready!")

    app.run(debug=True, host='0.0.0.0', port=5000)
