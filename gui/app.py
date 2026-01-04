
import os
import secrets
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import sys

# Import Core Engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daemon_nfr import NFREngine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'gui_uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit
app.secret_key = secrets.token_hex(16)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Engine
engine = NFREngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = filename + ".dmn"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    file.save(input_path)
    
    try:
        # Run compression (with finetuning by default for now)
        # Note: In a real app we'd use a background thread/celery
        print(f"Compressing {filename}...")
        engine.compress(input_path, output_path, finetune=True)
        
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        ratio = compressed_size / original_size if original_size > 0 else 0
        
        return jsonify({
            'status': 'success',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': f"{ratio:.2f}",
            'download_url': f"/download/{output_filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decompress', methods=['POST'])
def decompress():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    
    if not filename.endswith('.dmn'):
        return jsonify({'error': 'Invalid file type. Must be .dmn'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)
    
    # Check for sidecar model? For now assume model is embedded or we use default
    # The current engine needs the model. 
    # For this GUI PoC, we will assume the engine instance still has the weights 
    # IF we just compressed it. But if it's a new file, we might fail without the model file.
    # To fix this robustness, we'd need model upload. 
    # For now, we'll try to decompress directly.
    
    # Try to deduce original name
    output_filename = filename.replace('.dmn', '') 
    if output_filename == filename: output_filename += ".restored"
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    try:
        print(f"Decompressing {filename}...")
        
        # Check if sidecar model exists, if not, warn?
        # engine.decompress requires a loaded model.
        
        engine.decompress(input_path, output_path)
        
        return jsonify({
            'status': 'success',
            'download_url': f"/download/{output_filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    print("Starting NFR GUI on http://127.0.0.1:5000")
    # Open browser automatically
    from threading import Timer
    import webbrowser
    Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True, use_reloader=False)
