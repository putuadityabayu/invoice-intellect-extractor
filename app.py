
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from src.pipeline import process_invoice
from src.utils.image_utils import download_image_from_url, is_valid_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp_uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "pdf"}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"}), 200


@app.route("/extract", methods=["POST"])
def extract_invoice_data():
    # Check if image URL is provided
    if "image_url" in request.form:
        image_url = request.form["image_url"]
        
        # Download image from URL
        temp_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_url_image.jpg")
        success, message = download_image_from_url(image_url, temp_file_path)
        
        if not success:
            return jsonify({"error": message}), 400
        
        if not is_valid_image(temp_file_path):
            os.remove(temp_file_path)
            return jsonify({"error": "Invalid image file downloaded from URL"}), 400
            
        # Process the invoice
        result = process_invoice(temp_file_path)
        
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return jsonify(result), 200
        
    # Check if uploaded file is provided
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        if not is_valid_image(file_path):
            os.remove(file_path)
            return jsonify({"error": "Invalid image file uploaded"}), 400
        
        # Process the invoice
        result = process_invoice(file_path)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify(result), 200
    else:
        return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
