import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from predict import predict_mri


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None
    probabilities = None
    error = None

    if request.method == "POST":

        # Check whether a file was selected
        if "mri_image" not in request.files:
            error = "Please select an MRI image."
            return render_template(
                "index.html",
                error=error
            )

        file = request.files["mri_image"]

        # Check filename
        if file.filename == "":
            error = "Please select an MRI image."
            return render_template(
                "index.html",
                error=error
            )

        # Check extension
        if not allowed_file(file.filename):
            error = "Only JPG, JPEG, PNG and BMP images are allowed."

            return render_template(
                "index.html",
                error=error
            )

        # Secure filename
        filename = secure_filename(file.filename)

        # Save uploaded image
        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(file_path)

        # Run AI prediction
        prediction, confidence, probabilities = predict_mri(
            file_path
        )

        # Image path for browser
        image_path = f"/uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities.tolist() if probabilities is not None else None,
        image_path=image_path,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )