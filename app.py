from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained CNN model
model = load_model('model.keras')

# Define class labels
class_labels = ['Disturbed', 'Merging', 'Round Smooth', 'In-between round smooth', 'Cigar round smooth', 'Barred Spiral', 'Unbarred tight spiral', 'Unbarred Loose spiral', 'Edge-on without bulge', 'Edge-on with bulge']

@app.route('/')
def frontpage():
    return render_template('frontpage.html')

@app.route('/classifier')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image
    img = image.load_img(file_path, target_size=(256, 256))  # Adjust target_size as per your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Debugging: Print the shape and values of the preprocessed image
    print("Image shape:", img_array.shape)
    print("Image values:", img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Debugging: Print the raw predictions and the predicted class
    print("Raw predictions:", predictions)
    print("Predicted class index:", predicted_class)
    print("Predicted class label:", class_labels[predicted_class])

    # Map the predicted class to the corresponding label
    predicted_label = class_labels[predicted_class]

    # Format raw predictions as percentages with class labels
    formatted_predictions = [
        f"{class_labels[i]}: {pred * 100:.2f}%" for i, pred in enumerate(predictions[0])
    ]

    return render_template('result.html', filename=file.filename, label=predicted_label, formatted_predictions=formatted_predictions)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)