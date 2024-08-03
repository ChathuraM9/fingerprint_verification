from flask import Flask, request, render_template
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.keras')

# Load registered fingerprint features
with open('fingerprint_features.pickle', 'rb') as f:
    registered_features = joblib.load(f)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (160, 160))
    image = np.stack((image,) * 3, axis=-1).astype('float32') / 255.0
    image = image.reshape(1, 160, 160, 3)
    return image

def authenticate_fingerprint(image):
    features = model.predict(image).flatten()
    dist = np.linalg.norm(features - registered_features)
    return dist < 0.01

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['fingerprint']
        file.save('uploaded_fingerprint.png')

        # Preprocess and authenticate the fingerprint
        image = preprocess_image('uploaded_fingerprint.png')
        is_authenticated = authenticate_fingerprint(image)
        return render_template('result.html', result=is_authenticated)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
