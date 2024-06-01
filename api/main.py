from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1.keras")
CLASS_NAMES = ['Cerscospora', 'Healthy', 'Leaf rust', 'Miner', 'Phoma']

# Load the trained Autoencoder model
AUTOENCODER = tf.keras.models.load_model("../autoencoder_model/autoencoder.keras")

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def is_anomaly(img):
    # Preprocess the input image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Get the reconstructed image from the Autoencoder
    reconstructed_img = AUTOENCODER.predict(img_array)

    # Calculate the reconstruction error
    reconstruction_error = tf.reduce_mean(tf.square(img_array - reconstructed_img))

    # Set a threshold for anomaly detection
    threshold = 0.1  # Adjust this value based on your validation

    if reconstruction_error > threshold:
        return True  # Anomaly detected
    else:
        return False  # Normal input

def predict_image_class(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = MODEL.predict(img_array)
    max_prob = np.max(predictions[0])
    if max_prob >= 0.5:
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = round(100 * max_prob, 2)
        return predicted_class, confidence
    else:
        return "Anomaly", round(100 * max_prob, 2)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))

    # Check if the input image is an anomaly
    if is_anomaly(image):
        return {"class": "Anomaly", "confidence": 0.0}

    # If not an anomaly, proceed with classification
    predicted_class, confidence = predict_image_class(image)
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)