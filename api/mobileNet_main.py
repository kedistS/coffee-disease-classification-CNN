import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

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

# Load the pre-trained MobileNet model for coffee leaf detection
mobilenet_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=True)

# Load your pre-trained coffee disease classification model
COFFEE_MODEL = tf.keras.models.load_model("../saved_models/coffee.keras")
CLASS_NAMES = ['Cerscospora', 'Healthy', 'Leaf rust', 'Miner', 'Phoma']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to 224x224
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def is_coffee_leaf(img):
    """
    Checks if the input image contains a coffee leaf.
    Returns True if a coffee leaf is detected, False otherwise.
    """
    img_array = preprocess_image(img)
    predictions = mobilenet_model.predict(img_array)
    predicted_class = tf.keras.applications.mobilenet.decode_predictions(predictions, top=1)[0][0][1]

    # Check if the predicted class is related to coffee leaves
    if 'coffee' in predicted_class.lower():
        return True
    else:
        return False

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    if is_coffee_leaf(image):
        img_batch = preprocess_image(image)
        predictions = COFFEE_MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    else:
        return {
            'error': "The input image does not appear to be a coffee leaf. Please provide a valid coffee leaf image."
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)