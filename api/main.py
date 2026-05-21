from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy","not_a_potato_leaf","not_potato"]

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model("potato_model.keras")
    return MODEL

@app.get("/ping")
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    model = get_model()
    predictions = model.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }