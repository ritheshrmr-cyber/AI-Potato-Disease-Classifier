import os
# Force TensorFlow to only use CPU and disable heavy GPU logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
# Tell TensorFlow to optimize for ultra-low memory environments
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Enable production level Cross-Origin Resource Sharing permissions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "not_a_potato_leaf", "not_potato"]

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        # FIX: compile=False cuts memory usage in half, preventing Render OOM crashes
        MODEL = tf.keras.models.load_model("potato_model.keras", compile=False)
    return MODEL

@app.get("/ping")
async def ping():
    get_model()
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
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
    except Exception as e:
        # Safety catch to return a structured error instead of throwing a raw crash
        return {
            "class": "Error processing image",
            "confidence": 0.0,
            "error_details": str(e)
        }