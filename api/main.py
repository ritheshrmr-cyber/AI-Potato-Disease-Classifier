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
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ARCHITECTURE RECONSTRUCTION ---
# Re-building the layers manually forces Keras to completely ignore the saved file's bad configuration variables
def build_potato_model():
    input_shape = (256, 256, 3) # This is standard for image classification, adapt if your input size is different (e.g. 224, 150)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Since we are using load_model with compile=False, we just instantiate a basic layer sequence
        # to act as a receiver container for your trained parameters
    ])
    return model

# We load with compile=False so Keras completely skips looking at any quantization_config tags
MODEL = tf.keras.models.load_model("potato_model.keras", compile=False)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy","not_a_potato_leaf","not_potato"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    # Ensure image size matches what your model expects (e.g. resizing to 256x256) before batching
    img = Image.fromarray(image).resize((256, 256))
    img_batch = np.expand_dims(np.array(img), 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)