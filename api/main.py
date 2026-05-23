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

# --- PERFECT BYPASS: REBUILD MODEL CONTEXT MANUALLY ---
def load_clean_prediction_model(model_path):
    """
    Loads structural layers dynamically to ignore broken serialization tags.
    """
    try:
        # Load without compilation parameters to ignore config key blocks
        base_model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        
        # Extract layers manually into a fresh memory container
        inputs = tf.keras.layers.Input(shape=(256, 256, 3))
        outputs = base_model(inputs, training=False)
        
        prediction_model = tf.keras.models.Model(inputs, outputs)
        return prediction_model
    except Exception:
        # Fallback if functional layer architecture varies
        return tf.keras.models.load_model(model_path, compile=False)

# Reconstruct a clean runtime architecture instantly 
MODEL = load_clean_prediction_model("potato_model.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "not_a_potato_leaf", "not_potato"]

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
    img = Image.fromarray(image).resize((256, 256))
    img_batch = np.expand_dims(np.array(img), 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)