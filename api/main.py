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

# --- THE FIX: STRIP UNRECOGNIZED ARGUMENTS FROM DENSE LAYER ---
class SafeDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        # Remove the offending key that causes the crash on Render
        config.pop('quantization_config', None)
        return super().from_config(config)

# Using compile=False and registering our SafeDense patch prevents the deserialization error
MODEL = tf.keras.models.load_model(
    "potato_model.keras", 
    compile=False, 
    safe_mode=False, 
    custom_objects={'Dense': SafeDense}
)

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
    # Ensure image size matches what your model expects
    img = Image.fromarray(image).resize((256, 256))
    img_batch = np.expand_dims(np.array(img), 0)
    
    predictions = MODEL.predict(img_batch)

    # Fixed index to properly access predictions list
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)