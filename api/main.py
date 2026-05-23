from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import json
import zipfile
import io
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

# --- THE DEEP REMEDY: MANUALLY STRIP BROKEN ARCHIVE METADATA ---
def clean_and_load_keras_model(model_path):
    """
    Forces open the model's zip archive, strips 'quantization_config' 
    directly from raw bytes, and serves Keras a completely clean file buffer.
    """
    try:
        patched_buffer = io.BytesIO()
        with zipfile.ZipFile(model_path, 'r') as old_zip:
            with zipfile.ZipFile(patched_buffer, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                for item in old_zip.infolist():
                    data = old_zip.read(item.filename)
                    
                    # If this is the file controlling layer properties, wipe out the bad keyword
                    if item.filename == "config.json":
                        config_string = data.decode('utf-8')
                        # Safely erase both variations of the broken quantization metadata
                        config_string = config_string.replace('"quantization_config": null,', '')
                        config_string = config_string.replace(', "quantization_config": null', '')
                        config_string = config_string.replace('"quantization_config": null', '')
                        data = config_string.encode('utf-8')
                        
                    new_zip.writestr(item, data)
        
        patched_buffer.seek(0)
        # Load the sanitized file straight into memory
        return tf.keras.models.load_model(patched_buffer, compile=False)
    except Exception as e:
        print(f"Fallback loading mechanism active due to: {e}")
        return tf.keras.models.load_model(model_path, compile=False)

# This will now safely instantiate your model architecture!
MODEL = clean_and_load_keras_model("potato_model.keras")

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