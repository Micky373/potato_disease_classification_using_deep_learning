from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO


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

MODEL = tf.keras.models.load_model('../saved_models/1')

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

def read_file_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
): 
    image = read_file_image(await file.read())
    image_batch = np.expand_dims(image,0)
    
    prediction = MODEL.predict(image_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        'class': predicted_class,
        'confidence':float(confidence)
    }
    
@app.get('/ping')
async def ping():
    return "Hey Micky how are you doing!!!"

if __name__ == "__main__":
    uvicorn.run('main:app',host='localhost',port=8000,reload=True)