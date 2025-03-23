from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the trained pneumonia detection model
MODEL_PATH = 'pneumonia_detection_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Image size should match training data
IMAGE_SIZE = [180, 180]

def preprocess_image(image):
    """Preprocess uploaded X-ray image for model prediction."""
    try:
        image = image.convert("RGB")  # Ensure it's in RGB format
        image = np.array(image) / 255.0  # Normalize pixel values
        image = tf.image.resize(image, IMAGE_SIZE)  # Resize image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint to predict pneumonia from uploaded X-ray image."""
    try:
        # Check file extension
        if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
            raise HTTPException(status_code=400, detail="Invalid file type. Upload a PNG or JPG image.")
        
        # Read file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))  # Open image from uploaded file
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)[0][0]

        # Determine result
        result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return JSONResponse(content={"prediction": result, "confidence": f"{confidence*100:.2f}%"})
    
    except HTTPException as e:
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)
    
    except Exception as e:
        return JSONResponse(content={"detail": f"Unexpected error: {str(e)}"}, status_code=500)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
