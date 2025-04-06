from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uvicorn
import os

app = FastAPI()

model = load_model("WasteClassificationModelWeights.hdf5")

@app.post("/waste_classification")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(contents)

    img = load_img(file_path, target_size=(180, 180))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_index = np.argmax(preds[0])
    confidence = float(preds[0][class_index])

    os.remove(file_path)

    return {
        "class": "Recycle" if class_index == 1 else "Organic",
        "confidence": round(confidence * 100, 2)
    }

if __name__ == "__main__":
    uvicorn.run("smartBinBackend:app", host="0.0.0.0", port=8000, reload=True)
