from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage import io
import numpy as np
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = StarDist2D.from_pretrained('2D_versatile_he')

@app.get("/")
def root():
    return {"message": "StarDist API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and decode image
        image_data = await file.read()
        np_img = io.imread(image_data, plugin='imageio')
        if np_img.shape[-1] == 4:
            np_img = np_img[..., :3]

        # Predict
        labels, _ = model.predict_instances(normalize(np_img))
        binary_mask = np.zeros_like(labels, dtype=np.uint8)
        binary_mask[labels > 0] = 255

        # Save result temporarily
        output_path = f"/tmp/{file.filename}_mask.png"
        io.imsave(output_path, binary_mask)

        # Return the mask as base64 image
        import base64
        with open(output_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        return {"mask_image": encoded}

    except Exception as e:
        return {"error": str(e)}
