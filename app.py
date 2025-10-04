from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import tempfile, os
from skimage import io
from stardist.models import StarDist2D
from csbdeep.utils import normalize

app = FastAPI()

# Load pretrained model once when server starts
model = StarDist2D.from_pretrained('2D_versatile_he')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    # Read image
    img = io.imread(temp_path)
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Predict nuclei
    labels, details = model.predict_instances(normalize(img))

    # Create binary mask
    binary_mask = np.zeros_like(labels, dtype=np.uint8)
    binary_mask[labels > 0] = 255

    # Save output mask
    output_path = temp_path.replace(".png", "_mask.png")
    io.imsave(output_path, binary_mask)

    # Return the mask file
    return FileResponse(output_path, media_type="image/png")
