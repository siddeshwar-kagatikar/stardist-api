import io
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imread
from skimage import img_as_ubyte
import numpy as np
from PIL import Image

# 🧩 Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 🟢 Allow CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Load model once
logger.info("Loading StarDist2D model...")
model = StarDist2D.from_pretrained("2D_versatile_he")
logger.info("✅ Model loaded successfully")

@app.get("/")
def home():
    return {"message": "StarDist API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Received file: {file.filename}")

        contents = await file.read()
        image = imread(io.BytesIO(contents))
        logger.info(f"Image shape: {image.shape}")

        # Normalize and predict
        img_norm = normalize(image, 1, 99.8)
        labels, _ = model.predict_instances(img_norm)
        logger.info("✅ Prediction complete")

        # Convert to uint8 grayscale image
        mask = img_as_ubyte(labels > 0)
        pil_img = Image.fromarray(mask)

        # Convert to bytes for response
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        logger.info(f"📤 Sending segmented mask for {file.filename}")
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.exception(f"❌ Error during prediction: {e}")
        return {"error": str(e)}
