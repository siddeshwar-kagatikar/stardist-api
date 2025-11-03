import io
import logging
import base64
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imread
from skimage import img_as_ubyte
from PIL import Image
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 🧩 Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# 🟢 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Load StarDist model
logger.info("Loading StarDist2D model...")
stardist_model = StarDist2D.from_pretrained("2D_versatile_he")
logger.info("✅ StarDist model loaded successfully")

# 🧠 Load U-Net model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

logger.info("Loading U-Net model...")
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # ⚠️ None since custom weights
    in_channels=3,
    classes=1,
).to(DEVICE)

unet_model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))
unet_model.eval()
logger.info("✅ U-Net model loaded successfully")

# ============================================================
# 🔧 Preprocessing transform (same as Kaggle code)
# ============================================================
val_tfms = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


@app.get("/")
def home():
    return {"message": "StarDist + U-Net API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Received file: {file.filename}")
        contents = await file.read()
        image = imread(io.BytesIO(contents))
        logger.info(f"Image shape: {image.shape}")

        # ============================================================
        # 🧹 1️⃣ BASIC PREPROCESSING (RGB conversion, dtype consistency)
        # ============================================================
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = image.astype(np.uint8)

        # ============================================================
        # 🧬 2️⃣ StarDist prediction (unchanged)
        # ============================================================
        img_norm = normalize(image, 1, 99.8)
        labels, _ = stardist_model.predict_instances(img_norm)
        mask_stardist = img_as_ubyte(labels > 0)

        pil_star = Image.fromarray(mask_stardist)
        buf_star = io.BytesIO()
        pil_star.save(buf_star, format="PNG")
        star_b64 = base64.b64encode(buf_star.getvalue()).decode("utf-8")

        # ============================================================
        # 🧠 3️⃣ U-Net PREPROCESSING (Kaggle-style)
        # ============================================================
        # Resize to training input size
        img_resized = cv2.resize(image, (256, 256))

        # Apply Albumentations transform
        transformed = val_tfms(image=img_resized)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        # ============================================================
        # 🧩 4️⃣ U-Net INFERENCE + POSTPROCESSING (same as Kaggle)
        # ============================================================
        with torch.no_grad():
            preds = torch.sigmoid(unet_model(img_tensor))
            preds = (preds > 0.5).float().cpu().numpy()

        # Convert prediction to mask and scale to 0–255
        mask = (preds[0, 0] * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # back to original size

        # Encode to Base64
        pil_unet = Image.fromarray(mask)
        buf_unet = io.BytesIO()
        pil_unet.save(buf_unet, format="PNG")
        unet_b64 = base64.b64encode(buf_unet.getvalue()).decode("utf-8")

        logger.info("✅ Prediction complete")
        return JSONResponse({
            "stardist_mask": f"data:image/png;base64,{star_b64}",
            "unet_mask": f"data:image/png;base64,{unet_b64}"
        })

    except Exception as e:
        logger.exception(f"❌ Error during prediction: {e}")
        return {"error": str(e)}
