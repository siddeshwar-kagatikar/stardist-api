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

# ✅ Latest Cellpose (v3/v4)
from cellpose import models

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------------------------
# CORS setup
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load StarDist model
# ---------------------------
logger.info("Loading StarDist2D model...")
stardist_model = StarDist2D.from_pretrained("2D_versatile_he")
logger.info("✅ StarDist model loaded successfully")

# ---------------------------
# Load U-Net model
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

logger.info("Loading U-Net model...")
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(DEVICE)

unet_model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))
unet_model.eval()
logger.info("✅ U-Net model loaded successfully")

# ---------------------------
# Load Cellpose model (same as Colab)
# ---------------------------
use_gpu = torch.cuda.is_available()
logger.info(f"Using GPU for Cellpose: {use_gpu}")
cellpose_model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')
logger.info("✅ Cellpose model (v4 style) loaded successfully")

# ---------------------------
# Albumentations for UNet
# ---------------------------
val_tfms = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


@app.get("/")
def home():
    return {"message": "StarDist + U-Net + Cellpose API (latest) is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Received file: {file.filename}")
        contents = await file.read()
        image = imread(io.BytesIO(contents))
        logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # ---------------------------
        # Preprocess input
        # ---------------------------
        # Ensure a HxW x3 uint8 image for downstream models (StarDist, U-Net, Cellpose).
        # Keep the original intensities (don't normalize to [0,1] for Cellpose here).
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.ndim == 3 and image.shape[2] > 4:
            image = image[:, :, :3]

        # If image is float in [0,1], scale to 0-255 and cast to uint8.
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                # floats but already in 0-255 range
                image = np.clip(image, 0, 255).round().astype(np.uint8)
        else:
            # integer types → cast to uint8 safely
            image = image.astype(np.uint8)

        # ============================================================
        # 1️⃣ StarDist prediction
        # ============================================================
        img_norm = normalize(image, 1, 99.8)
        labels, _ = stardist_model.predict_instances(img_norm)
        mask_stardist = img_as_ubyte(labels > 0)

        pil_star = Image.fromarray(mask_stardist)
        buf_star = io.BytesIO()
        pil_star.save(buf_star, format="PNG")
        star_b64 = base64.b64encode(buf_star.getvalue()).decode("utf-8")

        # ============================================================
        # 2️⃣ U-Net prediction
        # ============================================================
        img_resized = cv2.resize(image, (256, 256))
        transformed = val_tfms(image=img_resized)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = torch.sigmoid(unet_model(img_tensor))
            preds = (preds > 0.5).float().cpu().numpy()

        mask_unet = (preds[0, 0] * 255).astype(np.uint8)
        mask_unet = cv2.resize(mask_unet, (image.shape[1], image.shape[0]))

        pil_unet = Image.fromarray(mask_unet)
        buf_unet = io.BytesIO()
        pil_unet.save(buf_unet, format="PNG")
        unet_b64 = base64.b64encode(buf_unet.getvalue()).decode("utf-8")

        # ============================================================
        # 3️⃣ Cellpose prediction (fixed to match Colab behavior)
        # ============================================================

        # Prepare input for Cellpose: use raw uint8 intensities, no manual normalization
        img_cp = image.copy()

        # If grayscale (already expanded earlier), ensure shape is HxW or HxWx3
        # Cellpose accepts HxW (single-channel) or HxWxC; using HxWx3 is fine.
        # We'll keep HxWx3 as in Colab.
        if img_cp.ndim == 2:
            img_cp = img_cp
        elif img_cp.ndim == 3 and img_cp.shape[2] == 3:
            # OK
            pass
        else:
            # Fallback: make it HxW x 3
            if img_cp.ndim == 3:
                img_cp = img_cp[:, :, :3]
            else:
                img_cp = np.stack([img_cp] * 3, axis=-1)

        # IMPORTANT: Do NOT divide by 255.0 here. Let Cellpose handle normalization internally.
        # If the image is float (rare after our casting), convert to uint8:
        if img_cp.dtype != np.uint8:
            # handle any stray float arrays
            if np.issubdtype(img_cp.dtype, np.floating) and img_cp.max() <= 1.0:
                img_cp = (np.clip(img_cp, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                img_cp = img_cp.astype(np.uint8)

        # Use same channel spec you used in Colab. For grayscale/brightfield RGB experiments [0,0] is typical.
        cp_channels = [0, 0]

        # Run Cellpose inference
        masks, flows, styles = cellpose_model.eval(
            img_cp,
            diameter=None,
            channels=cp_channels,
        )

        # Convert instance labels to binary mask (as in Colab)
        mask_cellpose = (masks > 0).astype(np.uint8) * 255

        pil_cellpose = Image.fromarray(mask_cellpose)
        buf_cell = io.BytesIO()
        pil_cellpose.save(buf_cell, format="PNG")
        cell_b64 = base64.b64encode(buf_cell.getvalue()).decode("utf-8")

        logger.info("✅ Prediction complete (StarDist + U-Net + Cellpose)")

        return JSONResponse({
            "stardist_mask": f"data:image/png;base64,{star_b64}",
            "unet_mask": f"data:image/png;base64,{unet_b64}",
            "cellpose_mask": f"data:image/png;base64,{cell_b64}"
        })

    except Exception as e:
        logger.exception(f"❌ Error during prediction: {e}")
        return {"error": str(e)}
