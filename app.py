import io
import logging
import base64
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
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

# ---------------------------
# Model definitions (initialized as None)
# ---------------------------
# We will load these in the 'startup' lifespan event
models_dict = {
    "stardist_model": None,
    "unet_model": None,
    "cellpose_model": None,
    "val_tfms": None,
    "device": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------------------
    # Load models on startup
    # ---------------------------
    logger.info("Server startup: Loading models...")
    
    # 1. Load StarDist model
    logger.info("Loading StarDist2D model...")
    models_dict["stardist_model"] = StarDist2D.from_pretrained("2D_versatile_he")
    logger.info("✅ StarDist model loaded successfully")

    # 2. Load U-Net model
    models_dict["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {models_dict['device']}")

    logger.info("Loading U-Net model...")
    unet_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Not loading pretrained ImageNet weights
        in_channels=3,
        classes=1,
    ).to(models_dict["device"])

    # Load your trained weights
    try:
        unet_model.load_state_dict(torch.load("unet_model.pth", map_location=models_dict["device"]))
        unet_model.eval()
        models_dict["unet_model"] = unet_model
        logger.info("✅ U-Net model loaded successfully")
    except FileNotFoundError:
        logger.error("❌ 'unet_model.pth' not found. The U-Net endpoint will fail.")
    except Exception as e:
        logger.error(f"❌ Error loading U-Net model: {e}")

    # 3. Load Cellpose model
    use_gpu = torch.cuda.is_available()
    logger.info(f"Using GPU for Cellpose: {use_gpu}")
    models_dict["cellpose_model"] = models.CellposeModel(gpu=use_gpu, model_type='nuclei')
    logger.info("✅ Cellpose model (v4 style) loaded successfully")

    # 4. Albumentations for UNet
    models_dict["val_tfms"] = A.Compose([
        A.Normalize(), # Normalizes pixel values
        ToTensorV2(),  # Converts to PyTorch tensor
    ])
    
    logger.info("✅ All models loaded. Server is ready.")
    
    yield
    
    # ---------------------------
    # Clean up models on shutdown (optional)
    # ---------------------------
    logger.info("Server shutdown: Clearing models...")
    models_dict.clear()


app = FastAPI(lifespan=lifespan)

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


@app.get("/")
def home():
    """Home endpoint to check if the API is running."""
    return {"message": "StarDist + U-Net + Cellpose API (latest) is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.
    Receives an image file, runs prediction with StarDist, U-Net, and Cellpose,
    and returns the binary masks as base64-encoded PNG strings.
    """
    try:
        # Access models from the dictionary
        stardist_model = models_dict.get("stardist_model")
        unet_model = models_dict.get("unet_model")
        cellpose_model = models_dict.get("cellpose_model")
        val_tfms = models_dict.get("val_tfms")
        DEVICE = models_dict.get("device")

        if not all([stardist_model, unet_model, cellpose_model, val_tfms, DEVICE]):
            logger.error("❌ Models are not loaded. Check startup logs.")
            return JSONResponse(status_code=500, content={"error": "Models are not initialized."})
            
        logger.info(f"📥 Received file: {file.filename}")
        contents = await file.read()
        
        # Read image from bytes
        image = imread(io.BytesIO(contents))
        logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # ---------------------------
        # Preprocess input
        # ---------------------------
        original_shape = image.shape[:2] # Save original H, W

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.ndim == 3 and image.shape[2] > 4:
            logger.warning(f"Image has {image.shape[2]} channels. Taking first 3.")
            image = image[:, :, :3]
        elif image.ndim == 3 and image.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                logger.info("Image is float [0, 1]. Scaling to [0, 255] and casting to uint8.")
                image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                logger.info("Image is float [0, 255]. Clipping and casting to uint8.")
                image = np.clip(image, 0, 255).round().astype(np.uint8)
        elif image.dtype != np.uint8:
             logger.warning(f"Image dtype is {image.dtype}. Casting to uint8. This may result in data loss.")
             if image.dtype == np.uint16:
                 image = (image / 256).astype(np.uint8)
             else:
                 image = image.astype(np.uint8)
        
        # ============================================================
        # 1️⃣ StarDist prediction
        # ============================================================
        logger.info("Running StarDist prediction...")
        img_norm = normalize(image, 1, 99.8, axis=(0,1))
        labels, _ = stardist_model.predict_instances(img_norm)
        mask_stardist = (labels > 0).astype(np.uint8) * 255
        logger.info("StarDist prediction complete.")

        pil_star = Image.fromarray(mask_stardist)
        buf_star = io.BytesIO()
        pil_star.save(buf_star, format="PNG")
        star_b64 = base64.b64encode(buf_star.getvalue()).decode("utf-8")

        # ============================================================
        # 2️⃣ U-Net prediction
        # ============================================================
        logger.info("Running U-Net prediction...")
        img_resized = cv2.resize(image, (256, 256))
        transformed = val_tfms(image=img_resized)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = torch.sigmoid(unet_model(img_tensor))
            preds = (preds > 0.5).float().cpu().numpy()

        mask_unet = (preds[0, 0] * 255).astype(np.uint8)
        mask_unet = cv2.resize(mask_unet, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        logger.info("U-Net prediction complete.")

        pil_unet = Image.fromarray(mask_unet)
        buf_unet = io.BytesIO()
        pil_unet.save(buf_unet, format="PNG")
        unet_b64 = base64.b64encode(buf_unet.getvalue()).decode("utf-8")

        # ============================================================
        # 3️⃣ Cellpose prediction
        # ============================================================
        logger.info("Running Cellpose prediction...")
        img_cp = image.copy()
        cp_channels = [0, 0]
        masks, flows, styles = cellpose_model.eval(
            img_cp,
            diameter=None,
            channels=cp_channels,
            do_3D=False
        )
        mask_cellpose = (masks > 0).astype(np.uint8) * 255
        logger.info("Cellpose prediction complete.")

        pil_cellpose = Image.fromarray(mask_cellpose)
        buf_cell = io.BytesIO()
        pil_cellpose.save(buf_cell, format="PNG")
        cell_b64 = base64.b64encode(buf_cell.getvalue()).decode("utf-8")

        logger.info("✅ All predictions complete.")

        return JSONResponse({
            "stardist_mask": f"data:image/png;base64,{star_b64}",
            "unet_mask": f"data:image/png;base64,{unet_b64}",
            "cellpose_mask": f"data:image/png;base64,{cell_b64}"
        })

    except Exception as e:
        logger.exception(f"❌ Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly at http://127.0.0.1:8000")
    # Note: When running directly, uvicorn.run() will manage the lifespan events.
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)