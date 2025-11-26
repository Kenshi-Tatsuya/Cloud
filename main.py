from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
import mediapipe as mp
import joblib
import base64
from io import BytesIO
from PIL import Image
import os  # 🔹 add this

# ------------------------------
# Load ML Model + Class Labels
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 🔹 folder where main.py lives
MODEL_PATH = os.path.join(BASE_DIR, "tamil_sign_twohand_model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "label_classes.npy")

print("MODEL_PATH:", MODEL_PATH)
print("LABELS_PATH:", LABELS_PATH)
print("Model file exists?", os.path.exists(MODEL_PATH))
print("Labels file exists?", os.path.exists(LABELS_PATH))

model = joblib.load(MODEL_PATH)
class_names = list(np.load(LABELS_PATH, allow_pickle=True))

print("Model loaded. Classes:", len(class_names))


# ---------------------------------
# FastAPI + CORS for frontend
# ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# Request schema
# ---------------------------------
class FrameData(BaseModel):
    image: str  # base64 string


# ---------------------------------
# Mediapipe setup
# ---------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------
# Helper: extract hand features
# ---------------------------------
def process_hand(landmarks):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = arr[0].copy()
    arr[:, :2] -= wrist[:2]

    scale = np.linalg.norm(arr[9, :2])
    if scale < 1e-6:
        scale = 1.0
    arr[:, :2] /= scale

    return arr.flatten().tolist()  # 63 floats


# ---------------------------------
# Prediction Route
# ---------------------------------
@app.post("/predict")
async def predict(data: FrameData):

    # ------------------------------
    # 1. Decode base64 → image
    # ------------------------------
    image_bytes = base64.b64decode(data.image.split(",")[1])
    pil_image = Image.open(BytesIO(image_bytes))
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ------------------------------
    # 2. Run MediaPipe Hands
    # ------------------------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)

    left_feats = [0.0] * 63
    right_feats = [0.0] * 63
    has_left = 0
    has_right = 0

    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            side = handed.classification[0].label.lower()
            feats = process_hand(lm.landmark)

            if side == "left":
                left_feats = feats
                has_left = 1
            else:
                right_feats = feats
                has_right = 1

    # No hands detected → return empty
    if not has_left and not has_right:
        return {"text": ""}

    # ------------------------------
    # 3. Build feature vector 
    # ------------------------------
    feat_vector = np.array(left_feats + right_feats + [has_left, has_right]).reshape(1, -1)

    # ------------------------------
    # 4. Predict label
    # ------------------------------
    try:
        probs = model.predict_proba(feat_vector)[0]
        idx = int(np.argmax(probs))
        label = class_names[idx]
    except Exception:
        label = ""

    return {"text": label}


# ---------------------------------
# Run server
# ---------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
