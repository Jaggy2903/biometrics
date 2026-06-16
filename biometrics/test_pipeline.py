import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from deepface import DeepFace

# ── Snelle test ───────────────────────────────────────────────────────────────
result = DeepFace.verify(
    img1_path         = "data/s01/reference.jpg",
    img2_path         = "data/s01/90cm.jpg",
    model_name        = "VGG-Face",
    distance_metric   = "cosine",
    enforce_detection = True,
    silent            = True,
)

print(f"Cosine distance : {result['distance']:.4f}")
print(f"Threshold       : {result['threshold']}")
print(f"Verified        : {result['verified']}")
print(f"\n✅ Pipeline werkt correct!")