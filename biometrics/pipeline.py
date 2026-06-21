# ============================================================================================
# Load in Libraries
# ============================================================================================

import os

# Helps with TensorFlow compatibility, and displays less unnecessary alerts
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Hide warnings
import warnings
warnings.filterwarnings("ignore")

# Displays only real errors
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Libraries required
from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import pandas as pd
import numpy as np
import time
from scipy.stats import spearmanr, kendalltau, kruskal, pearsonr

# ============================================================================================
# Experiment settings
# ============================================================================================

LIGHT_DISTANCES_CM = list(range(15, 136, 15))[::-1]  # 135,120,...,15
SUBJECTS           = ["s01", "s02", "s03", "s04", "s05",
                      "s06", "s07", "s08", "s09", "s10"]
DATA_DIR           = "./data"
MODEL              = "VGG-Face"
METRIC             = "cosine"
THRESHOLD          = 0.40
USE_CLAHE          = True
VALID_EXTENSIONS   = [".jpg", ".jpeg"]

# ============================================================================================
# Search for an image file by checking both .jpg and .jpeg extensions.
# ============================================================================================

def find_image(base_path_without_ext):
    for ext in VALID_EXTENSIONS:
        candidate = base_path_without_ext + ext
        if os.path.exists(candidate):
            return candidate
    return None

# ============================================================================================
# CLAHE preprocessing
# ============================================================================================

def apply_clahe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def load_img(path):
    return apply_clahe(path) if USE_CLAHE else path

def illuminance(dist_cm):
    return round((90 / dist_cm) ** 2, 4)

# ============================================================================================
# Reflection / glare metrics
# ============================================================================================

def variance_of_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def bright_pixel_ratio(gray, threshold=245):
    if gray is None or gray.size == 0:
        return None
    return float((gray >= threshold).sum()) / float(gray.size)

def crop_patch(img, center, half_size=20):
    x, y = int(center[0]), int(center[1])
    h, w = img.shape[:2]
    x1, x2 = max(0, x - half_size), min(w, x + half_size)
    y1, y2 = max(0, y - half_size), min(h, y + half_size)
    return img[y1:y2, x1:x2]

def extract_reflection_metrics(img_path):
    img = cv2.imread(img_path)
    empty = {
        "face_brightness": None,
        "face_contrast": None,
        "face_sharpness": None,
        "face_glare_ratio": None,
        "left_eye_glare": None,
        "right_eye_glare": None,
        "eye_glare_mean": None,
        "retinaface_score": None,
        "face_width_px": None,
        "face_height_px": None,
        "eye_distance_px": None,
    }

    if img is None:
        return empty

    try:
        resp = RetinaFace.detect_faces(img_path)
    except Exception:
        return empty

    if not isinstance(resp, dict) or len(resp) == 0:
        return empty

    face_key = list(resp.keys())[0]
    face = resp[face_key]

    if "facial_area" not in face or "landmarks" not in face:
        return empty

    x1, y1, x2, y2 = face["facial_area"]
    score = face.get("score", None)
    landmarks = face["landmarks"]

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return empty

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    left_eye_patch  = crop_patch(img, landmarks["left_eye"], 20)
    right_eye_patch = crop_patch(img, landmarks["right_eye"], 20)

    left_eye_glare = None
    right_eye_glare = None

    if left_eye_patch.size > 0:
        left_eye_glare = bright_pixel_ratio(
            cv2.cvtColor(left_eye_patch, cv2.COLOR_BGR2GRAY), threshold=245
        )

    if right_eye_patch.size > 0:
        right_eye_glare = bright_pixel_ratio(
            cv2.cvtColor(right_eye_patch, cv2.COLOR_BGR2GRAY), threshold=245
        )

    eye_vals = [v for v in [left_eye_glare, right_eye_glare] if v is not None]
    eye_glare_mean = sum(eye_vals) / len(eye_vals) if eye_vals else None

    lx, ly = landmarks["left_eye"]
    rx, ry = landmarks["right_eye"]
    eye_distance = float(np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2))

    return {
        "face_brightness": float(gray_face.mean()),
        "face_contrast": float(gray_face.std()),
        "face_sharpness": variance_of_laplacian(gray_face),
        "face_glare_ratio": bright_pixel_ratio(gray_face, threshold=245),
        "left_eye_glare": left_eye_glare,
        "right_eye_glare": right_eye_glare,
        "eye_glare_mean": eye_glare_mean,
        "retinaface_score": float(score) if score is not None else None,
        "face_width_px": int(x2 - x1),
        "face_height_px": int(y2 - y1),
        "eye_distance_px": eye_distance,
    }

# ============================================================================================
# Progress bar
# ============================================================================================

def progress(done, total, subject, dist_cm):
    pct    = done / total
    width  = 30
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {done}/{total}  {subject} @ {dist_cm}cm   ", end="", flush=True)

# ============================================================================================
# Main experiment
# ============================================================================================

records  = []
total    = len(SUBJECTS) * len(LIGHT_DISTANCES_CM)
done     = 0
t_start  = time.time()

clahe_label = "with CLAHE" if USE_CLAHE else "without CLAHE"
print(f"\n🚀 Pipeline started — {len(SUBJECTS)} subjects × {len(LIGHT_DISTANCES_CM)} distances = {total} comparisons ({clahe_label})\n")

for subject in SUBJECTS:
    subject_dir = os.path.join(DATA_DIR, subject)
    ref_path = find_image(os.path.join(subject_dir, "reference"))

    if ref_path is None:
        print(f"  ⚠ Reference not found (.jpg/.jpeg): {os.path.join(subject_dir, 'reference')} — subject skipped")
        done += len(LIGHT_DISTANCES_CM)
        continue

    ref_metrics = extract_reflection_metrics(ref_path)

    for dist_cm in LIGHT_DISTANCES_CM:
        done += 1
        test_path = find_image(os.path.join(subject_dir, f"{dist_cm}cm"))
        progress(done, total, subject, dist_cm)

        base = {
            "subject"         : subject,
            "light_dist_cm"   : dist_cm,
            "illuminance_rel" : illuminance(dist_cm),
            "threshold"       : THRESHOLD,
            "clahe"           : USE_CLAHE,
            "ref_path"        : ref_path,
            "test_path"       : test_path,
            **{f"ref_{k}": v for k, v in ref_metrics.items()}
        }

        if test_path is None:
            records.append({**base,
                            "cosine_dist"   : None,
                            "norm_dist"     : None,
                            "margin"        : None,
                            "confidence_pct": None,
                            "processing_s"  : None,
                            "verified"      : False,
                            "false_reject"  : True,
                            "detect_failed" : False,
                            "skip_reason"   : "testphoto_missing_jpg_jpeg",
                            })
            print(f"\n  ⚠ Test image missing (.jpg/.jpeg): {os.path.join(subject_dir, f'{dist_cm}cm')}")
            continue

        test_metrics = extract_reflection_metrics(test_path)

        try:
            t0 = time.time()

            result = DeepFace.verify(
                img1_path         = load_img(ref_path),
                img2_path         = load_img(test_path),
                model_name        = MODEL,
                distance_metric   = METRIC,
                enforce_detection = True,
                align             = True,
                silent            = True,
            )
            proc_s = round(time.time() - t0, 3)

            cosine      = round(result["distance"], 6)
            verified    = result["verified"]
            norm_dist   = round(cosine / THRESHOLD, 4)
            margin      = round(THRESHOLD - cosine, 6)

            if "confidence" in result and result["confidence"] is not None:
                confidence = round(float(result["confidence"]), 2)
            else:
                confidence = round(max(0.0, min(100.0, (1 - cosine / THRESHOLD) * 100)), 2)

            records.append({**base,
                            **{f"test_{k}": v for k, v in test_metrics.items()},
                            "cosine_dist"   : cosine,
                            "norm_dist"     : norm_dist,
                            "margin"        : margin,
                            "confidence_pct": confidence,
                            "processing_s"  : proc_s,
                            "verified"      : verified,
                            "false_reject"  : not verified,
                            "detect_failed" : False,
                            "skip_reason"   : None,
                            })

            status = "✓ MATCH " if verified else "✗ REJECT"
            print(f"\n  [{done:>3}/{total}] {subject} @ {dist_cm:>3}cm  "
                  f"cosine: {cosine:.4f}  norm: {norm_dist:.3f}  "
                  f"conf: {confidence:.1f}%  {proc_s:.2f}s  {status}")

        except ValueError:
            records.append({**base,
                            **{f"test_{k}": v for k, v in test_metrics.items()},
                            "cosine_dist"   : None,
                            "norm_dist"     : None,
                            "margin"        : None,
                            "confidence_pct": None,
                            "processing_s"  : None,
                            "verified"      : False,
                            "false_reject"  : True,
                            "detect_failed" : True,
                            "skip_reason"   : "no_face_detected",
                            })
            print(f"\n  [{done:>3}/{total}] {subject} @ {dist_cm:>3}cm  ⚠ No face detected")

        except Exception as e:
            records.append({**base,
                            **{f"test_{k}": v for k, v in test_metrics.items()},
                            "cosine_dist"   : None,
                            "norm_dist"     : None,
                            "margin"        : None,
                            "confidence_pct": None,
                            "processing_s"  : None,
                            "verified"      : False,
                            "false_reject"  : True,
                            "detect_failed" : True,
                            "skip_reason"   : f"error:{e}",
                            })
            print(f"\n  [{done:>3}/{total}] {subject} @ {dist_cm:>3}cm  ✗ Error: {e}")

# =============================================================================
# Save all collected results
# =============================================================================

elapsed = round(time.time() - t_start, 1)
df = pd.DataFrame(records)
out_csv = f"results_{'clahe' if USE_CLAHE else 'raw'}_15to135_with_glare.csv"
df.to_csv(out_csv, index=False)
print(f"\n\n✅ Done in {elapsed}s — {len(records)} rows saved in {out_csv}\n")

# =============================================================================
# Summary per distance
# =============================================================================

df_v = df[df["cosine_dist"].notna()]

summary = (
    df.groupby("light_dist_cm")
    .agg(
        n_total        = ("subject",         "count"),
        n_valid        = ("cosine_dist",     lambda x: x.notna().sum()),
        mean_cosine    = ("cosine_dist",     "mean"),
        median_cosine  = ("cosine_dist",     "median"),
        std_cosine     = ("cosine_dist",     "std"),
        min_cosine     = ("cosine_dist",     "min"),
        max_cosine     = ("cosine_dist",     "max"),
        q25            = ("cosine_dist",     lambda x: x.quantile(0.25)),
        q75            = ("cosine_dist",     lambda x: x.quantile(0.75)),
        mean_margin    = ("margin",          "mean"),
        mean_conf      = ("confidence_pct",  "mean"),
        mean_proc_s    = ("processing_s",    "mean"),
        FRR            = ("false_reject",    "mean"),
        n_detect_fail  = ("detect_failed",   "sum"),
        illuminance    = ("illuminance_rel", "first"),
        mean_eye_glare = ("test_eye_glare_mean", "mean"),
        mean_face_glare= ("test_face_glare_ratio", "mean"),
    )
    .reset_index()
    .sort_values("light_dist_cm", ascending=False)
)

summary["FRR_%"] = (summary["FRR"] * 100).round(1)
summary["IQR"]   = (summary["q75"] - summary["q25"]).round(4)
summary["CV_%"]  = (summary["std_cosine"] / summary["mean_cosine"] * 100).round(1)
summary["range"] = (summary["max_cosine"] - summary["min_cosine"]).round(4)

for col in ["mean_cosine","median_cosine","std_cosine","min_cosine","max_cosine",
            "mean_margin","mean_conf","mean_proc_s","mean_eye_glare","mean_face_glare"]:
    summary[col] = summary[col].round(4)

sum_csv = f"results_summary_{'clahe' if USE_CLAHE else 'raw'}_15to135_with_glare.csv"
summary.drop(columns=["FRR","q25","q75"]).to_csv(sum_csv, index=False)
print(f"📊 Summary saved in {sum_csv}\n")

# =============================================================================
# Correlation analysis
# =============================================================================

df_corr = df_v[["light_dist_cm", "cosine_dist", "illuminance_rel"]].dropna()

if len(df_corr) >= 4:
    pr, pp = pearsonr(df_corr["light_dist_cm"], df_corr["cosine_dist"])
    sr, sp = spearmanr(df_corr["light_dist_cm"], df_corr["cosine_dist"])
    kr, kp = kendalltau(df_corr["light_dist_cm"], df_corr["cosine_dist"])
    ir, ip = spearmanr(df_corr["illuminance_rel"], df_corr["cosine_dist"])

    print("\n── Correlation: light distance ↔ cosine distance ─────────────────────")
    print(f"  Pearson  r  = {pr:+.4f}  (p = {pp:.4f})  {'✓ signif.' if pp < 0.05 else '✗ not signif.'}")
    print(f"  Spearman ρ  = {sr:+.4f}  (p = {sp:.4f})  {'✓ signif.' if sp < 0.05 else '✗ not signif.'}")
    print(f"  Kendall  τ  = {kr:+.4f}  (p = {kp:.4f})  {'✓ signif.' if kp < 0.05 else '✗ not signif.'}")
    print(f"  Spearman ρ  (illuminance ↔ cosine) = {ir:+.4f}  (p = {ip:.4f})")

# Extra glare analysis
glare_cols = ["test_eye_glare_mean", "test_face_glare_ratio", "cosine_dist", "false_reject"]
df_glare = df_v[glare_cols].dropna()

if len(df_glare) >= 4:
    g1, gp1 = spearmanr(df_glare["test_eye_glare_mean"], df_glare["cosine_dist"])
    g2, gp2 = spearmanr(df_glare["test_face_glare_ratio"], df_glare["cosine_dist"])
    print("\n── Correlation: glare ↔ cosine distance ──────────────────────────────")
    print(f"  Spearman ρ (eye glare ↔ cosine)  = {g1:+.4f}  (p = {gp1:.4f})")
    print(f"  Spearman ρ (face glare ↔ cosine) = {g2:+.4f}  (p = {gp2:.4f})")

# =============================================================================
# Kruskal-Wallis H-test
# =============================================================================

groups = [g["cosine_dist"].dropna().values
          for _, g in df_v.groupby("light_dist_cm")
          if g["cosine_dist"].notna().sum() >= 2]

if len(groups) >= 2:
    H, p_kw = kruskal(*groups)
    print(f"\n── Kruskal-Wallis H-test ─────────────────────────────────────────────")
    print(f"  H = {H:.4f},  p = {p_kw:.4f}  "
          f"{'→ distributions differ significantly (p<0.05)' if p_kw < 0.05 else '→ no significant difference'}")

# =============================================================================
# Final summary
# =============================================================================

n_valid   = df[~df["detect_failed"]].shape[0]
n_failed  = int(df["detect_failed"].sum())
n_missing = int(df["skip_reason"].eq("testphoto_missing_jpg_jpeg").sum())
overall_frr = df["false_reject"].mean() * 100
avg_proc  = df["processing_s"].mean()

print(f"\n── Total ─────────────────────────────────────────────────────────────")
print(f"  Model: {MODEL}  |  Metric: {METRIC}  |  Threshold: {THRESHOLD}  |  CLAHE: {USE_CLAHE}")
print(f"  Comparisons: {len(records)}  |  Valid: {n_valid}  |  Detect-fail: {n_failed}  |  Missing: {n_missing}")
print(f"  Overall FRR: {overall_frr:.1f}%")
if pd.notna(avg_proc):
    print(f"  Mean processing time: {avg_proc:.2f}s per comparison")
print("─" * 110)