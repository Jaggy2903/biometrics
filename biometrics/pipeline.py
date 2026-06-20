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

# DeepFace performs face verification
from deepface import DeepFace
# OpenCV is used for image preprocessing
import cv2
# Pandas stores and exports the results
import pandas as pd
import time
# SciPy provides statistical tests for analysis
from scipy.stats import spearmanr, kendalltau, kruskal, pearsonr

# ============================================================================================
# Experiment settings
# LIGHT_DISTANCE_CM:
# Distance between the light source and the participant.
# Images are processed from the largest distance (135 cm) to the smallest distance (15 cm).
#
# THRESHOLD:
# Maximum cosine distance that is still considered a successful match.
#
# USE_CLAHE:
# Enables image contrast enhancement before face verification.
# ============================================================================================

# ── Configuration ──────────────────────────────────────────────────────────────
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
# Returns the full file path if the image exists.
# ============================================================================================

# ── Image Search (.jpg / .jpeg) ────────────────────────────────────────────
def find_image(base_path_without_ext):
    for ext in VALID_EXTENSIONS:
        candidate = base_path_without_ext + ext
        if os.path.exists(candidate):
            return candidate
    return None

# ============================================================================================
# Improve image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
# This technique increases local contrast without over-amplifying bright regions.
# The goal is to improve face visibility under different lighting conditions.
# ============================================================================================

# ── CLAHE preprocessing ───────────────────────────────────────────────────────
def apply_clahe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Kon afbeelding niet laden: {img_path}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# Load the image with or without CLAHE preprocessing
def load_img(path):
    return apply_clahe(path) if USE_CLAHE else path

# Calculate the relative light intensity using the inverse square law
def illuminance(dist_cm):
    """Relatieve lichtsterkte via inverse kwadraatwet (genormaliseerd op 90cm=1.0)."""
    return round((90 / dist_cm) ** 2, 4)

# ── Progress bar ──────────────────────────────────────────────────────────────
def progress(done, total, subject, dist_cm):
    pct    = done / total
    width  = 30
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {done}/{total}  {subject} @ {dist_cm}cm   ", end="", flush=True)

# ============================================================================================
# Main experiment
#
# Each participant is compared with every image taken under different
# lighting distances.
#
# For every comparison, DeepFace determines whether both images belong to the same person.
# ============================================================================================

# ── Hoofdloop ─────────────────────────────────────────────────────────────────
records  = []
total    = len(SUBJECTS) * len(LIGHT_DISTANCES_CM)
done     = 0
t_start  = time.time()

clahe_label = "met CLAHE" if USE_CLAHE else "zonder CLAHE"
print(f"\n🚀 Pipeline gestart — {len(SUBJECTS)} proefpersonen × {len(LIGHT_DISTANCES_CM)} afstanden = {total} vergelijkingen ({clahe_label})\n")

for subject in SUBJECTS:
    subject_dir = os.path.join(DATA_DIR, subject)
    ref_path = find_image(os.path.join(subject_dir, "reference"))

    if ref_path is None:
        print(f"  ⚠ Referentie niet gevonden (.jpg/.jpeg): {os.path.join(subject_dir, 'reference')} — subject overgeslagen")
        done += len(LIGHT_DISTANCES_CM)
        continue

    for dist_cm in LIGHT_DISTANCES_CM:
        done += 1
        test_path = find_image(os.path.join(subject_dir, f"{dist_cm}cm"))
        progress(done, total, subject, dist_cm)

        # Store information that is identical for every comparison
        base = {
            "subject"         : subject,
            "light_dist_cm"   : dist_cm,
            "illuminance_rel" : illuminance(dist_cm),
            "threshold"       : THRESHOLD,
            "clahe"           : USE_CLAHE,
            "ref_path"        : ref_path,
            "test_path"       : test_path,
        }

        # Skip the comparison if the test image is missing.
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
                            "skip_reason"   : "testfoto_ontbreekt_jpg_jpeg",
                            })
            print(f"\n  ⚠ Testfoto ontbreekt (.jpg/.jpeg): {os.path.join(subject_dir, f'{dist_cm}cm')}")
            continue

        try:
            t0 = time.time()

            # Perform face verification 
            # DeepFace detects, aligns, and compares both faces
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

            # Lower cosine distances indicate that the two faces are more similar
            cosine      = round(result["distance"], 6)
            verified    = result["verified"]

            # Express the distance relative to the decision threshold
            norm_dist   = round(cosine / THRESHOLD, 4)

            # Calculate how far the result is from the verification threshold
            margin      = round(THRESHOLD - cosine, 6)

            # Use the confidence value returned by DeepFace if available. 
            # Otherwise, estimate a confidence percentage from the cosine distance
            if "confidence" in result and result["confidence"] is not None:
                confidence = round(float(result["confidence"]), 2)
            else:
                confidence = round(max(0.0, min(100.0, (1 - cosine / THRESHOLD) * 100)), 2)

            # Store the results of this comparison
            records.append({**base,
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
            

        # No face could be detected in one of the images.
        # Record this separately from recognition errors
        except ValueError:
            records.append({**base,
                            "cosine_dist"   : None,
                            "norm_dist"     : None,
                            "margin"        : None,
                            "confidence_pct": None,
                            "processing_s"  : None,
                            "verified"      : False,
                            "false_reject"  : True,
                            "detect_failed" : True,
                            "skip_reason"   : "geen_gezicht",
                            })
            print(f"\n  [{done:>3}/{total}] {subject} @ {dist_cm:>3}cm  ⚠ Gezicht niet gedetecteerd")

        # Record unexpected errors without stopping the experiment
        except Exception as e:
            records.append({**base,
                            "cosine_dist"   : None,
                            "norm_dist"     : None,
                            "margin"        : None,
                            "confidence_pct": None,
                            "processing_s"  : None,
                            "verified"      : False,
                            "false_reject"  : True,
                            "detect_failed" : True,
                            "skip_reason"   : f"fout:{e}",
                            })
            print(f"\n  [{done:>3}/{total}] {subject} @ {dist_cm:>3}cm  ✗ Fout: {e}")


# ============================================================================= 
#  Save all collected results to a CSV file for later analysis. 
# =============================================================================

# ── Saving ───────────────────────────────────────────────────────────────────
elapsed = round(time.time() - t_start, 1)
df = pd.DataFrame(records)
out_csv = f"results_{'clahe' if USE_CLAHE else 'raw'}_15to135.csv"
df.to_csv(out_csv, index=False)
print(f"\n\n Klaar in {elapsed}s — {len(records)} rijen opgeslagen in {out_csv}\n")


# ============================================================================= 
# Calculate descriptive statistics for each lighting distance. 
#  These values summarize recognition performance under each condition. 
# =============================================================================

# ── Summary per light distance ─────────────────────────────────────────────
# Only valid rows
df_v = df[df["cosine_dist"].notna()]   

summary = (
    df.groupby("light_dist_cm")
    .agg(
        n_totaal       = ("subject",         "count"),
        n_valid        = ("cosine_dist",     lambda x: x.notna().sum()),
        gem_cosine     = ("cosine_dist",     "mean"),
        median_cosine  = ("cosine_dist",     "median"),
        std_cosine     = ("cosine_dist",     "std"),
        min_cosine     = ("cosine_dist",     "min"),
        max_cosine     = ("cosine_dist",     "max"),
        q25            = ("cosine_dist",     lambda x: x.quantile(0.25)),
        q75            = ("cosine_dist",     lambda x: x.quantile(0.75)),
        gem_marge      = ("margin",          "mean"),
        gem_conf       = ("confidence_pct",  "mean"),
        gem_proc_s     = ("processing_s",    "mean"),
        FRR            = ("false_reject",    "mean"),
        n_detect_fail  = ("detect_failed",   "sum"),
        illuminance    = ("illuminance_rel", "first"),
    )
    .reset_index()
    .sort_values("light_dist_cm", ascending=False)
)

summary["FRR_%"] = (summary["FRR"] * 100).round(1)
summary["IQR"]   = (summary["q75"] - summary["q25"]).round(4)
summary["CV_%"]  = (summary["std_cosine"] / summary["gem_cosine"] * 100).round(1)
summary["range"] = (summary["max_cosine"] - summary["min_cosine"]).round(4)

for col in ["gem_cosine","median_cosine","std_cosine","min_cosine","max_cosine",
            "gem_marge","gem_conf","gem_proc_s"]:
    summary[col] = summary[col].round(4)

sum_csv = f"results_summary_{'clahe' if USE_CLAHE else 'raw'}_15to135.csv"
summary.drop(columns=["FRR","q25","q75"]).to_csv(sum_csv, index=False)
print(f"📊 Samenvatting opgeslagen in {sum_csv}\n")

# ============================================================================= 
# Print a readable table in the terminal. 
# This allows quick inspection without opening the CSV file
# =============================================================================

# ── Print table ───────────────────────────────────────────────────────────────
SEP = "─" * 110
print("── Samenvatting per lichtafstand " + "─" * 78)
header = (f"{'Afst':>6}  {'Illum':>7}  {'gem':>7}  {'med':>7}  {'std':>6}  "
          f"{'IQR':>6}  {'CV%':>5}  {'range':>6}  {'marge':>7}  {'conf%':>6}  "
          f"{'FRR%':>5}  {'fail':>4}  {'proc_s':>6}")
print(header)
print(SEP)

for _, r in summary.iterrows():
    def f(v, fmt=".4f"):
        return f"{v:{fmt}}" if pd.notna(v) else "   n/a"

    print(
        f"{int(r.light_dist_cm):>5}cm  "
        f"{r.illuminance:>7.4f}  "
        f"{f(r.gem_cosine):>7}  "
        f"{f(r.median_cosine):>7}  "
        f"{f(r.std_cosine):>6}  "
        f"{f(r.IQR):>6}  "
        f"{f(r['CV_%'],'.1f'):>5}  "
        f"{f(r.range):>6}  "
        f"{f(r.gem_marge):>7}  "
        f"{f(r.gem_conf,'.1f'):>6}  "
        f"{r['FRR_%']:>5.1f}%  "
        f"{int(r.n_detect_fail):>4}  "
        f"{f(r.gem_proc_s,'.2f'):>6}"
    )

print(SEP)

# ============================================================================= 
# Statistical analysis: correlation tests 
# These test whether lighting distance or illuminance affects recognition. 
# =============================================================================

# ── Correlation-analysis ────────────────────────────────────────────────────────
df_corr = df_v[["light_dist_cm", "cosine_dist", "illuminance_rel"]].dropna()

if len(df_corr) >= 4:
    pr, pp = pearsonr(df_corr["light_dist_cm"], df_corr["cosine_dist"])
    sr, sp = spearmanr(df_corr["light_dist_cm"], df_corr["cosine_dist"])
    kr, kp = kendalltau(df_corr["light_dist_cm"], df_corr["cosine_dist"])
    ir, ip = spearmanr(df_corr["illuminance_rel"], df_corr["cosine_dist"])

    print("\n── Correlatie: lichtafstand ↔ cosine distance ──────────────────────────")
    print(f"  Pearson  r  = {pr:+.4f}  (p = {pp:.4f})  {'✓ sign.' if pp < 0.05 else '✗ niet sign.'}")
    print(f"  Spearman ρ  = {sr:+.4f}  (p = {sp:.4f})  {'✓ sign.' if sp < 0.05 else '✗ niet sign.'}")
    print(f"  Kendall  τ  = {kr:+.4f}  (p = {kp:.4f})  {'✓ sign.' if kp < 0.05 else '✗ niet sign.'}")
    print(f"  Spearman ρ  (illuminantie ↔ cosine) = {ir:+.4f}  (p = {ip:.4f})")

# ============================================================================= 
# Non-parametric group comparison 
# Tests whether different lighting distances produce different distributions. 
# =============================================================================

# ── Kruskal-Wallis H-test ─────────────────────────────────────────────────────
groups = [g["cosine_dist"].dropna().values
          for _, g in df_v.groupby("light_dist_cm")
          if g["cosine_dist"].notna().sum() >= 2]

if len(groups) >= 2:
    H, p_kw = kruskal(*groups)
    print(f"\n── Kruskal-Wallis H-test ──────────────────────────────────────────────")
    print(f"  H = {H:.4f},  p = {p_kw:.4f}  "
          f"{'→ distributies significant verschillend (p<0.05)' if p_kw < 0.05 else '→ geen significant verschil'}")


# ============================================================================= 
# Final experiment summary 
# Overall system performance and failure breakdown. 
# =============================================================================

# ── Final summary ───────────────────────────────────────────────────────────
n_valid   = df[~df["detect_failed"]].shape[0]
n_failed  = int(df["detect_failed"].sum())
n_missing = int(df["skip_reason"].eq("testfoto_ontbreekt_jpg_jpeg").sum())
overall_frr = df["false_reject"].mean() * 100
avg_proc  = df["processing_s"].mean()

print(f"\n── Totaal ─────────────────────────────────────────────────────────────")
print(f"  Model: {MODEL}  |  Metric: {METRIC}  |  Threshold: {THRESHOLD}  |  CLAHE: {USE_CLAHE}")
print(f"  Vergelijkingen: {len(records)}  |  Valide: {n_valid}  |  Detect-fout: {n_failed}  |  Ontbrekend: {n_missing}")
print(f"  Overall FRR: {overall_frr:.1f}%")
if pd.notna(avg_proc):
    print(f"  Gem. verwerkingstijd: {avg_proc:.2f}s per vergelijking")
print(SEP)