# Face Recognition Experiment with DeepFace

## Overview

This project evaluates how lighting distance affects face recognition performance using the **DeepFace** library. The system compares facial images taken under different lighting conditions and measures how recognition accuracy changes as illumination decreases.

The experiment uses multiple statistical methods to analyze the relationship between lighting distance and model performance.

---

## Core Idea

The pipeline compares:

* A **reference image** (baseline face per subject)
* Multiple **test images** taken at different light-source distances (15–135 cm)

For each comparison, the system determines:

* Whether both images belong to the same person
* How similar the faces are (cosine distance)
* How reliable the prediction is under different lighting conditions

---

## Technologies Used

* Python 3.9+
* DeepFace (face recognition framework)
* TensorFlow (backend for DeepFace)
* OpenCV (image preprocessing)
* Pandas (data storage and analysis)
* SciPy (statistical analysis)

---

## Installation

### 1. Install DeepFace and dependencies

The project is installed via pip:

```bash
pip install deepface
```

This automatically installs required dependencies such as:

* TensorFlow
* OpenCV
* NumPy
* Keras (legacy compatibility used)

---

### 2. Optional system setup

To reduce logs and improve stability:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

---

## Dataset Structure

The dataset must follow this structure:

```
data/
 ├── s01/
 │    ├── reference.jpg
 │    ├── 15cm.jpg
 │    ├── 30cm.jpg
 │    ├── ...
 ├── s02/
 │    ├── reference.jpg
 │    ├── 15cm.jpg
 │    ├── ...
```

Each subject contains:

* 1 reference image
* Multiple test images per lighting distance

Supported formats:

* .jpg
* .jpeg

---

## How the System Works

### 1. Image Loading

The system searches for images using:

* `.jpg`
* `.jpeg`

If no image exists, the comparison is skipped.

---

### 2. Preprocessing (CLAHE)

Optional image enhancement is applied:

* Converts image to LAB color space
* Enhances local contrast using CLAHE
* Improves visibility in low-light conditions

This step increases robustness under poor lighting.

---

### 3. Face Verification (DeepFace)

For each image pair:

DeepFace performs:

1. Face detection
2. Face alignment
3. Feature extraction
4. Vector comparison

Model used:

* VGG-Face

Distance metric:

* Cosine similarity

Output:

* Distance score
* Match / no match decision
* Confidence estimate

---

### 4. Lighting Model

Lighting intensity is calculated using the inverse square law:

[
I \propto \frac{1}{d^2}
]

Where:

* `d` = light distance in cm
* 90 cm is used as reference point (1.0 intensity)

---

### 5. Experiment Loop

For each subject:

* Load reference image
* Iterate over all lighting distances
* Compare against test images
* Store results in structured dataset

Each row contains:

* Subject ID
* Lighting distance
* Cosine distance
* Confidence
* Processing time
* Verification result
* Failure flags

---

## Output Files

### 1. Raw results

```
results_clahe_15to135.csv
```

Contains all individual comparisons.

---

### 2. Summary statistics

```
results_summary_clahe_15to135.csv
```

Aggregated metrics per lighting distance:

* Mean / median cosine distance
* Standard deviation
* False Reject Rate (FRR)
* Confidence averages
* Processing time

---

## Statistical Analysis

### 1. Correlation Tests

Used to measure relationship between lighting and performance:

* Pearson correlation (linear relationship)
* Spearman correlation (rank-based)
* Kendall correlation (robust ranking)
* Illuminance vs cosine distance

---

### 2. Kruskal-Wallis Test

Non-parametric test used to determine:

* Whether different lighting conditions produce significantly different results

---

## Key Metrics

### Cosine Distance

* Lower = more similar faces
* Threshold determines match decision

### False Reject Rate (FRR)

* Percentage of genuine matches incorrectly rejected

### Confidence Score

* Model certainty in match decision

### Processing Time

* Time required per face comparison

---

## Configuration Options

Inside the script:

```python
MODEL = "VGG-Face"
METRIC = "cosine"
THRESHOLD = 0.40
USE_CLAHE = True
```

You can modify:

* Model type (e.g. ArcFace, Facenet)
* Similarity threshold
* Image preprocessing

---

## Running the Experiment

Execute the script:

```bash
python pipeline.py
```

During execution:

* Progress bar shows real-time status
* Missing images are skipped automatically
* Errors are logged without stopping the pipeline

---

## Output Interpretation

If lighting distance increases:

* Cosine distance typically increases
* Recognition accuracy decreases
* False rejection rate increases

If CLAHE is enabled:

* Performance improves in low-light conditions
* Variance between conditions may reduce

---

## Repository Reference

DeepFace framework:
https://github.com/serengil/deepface

---

## Purpose of This Project

To evaluate:
* Impact of lighting distance on recognition stability
* Statistical significance of lighting effects on biometric systems

---
