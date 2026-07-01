<div align="center">

# 🏥 P.R.I.S.M.
### Patient Recognition, Interaction and Status Monitoring

**An intelligent, real-time healthcare monitoring pipeline that fuses skeleton-based action recognition, multi-person tracking, and a conversational RAG-LLM interface for caregiving environments.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-0F9D58?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![ONNX](https://img.shields.io/badge/ONNXRuntime-1.15%2B-005CED?style=for-the-badge)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [Key Features](#-key-features)
4. [Technology Stack](#-technology-stack)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Pipeline Deep Dive](#-pipeline-deep-dive)
9. [Action Classes](#-action-classes)
10. [JSON Output Format](#-json-output-format)
11. [RAG-LLM Query Interface](#-rag-llm-query-interface)
12. [Model Weights](#-model-weights)
13. [Configuration Reference](#-configuration-reference)

---

## 🔍 Overview

**P.R.I.S.M.** (Patient Recognition, Interaction and Status Monitoring) is a full-stack, real-time action recognition and monitoring system designed for **healthcare and caregiving environments**. It processes video footage from a scene and performs:

- **Multi-person detection and tracking** using a custom ReID model running via ONNX and BoT-SORT tracker
- **Face recognition & role tagging** using InsightFace and reference images categorized locally in `dataset/patient/` and `dataset/caregiver/` directories
- **3D skeleton extraction** using Google's MediaPipe Pose Landmarker
- **Skeleton-based action recognition** using a fine-tuned **EfficientGCN-B0** model (15-channel, 3D, ONNX optimized) trained on **45 NTU RGB+D action classes**
- **Interaction detection** between patients and caregivers
- **Structured JSON event logging** with per-frame metadata
- **Natural language querying** over the event log using a RAG + LLM pipeline (LFM-2.5-Thinking via Ollama and ChromaDB)

The goal is to provide automated, unobtrusive observational intelligence in healthcare scenarios — flagging distress actions (falling, chest pain, staggering), recognising routine activities, and enabling caregivers or supervisors to query what happened at any given moment.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT VIDEO                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  YOLO  Detector     │  (person class only)
                  └──────────┬──────────┘
                             │ detections [x1,y1,x2,y2,conf]
                  ┌──────────▼──────────────────────────┐
                  │   MemoryEnhancedBoTSORT Tracker     │
                  │  ┌──────────┐  ┌─────────────────┐  │
                  │  │NSA Kalman│  │ Custom ReID CNN │  │
                  │  │  Filter  │  │   (ONNX)        │  │
                  │  └──────────┘  └─────────────────┘  │
                  │  ┌──────────┐  ┌─────────────────┐  │
                  │  │   GMC    │  │Appearance Buffer│  │
                  │  │ORB+RANSAC│  │(gallery/re-ID)  │  │
                  │  └──────────┘  └─────────────────┘  │
                  └──────────┬──────────────────────────┘
                             │ stable track IDs + bboxes
          ┌──────────────────┼──────────────────────┐
          │                  │                      │
 ┌────────▼────────┐ ┌───────▼────────┐  ┌──────────▼───────────┐
 │ MediaPipe Pose  │ │  Interaction   │  │  Bounding Box        │
 │  Landmarker     │ │   Detector     │  │  Expansion (×1.2)    │
 │ (33 landmarks,  │ │(IoU+dist+persist│ └───────────┬──────────┘
 │   x, y, z)      │ │   temporal)    │              │
 └────────┬────────┘ └───────┬────────┘              │
          │                  │ interaction pairs     │
          │ [33,3] landmarks │                       │
 ┌────────▼────────┐         │                       │
 │ MP→NTU25 Mapper │         │                       │
 │ + OneEuro Filter│         │                       │
 │  (stabilisation)│         │                       │
 └────────┬────────┘         │                       │
          │ [25,3] joints    │                       │
 ┌────────▼──────────────────▼───┐                   │
 │         SkeletonBuffer        │                   │
 │  (180-frame rolling window)   │                   │
 │ ┌────────────────────────────┐│                   │
 │ │ generate_features() → 15ch ││                   │
 │ │  joint(6) + velocity(6)    ││                   │
 │ │  + bone(3)  [T,25,3→15ch]  ││                   │
 │ └────────────────────────────┘│                   │
 └────────┬──────────────────────┘                   │
          │ [1,15,T,25,M] tensor                     │
 ┌────────▼──────────────────┐                       │
 │ EfficientGCN-B0 (ONNX)    │                       │
 │  ┌───────┐ ┌────────────┐ │                       │
 │  │Spatial│ │STJoint Att │ │                       │
 │  │ Graph │ │ + CrossSt  │ │                       │
 │  │ Conv  │ │ Attention  │ │                       │
 │  └───────┘ └────────────┘ │                       │
 │  45-class softmax output  │                       │
 └────────┬──────────────────┘                       │
          │ (code, label, confidence)                │
 ┌────────▼──────────────────────────────────────────▼───┐
 │             Visualization & JSON Logging              │
 │  • Bounding boxes, skeletons, action labels on video  │
 │  • Face recognition & role tagging (InsightFace)      │
 │  • ActionEventLogger → structured JSON per second     │
 └────────┬──────────────────────────────────────────────┘
          │ output_action.mp4 + action_log.json
 ┌────────▼────────────────────────┐
 │     RAG + LLM Query Layer       │
 │  nest_rag.py (ChromaDB)         │
 │  SentenceTransformers (MiniLM)  │
 │  + LFM-2.5-Thinking via Ollama  │
 │  → Natural language Q&A         │
 └─────────────────────────────────┘
```

---

## ✨ Key Features

### 🎯 Multi-Person Detection & Tracking
- Person detection (class 0 only) with configurable confidence threshold
- **MemoryEnhancedBoTSORT** — a custom extension of BoT-SORT with:
  - **NSA Kalman Filter** — Noise Scale Adaptive; dynamically scales measurement trust based on detection confidence
  - **GMC (Global Motion Compensation)** — uses ORB features + RANSAC to compensate for camera motion
  - **Custom ReID backbone (ONNX-optimized)** (`ImprovedDeepSortWideResNet`) — a squeeze-excitation ResNet with GeM pooling, trained to 128-dim appearance embeddings
  - **Appearance gallery** per track for robust re-identification after occlusion

### 🦴 3D Skeleton Extraction
- **MediaPipe Pose Landmarker Heavy** (30M+ params) extracts 33 landmarks per person with x, y, z
- **MP→NTU25 mapping** remaps MediaPipe's 33 landmarks to the NTU RGB+D 25-joint skeleton format
- Z-depth is scaled by crop width to match NTU's depth-proportional convention
- **OneEuro filtering** smooths skeleton sequences per track to eliminate jitter
- `snap_missing_to_spine` snaps undetected joints to the spine, preventing zero-valued features

### 🧠 Action Recognition — EfficientGCN-B0 (3D, ONNX)
- Fine-tuned on **45 NTU RGB+D action classes** covering patient distress, daily activities, caregiver actions, and two-person interactions
- Now optimized as a **15-channel ONNX model** for faster and hardware-agnostic CPU/GPU inference
- **Three-stream feature extraction** from skeleton sequences:
  - **Joint stream** (6ch): absolute (x,y,z) + spine-relative (x,y,z)
  - **Velocity stream** (6ch): fast Δ (2-frame) + slow Δ (1-frame) per axis
  - **Bone stream** (3ch): bone delta vector (dx, dy, dz)
- **Cross-stream attention** soft-fuses the three streams before final classification
- **ST-Joint Attention** applies spatial (per joint) and temporal attention simultaneously
- **PredictionCache** holds predictions for N frames to avoid redundant inference every frame

### 🤝 Interaction Detection
- Detects when two tracked individuals are in proximity using:
  - IoU overlap threshold between bounding boxes
  - Centre-to-centre pixel distance (with dynamic scaling by bounding box diagonal)
  - **Temporal persistence** gate — N consecutive frames required before interaction is declared active
- Separate skeleton buffers for `(idA, idB)` pairs enable *joint* action recognition for the interacting individuals

### 📋 Structured JSON Logging
- **ActionEventLogger** captures a full event log in JSON, sampled once per second:
  - Session metadata (video path, FPS, resolution, model info, device)
  - Per-frame: bounding boxes, action codes, labels, confidence, category, interactions (IoU, pixel distance, skeleton spine distance)
  - Summary: action distribution, unique track IDs, total recognitions
- 45 classes are taxonomised into four semantic categories: `patient_specific`, `caregiver_specific`, `interaction_based`, `common`

### 👤 Face Recognition & Role Tagging
- Uses **InsightFace** for robust face detection and identity matching
- Reference images are organized into specific subdirectories (`dataset/patient/` and `dataset/caregiver/`)
- Automatically tags identities and assigns roles persistently across tracks

### 🗣 RAG-LLM Query Interface
- `nest_rag.py` ingests the JSON log into a **ChromaDB** vector store
- Uses `all-MiniLM-L6-v2` (SentenceTransformers) for semantic retrieval over the event log
- Interactive CLI powered by `lfm2.5-thinking` (Ollama) to ask natural-language questions
- Automatically generates a comprehensive video summary upon ingestion

---

## 🔧 Technology Stack

| Component | Technology |
|---|---|
| Person Detection | YOLO (Ultralytics) / ONNX |
| Multi-Object Tracking | BoT-SORT (custom — NSA Kalman + GMC + ReID via ONNX) |
| Face Recognition | InsightFace |
| Pose Estimation | MediaPipe Pose Landmarker Heavy |
| Skeleton Smoothing | One Euro Filter |
| Action Recognition | EfficientGCN-B0 (15-ch 3D ONNX, fine-tuned on NTU RGB+D) |
| Video Processing | OpenCV 4.8+ |
| Semantic Retrieval | ChromaDB, SentenceTransformers `all-MiniLM-L6-v2` |
| LLM Backend | LFM-2.5-Thinking via Ollama |

---

## 📁 Project Structure

```
NEST/
├── mediapipe_inference.py       # Main inference pipeline (detection → tracking →
│                                #   pose → action recognition → face id → logging)
├── finalfacerecognition.py      # Custom BoT-SORT implementation with face recognition
│                                #   (InsightFace, NSA Kalman, GMC, ONNX ReID backbone)
├── nest_rag.py                  # RAG pipeline: JSON → ChromaDB → LLM Q&A
├── requirements.txt             # Python dependencies
│
├── dataset/                     # Directory for face recognition reference images
│   ├── patient/                 # Patient reference images
│   └── caregiver/               # Caregiver reference images
│
├── nest_chroma_db/              # Persistent vector store for RAG
│
├── best_efficientgcn_b0_(2).onnx# 15-channel ONNX Action recognition model
├── best_model.onnx              # ONNX ReID model weights (BoT-SORT)
├── pose_landmarker_heavy.task   # MediaPipe pose model
├── yolo26s.onnx / .pt           # YOLO detection model
│
├── output_action.mp4            # Annotated output video (generated)
└── output_action_action_log.json# Structured action event log (generated)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.com/) installed for the LLM query interface

### 1. Clone the repository

```bash
git clone https://github.com/Manab-Bairagi/N.E.S.T.-Network-for-Evaluating-Status-Tracking.git
cd N.E.S.T.-Network-for-Evaluating-Status-Tracking
```

### 2. Install dependencies

> **Note:** InsightFace requires C++ Build Tools (Microsoft Visual C++ 14.0 or greater) to compile on Windows. Ensure it is installed before proceeding.
If running CPU only, you do not need to install CUDA. ONNXRuntime utilizes CPU Execution Provider natively.

```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe pose model

```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    'pose_landmarker_heavy.task'
)
print('Downloaded pose_landmarker_heavy.task')
"
```

### 4. Pull the LFM-2.5-Thinking model via Ollama (for RAG)

```bash
ollama pull lfm2.5-thinking:latest
```

---

## 🚀 Usage

### Step 1 — Run the Inference Pipeline

> **Note:** For face recognition, place your reference images in the correct subdirectories: `dataset/patient/` and `dataset/caregiver/` before running.

```bash
python mediapipe_inference.py
```

> Edit the bottom of `mediapipe_inference.py` (Cell 11) to point `video_path` at your input video before running.

**Key parameters you can tune in Cell 11:**

```python
process_video_with_action_recognition(
    video_path      = 'inpu.mp4',         # Input video
    output_path     = 'output_action.mp4',# Annotated output
    reid_model_path = 'best_model.onnx',  # ONNX ReID weights
    conf_threshold  = 0.25,               # YOLO detection confidence
    max_frames      = None,               # None = process all frames
    buffer_len      = 90,                # Temporal window (frames)
    bbox_scale      = 1.2,                # Crop expansion for pose
    iou_thresh      = 0.20,               # Interaction IoU threshold
    dist_thresh     = 150,                # Interaction distance (px)
    persist_frames  = 10,                 # Frames to confirm interaction
    inference_every = 8,                  # Run model every N frames
    dataset_dir     = 'dataset',          # Reference image parent directory
)
```

**Outputs:**
- `output_action.mp4` — annotated video with bounding boxes, skeletons, action labels, and interaction overlays
- `output_action_action_log.json` — structured JSON log

---

### Step 2 — Launch the RAG Pipeline

The `nest_rag.py` pipeline ingests the JSON log into a ChromaDB vector store, generates a video summary, and launches an interactive Q&A session.

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Launch the RAG pipeline
python nest_rag.py --json output_action_action_log.json
```

**Example interaction:**
```
>> Did anyone fall down during the session?
🧠 Answer:
At 2026-03-25T10:00:15, a patient was observed falling down, 
followed by a caregiver providing support shortly after.
--------------------------------------------------

>> What actions did the caregiver perform?
🧠 Answer:
The caregiver was observed patting someone on the back 
and supporting somebody at multiple timestamps during the session.
--------------------------------------------------

>> exit
```

---

## 🔬 Pipeline Deep Dive

### Feature Extraction — Three-Stream Skeleton Encoding (15-Channel)

Given a 90-frame skeleton sequence `[T, 25, 3]` per person, three feature streams are computed:

| Stream | Channels | Description |
|---|---|---|
| **Joint** | 6 | Absolute (x,y,z) normalized coords + Spine-relative (x,y,z) offsets |
| **Velocity** | 6 | Fast Δ (frame t+2 − t) + Slow Δ (frame t+1 − t), per axis |
| **Bone** | 3 | Bone delta vector (dx, dy, dz) |

All streams are concatenated into a `[15, T, 25, M]` tensor where M is the number of persons (1 for solo, 2 for an interacting pair). *Note: The bone length feature was removed in this version because MediaPipe depth scaling made it artificially stretch/shrink with distance, polluting the signal.*

### Tracker — MemoryEnhancedBoTSORT

The tracker combines four key components:

1. **NSA Kalman Filter** — adapts measurement covariance R inversely to detection confidence. Low-confidence detections (occluded, blurry) are treated with high noise, trusting the motion prediction more.

2. **GMC (Global Motion Compensation)** — detects ORB keypoints on a downscaled grayscale frame, matches them across consecutive frames, estimates an affine transform via RANSAC, and compensates all track positions accordingly.

3. **Custom ONNX ReID backbone** — `ImprovedDeepSortWideResNet` ported to ONNX with:
   - Efficient stem (32→32→MaxPool)
   - 3 residual stages (32→64→128) with Squeeze-and-Excitation
   - Generalized Mean Pooling (GeM) for robust global descriptor
   - BatchNorm neck + 128-dim L2-normalized embeddings

4. **Fused cost matrix** — matching uses `λ_IoU × IoU_cost + λ_ReID × ReID_cost`, prioritising spatial consistency while using appearance for disambiguation.

---

## 🎭 Action Classes

The model recognises **45 NTU RGB+D actions**, grouped into four healthcare-relevant categories:

### 🧑‍⚕️ Patient-Specific (25 actions)
`drink water` · `eat meal` · `brush teeth` · `reading` · `writing` · `put on glasses` · `take off glasses` · `jump up` · `sneeze/cough` · **`staggering`** · **`falling down`** · **`headache`** · **`chest pain`** · **`back pain`** · **`neck pain`** · **`nausea/vomiting`** · `fan self` · `squat down` · `apply cream on face` · `apply cream on hand` · `put object into bag` · `take object out of bag` · `open a box` · `move heavy objects` · `yawn`

### 👨‍⚕️ Caregiver-Specific (5 actions)
`pat on back` · `giving object` · `carry object` · `follow` · **`support somebody`**

### 🤝 Interaction-Based (10 actions)
`phone call` · `punch/slap` · `hugging` · `shaking hands` · `walking towards` · `walking apart` · `hit with object` · `wield knife` · `knock over` · `grab stuff`

### 🔄 Common (5 actions)
`drop` · `pick up` · `sit down` · `stand up` · `point finger`

> **Bold** = distress or clinically significant actions

---

## 📊 JSON Output Format

`output_action_action_log.json` structure:

```json
{
  "session": {
    "video_path": "Deepark.mp4",
    "output_video_path": "output_action.mp4",
    "processed_at": "2026-04-18T21:00:00",
    "fps": 25,
    "resolution": { "width": 1920, "height": 1080 },
    "model": "EfficientGCN-B0 (3D, NTU 45-class, 15-ch)",
    "device": "cuda",
    "num_action_classes": 45
  },
  "action_categories": {
    "patient_specific": [ { "code": "A043", "label": "falling down" }, "..." ],
    "caregiver_specific": [ "..." ],
    "interaction_based": [ "..." ],
    "common": [ "..." ]
  },
  "summary": {
    "total_frames_processed": 150,
    "unique_track_ids": [0, 1, 2],
    "total_recognitions": 87,
    "action_distribution": { "falling down": 12, "support somebody": 9 }
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "tracks": {
        "0": { "bbox": [120.0, 80.0, 320.0, 540.0], "has_skeleton": true }
      },
      "actions": {
        "0": { "code": "A043", "label": "falling down", "confidence": 0.82, "category": "patient_specific" }
      },
      "interactions": [
        {
          "track_a": 0, "track_b": 1, "interaction_active": true,
          "distance_px": 134.5, "iou": 0.21,
          "skeleton_spine_distance_px": 118.3
        }
      ]
    }
  ]
}
```

---

## 🗣 RAG-LLM Query Interface

The query system follows a **Retrieval-Augmented Generation (RAG)** pattern:

```
User Question
     │
     ▼
SentenceTransformer (all-MiniLM-L6-v2)
     │  encode query → 384-dim embedding
     ▼
Cosine Similarity over all event embeddings
     │  top-k most relevant events
     ▼
Prompt construction:
  "At <timestamp>, a <role> was performing <action>." × k
     │
     ▼
LFM-2.5-Thinking (via Ollama local inference)
     │  natural language answer
     ▼
Answer printed to terminal
```

---

## 🔑 Model Weights

| File | Description |
|---|---|
| `best_efficientgcn_b0_(2).onnx` | EfficientGCN-B0 15-channel action recognition weights (ONNX) |
| `best_model.onnx` | ReID backbone (BoT-SORT appearance model) in ONNX format |
| `pose_landmarker_heavy.task` | MediaPipe Pose Landmarker Heavy |
| `yolo*.onnx` or `yolo*.pt` | YOLO person detection models |

> All weight files should be placed in the **project root directory** alongside the scripts.

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `conf_threshold` | `0.25` | YOLO detection confidence cutoff |
| `buffer_len` | `180` | Skeleton buffer length (frames) used as temporal window |
| `bbox_scale` | `1.2` | Bounding box expansion factor for MediaPipe crop |
| `iou_thresh` | `0.20` | Minimum IoU to consider two persons as interacting |
| `dist_thresh` | `150` | Maximum centre-to-centre distance (px) for interaction |
| `persist_frames` | `10` | Consecutive frames required to confirm an interaction |
| `inference_every` | `8` | Run action model every N frames (uses cache in between) |
| `CONFIDENCE_THRESHOLD` | `0.10` | Minimum softmax confidence to report a prediction |
| `MAX_FRAMES` | `90` | Model's temporal window (training-fixed) |
| `NUM_JOINTS` | `25` | NTU 25-joint skeleton |
| `DROPOUT` | `0.30` | EfficientGCN-B0 dropout rate |

---

## 👥 Team

**P.R.I.S.M.** — *Patient Recognition,Interaction and status monitoring*

Built as a healthcare AI monitoring research project.

---

<div align="center">

*If you find this project useful, consider giving it a ⭐*

</div>
