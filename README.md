# Multimodal Ambivalence/Hesitancy Recognition in Videos

Solution for the **10th ABAW Competition - Ambivalence/Hesitancy (A/H) Video Recognition Challenge** at CVPR 2026, using the [BAH Dataset](https://arxiv.org/pdf/2505.19328).

This repository implements a multimodal approach combining **visual** (Action Units via Py-Feat), **audio** (Wav2Vec 2.0 embeddings), and **text** (BERT) features with multiple fusion strategies for binary A/H classification at the video level.



## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- FFmpeg (required for audio extraction)

## Installation

```bash
# Core dependencies
pip install numpy==1.26.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install pandas
pip install tqdm

# Feature extraction
pip install py-feat
pip install "scipy<1.14"
pip install mediapipe
pip install opencv-python
pip install librosa
pip install soundfile

# Analysis
pip install wordcloud
pip install pyyaml
```

Or install all at once:

```bash
pip install numpy==1.26.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers scikit-learn xgboost matplotlib pandas tqdm py-feat "scipy<1.14" mediapipe opencv-python librosa soundfile wordcloud pyyaml
```

> **Note:** Ensure FFmpeg is installed and available in your system PATH for audio feature extraction.

## Dataset Setup

1. Obtain the BAH dataset by following the instructions at [https://github.com/sbelharbi/bah-dataset](https://github.com/sbelharbi/bah-dataset).
2. Place the dataset contents inside the `data/` directory following the structure shown above.
3. The dataset should include `Videos/`, `cropped-aligned-faces/`, and `split/` directories with the train/val/test split files.

## Notebook Execution Order

All notebooks are located in `jupyter-notebooks/`. Run them in the following order:

### Phase 1: Data Exploration

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `analysis.ipynb` | Exploratory data analysis: class distribution, annotation statistics, transcript word clouds, and cue frequency analysis. |

### Phase 2: Feature Extraction

| # | Notebook | Description |
|---|----------|-------------|
| 2 | `extract_all_aus.ipynb` | Extract 20 Action Unit intensities from all videos using Py-Feat (GPU). Outputs `.npy` files to `data/au_features/`. |
| 3 | `extract_audio_features.ipynb` | Extract Wav2Vec 2.0 audio embeddings from all videos via FFmpeg + HuggingFace. Outputs `.npy` files to `data/audio_features/`. |
| 4 | `extract_blendshapes.ipynb` | Extract 52 MediaPipe blendshapes and 478-point face mesh landmarks. Outputs `.npy` files to `data/blendshape_features/` and `data/mesh_features/`. |

### Phase 3: Unimodal Models

| # | Notebook | Description |
|---|----------|-------------|
| 5 | `visual_model.ipynb` | Visual-only A/H classifiers using AUs: XGBoost on temporal stats, raw BiLSTM, and sliding-window BiLSTM+Attention. |
| 6 | `visual_model_blendshapes.ipynb` | Visual-only classifiers using blendshapes: XGBoost and BiLSTM+Attention. |
| 7 | `audio_model.ipynb` | Audio-only classifiers using Wav2Vec 2.0 embeddings: XGBoost with PCA and BiLSTM+Attention. |
| 8 | `text.ipynb` | Text-only binary classifier using BERT embeddings on video transcripts. |

### Phase 4: Analysis & Comparison

| # | Notebook | Description |
|---|----------|-------------|
| 9 | `visual.ipynb` | Exploratory AU extraction and visual analysis of AU patterns for selected participants. |
| 10 | `compare_au_vs_blendshapes.ipynb` | Side-by-side comparison of AU vs blendshape features for selected participants. |
| 11 | `statistical_analysis_au_vs_bs.ipynb` | Large-scale statistical comparison (Mann-Whitney U) of AUs vs blendshapes for A/H discrimination across all videos. |

### Phase 5: Multimodal Fusion

| # | Notebook | Description |
|---|----------|-------------|
| 12 | `multimodal_fusion.ipynb` | Multimodal fusion (visual AUs + audio + text) with 3 strategies: (A) concatenation, (B) divergence-based, (C) combined. |
| 13 | `multimodal_fusion_windowed.ipynb` | Same 3 fusion strategies but using sliding-window visual features instead of raw AUs. |
| 14 | `fusion_optimized_visual.ipynb` | Fusion B tested with 3 visual representations: AUs only (20d), blendshapes only (52d), and combined AU+BS (24d). |

### Phase 6: Inference

| # | Notebook | Description |
|---|----------|-------------|
| 15 | `inference_private_test.ipynb` | End-to-end inference pipeline for the ABAW private test set. Runs feature extraction + prediction using the best fusion model. Outputs `trial-0.txt`. |

## Key Results (Test Set Macro F1)

| Model                                              | Test F1 |
|----------------------------------------------------|---------|
| Text-only (BERT)                                   | 0.5837  |
| Audio-only (BiLSTM)                                | 0.6141  |
| Visual-only AU (Window BiLSTM)                     | 0.5436  |
| Visual-only BS (XGBoost)                           | 0.5712  |
| Fusion A - Implicited (raw AU + Text + Audio)      | 0.6604  |
| Fusion B - Divergence (raw AU + Text + Audio)      | 0.6808  |
| Fusion C - Combined (raw AU + Text + Audio)        | 0.6766  |
| Fusion C - Implicited (windowed AU + Text + Audio) | 0.6540  |
| Fusion C - Divergence (windowed AU + Text + Audio) | 0.6602  |
| Fusion C - Combined (windowed AU + Text + Audio)   | 0.6650  |

## Trained Model Checkpoints

The following model weights are saved in `data/` after training:

- `best_visual_lstm.pt` / `best_visual_window_lstm.pt` - Visual AU models
- `best_bs_lstm.pt` - Visual blendshape model
- `best_audio_lstm.pt` - Audio model
- `best_fusion_A.pt`, `best_fusion_B.pt`, `best_fusion_C.pt` - Multimodal fusion (raw AU)
- `best_win_fusion_A.pt`, `best_win_fusion_B.pt`, `best_win_fusion_C.pt` - Multimodal fusion (windowed)
- `best_fusionB_au.pt`, `best_fusionB_bs.pt`, `best_fusionB_combined.pt` - Optimized Fusion B variants
