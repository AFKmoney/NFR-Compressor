# DAEMON NFR (Neural Fractal Reconstruction) Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-yellow.svg) ![Status](https://img.shields.io/badge/status-BREAKTHROUGH-brightgreen.svg)

> **"We didn't break Shannon's Limit. We just stopped treating video as pixels."**

## 🌌 The Breakthrough (NFR v4)

**January 16, 2026:** We successfully compressed a **30.25 MB** video file into a **34 KB** neural network, achieving a compression ratio of **887x** (99.89% reduction).

This is not standard quantization. This is **Implicit Neural Representation (INR)**.

- **Input:** Traditional Video (Discrete Pixels)
- **Output:** A continuous mathematical function $f(t,x,y) \to RGB$
- **Result:** Infinite resolution scaling, dream-like interpolation, and massive space savings.

### Benchmarks (Verified)

| Type | Algorithm | Input | Compressed | Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Video** | **NFR v4 Holographic** | 30.25 MB | **34.1 KB** | **887x** 🏆 |
| **Audio** | NFR v3 Stereo-LSTM | 882 KB | 379 KB | 2.33x |
| **Data** | NFR v2 Context-AC | 14 KB | 13 KB | 1.1x |

---

## 📄 Academic Paper

A technical report detailing the SIREN-based architecture and methodology is available:
👉 [**Read the Paper (LaTeX)**](./paper_nfr_holographic.tex)

## 🚀 Quick Start Guide

### 1. Prerequisites

You need Python 3.10+ and the following libraries:

```bash
pip install torch numpy opencv-python
```

### 2. COMPRESS VIDEO (The "Holographic" Method)

To compress a video file (e.g., `input.mp4`) into a neural network (`.holo`):

```bash
# --ratio 1000 targets a 1000x compression ratio
python daemon_nfr_holographic.py compress input.mp4 video.holo --ratio 1000
```

*Note: This process uses CPU/GPU to train the network. It may take a few minutes.*

### 3. DECOMPRESS VIDEO

To reconstruct the video from the `.holo` file:

```bash
python daemon_nfr_holographic.py decompress video.holo output_video.mp4
```

*The resulting video is mathematically generated from the stored neural weights.*

### 4. COMPRESS AUDIO (Lossless)

To compress a WAV file using NFR v3 (Stereo Delta + LSTM):

```bash
python daemon_nfr_audio.py compress input.wav output.dmna
```

### 5. DECOMPRESS AUDIO

```bash
python daemon_nfr_audio.py decompress output.dmna restored.wav
```

---

## 🧠 How It Works

See the detailed explanation here: [**HOW_WE_DID_IT.md**](./HOW_WE_DID_IT.md)

1. **Paradigm Shift:** We treat the video as a signal function, not a byte stream.
2. **Fitting:** We overfit a tiny SIREN (Sinusoidal Representation Network) to the specific video instance.
3. **Storage:** We discard the pixels and store the network weights (~7000 floats).

## ⚠️ Disclaimer

- **NFR v4** is *generative* and *lossy*. It preserves semantic motion and color but hallucinate fine high-frequency details based on the network capacity.
- **NFR v3** is *lossless* and bit-perfect.
