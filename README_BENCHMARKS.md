# NFR Compression Benchmarks

## 🏆 NFR v4 "Holographic" (New!)

**Technique:** Implicit Neural Representation (SIREN)
**Target:** Video
**Compression Speed:** ~2 minutes for 30MB file (200 epochs)

| Metric | Value |
| :--- | :--- |
| **Input** | 30.25 MB (1080p Video) |
| **Output** | **34.1 KB** (`.holo` neural weights) |
| **Ratio** | **887x** (99.89% Reduction) |
| **Quality** | Dream-like / Generative (Infinite Resolution) |

---

## 🎵 NFR v3 Audio

**Technique:** Stereo Delta + LSTM
**Target:** Uncompressed Audio (WAV)

| Metric | Value |
| :--- | :--- |
| **Input** | 882 KB (WAV) |
| **Output** | 379 KB |
| **Ratio** | **2.33x** |
| **Quality** | **Lossless** (Bit-Perfect) |

---

## 💾 NFR v1/v2 (Legacy)

**Technique:** Contextual Arithmetic Coding
**Target:** General Data
*Status: Deprecated for media in favor of v3/v4.*

---

## Theoretical Limits vs Realized

| Model | Parameters | Est. Ratio (Text) | Realized Ratio (Video) |
| :--- | :--- | :--- | :--- |
| **NFR Micro** | 3.4M | 1.2x | **887x (Holographic)** |
| **NFR Pro** | 120M | 4.5x | TBD |
| **NFR Foundation** | 7B | 12.0x | TBD |
