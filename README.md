# NFR: Neural Fractal Reconstruction Engine
> *Breaking the Shannon Entropy Barrier with Deterministic Neural Context.*

![Version](https://img.shields.io/badge/version-1.0-blue) ![Python](https://img.shields.io/badge/python-3.9+-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

**NFR** is a next-generation data compression protocol that abandons traditional statistical frequency modeling (Huffman/LZ) in favor of **Neural Probability Estimation**. By treating data as a predictable sequence rather than a random stream, NFR maps files to high-precision arithmetic intervals based on the confidence of a neural network.

---

## 🚀 Key Features

*   **Neural Predictor Core:** Uses an LSTM (Long Short-Term Memory) network to predict the next byte based on context ($P(x_t | x_{t-k}...)$).
*   **Zero-Shot Adaptation:** The engine learns the specific grammar and patterns of *your* file during compression (Instance-Specific Overfitting). No pre-training required for basic usage.
*   **High-Precision Arithmetic Coding:** Custom 32-bit integer arithmetic coding kernel ensuring bit-perfect reconstruction without floating-point drift.
*   **Universal Format:** Can compress any byte stream (Text, Binary, DNA, Images).

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NFR-Engine.git
cd NFR-Engine

# Install dependencies
pip install torch numpy
```

*Note: CUDA is highly recommended for performance, but the engine runs on CPU automatically if CUDA is unavailable.*

## 📖 Usage

 The engine operates as a CLI tool via `daemon_nfr.py`.

### 1. Compress a File
To compress a file, use the `compress` command. The `--finetune` flag is recommended to let the model "learn" the file structure before compressing.

```bash
python daemon_nfr.py compress target.txt compressed.dmn --finetune --epochs 5
```
*   **Input:** `target.txt`
*   **Output:** `compressed.dmn` (Archive) + `target.txt.model` (The neural weights)

### 2. Decompress a File
To restore the original data, you need the `.dmn` archive and the corresponding model weights.

```bash
python daemon_nfr.py decompress compressed.dmn restored.txt --model target.txt.model
```
*   **Result:** `restored.txt` (Bit-perfect copy of source)

## 🧠 Architecture Principles

Traditional compression (ZIP, GZIP) relies on **Shannon Entropy**:
$$ L \approx -\log_2 P(x) $$
It assumes bytes are random variables defined by their global frequency.

**NFR** relies on **Contextual Entropy**:
$$ L \approx -\log_2 P(x_t \mid \text{Context}) $$
By understanding the *context* (e.g., "impor" -> "t"), the neural network assigns a near-100% probability to the next byte. In Arithmetic Coding, a high probability means the symbol consumes almost **0 bits** of storage space.

### Modules
1.  **`NeuralPredictor`**: PyTorch LSTM model predicting byte probabilities.
2.  **`DaemonArithmeticCoder`**: Maps probabilities to a single high-precision real number range.
3.  **`BitStream`**: Manages physical binary I/O and file headers.

## 📊 Benchmarks

See [BENCHMARKS](README_BENCHMARKS.md) for detailed performance reports on our current "Micro-Model" implementation.

| Metric | Traditional (Static) | NFR (Neural) |
| :--- | :--- | :--- |
| **Prediction** | Global Frequency | Local Context |
| **Adaptability** | Low (Fixed Algo) | High (Learns patterns) |
| **Limit** | Shannon Entropy | Kolmogorov Complexity |

## 📄 License

This project is open-source under the MIT License.

---
*Concept & Architecture by **Philippe-Antoine Robert**.*
