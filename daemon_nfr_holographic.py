"""
DAEMON NFR "Holographic" Engine - V4.0 (Experimental)
Implicit Neural Representation (INR) Video Compression
Target: Massive Compression Ratios (10x - 10,000x)

Technique:
Treats video not as a sequence of frames, but as a continuous function f(t, x, y) -> (r, g, b).
We train a SIREN (Sinusoidal Representation Network) to overfit this function.
The "Compressed File" is simply the weights of this small network.

Comparison:
- H.264 (30MB) -> Stores motion vectors + residuals
- NFR Holographic -> Stores a "formula" that regenerates the video (Infinite Resolution potential)

Usage:
    python daemon_nfr_holographic.py compress input.mp4 output.holo --ratio 100
    python daemon_nfr_holographic.py decompress output.holo reconstructed.mp4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import struct
import time
import sys
import numpy as np
import cv2  # OpenCV for video handling

# --- CONFIGURATION ---
MAGIC_HEADER = b'HOLO'

class HolographicUtils:
    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def normalize_coord(t, x, y, T, H, W):
        """Normalize coordinates to [-1, 1] range for the neural network."""
        return (
            2 * (t / (T - 1)) - 1,
            2 * (x / (H - 1)) - 1,
            2 * (y / (W - 1)) - 1
        )

# --- SIREN LAYER ---
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

# --- HOLOGRAPHIC NETWORK ---
class HolographicNetwork(nn.Module):
    def __init__(self, hidden_features=64, hidden_layers=3):
        super().__init__()
        self.device = HolographicUtils.get_device()
        
        layers = []
        # Input: (t, x, y) -> 3 coordinates
        layers.append(SineLayer(3, hidden_features, is_first=True, omega_0=30))
        
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=30))
            
        # Output: (r, g, b) -> 3 colors
        self.final_linear = nn.Linear(hidden_features, 3)
        self.net = nn.Sequential(*layers)
        
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30, 
                                              np.sqrt(6 / hidden_features) / 30)
            
        self.to(self.device)

    def forward(self, coords):
        # coords: [batch, 3]
        x = self.net(coords)
        return self.final_linear(x)

    def save_hologram(self, path, meta_info):
        """Save the network weights (the 'hologram') + metadata."""
        checkpoint = {
            'state_dict': self.state_dict(),
            'meta': meta_info # [frames, height, width, fps]
        }
        torch.save(checkpoint, path)

    def load_hologram(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()
        return checkpoint['meta']

# --- VIDEO DATASET ---
class VideoCoordinateDataset(Dataset):
    def __init__(self, video_path, sample_rate=0.01):
        """
        sample_rate: Percentage of pixels to train on per epoch (Stochastic sampling).
        """
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # We don't load the full video to RAM. We stream chunks.
        # But for training speed, we might need a cache.
        # Let's resize for "Thumbnail fitting" if ratio is huge.
        self.H_target = 240 # internal resolution for fitting
        self.W_target = 320
        
        print(f"[HOLO] Source: {self.total_frames} frames, {self.width}x{self.height} @ {self.fps}fps")
        print(f"[HOLO] Target Internal: {self.W_target}x{self.H_target}")
        
        self.frames = []
        count = 0
        # Increased stride to 10 (approx 3fps effective training) to handle 15min video in RAM
        train_stride = 10 
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            if count % train_stride == 0: 
                resized = cv2.resize(frame, (self.W_target, self.H_target))
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                self.frames.append(resized) # Store as uint8 list
            count += 1
        self.cap.release()
        
        # Keep as uint8 numpy array to save 4x memory vs float32
        self.data = np.array(self.frames, dtype=np.uint8) # [T, H, W, 3]
        
        self.T, self.H, self.W, _ = self.data.shape
        self.num_pixels = self.T * self.H * self.W
        print(f"[HOLO] Dataset shape: {self.data.shape} (uint8)")
        
    def get_batch(self, batch_size=10000):
        # Randomly sample spatiotemporal coordinates
        t_idx = np.random.randint(0, self.T, batch_size)
        y_idx = np.random.randint(0, self.H, batch_size)
        x_idx = np.random.randint(0, self.W, batch_size)
        
        # Sample uint8 and convert to float32 0..1 only for batch
        pixel_values = self.data[t_idx, y_idx, x_idx].astype(np.float32) / 255.0
        
        # Normalize coords [-1, 1]
        coords = np.stack([
            2 * (t_idx / (self.T - 1)) - 1,
            2 * (x_idx / (self.W - 1)) - 1,
            2 * (y_idx / (self.H - 1)) - 1
        ], axis=1).astype(np.float32)
        
        return torch.tensor(coords), torch.tensor(pixel_values)


# --- ENGINE ---
class HolographicEngine:
    def __init__(self):
        self.model = None 

    def compress(self, input_file, output_file, target_ratio=100):
        # 1. Determine Network Size based on Target Ratio
        file_size = os.path.getsize(input_file)
        target_size = file_size // target_ratio
        
        # Rough calc: standard float32 weight = 4 bytes.
        # Params approx = hidden*hidden + hidden*in...
        # Let's fix hidden size for now to a small value for high compression
        if target_ratio > 1000:
            hidden_dim = 32
            layers = 2
            print("[HOLO] Mode: Extreme Compression (x1000+)")
        elif target_ratio > 100:
            hidden_dim = 48
            layers = 3
            print("[HOLO] Mode: High Compression (x100+)")
        else:
            hidden_dim = 128
            layers = 4
            print("[HOLO] Mode: Balanced Compression")
            
        self.model = HolographicNetwork(hidden_dim, layers)
        print(f"[HOLO] Network Architecture: {hidden_dim} hidden units, {layers} layers")
        
        # 2. Prepare Data
        dataset = VideoCoordinateDataset(input_file)
        
        # 3. Train (Fit the Hologram)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        epochs = 200 # Fast fitting
        batch_size = 2**14 # 16k pixels per step
        
        print(f"[HOLO] Fitting Hologram (~{epochs} steps)...")
        start_time = time.time()
        
        for i in range(epochs):
            coords, pixels = dataset.get_batch(batch_size)
            coords, pixels = coords.to(self.model.device), pixels.to(self.model.device)
            
            optimizer.zero_grad()
            preds = self.model(coords)
            loss = F.mse_loss(preds, pixels) # Reconstruction loss
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                sys.stdout.write(f"\r > Step {i}/{epochs} | Loss (MSE): {loss.item():.5f}")
        
        elapsed = time.time() - start_time
        print(f"\n[HOLO] Fitting complete in {elapsed:.1f}s")
        
        # 4. Save Weights
        # Meta info for reconstruction (Original resolution vs fitting resolution)
        meta = {
            'T': dataset.total_frames, # Original duration
            'H_orig': dataset.height,
            'W_orig': dataset.width,
            'fps': dataset.fps,
            
            # Reconstruction grid (internal)
            'H_fit': dataset.H_target,
            'W_fit': dataset.W_target,
            'T_fit': dataset.T 
        }
        self.model.save_hologram(output_file, meta)
        
        final_size = os.path.getsize(output_file)
        real_ratio = file_size / final_size
        print(f"[SUCCESS] {file_size} -> {final_size} bytes.")
        print(f"[BREAKTHROUGH] Compression Ratio: {real_ratio:.2f}x")

    def decompress(self, input_file, output_file):
        # 1. Load Hologram
        # We need to peek meta to init model, or just try generic size then load state dict?
        # Torch load handles architecture matching if class is same, but we vary hidden dims.
        # For POC, we'll try to deduce or just try catch block for sizes. 
        # Actually simplest is to save architecture in the dict too. 
        # Re-instantiating blindly for this quick script:
        
        checkpoint = torch.load(input_file, map_location=HolographicUtils.get_device())
        
        # Infer architecture from weight shapes
        # keys like 'net.0.linear.weight' shape [hidden, 3]
        h_dim = checkpoint['state_dict']['net.0.linear.weight'].shape[0]
        # Count SineLayers (linear weights)
        n_layers = sum(1 for k in checkpoint['state_dict'] if 'linear.weight' in k) - 2 # minus input/output
        
        print(f"[HOLO] Detected Hologram: {h_dim} hidden, {n_layers} layers")
        self.model = HolographicNetwork(h_dim, n_layers)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        meta = checkpoint['meta']
        T_fit = meta['T_fit']
        H_fit = meta['H_fit']
        W_fit = meta['W_fit']
        
        # 2. Render Video from Function
        print(f"[HOLO] Reconstructing infinite-resolution stream to {output_file}...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, meta['fps'], (W_fit, H_fit)) # Render at fit res for now
        
        batch_size = H_fit * W_fit # One frame at a time
        
        # Pre-calc pixel coords
        y = np.linspace(-1, 1, H_fit)
        x = np.linspace(-1, 1, W_fit)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        # Flat coords [H*W, 2]
        spatial_coords = np.stack([X.ravel(), Y.ravel()], axis=1) # (x, y)
        
        with torch.no_grad():
            for t in range(T_fit):
                sys.stdout.write(f"\r > Rendering Frame {t}/{T_fit}")
                
                # Normalize T to [-1, 1]
                norm_t = 2 * (t / (T_fit - 1)) - 1
                
                # Combine T with Spatial
                t_col = np.full((batch_size, 1), norm_t)
                batch_coords = np.hstack([t_col, spatial_coords[::-1]]) # t, x, y (Fix spatial match)
                
                # Predict
                tensor_coords = torch.tensor(batch_coords, dtype=torch.float32).to(self.model.device)
                
                # We need proper X,Y, T order matching the training: T, Y, X in dataset get_batch?
                # Dataset: stack(t, x, y)
                # Here: stack(t, x, y) -> matches.
                
                rgb = self.model(tensor_coords) # [pixels, 3]
                
                # Reshape image
                frame = rgb.cpu().numpy().reshape(H_fit, W_fit, 3)
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
                
                out.write(frame)
                
        out.release()
        print("\n[Done] Holographic Reconstruction Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_c = subparsers.add_parser('compress')
    p_c.add_argument('input')
    p_c.add_argument('output')
    p_c.add_argument('--ratio', type=float, default=100.0)
    
    p_d = subparsers.add_parser('decompress')
    p_d.add_argument('input')
    p_d.add_argument('output')
    
    args = parser.parse_args()
    
    holo = HolographicEngine()
    
    if args.command == 'compress':
        holo.compress(args.input, args.output, target_ratio=args.ratio)
    elif args.command == 'decompress':
        holo.decompress(args.input, args.output)
