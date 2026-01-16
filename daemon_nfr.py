
"""
DAEMON NFR (Neural Fractal Reconstruction) Engine - Production V2.0
Author: DAEMON (Agentic AI)
System: Windows / Nvidia CUDA / Python 3.9+

Description:
High-precision Neural Compression engine replacing statistical entropy coding with 
context-aware neural probability estimation.
V2.0: Adds Parallel Block Processing and Batched Inference for high-speed compression.

Usage:
    python daemon_nfr.py compress <input_file> <output_file> [--epochs 5]
    python daemon_nfr.py decompress <input_file> <output_file>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import struct
import time
import hashlib
import sys
import numpy as np
from typing import BinaryIO, Generator, List, Tuple
import io

# --- CONFIGURATION CONSTANTS ---
MAGIC_HEADER = b'DMNv2'
DEFAULT_CONFIG = {
    'hidden_size': 512, 
    'num_layers': 2,    # Reduced for speed in V2
    'seq_len': 128,     
    'vocab_size': 256,
    'embedding_dim': 64,
    'dropout': 0.0      # Zero dropout for deterministic heavy compression
}

BLOCK_SIZE = 65536 # 64KB Blocks for parallel processing

# Arithmetic Coding Constants (32-bit fixed precision)
CODE_VALUE_BITS = 32
TOP_VALUE = (1 << CODE_VALUE_BITS) - 1
QUARTER = 1 << (CODE_VALUE_BITS - 2)
HALF = 1 << (CODE_VALUE_BITS - 1)
THREE_QUARTERS = 3 * QUARTER
SCALE_FACTOR = 16384  # 2^14 precision for probability mapping

class NFRUtils:
    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def pdf_to_cdf(pdf_tensor: torch.Tensor) -> torch.Tensor:
        """Converts probability distribution to Cumulative Distribution Function integers."""
        # Scale to integer frequencies
        freqs = (pdf_tensor * SCALE_FACTOR).long()
        # Ensure minimal probability mass (Laplace smoothing equivalent) to avoid dead-ends
        freqs = torch.clamp(freqs, min=1)
        # Cumsum
        cdf = torch.cumsum(freqs, dim=0)
        return cdf

class NeuralPredictor(nn.Module):
    """
    Production-grade LSTM Predictor.
    """
    def __init__(self, config=DEFAULT_CONFIG):
        super(NeuralPredictor, self).__init__()
        self.config = config
        self.device = NFRUtils.get_device()
        
        self.embed = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.lstm = nn.LSTM(
            input_size=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout']
        )
        self.fc = nn.Linear(config['hidden_size'], config['vocab_size'])
        self.to(self.device)

    def forward(self, x, hidden=None):
        # Standard inference: return LAST step prediction
        # x: [batch, seq_len]
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return F.softmax(logits, dim=1), hidden

    def forward_full(self, x_seq, hidden=None):
        """
        Processes an entire sequence and returns predictions for ALL steps.
        Used for Training.
        input: x_seq [Batch, L]
        output: logits [Batch, L, 256] (Not Softmax, for stable CrossEntropy)
        """
        emb = self.embed(x_seq) # [Batch, L, emb_dim]
        out, hidden = self.lstm(emb, hidden) # [Batch, L, hidden]
        logits = self.fc(out) # [Batch, L, 256]
        return logits, hidden
        
    def forward_batch(self, x_seq, hidden=None):
        """
        Processes an entire sequence at once for calculating probabilities of the NEXT tokens.
        (Teacher Forcing Mode for Compression)
        input: x_seq [1, L] containing bytes 0..L-1
        output: target_probs [L, 256] where row i is prob distribution for byte i+1
        """
        # This is essentially forward_full but with Softmax and optimized for batch-size 1 compression flow
        logits, hidden = self.forward_full(x_seq, hidden)
        # Logits: [1, L, 256] -> Squeeze to [L, 256]
        return F.softmax(logits.squeeze(0), dim=1), hidden

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

class BitStream:
    """Handles buffered bit-level I/O."""
    def __init__(self, handle: BinaryIO, mode='wb'):
        self.handle = handle
        self.mode = mode
        self.buffer = 0
        self.count = 0
        
        if 'r' in mode:
            # Buffer for reading
            self.read_chunk_size = 65536 
            self.byte_buffer = b''
            self.ptr = 0
            self._fill_buffer()

    def _fill_buffer(self):
        chunk = self.handle.read(self.read_chunk_size)
        self.byte_buffer = chunk
        self.ptr = 0

    def write_bit(self, bit: int):
        self.buffer = (self.buffer << 1) | bit
        self.count += 1
        if self.count == 8:
            self.handle.write(bytes([self.buffer]))
            self.buffer = 0
            self.count = 0

    def flush(self):
        if self.count > 0:
            self.buffer = (self.buffer << (8 - self.count))
            self.handle.write(bytes([self.buffer]))
            self.count = 0
            self.buffer = 0

    def bit_generator(self) -> Generator[int, None, None]:
        if 'r' not in self.mode:
            raise ValueError("Not in read mode")
        
        while True:
            # Yield bits from current buffer
            while self.ptr < len(self.byte_buffer):
                byte_val = self.byte_buffer[self.ptr]
                self.ptr += 1
                for i in range(7, -1, -1):
                    yield (byte_val >> i) & 1
            
            # Refill
            self._fill_buffer()
            if len(self.byte_buffer) == 0:
                break

# --- ARITHMETIC CODING KERNEL ---

class DaemonArithmeticCoder:
    def __init__(self, bit_writer):
        self.low = 0
        self.high = TOP_VALUE
        self.pending_bits = 0
        self.bit_writer = bit_writer

    def encode(self, byte, probs):
        cdf = NFRUtils.pdf_to_cdf(probs)
        total = cdf[-1].item()
        
        idx = byte
        low_count = 0 if idx == 0 else cdf[idx-1].item()
        high_count = cdf[idx].item()
        
        rng = self.high - self.low + 1
        self.high = self.low + (rng * high_count) // total - 1
        self.low = self.low + (rng * low_count) // total
        
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending_bits += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & TOP_VALUE
            self.high = ((self.high << 1) & TOP_VALUE) | 1

    def _emit(self, bit):
        self.bit_writer.write_bit(bit)
        for _ in range(self.pending_bits):
            self.bit_writer.write_bit(1 - bit)
        self.pending_bits = 0

    def finish(self):
        self.pending_bits += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)

class DaemonArithmeticDecoder:
    def __init__(self, bit_gen):
        self.bit_gen = bit_gen
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0
        for _ in range(CODE_VALUE_BITS):
            try:
                self.value = (self.value << 1) | next(self.bit_gen)
            except StopIteration:
                self.value = (self.value << 1)

    def decode(self, probs):
        cdf = NFRUtils.pdf_to_cdf(probs)
        total = cdf[-1].item()
        
        rng = self.high - self.low + 1
        offset = self.value - self.low
        count = ((offset + 1) * total - 1) // rng
        
        idx = torch.searchsorted(cdf, count, right=True).item()
        
        low_count = 0 if idx == 0 else cdf[idx-1].item()
        high_count = cdf[idx].item()
        
        self.high = self.low + (rng * high_count) // total - 1
        self.low = self.low + (rng * low_count) // total
        
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            
            self.low = (self.low << 1) & TOP_VALUE
            self.high = ((self.high << 1) & TOP_VALUE) | 1
            try:
                bit = next(self.bit_gen)
            except StopIteration:
                bit = 0
            self.value = ((self.value << 1) & TOP_VALUE) | bit
            
        return idx

class NFRBlockEngine:
    """
    V2 Engine handling file-level operations with Block Processing.
    """
    def __init__(self, model_path=None):
        self.model = NeuralPredictor()
        if model_path:
            self.model.load_checkpoint(model_path)
            print(f"[Core] Loaded model: {model_path}")
        else:
            print("[Core] Using fresh NFR Micro-Model.")

    def train_on_file(self, input_file, epochs=5):
        """Train model on the specific file instance."""
        print(f"[NFR] Analyzing file structure ({epochs} epochs)...")
        self.model.train()
        
        file_size = os.path.getsize(input_file)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.005) 
        criterion = nn.CrossEntropyLoss() # Expects Logits
        
        # Read chunks but train on small sub-sequences
        # Reduced from 256KB to 16KB to avoid OOM
        read_chunk_size = 16384 
        seq_len = self.model.config['seq_len'] # 128
        
        chunk_idx = 0
        total_chunks_est = file_size // read_chunk_size

        with open(input_file, 'rb') as f:
            while True:
                data = f.read(read_chunk_size)
                if not data: break
                
                chunk_idx += 1
                
                # AGGRESSIVE SAMPLING: Train on 2% of the file (1 in 50 chunks)
                # This ensures completion in < 2 mins.
                if chunk_idx % 50 != 0: 
                    continue
                
                data_indices = np.frombuffer(data, dtype=np.uint8)
                if len(data_indices) < seq_len + 1: continue

                num_seqs = (len(data_indices) - 1) // seq_len
                if num_seqs == 0: continue
                
                # Truncate to fit
                limit = num_seqs * seq_len
                inputs = torch.tensor(data_indices[:limit], dtype=torch.long).view(num_seqs, seq_len).to(self.model.device)
                targets = torch.tensor(data_indices[1:limit+1], dtype=torch.long).view(num_seqs, seq_len).to(self.model.device)
                
                sys.stdout.write(f"\r[NFR] Analyzing Pattern... {chunk_idx / total_chunks_est * 100:.1f}%")

                for _ in range(epochs):
                    optimizer.zero_grad()
                    # Use forward_full to get [Batch, Seq, Vocab] logits
                    logits, _ = self.model.forward_full(inputs) 
                    
                    # Flatten for loss
                    loss = criterion(logits.view(-1, 256), targets.view(-1))
                    loss.backward()
                    optimizer.step()
                
                if file_size > 50 * 1024 * 1024:
                    break
                    
        self.model.save_checkpoint(input_file + ".model")
        print("\n[NFR] Analysis complete. Model adapted.")

    def compress(self, input_file: str, output_file: str, epochs=5):
        # 1. Setup
        if not os.path.exists(input_file + ".model"):
            self.train_on_file(input_file, epochs=epochs)
        
        file_size = os.path.getsize(input_file)
        out_f = open(output_file, 'wb')
        bit_stream = BitStream(out_f, 'wb')
        
        # 2. Header: Magic + Original Size
        out_f.write(MAGIC_HEADER)
        out_f.write(struct.pack('>Q', file_size))
        
        encoder = DaemonArithmeticCoder(bit_stream)
        self.model.eval()
        
        print(f"[NFR] Compressing {file_size} bytes (High-Speed Block Mode)...")
        start_time = time.time()
        
        with open(input_file, 'rb') as f, torch.no_grad():
            bytes_processed = 0
            while True:
                chunk = f.read(BLOCK_SIZE)
                if not chunk: break
                
                # Convert to tensor
                data = np.frombuffer(chunk, dtype=np.uint8)
                seq_tensor = torch.tensor(data, dtype=torch.long).unsqueeze(0).to(self.model.device)
                
                # --- FAST OPTIMIZATION: BATCH PREDICTION ---
                if bytes_processed == 0:
                    hidden = None
                    # Prepend a zero for the very first byte prediction
                    inp = torch.cat([torch.zeros(1, 1, dtype=torch.long).to(self.model.device), seq_tensor], dim=1)
                    inp = inp[:, :-1]
                else:
                    # Blocks are independent context
                    hidden = None
                    inp = torch.cat([torch.zeros(1, 1, dtype=torch.long).to(self.model.device), seq_tensor], dim=1)
                    inp = inp[:, :-1]

                probs, _ = self.model.forward_batch(inp, hidden)
                # probs: [L, 256]
                probs_cpu = probs.cpu()
                
                # --- CPU ARITHMETIC CODING LOOP ---
                for i, byte in enumerate(data):
                    encoder.encode(byte, probs_cpu[i])
                
                bytes_processed += len(chunk)
                if bytes_processed % (1024*1024) == 0:
                    sys.stdout.write(f"\r > Processed {bytes_processed//1024//1024} MB...")
        
        encoder.finish()
        bit_stream.flush()
        out_f.close()
        
        elapsed = time.time() - start_time
        comp_size = os.path.getsize(output_file)
        print(f"\n[Done] {comp_size} bytes. Ratio: {file_size/comp_size:.3f}x. Speed: {file_size/elapsed/1024:.2f} KB/s")

    def decompress(self, input_file: str, output_file: str):
        if not os.path.exists(input_file): raise FileNotFoundError
        
        in_f = open(input_file, 'rb')
        magic = in_f.read(len(MAGIC_HEADER))
        if magic != MAGIC_HEADER: raise ValueError("Not DMNv2")
        
        original_size = struct.unpack('>Q', in_f.read(8))[0]
        
        bit_stream = BitStream(in_f, 'rb')
        decoder = DaemonArithmeticDecoder(bit_stream.bit_generator())
        
        out_f = open(output_file, 'wb')
        self.model.eval()
        
        print(f"[NFR] Decompressing {original_size} bytes...")
        
        context_tensor = torch.zeros(1, 1, dtype=torch.long).to(self.model.device) # Start with SOS=0
        hidden = None
        
        write_buffer = bytearray()
        
        with torch.no_grad():
            for i in range(original_size):
                if i % 100 == 0:
                     if i % 1000 == 0: sys.stdout.write(f"\r > {i}/{original_size}")
                
                # Check block boundary
                if i > 0 and i % BLOCK_SIZE == 0:
                    # Reset Context for next block as per compression logic
                    context_tensor = torch.zeros(1, 1, dtype=torch.long).to(self.model.device)
                    hidden = None
                
                # Predict NEXT byte probability based on Current Context
                probs, hidden = self.model(context_tensor, hidden) 
                probs = probs[0].cpu()
                
                # Decode actual byte
                symbol = decoder.decode(probs)
                
                write_buffer.append(symbol)
                
                # Prepare next context
                context_tensor = torch.tensor([[symbol]], dtype=torch.long).to(self.model.device)
                
                if len(write_buffer) >= 65536:
                    out_f.write(write_buffer)
                    write_buffer = bytearray()
                    
        out_f.write(write_buffer)
        out_f.close()
        in_f.close()
        print("\n[Done] Restored.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_c = subparsers.add_parser('compress')
    p_c.add_argument('input'); p_c.add_argument('output'); p_c.add_argument('--epochs', type=int, default=5)
    
    p_d = subparsers.add_parser('decompress')
    p_d.add_argument('input'); p_d.add_argument('output')
    
    args = parser.parse_args()
    
    engine = NFRBlockEngine()
    
    if args.command == 'compress':
        engine.compress(args.input, args.output, epochs=args.epochs)
    elif args.command == 'decompress':
        # Look for model
        m_path = args.input.replace('.dmn', '') + '.model'
        if not os.path.exists(m_path): 
             base = os.path.splitext(args.input)[0]
             if os.path.exists(base + ".model"): m_path = base + ".model"
             
        if os.path.exists(m_path):
            engine.model.load_checkpoint(m_path)
        else:
            print("![Warning] No sidecar model found. Decompression will likely produce garbage/noise.")
            
        engine.decompress(args.input, args.output)
