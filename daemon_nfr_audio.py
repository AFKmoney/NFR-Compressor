"""
DAEMON NFR Audio Engine - V3.0
Specialized Neural Compression for Audio Data

Key Features:
- Stereo Delta Encoding (L/R correlation)
- LSTM-based byte prediction
- Optimized for WAV/PCM input (uncompressed audio)

Usage:
    python daemon_nfr_audio.py compress input.wav output.dmna
    python daemon_nfr_audio.py decompress input.dmna output.wav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import struct
import time
import sys
import numpy as np
import wave
from typing import BinaryIO, Generator

# --- CONFIGURATION ---
MAGIC_HEADER = b'DMNA'
BLOCK_SIZE = 8192

# Arithmetic Coding Constants
CODE_VALUE_BITS = 32
TOP_VALUE = (1 << CODE_VALUE_BITS) - 1
QUARTER = 1 << (CODE_VALUE_BITS - 2)
HALF = 1 << (CODE_VALUE_BITS - 1)
THREE_QUARTERS = 3 * QUARTER
SCALE_FACTOR = 16384

class AudioNFRUtils:
    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def pdf_to_cdf(pdf_tensor: torch.Tensor) -> torch.Tensor:
        freqs = (pdf_tensor * SCALE_FACTOR).long()
        freqs = torch.clamp(freqs, min=1)
        return torch.cumsum(freqs, dim=0)
    
    @staticmethod
    def apply_stereo_delta(samples: np.ndarray, channels: int) -> np.ndarray:
        """Convert stereo L/R to L/Delta encoding for better compression."""
        if channels != 2:
            return samples
        stereo = samples.reshape(-1, 2)
        left = stereo[:, 0].astype(np.int32)
        right = stereo[:, 1].astype(np.int32)
        delta = np.clip(right - left, -32768, 32767).astype(np.int16)
        result = np.empty_like(samples)
        result[0::2] = left.astype(np.int16)
        result[1::2] = delta
        return result
    
    @staticmethod
    def reverse_stereo_delta(samples: np.ndarray, channels: int) -> np.ndarray:
        """Reverse delta encoding back to L/R."""
        if channels != 2:
            return samples
        left = samples[0::2].astype(np.int32)
        delta = samples[1::2].astype(np.int32)
        right = (left + delta).astype(np.int16)
        result = np.empty_like(samples)
        result[0::2] = left.astype(np.int16)
        result[1::2] = right
        return result

class AudioNeuralPredictor(nn.Module):
    """Simplified LSTM predictor for audio bytes."""
    def __init__(self, hidden_size=256, num_layers=2):
        super(AudioNeuralPredictor, self).__init__()
        self.device = AudioNFRUtils.get_device()
        self.hidden_size = hidden_size
        self.vocab_size = 256
        self.seq_len = 128
        
        self.embed = nn.Embedding(self.vocab_size, 64)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.to(self.device)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out[:, -1, :])
        return F.softmax(logits, dim=1), hidden

    def forward_full(self, x_seq, hidden=None):
        emb = self.embed(x_seq)
        out, hidden = self.lstm(emb, hidden)
        return self.fc(out), hidden

    def forward_batch(self, x_seq, hidden=None):
        logits, hidden = self.forward_full(x_seq, hidden)
        return F.softmax(logits.squeeze(0), dim=1), hidden

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

# --- ARITHMETIC CODING ---

class BitStream:
    def __init__(self, handle: BinaryIO, mode='wb'):
        self.handle = handle
        self.mode = mode
        self.buffer = 0
        self.count = 0
        if 'r' in mode:
            self.read_chunk_size = 65536
            self.byte_buffer = b''
            self.ptr = 0
            self._fill_buffer()

    def _fill_buffer(self):
        self.byte_buffer = self.handle.read(self.read_chunk_size)
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
        while True:
            while self.ptr < len(self.byte_buffer):
                byte_val = self.byte_buffer[self.ptr]
                self.ptr += 1
                for i in range(7, -1, -1):
                    yield (byte_val >> i) & 1
            self._fill_buffer()
            if len(self.byte_buffer) == 0:
                break

class ArithmeticCoder:
    def __init__(self, bit_writer):
        self.low = 0
        self.high = TOP_VALUE
        self.pending_bits = 0
        self.bit_writer = bit_writer

    def encode(self, symbol, probs):
        cdf = AudioNFRUtils.pdf_to_cdf(probs)
        total = cdf[-1].item()
        low_count = 0 if symbol == 0 else cdf[symbol-1].item()
        high_count = cdf[symbol].item()
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

class ArithmeticDecoder:
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
        cdf = AudioNFRUtils.pdf_to_cdf(probs)
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

# --- AUDIO ENGINE ---

class NFRAudioEngine:
    def __init__(self, model_path=None):
        self.model = AudioNeuralPredictor()
        if model_path and os.path.exists(model_path):
            self.model.load_checkpoint(model_path)
            print(f"[Audio] Loaded model: {model_path}")
        else:
            print("[Audio] Using fresh Audio Neural Predictor.")

    def train_on_audio(self, audio_bytes: bytes, epochs=3):
        print(f"[Audio] Learning audio patterns ({epochs} epochs)...")
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()
        
        data = np.frombuffer(audio_bytes, dtype=np.uint8)
        seq_len = 128
        chunk_size = 2048
        num_chunks = len(data) // chunk_size
        sample_indices = np.random.choice(num_chunks, max(1, num_chunks // 10), replace=False)
        
        for epoch in range(epochs):
            total_loss = 0
            for idx in sample_indices:
                start = idx * chunk_size
                chunk = data[start:start+chunk_size]
                if len(chunk) < seq_len + 1:
                    continue
                
                num_seqs = (len(chunk) - 1) // seq_len
                if num_seqs == 0:
                    continue
                
                limit = num_seqs * seq_len
                inputs = torch.tensor(chunk[:limit], dtype=torch.long).view(num_seqs, seq_len).to(self.model.device)
                targets = torch.tensor(chunk[1:limit+1], dtype=torch.long).view(num_seqs, seq_len).to(self.model.device)
                
                optimizer.zero_grad()
                logits, _ = self.model.forward_full(inputs)
                loss = criterion(logits.view(-1, 256), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            sys.stdout.write(f"\r[Audio] Epoch {epoch+1}/{epochs} Loss: {total_loss/max(1, len(sample_indices)):.4f}")
        print()

    def compress(self, input_file: str, output_file: str, epochs=3):
        print(f"[Audio] Reading WAV: {input_file}")
        
        with wave.open(input_file, 'rb') as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)
        
        print(f"[Audio] {channels}ch, {sample_width*8}bit, {framerate}Hz, {n_frames} frames")
        
        if sample_width == 2:
            samples = np.frombuffer(audio_data, dtype=np.int16)
        else:
            samples = np.frombuffer(audio_data, dtype=np.uint8)
        
        if channels == 2:
            print("[Audio] Applying stereo delta encoding...")
            samples = AudioNFRUtils.apply_stereo_delta(samples, channels)
        
        audio_bytes = samples.tobytes()
        
        self.train_on_audio(audio_bytes, epochs=epochs)
        self.model.save_checkpoint(input_file + ".model")
        
        out_f = open(output_file, 'wb')
        bit_stream = BitStream(out_f, 'wb')
        
        out_f.write(MAGIC_HEADER)
        out_f.write(struct.pack('>HBIQ', channels, sample_width, framerate, len(audio_bytes)))
        
        encoder = ArithmeticCoder(bit_stream)
        self.model.eval()
        
        print(f"[Audio] Compressing {len(audio_bytes)} bytes...")
        start_time = time.time()
        
        data = np.frombuffer(audio_bytes, dtype=np.uint8)
        
        with torch.no_grad():
            for i in range(0, len(data), BLOCK_SIZE):
                chunk = data[i:i+BLOCK_SIZE]
                seq_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(self.model.device)
                
                inp = torch.cat([torch.zeros(1, 1, dtype=torch.long).to(self.model.device), seq_tensor], dim=1)
                inp = inp[:, :-1]
                
                probs, _ = self.model.forward_batch(inp, None)
                probs_cpu = probs.cpu()
                
                for j, byte in enumerate(chunk):
                    encoder.encode(byte, probs_cpu[j])
                
                if i % (BLOCK_SIZE * 10) == 0:
                    sys.stdout.write(f"\r > {i}/{len(data)} bytes...")
        
        encoder.finish()
        bit_stream.flush()
        out_f.close()
        
        elapsed = time.time() - start_time
        comp_size = os.path.getsize(output_file)
        orig_size = len(audio_bytes)
        ratio = orig_size / comp_size if comp_size > 0 else 0
        print(f"\n[Done] {comp_size} bytes. Ratio: {ratio:.3f}x. Speed: {orig_size/elapsed/1024:.2f} KB/s")

    def decompress(self, input_file: str, output_file: str):
        in_f = open(input_file, 'rb')
        magic = in_f.read(len(MAGIC_HEADER))
        if magic != MAGIC_HEADER:
            raise ValueError("Not DMNA audio format")
        
        channels, sample_width, framerate, data_size = struct.unpack('>HBIQ', in_f.read(2+1+4+8))
        
        bit_stream = BitStream(in_f, 'rb')
        decoder = ArithmeticDecoder(bit_stream.bit_generator())
        
        self.model.eval()
        print(f"[Audio] Decompressing {data_size} bytes...")
        
        context = torch.zeros(1, 1, dtype=torch.long).to(self.model.device)
        hidden = None
        result = bytearray()
        
        with torch.no_grad():
            for i in range(data_size):
                if i % 1000 == 0:
                    sys.stdout.write(f"\r > {i}/{data_size}")
                
                if i > 0 and i % BLOCK_SIZE == 0:
                    context = torch.zeros(1, 1, dtype=torch.long).to(self.model.device)
                    hidden = None
                
                probs, hidden = self.model(context, hidden)
                probs = probs[0].cpu()
                
                symbol = decoder.decode(probs)
                result.append(symbol)
                context = torch.tensor([[symbol]], dtype=torch.long).to(self.model.device)
        
        in_f.close()
        
        samples = np.frombuffer(bytes(result), dtype=np.int16)
        
        if channels == 2:
            samples = AudioNFRUtils.reverse_stereo_delta(samples, channels)
        
        with wave.open(output_file, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(framerate)
            wav.writeframes(samples.tobytes())
        
        print(f"\n[Done] Restored to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFR Audio Compressor v3")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_c = subparsers.add_parser('compress')
    p_c.add_argument('input')
    p_c.add_argument('output')
    p_c.add_argument('--epochs', type=int, default=3)
    
    p_d = subparsers.add_parser('decompress')
    p_d.add_argument('input')
    p_d.add_argument('output')
    
    args = parser.parse_args()
    
    engine = NFRAudioEngine()
    
    if args.command == 'compress':
        engine.compress(args.input, args.output, epochs=args.epochs)
    elif args.command == 'decompress':
        m_path = args.input.replace('.dmna', '') + '.model'
        if not os.path.exists(m_path):
            base = os.path.splitext(args.input)[0]
            if os.path.exists(base + ".wav.model"):
                m_path = base + ".wav.model"
        if os.path.exists(m_path):
            engine.model.load_checkpoint(m_path)
        else:
            print("[!] Warning: No model found, decompression may fail")
        engine.decompress(args.input, args.output)
