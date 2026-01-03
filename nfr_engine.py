
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import struct
import os
import sys

# --- HYPERPARAMETRES DU NOYAU ---
CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'hidden_size': 512,
    'num_layers': 3,     
    'seq_len': 64,       
    'vocab_size': 256,   
    'lr': 0.001,
    'epochs': 5 # Low for PoC speed
}

# --- CONFIGURATION DU NOYAU DAEMON (ARITHMETIC CODING) ---
CODE_VALUE_BITS = 32
TOP_VALUE = (1 << CODE_VALUE_BITS) - 1
QUARTER = 1 << (CODE_VALUE_BITS - 2)
HALF = 1 << (CODE_VALUE_BITS - 1)
THREE_QUARTERS = 3 * QUARTER

class DaemonArithmeticCoder:
    """
    Implémentation haute précision d'un codeur arithmétique.
    Conçu pour s'interfacer avec les tensors de probabilité du NeuralPredictor.
    """
    def __init__(self):
        self.low = 0
        self.high = TOP_VALUE
        self.pending_bits = 0
        self.buffer = []  # Buffer de bits compressés

    def _transmit_bit(self, bit):
        """Écrit un bit dans le buffer (et gère les bits en attente/carry)."""
        self.buffer.append(bit)
        # Gestion des bits en attente (cas de sous-dépassement)
        while self.pending_bits > 0:
            self.buffer.append(1 - bit)
            self.pending_bits -= 1

    def encode_symbol(self, symbol_idx, pdf_tensor):
        """
        Encode un symbole basé sur la distribution de probabilité fournie par le NeuralPredictor.
        Args:
            symbol_idx (int): L'octet réel à encoder (0-255).
            pdf_tensor (torch.Tensor): Tensor 1D de probabilités (somme = 1.0).
        """
        # 1. Conversion des probabilités flottantes en fréquences cumulatives entières
        scale_factor = 16384 # 2^14
        
        # Calcul rapide de la CDF
        cdf = torch.cumsum(pdf_tensor, dim=0)
        
        # S'assurer que le dernier élément est exactement scale_factor
        cdf_int = (cdf * scale_factor).long()
        cdf_int[-1] = scale_factor 
        
        # Récupération des bornes pour le symbole actuel
        low_count = 0 if symbol_idx == 0 else cdf_int[symbol_idx - 1].item()
        high_count = cdf_int[symbol_idx].item()
        total_count = scale_factor

        # 2. Rétrécissement de l'intervalle global
        current_range = self.high - self.low + 1
        self.high = self.low + (current_range * high_count) // total_count - 1
        self.low = self.low + (current_range * low_count) // total_count

        # 3. Renormalisation (Emission de bits)
        while True:
            if self.high < HALF:
                self._transmit_bit(0)
            elif self.low >= HALF:
                self._transmit_bit(1)
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

    def finish(self):
        """Finalise le flux (flush des derniers bits)."""
        self.pending_bits += 1
        if self.low < QUARTER:
            self._transmit_bit(0)
        else:
            self._transmit_bit(1)
        return self.buffer

class DaemonDecoder:
    """
    Décodeur symétrique.
    """
    def __init__(self, bitstream):
        self.bitstream = bitstream
        self.bit_idx = 0
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0
        
        # Initialisation du buffer interne avec les premiers bits
        for _ in range(CODE_VALUE_BITS):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self):
        if self.bit_idx >= len(self.bitstream):
            return 0 # Padding final
        bit = self.bitstream[self.bit_idx]
        self.bit_idx += 1
        return bit

    def decode_symbol(self, pdf_tensor):
        scale_factor = 16384
        cdf = torch.cumsum(pdf_tensor, dim=0)
        cdf_int = (cdf * scale_factor).long()
        cdf_int[-1] = scale_factor
        
        current_range = self.high - self.low + 1
        # Mapping inverse
        mapped_value = ((self.value - self.low + 1) * scale_factor - 1) // current_range
        
        # Recherche du symbole correspondant
        # torch.searchsorted trouve l'index où insérer mapped_value pour garder l'ordre.
        # cdf_int[i-1] <= mapped_value < cdf_int[i]
        symbol_idx = torch.searchsorted(cdf_int, mapped_value).item()
        
        # Mise à jour des bornes (exactement comme l'encodeur)
        low_count = 0 if symbol_idx == 0 else cdf_int[symbol_idx - 1].item()
        high_count = cdf_int[symbol_idx].item()
        
        self.high = self.low + (current_range * high_count) // scale_factor - 1
        self.low = self.low + (current_range * low_count) // scale_factor
        
        # Renormalisation
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
            self.value = ((self.value << 1) & TOP_VALUE) | self._read_bit()
            
        return symbol_idx

class NeuralPredictor(nn.Module):
    def __init__(self, config):
        super(NeuralPredictor, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.lstm = nn.LSTM(config['hidden_size'], config['hidden_size'], 
                            config['num_layers'], batch_first=True)
        self.fc = nn.Linear(config['hidden_size'], config['vocab_size'])
        
    def forward(self, x, hidden):
        embedded = self.embed(x)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden

def entropy_benchmark(data):
    if len(data) == 0: return 0
    counts = np.bincount(data, minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))

def train_model(model, data_indices):
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    print("[SYSTEM] Démarrage de l'entrainement...")
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        chunks = 0
        for i in range(0, len(data_indices) - CONFIG['seq_len'], CONFIG['seq_len']):
            input_seq = torch.tensor(data_indices[i:i+CONFIG['seq_len']], dtype=torch.long).unsqueeze(0).to(CONFIG['device'])
            target_seq = torch.tensor(data_indices[i+1:i+CONFIG['seq_len']+1], dtype=torch.long).unsqueeze(0).to(CONFIG['device'])
            
            model.zero_grad()
            output, _ = model(input_seq, None)
            
            loss = criterion(output.view(-1, 256), target_seq.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch_loss += loss.item()
            chunks += 1
            
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}: Loss = {epoch_loss/chunks:.4f}")
    return model

# Helpers bits conversions
def bits_to_bytes(bits):
    b = bytearray()
    idx = 0
    while idx < len(bits):
        byte = 0
        for i in range(8):
            if idx < len(bits):
                byte = (byte << 1) | bits[idx]
                idx += 1
            else:
                byte = (byte << 1)
        b.append(byte)
    return bytes(b)

def bytes_to_bits(data):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def main():
    print("--- DAEMON NFR Engine (Integrated High-Precision Arithmetic) ---\n")
    
    # 1. DATA
    text_data = b"import numpy as np\ndef function(x): return x * x\n" * 20
    text_data += b"Random seed data: " + os.urandom(100)
    
    original_size = len(text_data)
    print(f"Original Text Data Size: {original_size} bytes")
    print(f"Shannon Entropy: {entropy_benchmark(np.frombuffer(text_data, dtype=np.uint8)):.4f} bits/byte")
    
    data_indices = np.frombuffer(text_data, dtype=np.uint8)
    
    # 2. MODEL
    model = NeuralPredictor(CONFIG).to(CONFIG['device'])
    model = train_model(model, data_indices)
    
    # 3. COMPRESSION
    encoder = DaemonArithmeticCoder()
    context = [0] * CONFIG['seq_len'] # Zero padding start
    
    print("\n[COMPRESSION START]")
    model.eval()
    with torch.no_grad():
        for i, byte in enumerate(data_indices):
            byte = int(byte)
            
            # Predict
            seq_tensor = torch.tensor([context[-CONFIG['seq_len']:]], dtype=torch.long).to(CONFIG['device'])
            logits, _ = model(seq_tensor, None)
            
            # Get specific probabilities for next token
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=0) # Tensor on device
            
            # Encode
            encoder.encode_symbol(byte, probs)
            
            # Update Context
            context.append(byte)
            
            if i % 1000 == 0:
                 sys.stdout.write(f"\rProcessed {i}/{len(data_indices)} bytes")
    
    print("\nFinalizing stream...")
    compressed_bits = encoder.finish()
    compressed_bytes = bits_to_bytes(compressed_bits)
    compressed_size = len(compressed_bytes)
    
    print(f"Compressed Size: {compressed_size} bytes")
    print(f"Ratio: {original_size / compressed_size:.2f}x ({compressed_size/original_size*100:.2f}%)")
    
    # 4. DECOMPRESSION
    print("\n[DECOMPRESSION START]")
    # Convert bytes back to bits list for the decoder
    bits_for_decoder = bytes_to_bits(compressed_bytes)
    # Important: The decoder might have read extra padding zero-bits at the end during bit assembling, 
    # but the logic relies on exact bit count or implicit stop. 
    # Usually we pass the length of original data to know when to stop decoding.
    
    decoder = DaemonDecoder(bits_for_decoder)
    decoded_data = []
    context = [0] * CONFIG['seq_len'] # Reset context
    
    model.eval()
    with torch.no_grad():
        for i in range(original_size):
            # Predict (Must match encoder exactly)
            seq_tensor = torch.tensor([context[-CONFIG['seq_len']:]], dtype=torch.long).to(CONFIG['device'])
            logits, _ = model(seq_tensor, None)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=0)
            
            # Decode
            sym = decoder.decode_symbol(probs)
            decoded_data.append(sym)
            
            # Update Context
            context.append(sym)
            
            if i % 1000 == 0:
                 sys.stdout.write(f"\rDecoded {i}/{original_size} bytes")

    decoded_bytes = bytes(decoded_data)
    
    print("\n\n--- INTEGRITY CHECK ---")
    if decoded_bytes == text_data:
        print(">> SUCCESS: Perfect Reconstruction.")
    else:
        print(">> FAILURE: Data mismatch.")
        print(f"Original len: {len(text_data)}, Decoded len: {len(decoded_bytes)}")
        # Diff
        for j in range(min(len(text_data), len(decoded_bytes))):
             if text_data[j] != decoded_bytes[j]:
                 print(f"First mismatch at index {j}: Orig={text_data[j]} vs Dec={decoded_bytes[j]}")
                 break

if __name__ == "__main__":
    main()
