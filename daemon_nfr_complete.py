
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import struct
import time
import numpy as np

# --- DAEMON CONFIGURATION ---
CODE_VALUE_BITS = 32
TOP_VALUE = (1 << CODE_VALUE_BITS) - 1
QUARTER = 1 << (CODE_VALUE_BITS - 2)
HALF = 1 << (CODE_VALUE_BITS - 1)
THREE_QUARTERS = 3 * QUARTER

# NOTE: MODEL_SEQ_LEN must match the seq_len of the trained model
CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'hidden_size': 512,
    'num_layers': 3,     
    'seq_len': 64,       
    'vocab_size': 256,   
    # Training params
    'lr': 0.001,
    'epochs': 5 
}
MODEL_SEQ_LEN = CONFIG['seq_len']

class BitStreamIO:
    """Handles physical read/write of bits in a .DMN file"""
    def __init__(self, filename, mode='wb'):
        self.file = open(filename, mode)
        self.buffer = 0
        self.count = 0
        self.mode = mode
        # For reading
        if 'r' in mode:
            # We will read the whole file content once for simplicity 
            # (In production, streaming read is better)
            self.read_buffer = self.file.read() 
            self.read_ptr = 0 # position in byte stream
            self._iterator = self._bit_generator()

    def write_header(self, original_size):
        """Write 8-byte file size header."""
        self.file.write(struct.pack('>Q', original_size)) # Big-endian unsigned long long

    def read_header(self):
        """Read 8-byte file size header."""
        # Read first 8 bytes
        header_bytes = self.read_buffer[:8]
        self.read_ptr = 8 # advance pointer
        original_size = struct.unpack('>Q', header_bytes)[0]
        return original_size

    def write_bit(self, bit):
        self.buffer = (self.buffer << 1) | bit
        self.count += 1
        if self.count == 8:
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.count = 0

    def close_write(self):
        # Padding last byte if necessary
        if self.count > 0:
            self.buffer = (self.buffer << (8 - self.count))
            self.file.write(bytes([self.buffer]))
        self.file.close()

    def read_bit_stream(self):
        return self._iterator

    def _bit_generator(self):
        # Generator to read bit by bit, from read_ptr (after header)
        while self.read_ptr < len(self.read_buffer):
            byte = self.read_buffer[self.read_ptr]
            self.read_ptr += 1
            for i in range(7, -1, -1):
                yield (byte >> i) & 1

def pdf_to_cdf(pdf_tensor):
    scale = 16384
    freqs = (pdf_tensor * scale).long()
    freqs = torch.clamp(freqs, min=1)
    cdf = torch.cumsum(freqs, dim=0)
    return cdf

class DaemonArithmeticCoder:
    """Mathematical Compression Engine"""
    def __init__(self, bit_writer=None):
        self.low = 0
        self.high = TOP_VALUE
        self.pending_bits = 0
        self.bit_writer = bit_writer

    def encode_symbol(self, symbol_idx, pdf_tensor):
        cdf_int = pdf_to_cdf(pdf_tensor)
        
        low_count = 0 if symbol_idx == 0 else cdf_int[symbol_idx - 1].item()
        high_count = cdf_int[symbol_idx].item()
        total_count = cdf_int[-1].item()

        current_range = self.high - self.low + 1
        self.high = self.low + (current_range * high_count) // total_count - 1
        self.low = self.low + (current_range * low_count) // total_count

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
        while self.pending_bits > 0:
            self.bit_writer.write_bit(1 - bit)
            self.pending_bits -= 1

    def finish(self):
        self.pending_bits += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)

class DaemonDecoder:
    """Mathematical Decompression Engine"""
    def __init__(self, bit_iterator):
        self.bit_iterator = bit_iterator
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0
        # Initial fill
        for _ in range(CODE_VALUE_BITS):
            try:
                self.value = (self.value << 1) | next(self.bit_iterator)
            except StopIteration:
                self.value = (self.value << 1) # Padding

    def decode_symbol(self, pdf_tensor):
        cdf_int = pdf_to_cdf(pdf_tensor)
        scale_factor = cdf_int[-1].item() # Variable scale
        
        current_range = self.high - self.low + 1
        mapped_value = ((self.value - self.low + 1) * scale_factor - 1) // current_range
        
        # Binary search like
        symbol_idx = torch.searchsorted(cdf_int, mapped_value, right=True).item()
        
        low_count = 0 if symbol_idx == 0 else cdf_int[symbol_idx - 1].item()
        high_count = cdf_int[symbol_idx].item()
        
        self.high = self.low + (current_range * high_count) // scale_factor - 1
        self.low = self.low + (current_range * low_count) // scale_factor
        
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
                bit = next(self.bit_iterator)
            except StopIteration:
                bit = 0
            self.value = ((self.value << 1) & TOP_VALUE) | bit
            
        return symbol_idx

class NeuralPredictor(nn.Module):
    """
    Autoregressive Model for Probability Density Estimation.
    (Merged from nfr_engine.py)
    """
    def __init__(self, config):
        super(NeuralPredictor, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.lstm = nn.LSTM(config['hidden_size'], config['hidden_size'], 
                            config['num_layers'], batch_first=True)
        self.fc = nn.Linear(config['hidden_size'], config['vocab_size'])
        
    def forward(self, x, hidden):
        # x: [batch, seq_len]
        # In this pipeline, x is [1, seq_len]
        embedded = self.embed(x)
        output, hidden = self.lstm(embedded, hidden)
        
        # We need logits for the LAST element of the sequence
        # to predict the NEXT one.
        # output shape: [1, seq_len, hidden]
        last_step_output = output[:, -1, :] # [1, hidden]
        
        prediction_logits = self.fc(last_step_output) # [1, vocab_size]
        probs = F.softmax(prediction_logits, dim=1)
        return probs

def train_network(model, data_bytes):
    """Mini Training Loop for PoC (Overfitting on file)"""
    data_indices = np.frombuffer(data_bytes, dtype=np.uint8)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    print(f"[SYSTEM] Training NeuralPredictor on {len(data_bytes)} bytes...")
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        chunks = 0
        for i in range(0, len(data_indices) - CONFIG['seq_len'], CONFIG['seq_len']):
            input_seq = torch.tensor(data_indices[i:i+CONFIG['seq_len']], dtype=torch.long).unsqueeze(0).to(CONFIG['device'])
            target_seq = torch.tensor(data_indices[i+1:i+CONFIG['seq_len']+1], dtype=torch.long).unsqueeze(0).to(CONFIG['device'])
            
            model.zero_grad()
            # Note: Model returns probs in forward() for inference.
            # For training, we need to adapt slightly or use internal logic.
            # We use LSTM logic directly here to simplify (copied from nfr_engine.py)
            
            embedded = model.embed(input_seq)
            output, _ = model.lstm(embedded)
            logits = model.fc(output) # [1, seq_len, 256]
            
            loss = criterion(logits.view(-1, 256), target_seq.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch_loss += loss.item()
            chunks += 1
            
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}: Loss = {epoch_loss/chunks:.4f}")
    return model

# --- MASTER FUNCTIONS ---

def compress_file(input_path, output_path, model):
    print(f">> DAEMON: COMPRESSING {input_path}...")
    start_time = time.time()
    
    # 1. Read Raw Data
    with open(input_path, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    
    # 2. Train model on this file (Archive Self-contained PoC Mode)
    # In practice, we would typically have a universal pre-trained model
    # or store compressed weights in the header.
    # Here, we assume the recipient has the same training script
    # and will re-train (which is cheating), OR that the model is already good.
    # For this "Archive" PoC, we will train the model "fresh".
    train_network(model, data)

    # 3. Initialize Output Stream
    bit_io = BitStreamIO(output_path, 'wb')
    
    # 4. WRITE HEADER (File Size)
    print(f">> HEADER: Writing Size {original_size}")
    bit_io.write_header(original_size)
    
    encoder = DaemonArithmeticCoder(bit_io)
    
    # 5. Initialize Context (Zero Padding)
    context = [0] * MODEL_SEQ_LEN
    
    model.eval()
    with torch.no_grad():
        for i, byte in enumerate(data):
            if i % 100 == 0:
                print(f"\r>> PROGRESS: {i}/{len(data)} bytes", end="")
            
            # Prepare Tensor Context (Shape: [1, seq_len])
            input_tensor = torch.tensor([context], dtype=torch.long).to(CONFIG['device'])
            
            # NEURAL PREDICTION
            probs = model(input_tensor, None) # Returns probs [1, 256] (see new forward)
            probs = probs[0].cpu() # On CPU for arithmetic
            
            # ARITHMETIC ENCODING
            encoder.encode_symbol(byte, probs)
            
            # Update Context (Shift)
            context.pop(0)
            context.append(byte)

    encoder.finish()
    bit_io.close_write()
    
    compressed_size = os.path.getsize(output_path)
    ratio = (1 - compressed_size/original_size) * 100
    
    print(f"\n>> DONE. Ratio: {ratio:.2f}% | Time: {time.time() - start_time:.2f}s")
    print(f">> FILE CREATED: {output_path} ({compressed_size} bytes)")

def decompress_file(input_path, output_path, model):
    print(f">> DAEMON: DECOMPRESSING {input_path}...")
    
    bit_io = BitStreamIO(input_path, 'rb')
    
    # 1. READ HEADER
    original_size = bit_io.read_header()
    print(f">> HEADER: Read Size {original_size}")
    
    bit_iterator = bit_io.read_bit_stream()
    decoder = DaemonDecoder(bit_iterator)
    
    decoded_data = bytearray()
    context = [0] * MODEL_SEQ_LEN
    
    # IMPORTANT NOTE: The model must be in the SAME STATE as during compression.
    # If we did "live" training during compression,
    # the decompressor must have exact weights.
    # For this PoC, the 'model' object passed as argument is already trained (same memory session).
    # In real usage, weights should be loaded.
    
    model.eval()
    with torch.no_grad():
        for i in range(original_size):
            if i % 100 == 0:
                print(f"\r>> PROGRESS: {i}/{original_size} bytes", end="")
                
            input_tensor = torch.tensor([context], dtype=torch.long).to(CONFIG['device'])
            
            # NEURAL PREDICTION
            probs = model(input_tensor, None)
            probs = probs[0].cpu()
            
            # ARITHMETIC DECODING
            symbol = decoder.decode_symbol(probs)
            
            decoded_data.append(symbol)
            
            # Update Context
            context.pop(0)
            context.append(symbol)
            
    with open(output_path, 'wb') as f:
        f.write(decoded_data)
    print("\n>> DECOMPRESSION FINISHED.")

# --- EXECUTION ---
if __name__ == "__main__":
    # Test Data: Complex Source Code + Binary (Pattern is important to see compression)
    dummy_data = b"DAEMON_IS_WATCHING_YOU_" * 20 + os.urandom(50)
    with open("daemon_test.bin", "wb") as f:
        f.write(dummy_data)
        
    # Initialize Model
    model = NeuralPredictor(CONFIG).to(CONFIG['device'])
    
    # 1. Compression (Including on-the-fly training)
    compress_file("daemon_test.bin", "daemon_test.dmn", model)
    
    # 2. Decompression (.dmn file contains header size)
    # model has trained weights because we are in the same memory session.
    decompress_file("daemon_test.dmn", "daemon_restored.bin", model)
    
    # Verification
    with open("daemon_restored.bin", "rb") as f:
        restored = f.read()
    
    if restored == dummy_data:
        print(">> INTEGRITY CONFIRMED: 100%")
    else:
        print(">> CRITICAL ERROR IN THE MATRIX.")
        print(f"Original: {len(dummy_data)}, Restored: {len(restored)}")
        for j in range(min(len(dummy_data), len(restored))):
             if dummy_data[j] != restored[j]:
                 print(f"First mismatch at index {j}: Orig={dummy_data[j]} vs Dec={restored[j]}")
                 print(f"Context around mismatch (Orig): {dummy_data[max(0, j-5):j+5]}")
                 print(f"Context around mismatch (Rest): {restored[max(0, j-5):j+5]}")
                 break
        if len(dummy_data) != len(restored):
            print("Length mismatch as well.")
