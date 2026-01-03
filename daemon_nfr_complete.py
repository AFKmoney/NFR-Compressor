
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import struct
import time
import numpy as np

# --- CONFIGURATION DAEMON ---
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
    """Gère l'écriture/lecture physique des bits dans un fichier .DMN"""
    def __init__(self, filename, mode='wb'):
        self.file = open(filename, mode)
        self.buffer = 0
        self.count = 0
        self.mode = mode
        # Pour la lecture
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
        # Padding du dernier octet si nécessaire
        if self.count > 0:
            self.buffer = (self.buffer << (8 - self.count))
            self.file.write(bytes([self.buffer]))
        self.file.close()

    def read_bit_stream(self):
        return self._iterator

    def _bit_generator(self):
        # Générateur pour lire bit par bit, à partir de read_ptr (après header)
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
    """Moteur mathématique de compression"""
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
    """Moteur mathématique de décompression"""
    def __init__(self, bit_iterator):
        self.bit_iterator = bit_iterator
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0
        # Remplissage initial
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
    Modèle Autoregressif pour l'estimation de densité de probabilité.
    (Fusionné depuis nfr_engine.py)
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
        # Dans ce pipeline, x est [1, seq_len]
        embedded = self.embed(x)
        output, hidden = self.lstm(embedded, hidden)
        
        # Nous avons besoin des logits pour le DERNIER élément de la séquence 
        # pour prédire le PROCHAIN.
        # output shape: [1, seq_len, hidden]
        last_step_output = output[:, -1, :] # [1, hidden]
        
        prediction_logits = self.fc(last_step_output) # [1, vocab_size]
        probs = F.softmax(prediction_logits, dim=1)
        return probs

def train_network(model, data_bytes):
    """Mini boucle d'entrainement pour le PoC (Overfitting sur le fichier)"""
    data_indices = np.frombuffer(data_bytes, dtype=np.uint8)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    print(f"[SYSTEM] Entrainement du NeuralPredictor sur {len(data_bytes)} octets...")
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        chunks = 0
        for i in range(0, len(data_indices) - CONFIG['seq_len'], CONFIG['seq_len']):
            input_seq = torch.tensor(data_indices[i:i+CONFIG['seq_len']], dtype=torch.long).unsqueeze(0).to(CONFIG['device'])
            target_seq = torch.tensor(data_indices[i+1:i+CONFIG['seq_len']+1], dtype=torch.long).unsqueeze(0).to(CONFIG['device'])
            
            model.zero_grad()
            # Note: Le modèle retourne les probs dans forward() pour l'inférence.
            # Pour l'entrainement, il faut modifier un peu ou utiliser la logique interne.
            # On va utiliser la logique LSTM directement ici pour simplifier (copié de nfr_engine.py)
            
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

# --- FONCTIONS MAÎTRESSES ---

def compress_file(input_path, output_path, model):
    print(f">> DAEMON: COMPRESSION DE {input_path}...")
    start_time = time.time()
    
    # 1. Lire les données brutes
    with open(input_path, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    
    # 2. Entrainer le modèle sur ce fichier (Mode Archive Self-contained PoC)
    # Dans la pratique, on aurait un modèle pré-entrainé universel 
    # ou on stokerait les poids compressés dans le header.
    # Ici, nous supposons que le destinataire a le même script d'entrainement 
    # et va ré-entrainer (ce qui est triché), OU que le modèle est déjà bon.
    # Pour ce PoC "Archive", nous allons entrainer le modèle "fresh".
    train_network(model, data)

    # 3. Initialiser le flux de sortie
    bit_io = BitStreamIO(output_path, 'wb')
    
    # 4. ÉCRIRE LE HEADER (Taille du fichier)
    print(f">> HEADER: Writing Size {original_size}")
    bit_io.write_header(original_size)
    
    encoder = DaemonArithmeticCoder(bit_io)
    
    # 5. Initialiser le contexte (Padding de zéros)
    context = [0] * MODEL_SEQ_LEN
    
    model.eval()
    with torch.no_grad():
        for i, byte in enumerate(data):
            if i % 100 == 0:
                print(f"\r>> PROGRES: {i}/{len(data)} bytes", end="")
            
            # Préparer le contexte tensoriel (Shape: [1, seq_len])
            input_tensor = torch.tensor([context], dtype=torch.long).to(CONFIG['device'])
            
            # PRÉDICTION NEURONALE
            probs = model(input_tensor, None) # Retourne probs [1, 256] (voir nouveau forward)
            probs = probs[0].cpu() # Sur CPU pour arithmétique
            
            # ENCODAGE ARITHMÉTIQUE
            encoder.encode_symbol(byte, probs)
            
            # Mise à jour du contexte (Glissement)
            context.pop(0)
            context.append(byte)

    encoder.finish()
    bit_io.close_write()
    
    compressed_size = os.path.getsize(output_path)
    ratio = (1 - compressed_size/original_size) * 100
    
    print(f"\n>> TERMINÉ. Ratio: {ratio:.2f}% | Temps: {time.time() - start_time:.2f}s")
    print(f">> FICHIER CRÉÉ: {output_path} ({compressed_size} bytes)")

def decompress_file(input_path, output_path, model):
    print(f">> DAEMON: DÉCOMPRESSION DE {input_path}...")
    
    bit_io = BitStreamIO(input_path, 'rb')
    
    # 1. LIRE LE HEADER
    original_size = bit_io.read_header()
    print(f">> HEADER: Read Size {original_size}")
    
    bit_iterator = bit_io.read_bit_stream()
    decoder = DaemonDecoder(bit_iterator)
    
    decoded_data = bytearray()
    context = [0] * MODEL_SEQ_LEN
    
    # NOTE IMPORTANTE: Le modèle doit être dans le MÊME ÉTAT que lors de la compression.
    # Si nous avons fait un training "live" lors de la compression, 
    # le décompresseur doit avoir les poids exacts.
    # Pour ce PoC, l'objet 'model' passé en argument est déjà entrainé (car on est dans le même script).
    # Dans un vrai usage, il faudrait charger les poids.
    
    model.eval()
    with torch.no_grad():
        for i in range(original_size):
            if i % 100 == 0:
                print(f"\r>> PROGRES: {i}/{original_size} bytes", end="")
                
            input_tensor = torch.tensor([context], dtype=torch.long).to(CONFIG['device'])
            
            # PRÉDICTION NEURONALE
            probs = model(input_tensor, None)
            probs = probs[0].cpu()
            
            # DÉCODAGE ARITHMÉTIQUE
            symbol = decoder.decode_symbol(probs)
            
            decoded_data.append(symbol)
            
            # Mise à jour contexte
            context.pop(0)
            context.append(symbol)
            
    with open(output_path, 'wb') as f:
        f.write(decoded_data)
    print("\n>> DÉCOMPRESSION TERMINÉE.")

# --- EXECUTION ---
if __name__ == "__main__":
    # Test Data: Code source complexe + binaire (le pattern est important pour voir la compression)
    dummy_data = b"DAEMON_IS_WATCHING_YOU_" * 20 + os.urandom(50)
    with open("daemon_test.bin", "wb") as f:
        f.write(dummy_data)
        
    # Initialisation du modèle
    model = NeuralPredictor(CONFIG).to(CONFIG['device'])
    
    # 1. Compression (Incluant le training on-the-fly)
    compress_file("daemon_test.bin", "daemon_test.dmn", model)
    
    # 2. Décompression (Le fichier .dmn contient maintenant la taille en header)
    # model a les poids entrainés car on est dans la même session mémoire.
    decompress_file("daemon_test.dmn", "daemon_restored.bin", model)
    
    # Vérification
    with open("daemon_restored.bin", "rb") as f:
        restored = f.read()
    
    if restored == dummy_data:
        print(">> INTEGRITÉ CONFIRMÉE: 100%")
    else:
        print(">> ERREUR CRITIQUE DANS LA MATRICE.")
        print(f"Original: {len(dummy_data)}, Restored: {len(restored)}")
        for j in range(min(len(dummy_data), len(restored))):
             if dummy_data[j] != restored[j]:
                 print(f"First mismatch at index {j}: Orig={dummy_data[j]} vs Dec={restored[j]}")
                 print(f"Context around mismatch (Orig): {dummy_data[max(0, j-5):j+5]}")
                 print(f"Context around mismatch (Rest): {restored[max(0, j-5):j+5]}")
                 break
        if len(dummy_data) != len(restored):
            print("Length mismatch as well.")
