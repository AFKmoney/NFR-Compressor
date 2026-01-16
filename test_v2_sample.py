
import os
import time
from daemon_nfr import NFRBlockEngine

def test_sample():
    print("[TEST] Creating sample...")
    # Create 1KB random file (but with pattern so compression is possible/visible)
    data = b"DAEMON_NFR_V2_Is_Great_" * 50
    with open("sample_v2.bin", "wb") as f:
        f.write(data)
        
    engine = NFRBlockEngine()
    
    print("[TEST] Compressing...")
    # epochs=1 for speed
    engine.compress("sample_v2.bin", "sample_v2.dmn", epochs=1)
    
    print("[TEST] Decompressing...")
    # Delete model to ensure it loads from disk if logic requires (but we just saved it)
    engine.decompress("sample_v2.dmn", "sample_v2_restored.bin")
    
    # Verify
    with open("sample_v2_restored.bin", "rb") as f:
        restored = f.read()
        
    if restored == data:
        print("[TEST] SUCCESS: Bit-perfect reconstruction.")
    else:
        print("[TEST] FAILURE: Data mismatch.")
        print(f"Orig: {len(data)}, Restored: {len(restored)}")

if __name__ == "__main__":
    test_sample()
