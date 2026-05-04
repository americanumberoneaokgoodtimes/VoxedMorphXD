import numpy as np
import voxed

def test_parse_time():
    print("Testing parse_time_to_seconds...")
    assert voxed.parse_time_to_seconds("27s") == 27.0
    assert voxed.parse_time_to_seconds("1m30s") == 90.0
    assert voxed.parse_time_to_seconds("01:10") == 70.0
    assert voxed.parse_time_to_seconds("00:00:27") == 27.0
    print("  [OK] parse_time_to_seconds verified.")

def test_lpc():
    print("Testing LPC pole extraction...")
    # Generate a simple sine wave
    sr = 16000
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(sr)
    
    poles = voxed.get_lpc_poles(audio[:1024])
    print(f"  [INFO] Extracted {len(poles)} poles.")
    assert len(poles) > 0
    
    # Check if poles_to_poly works
    poly = voxed.poles_to_poly(poles)
    assert len(poly) > 0
    print("  [OK] LPC logic verified.")

if __name__ == "__main__":
    try:
        test_parse_time()
        test_lpc()
        print("\n=== ALL TESTS PASSED ===")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
