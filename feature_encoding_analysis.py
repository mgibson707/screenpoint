import cv2
import numpy as np
import pickle
import gzip
import json
import struct

view = cv2.imread('example/view.jpg', 0)
orb = cv2.ORB_create(nfeatures=1000)
kp, des = orb.detectAndCompute(view, None)

print("="*70)
print("FEATURE ENCODING ANALYSIS")
print("="*70)

print(f"\nDetected {len(kp)} keypoints")

# 1. Show what's in a keypoint
print("\n1. KEYPOINT DATA STRUCTURE")
print("-"*70)
print(f"Example keypoint:")
k = kp[0]
print(f"  pt (x, y):  {k.pt}  (float32, float32)")
print(f"  size:       {k.size}  (float32)")
print(f"  angle:      {k.angle}  (float32)")
print(f"  response:   {k.response}  (float32)")
print(f"  octave:     {k.octave}  (int)")
print(f"  class_id:   {k.class_id}  (int)")

# 2. Show descriptor structure
print("\n2. ORB DESCRIPTOR STRUCTURE")
print("-"*70)
print(f"Descriptor shape: {des.shape}")
print(f"Descriptor dtype: {des.dtype}")
print(f"Descriptor size:  {des.nbytes} bytes ({des.nbytes/1024:.1f} KB)")
print(f"\nExample descriptor (first keypoint):")
print(f"  {des[0][:16]}... (showing first 16 of 32 bytes)")
print(f"  As binary: {''.join(format(b, '08b') for b in des[0][:4])}... (256 bits total)")

# 3. Different encoding schemes
print("\n3. ENCODING SCHEMES COMPARISON")
print("-"*70)

# Method 1: Pickle (what we used before)
kp_data_full = [(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]
pickle_full = pickle.dumps((kp_data_full, des))
pickle_gzip = gzip.compress(pickle_full)

print(f"\nMethod 1: Pickle + Gzip (current)")
print(f"  Keypoint data: {[(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp][:1]}")
print(f"  Size: {len(pickle_gzip)/1024:.1f} KB")

# Method 2: Minimal pickle (drop octave, class_id, response - not needed for matching)
kp_data_minimal = [(k.pt[0], k.pt[1], k.size, k.angle) for k in kp]
pickle_minimal = pickle.dumps((kp_data_minimal, des))
pickle_minimal_gzip = gzip.compress(pickle_minimal)

print(f"\nMethod 2: Pickle Minimal (x, y, size, angle only)")
print(f"  Keypoint data: {[(k.pt[0], k.pt[1], k.size, k.angle) for k in kp][:1]}")
print(f"  Size: {len(pickle_minimal_gzip)/1024:.1f} KB")
print(f"  Savings: {(1 - len(pickle_minimal_gzip)/len(pickle_gzip))*100:.1f}%")

# Method 3: Binary struct packing (most efficient)
# Pack as: num_keypoints (4 bytes), then for each keypoint:
#   x, y, size, angle (4 floats = 16 bytes)
#   descriptor (32 bytes)
# Total per keypoint: 48 bytes

binary_data = bytearray()
binary_data.extend(struct.pack('<I', len(kp)))  # num keypoints (unsigned int)

for k, d in zip(kp, des):
    # Pack keypoint: x, y, size, angle (4 floats)
    binary_data.extend(struct.pack('<4f', k.pt[0], k.pt[1], k.size, k.angle))
    # Pack descriptor: 32 bytes
    binary_data.extend(d.tobytes())

binary_gzip = gzip.compress(bytes(binary_data))

print(f"\nMethod 3: Binary struct packing")
print(f"  Format: num_kp(4) + [(x,y,size,angle)(16) + descriptor(32)]")
print(f"  Uncompressed: {len(binary_data)} bytes ({len(binary_data)/1024:.1f} KB)")
print(f"  Compressed: {len(binary_gzip)/1024:.1f} KB")
print(f"  Bytes per keypoint: {len(binary_data)/len(kp):.1f}")
print(f"  Savings vs pickle: {(1 - len(binary_gzip)/len(pickle_gzip))*100:.1f}%")

# Method 4: NumPy native format (.npz compressed)
import io
npz_buffer = io.BytesIO()
kp_array = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)
np.savez_compressed(npz_buffer, keypoints=kp_array, descriptors=des)
npz_size = npz_buffer.tell()

print(f"\nMethod 4: NumPy compressed (.npz)")
print(f"  Size: {npz_size/1024:.1f} KB")
print(f"  Savings vs pickle: {(1 - npz_size/len(pickle_gzip))*100:.1f}%")

# Method 5: JSON (for comparison - not recommended)
kp_json = json.dumps({
    'keypoints': kp_data_minimal,
    'descriptors': des.tolist()
})
json_gzip = gzip.compress(kp_json.encode())

print(f"\nMethod 5: JSON + Gzip (for comparison)")
print(f"  Size: {len(json_gzip)/1024:.1f} KB")
print(f"  Note: JSON is human-readable but inefficient")

# Summary table
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n{'Method':<30} {'Size (KB)':<12} {'Efficiency':<15}")
print("-"*70)
methods = [
    ("Pickle (full)", len(pickle_gzip)/1024, "Baseline"),
    ("Pickle (minimal)", len(pickle_minimal_gzip)/1024,
     f"{(1-len(pickle_minimal_gzip)/len(pickle_gzip))*100:.1f}% smaller"),
    ("Binary struct", len(binary_gzip)/1024,
     f"{(1-len(binary_gzip)/len(pickle_gzip))*100:.1f}% smaller"),
    ("NumPy compressed", npz_size/1024,
     f"{(1-npz_size/len(pickle_gzip))*100:.1f}% smaller"),
    ("JSON", len(json_gzip)/1024,
     f"{(len(json_gzip)/len(pickle_gzip)-1)*100:.1f}% larger"),
]

for method, size, eff in methods:
    print(f"{method:<30} {size:>8.1f}      {eff:<15}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
BEST: Binary struct packing (Method 3)
  ✅ Most efficient: ~30 KB for 1000 keypoints
  ✅ Language agnostic (can decode in any language)
  ✅ Simple format: easy to implement
  ✅ Fixed size per keypoint (predictable bandwidth)

Format specification:
  Header:
    - num_keypoints: 4 bytes (uint32, little-endian)

  Per keypoint (48 bytes each):
    - x:          4 bytes (float32)
    - y:          4 bytes (float32)
    - size:       4 bytes (float32)
    - angle:      4 bytes (float32)
    - descriptor: 32 bytes (uint8[32])

  Then gzip compress everything.

Decoding (pseudo-code):
  data = gzip.decompress(received_bytes)
  num_kp = struct.unpack('<I', data[0:4])[0]

  for i in range(num_kp):
      offset = 4 + i * 48
      x, y, size, angle = struct.unpack('<4f', data[offset:offset+16])
      descriptor = data[offset+16:offset+48]  # 32 bytes

      keypoints.append(KeyPoint(x, y, size, angle))
      descriptors.append(descriptor)

Alternative: NumPy .npz (Method 4) if both sides use Python
  ✅ Almost as efficient
  ✅ Dead simple: np.savez_compressed() / np.load()
  ❌ Python-specific format
""")

# Show actual decoding example
print("\n" + "="*70)
print("DECODING EXAMPLE (Binary struct)")
print("="*70)

# Decode the binary format we created
decoded_data = gzip.decompress(binary_gzip)
num_kp = struct.unpack('<I', decoded_data[0:4])[0]
print(f"\nDecoded {num_kp} keypoints")

# Decode first keypoint
x, y, size, angle = struct.unpack('<4f', decoded_data[4:20])
descriptor = np.frombuffer(decoded_data[20:52], dtype=np.uint8)

print(f"\nFirst keypoint:")
print(f"  x={x:.2f}, y={y:.2f}, size={size:.2f}, angle={angle:.2f}")
print(f"  descriptor: {descriptor[:8]}... ({len(descriptor)} bytes)")

# Verify it matches original
print(f"\nOriginal first keypoint:")
print(f"  x={kp[0].pt[0]:.2f}, y={kp[0].pt[1]:.2f}, size={kp[0].size:.2f}, angle={kp[0].angle:.2f}")
print(f"  descriptor: {des[0][:8]}... ({len(des[0])} bytes)")
print(f"\n✅ Match: {np.allclose([x,y,size,angle], [kp[0].pt[0],kp[0].pt[1],kp[0].size,kp[0].angle])}")
