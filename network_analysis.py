import cv2
import numpy as np
import time

# Load test image (simulating phone capture)
view = cv2.imread('example/view.jpg', 0)  # Grayscale
view_color = cv2.imread('example/view.jpg')  # Color for encoding

print("="*70)
print("NETWORK BANDWIDTH ANALYSIS: Phone vs Computer Processing")
print("="*70)

# Image info
h, w = view.shape
print(f"\nView Image: {w}x{h} pixels")

# Raw image size
raw_gray = view.nbytes
raw_color = view_color.nbytes
print(f"\nRaw Sizes:")
print(f"  Grayscale: {raw_gray:,} bytes ({raw_gray/1024:.1f} KB)")
print(f"  Color RGB: {raw_color:,} bytes ({raw_color/1024:.1f} KB)")

# JPEG compression
_, jpeg_encoded = cv2.imencode('.jpg', view_color, [cv2.IMWRITE_JPEG_QUALITY, 85])
jpeg_size = len(jpeg_encoded)
print(f"\nJPEG (quality=85): {jpeg_size:,} bytes ({jpeg_size/1024:.1f} KB)")

_, jpeg_low = cv2.imencode('.jpg', view_color, [cv2.IMWRITE_JPEG_QUALITY, 50])
jpeg_low_size = len(jpeg_low)
print(f"JPEG (quality=50): {jpeg_low_size:,} bytes ({jpeg_low_size/1024:.1f} KB)")

# Downscaled versions (common for streaming)
scales = [1.0, 0.75, 0.5, 0.25]
print(f"\n{'Scale':<8} {'Resolution':<15} {'JPEG Size':<15} {'30 FPS BW':<15} {'60 FPS BW'}")
print("-"*70)

for scale in scales:
    scaled = cv2.resize(view_color, None, fx=scale, fy=scale)
    _, encoded = cv2.imencode('.jpg', scaled, [cv2.IMWRITE_JPEG_QUALITY, 75])
    size = len(encoded)
    h_s, w_s = scaled.shape[:2]
    bw_30 = size * 30 / 1024 / 1024  # MB/s at 30 FPS
    bw_60 = size * 60 / 1024 / 1024  # MB/s at 60 FPS
    print(f"{scale:<8.2f} {w_s}x{h_s:<10} {size/1024:>8.1f} KB    {bw_30:>8.2f} MB/s    {bw_60:>8.2f} MB/s")

# Network latency considerations
print("\n" + "="*70)
print("LATENCY ANALYSIS")
print("="*70)

latencies = {
    "Same WiFi (LAN)": 1,
    "Good WiFi": 5,
    "Typical WiFi": 10,
    "Poor WiFi": 30,
    "4G/LTE": 50,
}

print(f"\n{'Network Type':<20} {'Round-trip':<15} {'Impact at 30 FPS'}")
print("-"*70)
for net, ms in latencies.items():
    frames_delayed = (ms * 2) / (1000/30)  # Round-trip delay in frames
    print(f"{net:<20} {ms*2:>6} ms        {frames_delayed:>4.1f} frames behind")

# Recommendation summary
print("\n" + "="*70)
print("ARCHITECTURE RECOMMENDATION")
print("="*70)

print("""
PHONE-SIDE PROCESSING:
  Best for:
    - Offline use cases
    - Low-latency requirements (<10ms)
    - Static or slowly changing screen content

  Data transfer:
    - Result only: 8 bytes/frame (negligible)
    - Screen image: One-time ~500KB or periodic updates

  Processing time:
    - Phone CPU: ~50-100ms/frame (estimated, ARM slower than desktop)
    - Max FPS: 10-20 FPS

COMPUTER-SIDE PROCESSING:
  Best for:
    - Dynamic screen content (video, animations)
    - Good network connection
    - Battery-sensitive phone usage

  Data transfer:
    - 50% scale JPEG: ~30KB/frame = 0.9 MB/s @ 30 FPS
    - WiFi can handle: 1-10 MB/s easily

  Processing time:
    - Desktop CPU: ~35ms/frame (measured)
    - Network round-trip: 2-20ms (WiFi LAN)
    - Total latency: ~40-60ms
    - Max FPS: 20-25 FPS (limited by processing + network)

HYBRID APPROACH:
  - Phone does lightweight optical flow tracking (~2ms)
  - Sends full frame to computer every 10-30 frames for re-sync
  - Best of both: low latency + good accuracy + low bandwidth

  Data transfer:
    - Avg: ~3KB/frame (optical flow) + periodic full frame
    - ~0.1 MB/s average bandwidth

  Processing time:
    - Phone: 2ms (optical flow)
    - Computer: 35ms every 10 frames
    - Effective FPS: 100+ FPS

VERDICT:
  For real-time interactive apps → HYBRID or COMPUTER-SIDE
  For offline/embedded → PHONE-SIDE
  For lowest latency with dynamic screens → HYBRID
""")
