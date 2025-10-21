import screenpoint
import cv2
import time
import numpy as np

# Load test images
screen = cv2.imread('example/screen.png', 0)
view = cv2.imread('example/view.jpg', 0)

print(f"Image sizes - Screen: {screen.shape}, View: {view.shape}")
print("="*60)

# Benchmark ORB
print("\nBenchmarking ORB...")
orb_times = []
for i in range(10):
    start = time.time()
    x, y = screenpoint.project(view, screen, algorithm='orb')
    elapsed = time.time() - start
    orb_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.2f}ms - Result: ({x}, {y})")

orb_avg = np.mean(orb_times)
orb_std = np.std(orb_times)
orb_fps = 1.0 / orb_avg

print(f"\nORB Results:")
print(f"  Average: {orb_avg*1000:.2f}ms ± {orb_std*1000:.2f}ms")
print(f"  FPS: {orb_fps:.2f}")
print(f"  Min: {min(orb_times)*1000:.2f}ms, Max: {max(orb_times)*1000:.2f}ms")

# Benchmark SIFT
print("\n" + "="*60)
print("\nBenchmarking SIFT...")
sift_times = []
for i in range(10):
    start = time.time()
    x, y = screenpoint.project(view, screen, algorithm='sift')
    elapsed = time.time() - start
    sift_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.2f}ms - Result: ({x}, {y})")

sift_avg = np.mean(sift_times)
sift_std = np.std(sift_times)
sift_fps = 1.0 / sift_avg

print(f"\nSIFT Results:")
print(f"  Average: {sift_avg*1000:.2f}ms ± {sift_std*1000:.2f}ms")
print(f"  FPS: {sift_fps:.2f}")
print(f"  Min: {min(sift_times)*1000:.2f}ms, Max: {max(sift_times)*1000:.2f}ms")

# Comparison
print("\n" + "="*60)
print("\nCOMPARISON:")
print(f"  Speedup: {sift_avg/orb_avg:.2f}x faster with ORB")
print(f"  FPS Improvement: {orb_fps:.2f} FPS (ORB) vs {sift_fps:.2f} FPS (SIFT)")
print(f"  Time saved per frame: {(sift_avg - orb_avg)*1000:.2f}ms")
