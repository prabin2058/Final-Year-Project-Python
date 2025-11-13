## Manual Face Detection (from-scratch inference)

This project contains a manual implementation of a real-time face detector in JavaScript. No third‑party detection libraries are used; only a pretrained cascade model file is included as data.

### Components
* `manual_detect.js`: Grayscale conversion, cascade unpacking, sliding-window scanning, decision-tree classification, non‑max suppression (clustering), and temporal memory.
* `examples/webcam.html`: Minimal demo that captures webcam frames, runs detection, and draws results.
* `examples/facefinder`: Pretrained cascade binary (data only).

### How it works
1. Load the pretrained cascade from `examples/facefinder`.
2. Convert each frame to grayscale.
3. Slide a window over multiple scales and classify regions using a cascade of decision trees based on pixel intensity comparisons.
4. Cluster overlapping detections and render circles for faces.

### Run the demo
Serve the `examples/` folder via an HTTP server and open `examples/webcam.html` in your browser. Grant webcam permission.

On Windows, a quick way is PowerShell from the project root:
```
pwsh -NoProfile -Command "cd examples; python -m http.server 8000"
```
Then open `http://localhost:8000/webcam.html`.

### License
MIT.
