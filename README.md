# ✨ AirDraw 3D – AI Powered Gesture-Based 3D Shape Creator

AirDraw 3D is an advanced **air-drawing & gesture-controlled 3D shape generation system**.  
It uses **computer vision**, **MediaPipe hand tracking**, **Kalman filtering**, and an  
**ultra-accurate multi-shape recognition system** to detect your drawn shapes in mid-air  
and convert them into **3D wireframe models**.

---

## 🚀 Features

### ✅ **Gesture-Controlled Air Drawing**
- **☝ Index Finger Up (Hover)** → Ready to draw  
- **✍ Hover for 0.4 seconds** → Starts drawing automatically  
- **👉 Index Finger draws in air**  
- **✊ Fist** → Stops drawing, detects shape, generates 3D output  
- **✋ Palm** → Clears screen & resets all drawings  

---

## 🎯 **Supported Shapes (Ultra Accurate Detection)**

### ✅ Geometric Shapes
- Circle  
- Oval  
- Triangle  
- Square  
- Rectangle  
- Diamond  
- Parallelogram  
- Trapezium  
- Pentagon  
- Hexagon  
- Heptagon  
- Octagon  

### ✅ Special Complex Shapes (Template + Hu Moments)
- ⭐ Star  
- ❤️ Heart  
- ➡ Arrow  
- ➕ Plus  
- ❌ Cross  
- ∞ Infinity  

---

## 🔮 **3D Shape Output**
After locking with a **Fist gesture**, your air-drawn shape is converted to a 3D figure:

- Square → Cube  
- Rectangle → Cuboid  
- Triangle → Triangular Prism  
- Pentagon → Pentagonal Prism  
- Circle/Oval → Sphere / Elliptic structure  
- Hexagon, Heptagon, Octagon → Polygonal Prisms  
- Heart / Arrow / Star → Extruded 3D outline  

---

## ⚙️ **Technology Stack**
- **Python 3.10+**
- **OpenCV** (image processing, 3D drawing)
- **MediaPipe Hands** (hand tracking)
- **NumPy** (math + geometry)
- **Kalman Filter** (smooth fingertip tracking)
- **Hu Moments** (complex shape recognition)
- **RDP Polygon Approximation** (shape corner detection)
- **Morphology + Contour Extraction** (clean outlines)

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/airdraw-3d.git
cd airdraw-3d

pip install opencv-python mediapipe numpy

python gestureRecognition.py

HOVER_FRAMES_TO_START = 12
HOVER_MOVEMENT_PX = 10
if best_dist < 3.8:
