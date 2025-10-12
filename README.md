# 🐵 Face + Hand Expression Meme Display

A fun and interactive Python project using **MediaPipe** and **OpenCV** that detects facial expressions and hand gestures to display different meme images in real time.

**GitHub:** [https://github.com/pathipat11/mediapipe-project.git](https://github.com/pathipat11/mediapipe-project.git)

Perfect for learning:

* Real-time computer vision with OpenCV
* MediaPipe Face Mesh & Hands tracking
* Gesture-based interaction
* Landmark distance analysis

## 🎭 Expressions Supported

| Expression    | Trigger Condition                  |
| ------------- | ---------------------------------- |
| **Serious**   | Default (no gesture detected)      |
| **Shy**       | Index finger near the mouth        |
| **Thinking**  | Index finger raised above forehead |
| **Surprised** | Mouth open                         |

Each expression automatically switches to a corresponding meme image.

## ✨ Features

* 🧠 Real-time face and hand tracking using **MediaPipe**
* 🖼️ Dual-window display: Camera Input + Meme Output
* 🧩 Expression detection with smooth overlay labels
* 🎨 Transparent overlay labels for cleaner visualization
* 🔄 Automatic meme switching based on detected expressions

## 🧰 Requirements

* **Python 3.11**
* Webcam (built-in or USB)
* 4 meme images in `assets/` directory:

  * `the-monkey-serious-meme.png`
  * `the-monkey-shy-meme.png`
  * `the-monkey-thinking-meme.png`
  * `the-monkey-surprised-meme.png`

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/pathipat11/mediapipe-project.git
cd mediapipe-project
```

### 2. Install Dependencies

```bash
python3.11 -m pip install -r requirements.txt
```

### 3. Add Meme Images

Place your 4 meme images inside the `assets/` folder. Make sure they match the filenames listed above.

### 4. Run the Program

**Windows:**

```bash
python3.11 main.py
```

**macOS/Linux:**

```bash
python3 main.py
```

## 🕹️ Controls

* Press **'q'** to quit the program.

## 🧠 How It Works

1. **Face Mesh Detection:** Tracks 468 facial landmarks.
2. **Hand Tracking:** Tracks 21 landmarks per hand.
3. **Expression Logic:** Combines mouth distance + finger position to classify emotion.
4. **Meme Output:** Displays the corresponding meme image in a separate window.

### Detection Flow

1. Mouth open → `Surprised`
2. Index finger near mouth → `Shy`
3. Index finger above forehead → `Thinking`
4. None of the above → `Serious`

## 🧩 Technical Details

* **Face Detection:** MediaPipe Face Mesh
* **Hand Detection:** MediaPipe Hands
* **Drawing:** OpenCV for overlay visualization
* **Resolution:** 960×720 (for both windows)
* **Overlay Labels:** Semi-transparent rounded rectangles with text

## 💡 Customization Ideas

* Add new gestures (peace sign, thumbs up)
* Integrate sound effects per expression
* Save frames when expression changes
* Add emoji overlays or filters
* Build a GUI slider for sensitivity tuning

## 🚑 Troubleshooting

### Webcam Not Detected

* Check if another app is using the camera
* Change camera index in `main.py`: `cap = cv2.VideoCapture(1)`

### Meme Images Not Found

* Make sure all 4 PNGs exist in `assets/`
* Use correct filenames (case-sensitive)

### Detection Too Sensitive

Adjust the thresholds in `main.py`:

```python
MOUTH_OPEN_THRESHOLD = 0.045
HAND_FACE_DISTANCE_THRESHOLD = 0.08
```

Increase or decrease these values as needed.

## 🧩 Project Structure

```
mediapipe-project/
├── assets/              # Folder containing meme images
│     ├── the-monkey-serious-meme.png
│     ├── the-monkey-shy-meme.png
│     ├── the-monkey-thinking-meme.png
│     └── the-monkey-surprised-meme.png
├── main.py              # Main application script (well-commented for learning)
├── requirements.txt     # Python dependencies
├── run.bat              # Windows batch file to run with Python 3.11
├── run.sh               # macOS/Linux shell script to run with Python 3.11
├── README.md            # This file - project overview and documentation
├── QUICKSTART.md        # Quick 5-minute setup guide
├── TUTORIAL.md          # Detailed tutorial on how the code works
├── IMAGE_GUIDE.md       # Guide for finding and preparing images
├── SETUP_MACOS.md       # macOS-specific setup guide
├── CONTRIBUTING.md      # Guide for contributors
├── LICENSE              # MIT License
├── .gitignore           # Git ignore file
├── apple.png            # Meme image (normal state) - YOU NEED TO ADD THIS
└── appletongue.png      # Meme image (tongue out) - YOU NEED TO ADD THIS
```

## 🧾 Dependencies

* `mediapipe==0.10.7`
* `opencv-python==4.8.1.78`
* `numpy==1.24.3`

## 🧑‍💻 Contributing

Pull requests are welcome! You can:

* Improve expression logic
* Add new meme reactions
* Enhance overlay UI
* Optimize performance

## 🪟 Platform Support

| Platform         | Status            | Notes                               |
| ---------------- | ----------------- | ----------------------------------- |
| 🪟 Windows 10/11 | ✅ Fully Supported | Recommended for beginners           |
| 🍎 macOS 10.14+  | ✅ Supported       | Camera permission required          |
| 🐧 Linux         | ✅ Supported       | Add user to `video` group if needed |

## 📜 License

Released under the **MIT License** — free for personal and educational use.

---

🎥 *Created by Pathipat Mattra — A computer vision playground built with love and monkeys!* 🐒
