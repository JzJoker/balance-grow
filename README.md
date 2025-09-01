# Hairline Distance Measurement Demo

## Overview

This project demonstrates **real-time hair segmentation and eyebrow-to-hairline distance measurement** using **Google’s MediaPipe models** and a **Self-Trained Segmentation Model**.
It uses:

* **Hair segmentation** – to detect hair regions.
* **Face mesh / landmarks** – to detect eyes and eyebrows.
* **Python + OpenCV** – for live camera capture and overlay.

The main goal is to calculate **vertical distances from eyebrows to hairline** and visualize them in real time.

---

## Features

* Real-time video capture and processing.
* Hair segmentation mask visualization using MediaPipe (default) or a self-trained U-Net model.
* Automatic detection of eyebrows using MediaPipe Face Mesh.
* Vertical line overlay from eyebrows to hairline.
* Distance metrics calculation (pixel-based).

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/JzJoker/balance-grow.git
cd balance-grow
```

2. **Install dependencies** (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Model Demo

This will run hair segmentation on a **single input image** and display a mask overlay.

#### Command:

```bash
cd src
python inference_segmentation.py --img ..\data\test\no_hand+balding+wet.png
```

#### Parameters:

| Parameter | Type    | Default      | Description                                                                |
| --------- | ------- | ------------ | -------------------------------------------------------------------------- |
| `--img`   | `str`   | **required** | Path to the input image.                                                   |
| `--flip`  | `bool`  | `False`      | Optionally flip the input image horizontally.                              |
| `--out`   | `str`   | `output.png` | Path to save the output overlay image.                                     |
| `--alpha` | `float` | `0.5`        | Transparency of the overlay (0.0 = fully transparent, 1.0 = fully opaque). |


**Example Output:**

![Overlay Demo](screenshots/overlay_demo.png)

**Notes:**

* The overlay shows **hair in red** and **skin in green**.
* High-confidence pixels are more opaque; lower-confidence areas appear faint.
* Works with both **your trained U-Net model** or **MediaPipe hair segmentation**, depending on what’s loaded in the repo.
* Useful for **testing single images before moving to real-time webcam demos**.

### Run Real-Time Mapping Demo
```bash
python main.py
```

* Opens your webcam (or image input).
* Shows hair segmentation overlay and eyebrow-to-hairline lines.
* Prints distances in pixels in the console or overlays them on the video feed.

### Parameters

| Parameter     | Description                                        | Default |
|---------------|----------------------------------------------------|---------|
| `--img`       | Path to input image                                | *Required* |
| `--flip`      | Bool to flip image horizontally                   | False   |
| `--out`       | Path to save output overlay                        | `output.png` |
| `--alpha`     | Overlay transparency for hair mask                | 0.5     |
| `--threshold` | Confidence threshold for hair segmentation mask   | 0.7     |


---

## License & Attribution

This project is licensed under the **MIT License**.

**Attribution:**

* This project uses **MediaPipe**: © Google, licensed under Apache 2.0.

  * MediaPipe GitHub: [https://github.com/google/mediapipe](https://github.com/google/mediapipe)
  * Apache 2.0 license: [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Future Work
* Implement Google's MediaPipe face mesh and hair segmentation models
* Create desktop demo for real-time mapping
* Port to **iOS** for on-device real-time measurement.
* Add metrics output in millimeters using a reference scale.
* Optimize for **higher FPS** on mobile devices.

---

## Screenshots / GIFs

*(Optional: Add a few images or short GIFs showing the overlay lines and distances.)*
