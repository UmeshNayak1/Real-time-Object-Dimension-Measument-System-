📏 Real-Time Object Dimension Measurement System

A computer vision–based system that measures the dimensions of real-world objects in real time using a camera.
The system uses image processing techniques to detect objects, calculate their size, and display measurements instantly.

This project demonstrates the practical application of computer vision and image processing using OpenCV to estimate object dimensions from images or live camera input. Real-time measurement systems like this are widely used in manufacturing, robotics, and automation environments where quick size estimation is required.

🚀 Features

✔ Real-time object detection using a webcam
✔ Automatic contour detection and object boundary extraction
✔ Measurement of width and height of objects
✔ Real-time display of calculated dimensions
✔ Image processing using OpenCV
✔ Works with both images and live camera feed

🧠 How It Works

The system follows these steps:

1️⃣ Image Acquisition
The camera captures a frame containing the object.

2️⃣ Preprocessing
The image is converted to grayscale and filtered to remove noise.

3️⃣ Edge Detection
Edges of the object are detected using computer vision algorithms.

4️⃣ Contour Detection
Contours are identified to locate the object boundaries.

5️⃣ Dimension Calculation
The system calculates the width and height of the object using pixel-to-real-world scaling.

6️⃣ Visualization
The measured dimensions are displayed directly on the screen.

🛠️ Tech Stack

Programming Language

Python

Libraries

OpenCV

NumPy

Tools

Webcam / Camera

Python IDE (VS Code / PyCharm)

📂 Project Structure
Real-time-Object-Dimension-Measument-System
│
├── measure_object_size.py
├── measure_object_size_camera.py
├── object_detector.py
├── images/
│   └── sample_images
├── output/
│   └── result_images
└── README.md
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/UmeshNayak1/Real-time-Object-Dimension-Measument-System-.git
2️⃣ Navigate to the Project Folder
cd Real-time-Object-Dimension-Measument-System-
3️⃣ Install Dependencies
pip install opencv-python numpy
▶️ Usage
Run with Webcam
python measure_object_size_camera.py
Run with Image Input
python measure_object_size.py

The program will detect objects and display their measured dimensions on the screen.

📊 Applications

This system can be used in many fields:

📦 E-commerce product measurement

🏭 Industrial quality control

🤖 Robotics and automation

🚗 Autonomous systems

📏 Educational computer vision projects

📸 Output Example

Example output shows detected objects with bounding boxes and dimension labels.

Width: XX cm
Height: XX cm
🎯 Future Improvements

Improve measurement accuracy

Add deep learning based object detection

Support multiple object measurements simultaneously

Build a GUI interface

Deploy as a web application
