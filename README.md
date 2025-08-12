# 🍎 Fruit Freshness Classification using EfficientNet-B0

This project uses a trained **EfficientNet-B0** deep learning model to classify fruits as **fresh** or **rotten**.  
It works in **real-time via your laptop camera** or by **uploading fruit images**.

Supported **10 classes**:

1. freshapples  
2. freshbanana  
3. freshmango  
4. freshoranges  
5. freshtomato  
6. rottenapples  
7. rottenbanana  
8. rottenmango  
9. rottenoranges  
10. rottentomato  

---

## 🛠 How to Set Up and Run

Follow these steps in order:

### 1. Download and Extract
Download this repository as a `.zip` file and extract it to a location on your computer.

### 2. Open in VS Code
Open Visual Studio Code, then go to File → Open Folder and select the extracted project folder.

### 3. Create a Virtual Environment
Open the VS Code terminal
python -m venv venv

### 4. Activate the Virtual Environment
venv\Scripts\activate

### 5. Install Dependencies
pip install -r requirements.txt

### 6. Run the Flask Application
python video_test_st.py

### 7. Open the Web App
After running, Flask will display:
Running on http://127.0.0.1:5000/
Open this link in your browser.

### 8. Test the Model
For real-time testing: Allow camera access in your browser and show a fruit to your laptop camera.
For image upload testing: Click the "Upload Image" button in the web app and select a fruit image.

### Model Performance
Training Accuracy: 99.81%
Test Accuracy: 99.72%

### Requirements
Python 3.8 or higher
Flask
OpenCV
TensorFlow / PyTorch (depending on your model)
NumPy
Other dependencies listed in requirements.txt




