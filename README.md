# ✍️Handwriting Recognition with CNN-RNN-CTC

A simple Python application for **handwritten text recognition** using a hybrid **CNN-RNN architecture** with **CTC (Connectionist Temporal Classification) loss**. The project leverages **TensorFlow/Keras**, **OpenCV**, and the **IAM Handwriting Database** from Kaggle to recognize words from images of handwritten text.

## 🚀 Features

* 🧠 Hybrid CNN-RNN Model
* 🔗 CTC Loss Function
* 🧹 Data Preprocessing
* ⚡ Custom Data Pipeline
* 📚 Kaggle IAM Handwriting Dataset
* 📝 Model Evaluation & Prediction
* 🛠️ Flexible Architecture

## 📦 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/canonee/BIAI-proj.git
    cd BIAI-proj
    ```

2. **Install dependencies:**
    ```bash
    pip install tensorflow opencv-python numpy scikit-learn
    ```

3. **Prepare the dataset:**
* No need to download the dataset! All necessary data is already included in the repository under the DATA/ directory.

4. **Train the model:**
    ```bash
    python main.py
    ```

5. **Run prediction on a single image:**
    ```bash
    python testModel.py path/to/image.png
    ```

## 📂 Project Structure
    .
    ├── DATA/
    │   ├── words.new.txt
    │   └── iam_words/
    │       ├── words.txt		# Official IAM words metadata file
    │       └── words/
    │           ├── a01/
    │           ├── a02/
    │           ├── a03/
    │           └── ...		# Folders with word image files
    ├── DOC/
    │   └── Report.pdf		# Detailed project report and analysis
    ├── IMPL/
    │   ├── main.py			# Main training script and model definition
    │   └── testModel.py		# Script for loading a trained model and predicting single images

## ℹ️ Information
* **Version:** v1.1
* **Dataset:** Kaggle - IAM Handwriting Database
* **Technologies:** TensorFlow, Keras, OpenCV, NumPy, scikit-learn
* **Note:** For best results, ensure the dataset is correctly extracted and all dependencies are installed. The model can be further tuned by modifying network parameters in main.py as described in the report.
