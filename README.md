# Multi-Class Flower Classification using CNN

##  Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify flower images into **17 different categories**.

The model learns from labeled flower images and predicts the class of unseen images.

---

##  Objective

* Build an image classification model using deep learning
* Classify flower images into 17 classes
* Achieve good prediction accuracy on unseen data

---

##  Project Structure

```
project_yourname/
│
├── checkpoints/
│   └── final_weights.pth
│
├── data/
│   ├── daisy/
│   ├── blueball/
│   └── ... (total 17 folders)
│
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── interface.py
├── config.py
├── README.md
```

---

## Dataset

* Source: Kaggle Flower Dataset
* Number of classes: 17
* Format: RGB images
* Data used: Sample images (70 per class)

---

##  Requirements

Install dependencies:

```
pip install torch torchvision pillow
```

---

##  How to Run

###  Step 1: Train the Model

```
python train.py
```

This will:

* Load images from `data/`
* Train the CNN model
* Save weights in `checkpoints/final_weights.pth`

---

###  Step 2: Run Prediction

```
python predict.py
```

This will:

* Load trained model
* Predict class labels for test images

---

##  Model Architecture

The CNN model consists of:

* 3 Convolution layers (feature extraction)
* ReLU activation
* MaxPooling layers
* Fully connected layers
* Output layer (17 classes)

---

##  Input

* RGB flower image

##  Output

* Predicted class label (0–16)

---

##  Configuration

All hyperparameters are defined in `config.py`:

* Batch size
* Epochs
* Learning rate
* Image size

---

##  Methodology

1. Load dataset using custom Dataset class
2. Preprocess images (resize, tensor conversion)
3. Train CNN using cross-entropy loss
4. Evaluate predictions

---

##  Results

* Model trained successfully
* Produces predictions for unseen images
* Expected accuracy: 70%–90%

---

##  Notes

* Ensure `data/` folder is correctly structured
* Ensure `checkpoints/` folder exists before training
* Only sample dataset is included (not full dataset)

---

##  Interface

The `interface.py` file maps:

* Model → `TheModel`
* Trainer → `the_trainer`
* Predictor → `the_predictor`
* Dataset → `TheDataset`
* Dataloader → `the_dataloader`

This allows automated grading.

---
