# Time of Day Image Classifier
This project implements a Convolutional Neural Network (CNN) to classify images based on the time of day they represent (e.g., morning, afternoon, evening, night).

## Overview

This image classifier uses a CNN to categorize images into different times of the day. It's trained on the "Time of Day Dataset" from Kaggle, which contains images labeled with different times of day. The model processes images and predicts the corresponding time of day category.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pillow (PIL)
- Matplotlib
- scikit-learn
- Kaggle API

## Setup

1. Clone this repository:
   ```
   git clone https[://github.com/yourusername/time-of-day-classifier.git](https://github.com/vedant0321/Day_time_detection)
   cd time-of-day-classifier
   ```

2. Install the required packages:
   ```
   pip install tensorflow numpy pillow matplotlib scikit-learn kaggle
   ```

3. Set up your Kaggle API credentials:
   - Download your Kaggle API key (`kaggle.json`) from your Kaggle account settings.
   - Place the `kaggle.json` file in your home directory:
     ```
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. Download and extract the dataset:
   ```
   kaggle datasets download -d aymenkhouja/timeofdaydataset
   unzip timeofdaydataset.zip -d timeofdaydataset
   ```

## Usage

1. Prepare the data:
   The `load_images` function in the script loads and preprocesses the images from the dataset directory.

2. Train the model:
   Run the main script to train the model:
   ```
   python time_of_day_classifier.py
   ```
   This script will:
   - Load and preprocess the data
   - Split the data into training and testing sets
   - Define and compile the CNN model
   - Train the model
   - Evaluate the model and display results

3. Visualize results:
   The script automatically generates:
   - A plot of training and validation accuracy
   - Predictions on a subset of test images
   - A confusion matrix

4. Classify new images:
   To classify new images, you can use the `preprocess_image` and prediction code at the end of the script. Modify it to accept command-line arguments for image paths.

## Model Architecture

The CNN architecture consists of:
- Input layer: (100, 100, 3) - RGB images resized to 100x100
- 3 Convolutional layers with ReLU activation (32, 64, and 128 filters)
- 3 MaxPooling layers
- Flatten layer
- Dense layer with 256 units and ReLU activation
- Output Dense layer with softmax activation (number of classes)

The model is compiled with:
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Metric: Accuracy

## Results

The model achieves an accuracy of 81.99% on the test set. 

The script generates:
- Learning curves showing training and validation accuracy over epochs
- A subset of predicted images with their true and predicted labels
- A confusion matrix to visualize the model's performance across different classes
