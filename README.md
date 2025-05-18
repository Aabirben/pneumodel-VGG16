Pneumonia Detection using VGG16 - Dataset Preprocessing and Model Development
==========================================================================

Project Overview
----------------

This project aims to detect pneumonia from chest X-ray images by leveraging a deep learning model based on the VGG16 architecture. It includes detailed dataset preprocessing to prepare images for the model and the design of a robust classifier to distinguish between Normal and Pneumonia cases.

Dataset Preprocessing
---------------------

### 1. Initial Dataset Overview

-The original dataset is located at /content/drive/MyDrive/Pneumonia_Project/pneumonia_data/chest_xray.

-Dataset split into train, validation (val), and test sets.

-Classes: NORMAL and PNEUMONIA.

-Initial imbalance:

Train: 1341 NORMAL, 3875 PNEUMONIA

Val: 8 NORMAL, 8 PNEUMONIA

Test: 234 NORMAL, 390 PNEUMONIA

This imbalance risks biasing the model toward the Pneumonia class.

### 2. Data Augmentation for Balancing

-Augmented images were added (prefixed with "aug_") to train and val sets to balance classes.

-Post-augmentation:

Train: 3875 NORMAL (including 2534 augmented), 3875 PNEUMONIA (total 7750 images)

Val: 50 NORMAL (42 augmented), 50 PNEUMONIA (42 augmented)

-Test remains unchanged to reflect real-world distribution.

### 3. Preprocessing Pipeline

-Each image undergoes the following steps via the process_and_save_image function:

-Convert to grayscale.

-Normalize pixel values between 0 and 1.

-Apply histogram equalization to enhance contrast.

-Re-normalize to 0–255 range.

-Resize to 224x224 pixels (required by VGG16).

-Convert grayscale image to RGB by duplicating channels.

-Save processed images maintaining folder structure under chest_xray_processed_v4.

### 4. Validation and Output

-Visual checks confirmed contrast improvement.

-Final processed dataset contains 8474 images (including augmentations), all 224x224 RGB.

-This ensures compatibility with VGG16 and enhances feature extraction.

Model Development
-----------------

### 1. Model Architecture

-Base model: VGG16 pre-trained on ImageNet (without top fully connected layers).

-Input size: 224x224x3 RGB images.

-Preprocessing input using VGG16's preprocess_input for consistent normalization.

### 2. Custom Classifier Layers

-Convolutional layers frozen to preserve learned features.

Added layers:

-GlobalAveragePooling2D to reduce dimensions.

-Two Dense layers with 1024 neurons + ReLU activations.

-Two Dropout layers (0.5 rate) to reduce overfitting.

-Dense layer with 512 neurons + ReLU.

-Final Dense layer with 1 neuron and sigmoid activation for binary classification.

### 3. Training Details

-Optimizer: Adam.

-Loss: Binary Crossentropy.

-Metric: Accuracy.

-Data fed through a custom generator with batch size 32 and shuffling for train.

-Trained for 10 epochs with checkpoints saved automatically and manually at select epochs.

-Results:

Training accuracy ~98%

Validation accuracy 68–78%

Test accuracy ~81.5%

### 4. Addressing Class Imbalance Bias

-Despite balanced training set, test set remains imbalanced, causing bias toward Pneumonia.

-Observed high recall but lower precision on Pneumonia class.

-Fine-tuning last convolutional blocks and increasing dropout rate helped reduce overfitting and bias.

-Final model checkpoints provide options to select based on trade-offs between precision and recall.

How to Use
----------

### Dataset Preparation

-Place your original dataset in the folder path described.

-Run the preprocessing script to generate processed images in chest_xray_processed_v4.

### Model Training

-Use the provided training script.

-Ensure to use the custom data generator for efficient batch loading.

-Monitor model checkpoints saved during training.

### Evaluation

-Use test set without augmentation to evaluate performance under real conditions.

-Analyze precision, recall, and accuracy metrics for model robustness.

Contact
-------

For any questions or support, please contact Pneumodel team at [aabirbenhamamouche@gmail.com].
