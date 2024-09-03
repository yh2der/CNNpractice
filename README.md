# CNNpractice

## Project Overview

This project focuses on image classification tasks using PyTorch and the CIFAR-10 dataset. We have developed four different versions of Convolutional Neural Network (CNN) models and conducted various experiments, including the application of techniques such as Dropout and Batch Normalization.

## Dataset

### The CIFAR-10 dataset used in this project can be accessed via Google Drive:
    It will download while training so no need to download to your local.
[CIFAR-10 Dataset Link](https://drive.google.com/drive/folders/1P_tMDpkjWsZY8JXhQhG-HHeGbsHkDkEr?usp=drive_link)

## CNN Models

This project implements four different Convolutional Neural Network (CNN) models for image classification:

1. **Advanced CNN with Residual Connections**: A deeper model featuring residual blocks and batch normalization.

2. **Simple CNN with Dropout**: A basic CNN model that heavily utilizes dropout for regularization.

3. **CNN with Batch Normalization and Dropout**: A model that combines both batch normalization and dropout throughout its architecture.

4. **CNN with Selective Regularization**: A model using batch normalization in all layers but applying dropout only in fully connected layers.

Each model is designed to explore different aspects of CNN architecture and regularization techniques.

## Getting Started

1. **Set Up Python Environment**: Ensure your Python environment is ready with all necessary libraries installed.

2. **Open Notebook in Google Colab**: Open the training `.ipynb` file in Google Colab, confirming the model is correctly imported and the filename is accurate.

3. **Import Model**: Use the file explorer on the left to upload and import the desired model.

4. **Run Training**: Execute the training script to generate the `.pt` file.

5. **Run Test**: Execute the tset script with the `.pt` file.

6. **Refine the Model**: Analyze results and make adjustments to improve the model.
