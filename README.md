# Grammatical Error Correction in Tamil Language Using Language Models
This repository contains code for training and evaluating various transformer-based and recurrent models for Grammatical Error Correction (GEC) in the Tamil language. The models included are BERT, T5, mBART, and LSTM.

# Overview
Grammatical Error Correction (GEC) aims to automatically detect and correct grammatical errors in text. This repository demonstrates the application of several advanced NLP models to perform GEC in Tamil, an under-resourced language in the field of NLP. The models used in this project are:

BERT (Bidirectional Encoder Representations from Transformers)

T5 (Text-to-Text Transfer Transformer)

mBART (Multilingual BART)

LSTM (Long Short-Term Memory)

# Dataset
The dataset used for this project is sourced from Kaggle and consists of Tamil sentences with errors and their corresponding corrections. The dataset is preprocessed to remove null values, tokenized, lemmatized, and encoded for model training.

# Models
# BERT
The BERT model is used in an encoder-decoder setup for sequence-to-sequence tasks. The code includes data preprocessing, model initialization, and training using K-Fold Cross-Validation.
# T5
T5 is fine-tuned for GEC tasks using a subset of the dataset. The model is trained using sequence-to-sequence tasks with token-level accuracy as the evaluation metric.
# mBART
mBART is employed for grammatical error correction by generating corrected sentences. The model's accuracy is evaluated on a sample dataset.
# LSTM
The LSTM model is developed with regularization and hyperparameter tuning. It includes bidirectional LSTM layers with dropout to prevent overfitting.

# Data Augmentation
To increase the diversity of the training data, several data augmentation techniques were applied, including synonym replacement, random insertion, and word swapping.

# Training and Evaluation
Each model is trained and evaluated using various metrics, including training and validation losses and accuracies. Detailed metrics for each fold during K-Fold Cross-Validation are provided for the BERT model.

# Running the code

Download the zip file and extract the code (.ipynb) file and the dataset (.csv) file

Upload the dataset file (Error Annotated Corpus.csv), in the project directory or in the location of code run (like Google Colab)

Upload the code (Language_Models_for_GEC_in_Tamil.ipynb) file in Google Colab or Jupyter  

Connect to a suitable GPU or CPU

Run each of the cells in order

# Results

1. BERT

- Mean Accuracy: 96.17%

- Mean Loss: 0.182

2. T5

- Evaluation Accuracy: 94.34%

- Evaluation Loss: 0.1485

3. mBART

- Accuracy: 66.67%

4. LSTM

- Validation Accuracy: 56.37%

- Validation Loss: 2.6524

# Future Work
Future research can explore additional transformer models like GPT-3, Transformer-XL, or RoBERTa, experiment with ensemble methods, and systematically explore different hyperparameters to further optimize model performance. Utilizing more extensive and diverse datasets can enhance model robustness and generalization.


