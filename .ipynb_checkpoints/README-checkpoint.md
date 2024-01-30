# Biophysics & AI

Exploring the intersection of artificial intelligence and the intricate realms of proteins and genomics has been a captivating journey for me. Delving into the intricacies of molecular biology, my experiments with AI have been centered on deciphering complex biological data, elucidating protein structures, and unraveling genomic mysteries. Through a multifaceted approach, I've harnessed AI algorithms to analyze vast datasets, predict protein interactions, and contribute to advancements in genomics research. This immersive exploration has not only deepened my understanding of the biological intricacies at play but has also unveiled the potential of AI in revolutionizing our comprehension of the fundamental building blocks of life.


## ProteinStructurePredictionCNN

> Reference

> Jian Zhou and Olga G. Troyanskaya (2014) - "Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction" - https://arxiv.org/pdf/1403.1347.pdf

> Sheng Wang et al. (2016) - "Protein Secondary Structure Prediction Using Deep Convolutional Neural Fields" - https://arxiv.org/pdf/1512.00843.pdf



## CodonCraft ProGen: Precision Translation Model for Optimal Bacterial Expression (with Attention)

> 0. Basic LSTM model (RNN)

> 1. BERT (with Attention)

> 2. GPT-esk Transformer (uni-directional)


# Data

> https://www.ncbi.nlm.nih.gov/home/develop/api/

# Project Structure

```
project_root/
|-- data/
|   |-- raw/              # Raw data files
|   |-- processed/        # Processed and preprocessed data
|   |-- dataset.py        # Custom dataset classes and data loading utilities
|
|-- models/
|   |-- architecture.py   # Model architecture definition
|   |-- loss.py           # Custom loss functions
|   |-- metrics.py        # Evaluation metrics
|   |-- train.py          # Training script
|   |-- predict.py        # Inference script
|
|-- utils/
|   |-- helpers.py        # Utility functions
|   |-- visualization.py  # Visualization functions
|
|-- config/
|   |-- config.yaml       # Configuration file for hyperparameters
|
|-- notebooks/            # Jupyter notebooks for experimentation and analysis
|
|-- experiments/
|   |-- experiment_1/     # Directory for experiment 1 (can have multiple experiments)
|       |-- logs/         # TensorBoard logs, training/validation metrics
|       |-- saved_models/ # Saved model checkpoints
|
|-- requirements.txt       # Python dependencies file
|-- README.md              # Project documentation
```