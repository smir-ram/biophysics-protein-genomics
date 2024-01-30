![Banner Image](ProteinStructurePredictionCNN/images/banner-bpg.png)

# Biophysics & AI

Exploring the intersection of artificial intelligence and the intricate realms of proteins and genomics has been a captivating journey for me. Delving into the intricacies of molecular biology, my experiments with AI have been centered on deciphering complex biological data, elucidating protein structures, and unraveling genomic mysteries. Through a multifaceted approach, I've harnessed AI algorithms to analyze vast datasets, predict protein interactions, and contribute to advancements in genomics research. This immersive exploration has not only deepened my understanding of the biological intricacies at play but has also unveiled the potential of AI in revolutionizing our comprehension of the fundamental building blocks of life.



## protein-ligand interaction
To-be-moved

## molecular property prediction (small protein/ligand: binding)
To-be-moved

## molecule generation (small protein/ligand)
To-be- moved

## ProteinStructurePredictionCNN

> Jian Zhou and Olga G. Troyanskaya (2014) - "Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction" - https://arxiv.org/pdf/1403.1347.pdf

> Sheng Wang et al. (2016) - "Protein Secondary Structure Prediction Using Deep Convolutional Neural Fields" - https://arxiv.org/pdf/1512.00843.pdf

This algorithm leverages the combined ability of two deep learning techniques, including a combination of convolutional neural networks (CNNs) and generative stochastic networks (GSNs), to achieve state-of-the-art accuracy in predicting the secondary structure of proteins. By effectively capturing complex dependencies and patterns in protein sequences, DCGSN offers a powerful tool for researchers and practitioners in the field of bioinformatics and structural biology to improve the accuracy of secondary structure prediction, ultimately advancing our understanding of protein functionality and interactions. Additionally, Deep Convolutional Neural Fields (DeepCNF) is used to refine the protein secondary structure prediction; this combines the power of deep learning and graphical models to enhance accuracy further. The method leverages deep convolutional neural networks to capture informative features from protein sequences, facilitating precise secondary structure predictions to build whole protein structures.

-[ ] Transformer (with attention) implementations (AA sequence)

## CodonCraft ProGen: Precision Translation Model for Optimal Bacterial Expression (with Attention)

> 0. Basic LSTM model (RNN)

> 1. BERT (with Attention)

> 2. GPT-esk Transformer (uni-directional)

-[x] project: code moved from private git
-[ ] data needs to anonymized 


# Data

## CodonCraft

> https://www.ncbi.nlm.nih.gov/home/develop/api/

## ProteinStructurePredictions

> CullPDB53 Dataset (6125 proteins):The CullPDB53 dataset is a non-redundant set of protein structures from the Protein Data Bank (PDB). https://www.rcsb.org/.

> The CB513 dataset is often used for protein secondary structure prediction. https://www.princeton.edu/~jzthree/datasets/ICML2014/.

> The Critical Assessment of Structure Prediction (CASP) datasets are used for protein structure prediction and related tasks.  http://predictioncenter.org/.

> CAMEO Test Proteins (6 months): The CAMEO (Continuous Automated Model EvaluatiOn) test proteins are used for protein structure prediction evaluation. http://www.cameo3d.org/sp/6-months/.

> JPRED Training and Test Data (1338 training and 149 test proteins): The JPRED dataset provides training and test data for protein secondary structure prediction. http://www.compbio.dundee.ac.uk/jpred4/about.shtml.

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