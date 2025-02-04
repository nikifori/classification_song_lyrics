# Classifying Genres from Song Lyrics with DistilBERT
The challenge of comprehending song lyrics is addressed via natural language processing (NLP) in this exercise, with special focus on genre classification. Our experiments, utilizing a DistilBERT model for genre classification, provide positive findings with 67\% accuracy. To further evaluate model performance, DistilBERT was compared with other BERT-based models, including BERT-uncased and RoBERTa, as well as a baseline logistic regression model. The final assessment considered both predictive accuracy and computational efficiency, highlighting the trade-offs between model size and classification performance. 

# How to train DistilBERT, BERT, RoBERTa 

## Overview
This document provides a comprehensive guide to training DistilBERT, BERT, and RoBERTa models using a Lightning-based training script and a configuration file. The training process is designed for classification of song lyrics and leverages distributed training via PyTorch Lightning.

---
## Configuration File Breakdown
### **Training Configuration (`training_config`):**

- **`exp_folder`**: Path where experimental results are saved.
- **`exp_name`**: Name of the experiment.
- **`pre_validate`**: Run validation before training.
- **`max_epochs`**: Number of epochs to train.
- **`train_batch_size`**: Batch size for training.
- **`seed`**: Random seed for reproducibility.
- **`strategy`**: Training strategy (Distributed Data Parallel (DDP)). Recommended: `ddp_find_unused_parameters_true`
- **`precision`**: Floating point precision.
- **`enable_checkpointing`**: Whether to save model checkpoints.
- **`num_nodes`**: Number of compute nodes used in training.
- **`enable_model_summary`**: Show model summary.
- **`enable_progress_bar`**: Show progress bar during training.
- **`accumulate_grad_batches`**: Number of batches to accumulate gradients over.
- **`gradient_clip_val`**: Gradient clipping threshold.
- **`log_every_n_steps`**: Logging frequency.
- **`check_val_every_n_epoch`**: Frequency of validation checks.
- **`num_sanity_val_steps`**: Number of validation steps before training starts.
- **`save_top_k_checkpoints`**: Number of best checkpoints to save.
- **`every_n_epochs_model_save`**: Save model every N epochs.

### **Dataset Configuration (`dataset_config`):**
- **`csv_path`**: Path to the dataset file.
- **`random_seed`**: Random seed for dataset handling.
- **`num_workers`**: Number of workers for data loading.

### **Model Configuration (`model_config`):**
- **`transformer_model`**: Specifies which transformer model to use.
  - Options include:
    - `distilbert/distilbert-base-uncased`
    - `FacebookAI/roberta-base`
    - `google-bert/bert-base-uncased`
- **`use_activation_func_before_class_layer`**: Whether to apply an activation function before classification.

### **Neptune Configuration (`neptune_config`):**
- **`use_neptune`**: Whether to enable Neptune logging.
- **`project`**: Neptune project name.
- **`tags`**: Tags for tracking experiments.
- **`capture_stdout, capture_stderr, capture_hardware_metrics, capture_traceback`**: Logging options.

---
## Training Script (`training.py`)

### **Script Overview**
This script is used to train transformer-based models using PyTorch Lightning. It includes functionalities such as dataset loading, model initialization, training, validation, and testing.

### **Main Components**
1. **Setup Neptune Logger (`setup_neptune_logger`)**
   - Configures Neptune logging if enabled.

2. **Load Configuration (`safe_load_yaml`)**
   - Reads the YAML configuration file.

3. **Train Function (`train`)**
   - Loads dataset
   - Initializes model
   - Configures data loaders
   - Sets up the PyTorch Lightning `Trainer`
   - Runs pre-validation (if enabled)
   - Trains the model
   - Tests the best model based on validation loss

4. **Command-Line Arguments**
   - `--config`: Path to YAML config file
   - `--gpu`: GPU device selection

### **Dataset Handling (`Genius_dataset`)**
The dataset is loaded using a custom `Genius_dataset` class, which splits the data into training, validation, and test sets.

### **Model Definition (`GenreClassifier_lightning`)**
The model is initialized using one of the specified transformer models (DistilBERT, BERT, or RoBERTa) with a classification head.

### **Training and Validation**
- Uses **DDP (Distributed Data Parallel)** for multi-GPU training.
- Employs gradient accumulation and clipping for stability.
- Logs learning rate and validation loss.
- Saves checkpoints of the best-performing models.

### **Testing the Model**
After training, the model is tested using the best checkpoint saved based on validation loss.

---
## Running the Training Script
To start training, run:
```sh
python training.py --config path/to/config.yaml --gpu 0
```

### **Example Usage**
To train BERT base uncased:
```yaml
model_config:
  transformer_model: "google-bert/bert-base-uncased"
```
To train RoBERTa base:
```yaml
model_config:
  transformer_model: "FacebookAI/roberta-base"
```
To train DistilBERT:
```yaml
model_config:
  transformer_model: "distilbert/distilbert-base-uncased"
```

---
## Conclusion
This guide provides an overview of training DistilBERT, BERT, and RoBERTa models for song lyric classification using PyTorch Lightning. The setup allows for easy experimentation with different models, logging, and distributed training.

