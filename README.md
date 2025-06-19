# 🧠 Consumer Complaint Classification & Summarization

## 📅 Dataset Overview

- **Date Range**: January 1, 2018 – November 11, 2024  
- **Total Records**: ~2 million consumer complaints  
- **Note**: Data collection, cleaning, sampling, summarization, and modeling were done entirely by me.

---

## 🫖 Label Distribution – Company Response to Consumer

| Response Type               | Original Count |
|----------------------------|----------------|
| Closed with explanation     | 100,000        |
| Closed with non-monetary relief | 100,000    |
| Closed with monetary relief | 12,703         |
| Untimely response           | 2,111          |
| Closed                     | 1,378          |

---

## ⚖️ Sampling and Model Training Details

### Long Narratives (~215K records)

| Response Type               | Sampled Count |
|----------------------------|---------------|
| Closed with explanation     | 25,500        |
| Closed with non-monetary relief | 25,500   |
| Closed with monetary relief | 12,703        |
| Untimely response           | 2,111         |
| Closed                     | 1,378         |

### Summarized Complaints (~67K records)

Used for faster training and better generalization.

---

## 🧰 Why Summarization?

| Aspect             | Truncated Complaint | Summarized Complaint |
|--------------------|---------------------|-----------------------|
| Length             | Fixed at 512 tokens | Shorter (128–256 tokens) |
| Noise              | High                | Low (core content only) |
| Semantic Focus     | Broad               | Sharp and focused     |
| Training Efficiency| Slower              | Faster                |
| Generalization     | May overfit         | Learns true semantics |

**Summarization Model**: `BART-large-CNN`  
**ROUGE Scores**:
- ROUGE-1: 0.7361  
- ROUGE-2: 0.7020  
- ROUGE-L: 0.6813  
- **Avg ROUGE**: 0.71  
- *(Baseline T5-small: ~0.21)*

---

## 📈 Project Overview

Multi-Class Classification of Company Responses using Transformer Models:

- **Model 1**: Trained on long-form narratives  
- **Model 2**: Trained on BART-generated summaries  

---

###✅ Baseline Assumption
If we always predict "Closed with explanation", then:

True Positives (TP) = 25,500

False Positives (FP) = 76,192 - 25,500 = 50,692

False Negatives (FN) = 0 (since we never predict any other class)

Precision = TP / (TP + FP) = 25,500 / 76,192 ≈ 0.3345
Recall = TP / (TP + FN) = 25,500 / 25,500 = 1.0
F1 = 2 × (Precision × Recall) / (Precision + Recall)
= 2 × (0.3345 × 1) / (0.3345 + 1) ≈ 0.501

## ✅ Validation Performance

| Model                   | F1-Score | Accuracy | Validation Loss |
|------------------------|----------|----------|-----------------|
| Summarized Complaints  | 0.82     | 0.84     | 0.454           |
| Long Narratives        | 0.81     | 0.77     | 0.67            |

---

## ✅ Model Architecture

- **Base Model**: `bert-base-uncased`  
- **Task**: Multi-class classification (5 classes)  
- **Output Layer**: Softmax over 5 response types  

---

## 🪚 Data Preprocessing

### Long Narratives:
- Concatenated fields: Narrative + Product + Issue + Company  
- Preprocessed using regex + lowercasing + legal ref removal  
- Truncated to 512 tokens  

### Summarized Complaints:
- Concatenated summary + metadata  
- Tokenized to 256 tokens max  

---

## 🔠 Label Handling & Balancing

- Label encoding with `LabelEncoder`  
- Long Narrative Model: Downsampling + class weights  
- Summary Model: Sampled dataset (balanced)

---

## 🧪 Tokenization & Dataset Prep

- **Tokenizer**: `bert-base-uncased`  
- `batch_encode_plus` with padding, truncation, attention masks  
- Converted to PyTorch `TensorDataset`

---

## 🏋️️ Model Training Setup

| Parameter         | Long Model | Summarized Model |
|------------------|------------|------------------|
| Train/Val Split  | 80/20      | 85/15            |
| Loss Function    | CrossEntropy (weighted) | CrossEntropy |
| Optimizer        | AdamW (lr=2e-5) | AdamW (lr=2e-5) |
| Scheduler        | Linear warmup | Linear warmup   |
| Batch Size       | 16         | 16               |
| Epochs           | 3          | 4                |
| Device           | GPU (cuda) | GPU (cuda)       |

---

## 📊 Evaluation Metrics

- Loss  
- Accuracy  
- F1-Score (weighted)  
> Tracked across all epochs for training and validation

---

## 📀 Model Saving & Loading

### Save:
```python
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

### Load:
```python
from transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
```

### 🔧 Summary of Key Techniques:
| Component        | Implementation                           |
| ---------------- | ---------------------------------------- |
| Model            | `bert-base-uncased`                      |
| Input Variants   | Full narrative / BART summary            |
| Truncation Limit | 512 (narrative), 256 (summary)           |
| Cleaning         | Regex + NLTK                             |
| Class Balancing  | Downsampling + class weights (long only) |
| Loss Function    | CrossEntropy                             |
| Optimization     | AdamW + Linear Scheduler                 |
| Evaluation       | Accuracy, Loss, F1-Score                 |



### 🌐 Streamlit App Overview
Page 1: 📝 Summary Generation
Input: Raw complaint

Model: summarizer.summarize_text() (BART)

Output: Clean summary

Summary stored in st.session_state.summary

Page 2: 🧾 Complaint Classification
Input: Narrative, Summary, Metadata

Models: Both BERT models

Output: Predicted response type (with probabilities)

| Component      | Code Used                     | Description                        |
| -------------- | ----------------------------- | ---------------------------------- |
| Summarization  | `summarizer.summarize_text()` | Generates summary using BART       |
| Session Memory | `st.session_state.summary`    | Maintains summary across app pages |
| Classification | `predict_with_probability()`  | Classifies using BERT              |




