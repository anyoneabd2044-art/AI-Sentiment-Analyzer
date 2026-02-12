# AI-Sentiment-Analyzer
AI-Sentiment-Analyzer Project made using Python ,Transformers (Hugging Face) ,PyTorch and Matplotlib



 AI-Based Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Model](https://img.shields.io/badge/Model-DistilBERT-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)



##  Project Overview

This project is an AI-powered **Sentiment Analysis System** built using **Hugging Face Transformers**.
It uses a pretrained **DistilBERT transformer model** to classify text into **Positive** or **Negative** sentiment with a confidence score.

The system also visualizes prediction results using graphical representation.



##  Model & Technical Details

* **Model:** DistilBERT (Transformer-based architecture)
* **Architecture:** Self-Attention Mechanism (Bidirectional Encoder)
* **Training Dataset:** Stanford Sentiment Treebank (SST-2)
* **Benchmark Accuracy:** ~91–93%
* **Inference Framework:** PyTorch

This project uses a pretrained deep learning model for real-time sentiment classification.



##  Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Matplotlib
* Google Colab

---

##  System Workflow

```
User Input
     ↓
Tokenization
     ↓
Transformer Model (DistilBERT)
     ↓
Classification Layer
     ↓
Softmax Probability
     ↓
Sentiment + Confidence Score
     ↓
Visualization
```

---

##  Features

✔ Real-time sentiment prediction
✔ Confidence score display
✔ Transformer-based deep learning model
✔ Visual representation using bar charts
✔ Easily extendable for batch analysis

---

##  How To Run

### Install Dependencies

```bash
pip install transformers torch matplotlib
```

### Run Python Script

```bash
python sentiment_analyzer.py
```

Or open the notebook in **Google Colab** and run all cells.

---

##  Sample Output

**Input:**

```
I really love this product. It works perfectly!
```

**Output:**

```
Sentiment: POSITIVE
Confidence Score: 0.98






##  Real-World Applications

* Product review analysis
* Customer feedback monitoring
* Social media sentiment tracking
* Brand reputation analysis
* Opinion mining

---

##  Future Improvements

* Multi-class sentiment (Positive / Neutral / Negative)
* Batch processing using CSV files
* Sentiment analytics dashboard
* Confusion matrix and evaluation metrics
* Web-based UI using Flask or Streamlit

---


##  Author
Mohd Abdullah arif
Aspiring AI & Machine Learning Engineer

GitHub:https://github.com/anyoneabd2044-art/AI-Sentiment-Analyzer




