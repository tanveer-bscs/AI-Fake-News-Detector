# AI Fake News & Misinformation Detector

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.59%25-brightgreen)

---

## Overview

AI Fake News & Misinformation Detector is a Machine Learning project that detects whether a news article is **REAL** or **FAKE** using Natural Language Processing (NLP).

The system cleans the news article, converts it into numerical features using **TF-IDF Vectorization**, and predicts authenticity using the **Passive Aggressive Classifier**.

In addition to prediction, the project also generates:

- Confidence Score
- Risk Level
- Prediction History
- Automatic Report Generation

This project demonstrates an end-to-end Machine Learning pipeline from data preprocessing to real-world prediction.

---

# Features

- Fake News Detection
- Natural Language Processing (NLP)
- Text Cleaning
- TF-IDF Vectorization
- Passive Aggressive Classifier
- Confidence Score Calculation
- Risk Level Analysis
- Prediction History Logging
- Automatic Report Generation
- Interactive User Prediction

---

# Dataset

Dataset contains over **44,000 news articles**.

- Fake News Articles
- Real News Articles

Total Samples:

- Fake News: **23,481**
- Real News: **21,417**

Combined Dataset:

**44,898 News Articles**

---

# Machine Learning Pipeline

Dataset

↓

Data Cleaning

↓

Merge Dataset

↓

Shuffle Dataset

↓

Feature Engineering

↓

TF-IDF Vectorization

↓

Train-Test Split

↓

Passive Aggressive Classifier

↓

Prediction

↓

Confidence Score

↓

Risk Analysis

↓

History Saving

↓

Report Generation

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- Machine Learning
- TF-IDF Vectorizer
- Passive Aggressive Classifier

---

# Project Structure

```
AI-Fake-News-Detector
│
├── modules
│   ├── cleaner.py
│   ├── detector.py
│   ├── history.py
│   └── report_generator.py
│
├── reports
│
├── Fake.csv
├── True.csv
├── fake_news.csv
├── history.txt
├── main.py
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/tanveer-bscs/AI-Fake-News-Detector.git
```

Move into the project directory

```bash
cd AI-Fake-News-Detector
```

Install dependencies

```bash
pip install pandas scikit-learn
```

Run the project

```bash
python main.py
```

---

# Example Prediction

Input

```
Donald Trump says aliens are living in the White House.
```

Output

```
Result : FAKE NEWS

Confidence Score : 1.16

Risk Level : MEDIUM

Recommendation:

This news may be misleading or fake.
Verify it before sharing.
```

---

# Model Performance

Training Samples

```
35,918
```

Testing Samples

```
8,980
```

Model Accuracy

```
99.59%
```

---

# Future Improvements

- Streamlit Web Application
- Flask API
- BERT Transformer Model
- Explainable AI (XAI)
- News API Integration
- Real-Time Fake News Detection
- Deep Learning Models
- Fake Image Detection

---

# Learning Outcomes

This project demonstrates practical experience with:

- Machine Learning
- Natural Language Processing
- Text Classification
- Feature Engineering
- Data Cleaning
- Model Evaluation
- Python Programming
- Modular Programming
- Git & GitHub

---

# Author

## Tanveer Ahmad

BS Computer Science Student

AI • Machine Learning • NLP Enthusiast

### GitHub

https://github.com/tanveer-bscs

### LinkedIn

https://www.linkedin.com/in/tanveer-bscs

---

# License

This project is developed for educational, research, and portfolio purposes.
