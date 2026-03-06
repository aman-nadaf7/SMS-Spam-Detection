# SMS Spam Detection using Naive Bayes

A lightweight machine learning pipeline to classify SMS messages as 'spam' or 'ham' (legitimate). This was built as a foundational NLP project to understand text vectorization and probabilistic classification.

## The Goal
I wanted to see how well a simple Naive Bayes model could handle the messiness of SMS language (slang, caps, punctuation). For a startup, this is a great baseline because it's fast and doesn't require a GPU.

## Tech Stack
* **Python** (Logic)
* **Scikit-Learn** (Modeling & TF-IDF)
* **Pandas** (Data handling)

##  Results
The model achieved a **98% accuracy** on the test set. More importantly, I optimized for **Precision**, ensuring that legitimate messages are almost never accidentally marked as spam.

## How to run
1. Clone the repo: `git clone <your-link>`
2. Install requirements: `pip install -r requirements.txt`
3. Run the script: `python src/predictor.py`
