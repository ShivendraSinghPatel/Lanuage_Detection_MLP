# Language Detection Model

This repository contains a machine learning-based language detection model that identifies the language of a given text. The model is trained using a dataset of text samples in multiple languages and utilizes natural language processing (NLP) techniques for accurate predictions.

## Features
- Detects multiple languages from text input.
- Uses machine learning and NLP techniques.
- Trained on a dataset containing diverse language samples.
- Provides predictions with high accuracy.

## Dataset
The model is trained using `language.csv`, which contains text samples labeled with their respective languages.

## Installation
To use this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/language-detection.git
cd language-detection
pip install -r requirements.txt
```

## Usage
You can run the model using the provided Jupyter Notebook (`Language Detection.ipynb`).

### Steps to Run:
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Language\ Detection.ipynb
   ```
2. Run the notebook cells to train and test the language detection model.
3. Input text samples to check language predictions.

## Dependencies
Ensure you have the following dependencies installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK

Install missing dependencies using:
```bash
pip install pandas numpy scikit-learn nltk
```

## Model Training
The model is trained using machine learning techniques, including:
- Text preprocessing (tokenization, stopword removal, etc.)
- Feature extraction (TF-IDF, CountVectorizer)
- Classification using ML algorithms (e.g., Naïve Bayes, Logistic Regression, SVM)

## Contributing
Feel free to fork this repository and submit pull requests for improvements.


Happy coding!

