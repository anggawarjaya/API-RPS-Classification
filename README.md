
# 📄 Flask API Docx Classifier with NLP, Wordcloud, and SVM

A web-based Flask API application for classifying `.docx` documents using Natural Language Processing (NLP), visualizing word distributions, and predicting classes (e.g., `RPS` vs `Bukan_RPS`) using a machine learning model based on LSI + SVM.

---

## 🚀 Features

- Upload `.docx` documents
- Automatic text extraction and cleaning (lowercasing, stemming, stopword removal)
- WordCloud and Barplot visualization (before & after preprocessing)
- Text vectorization using TF-IDF and LSI (SVD)
- Classification using trained SVM model
- Probabilities for each class shown
- Auto-cleanup of uploaded files after 5 minutes

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/anggawarjaya/API-RPS-Classification.git
cd API-RPS-Classification
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install dependencies

```bash
pip install -r completed_requirements.txt
```

---

## 📂 Folder Structure

```
project/
│
├── app.py                  # Main Flask app
├── completed_requirements.txt
├── .gitignore
│
├── model/                  # Your saved ML models (vectorizer, SVD, SVM)
│   ├── vectorizer.pkl
│   ├── svd_lsi.pkl
│   └── svm_model.pkl
│
├── static/
│   ├── uploads/            # Uploaded .docx files (temporary)
│   └── plots/              # Wordclouds and barplots
```

---

## ⚙️ Usage

### Run the Flask server

```bash
python app.py
```

By default, it will be hosted at:

```
http://localhost:5000
```

### Send a POST request to `/predict`

You can use [Postman](https://www.postman.com/) or `curl`:

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/your/file.docx" \
  -F "filename=nama_file" \
  -F "description=deskripsi_file"
```

You will receive a JSON response with:
- Predicted class
- Class probabilities
- Extracted text (original and cleaned)
- TF-IDF tokens
- LSI vector
- Links to wordcloud and barplot images

---

## 📦 Dependencies

Listed in `completed_requirements.txt`:

```
flask
python-docx
sastrawi
nltk
matplotlib
wordcloud
joblib
scikit-learn
```

---

## 📌 Notes

- Make sure you already downloaded NLTK corpora:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

- Uploaded files and generated images will be deleted automatically after 5 minutes.

---

## 👨‍💻 Author

Created by [Angga Warjaya](https://github.com/anggawarjaya)  
Contact: awfproduction.ing@gmail.com

---

## 📃 License

This project is licensed under the MIT License.
