from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from docx import Document
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import joblib
import nltk
import json
import uuid
import numpy as np
import threading
import time
from waitress import serve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
PLOT_FOLDER = './static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Stopwords dan stemmer
stopwords_id = set(stopwords.words('indonesian'))
stopwords_en = set(stopwords.words('english'))
stopwords_fr = set(stopwords.words('french'))
combined_stopwords = stopwords_id.union(stopwords_en).union(stopwords_fr)
stemmer = StemmerFactory().create_stemmer()

# Load model
vectorizer = joblib.load('./model/vectorizer.pkl')
svd = joblib.load('./model/svd_lsi.pkl')
svm_model = joblib.load('./model/svm_model.pkl')

def extract_text(docx_path):
    doc = Document(docx_path)
    return ' '.join([p.text for p in doc.paragraphs]).strip()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in combined_stopwords and len(t) > 2]
    stemmed = stemmer.stem(' '.join(tokens))
    return stemmed

def generate_wordcloud_and_barplot(text, filename_prefix):
    tokens = word_tokenize(text)
    words = [t for t in tokens if t.isalpha()]
    counter = Counter(words)
    if not counter:
        return None, None

    # Wordcloud
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counter)
    wc_path = os.path.join(PLOT_FOLDER, f'{filename_prefix}_wordcloud.png')
    wc.to_file(wc_path)

    # Barplot
    plt.figure(figsize=(10, 5))
    most_common = counter.most_common(20)
    labels, values = zip(*most_common)
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    bp_path = os.path.join(PLOT_FOLDER, f'{filename_prefix}_barplot.png')
    plt.tight_layout()
    plt.savefig(bp_path)
    plt.close()

    return wc_path, bp_path

def delete_file_after_delay(file_paths, delay=300):
    def delete_files():
        time.sleep(delay)
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")
    threading.Thread(target=delete_files, daemon=True).start()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    filename = request.form.get('filename')
    description = request.form.get('description')

    if not file or not filename or not description:
        return jsonify({'error': 'Missing required fields'}), 400

    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    safe_filename = secure_filename(filename) + '_' + unique_id + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)

    filename_prefix = secure_filename(filename) + '_' + unique_id

    # Ekstraksi & praproses
    extracted_text = extract_text(filepath)
    cleaned_text = preprocess_text(extracted_text)

    # Visualisasi
    wc1, bp1 = generate_wordcloud_and_barplot(extracted_text, filename_prefix + '_before')
    wc2, bp2 = generate_wordcloud_and_barplot(cleaned_text, filename_prefix + '_after')

    # Vektorisasi
    tfidf_vector = vectorizer.transform([cleaned_text])
    lsi_vector = svd.transform(tfidf_vector)

    # Ambil tfidf bernilai
    tfidf_vals = tfidf_vector.toarray()[0]
    tfidf_dict = {word: float(tfidf_vals[idx]) for word, idx in vectorizer.vocabulary_.items() if tfidf_vals[idx] > 0}

    # Prediksi
    pred_class = int(svm_model.predict(lsi_vector)[0])
    probas = svm_model.predict_proba(lsi_vector)[0]
    class_names = svm_model.classes_
    probas_dict = {str(class_names[i]): float(probas[i]) for i in range(len(class_names))}

    base_url = request.host_url.rstrip('/')

    response = {
        'prediksi_kelas': 'RPS' if pred_class == 1 else 'Bukan_RPS',
        'probabilitas': probas_dict,
        'extracted_text': extracted_text,
        'cleaned_text': cleaned_text,
        'tfidf': tfidf_dict,
        'lsi_vector': lsi_vector[0].tolist(),
        'wordcloud_before': f"{base_url}/static/plots/{os.path.basename(wc1)}" if wc1 else None,
        'barplot_before': f"{base_url}/static/plots/{os.path.basename(bp1)}" if bp1 else None,
        'wordcloud_after': f"{base_url}/static/plots/{os.path.basename(wc2)}" if wc2 else None,
        'barplot_after': f"{base_url}/static/plots/{os.path.basename(bp2)}" if bp2 else None,
    }

    files_to_delete = [filepath]
    if wc1: files_to_delete.append(wc1)
    if bp1: files_to_delete.append(bp1)
    if wc2: files_to_delete.append(wc2)
    if bp2: files_to_delete.append(bp2)
    delete_file_after_delay(files_to_delete, delay=300)

    return jsonify(response)

@app.route('/static/plots/<path:filename>')
def download_file(filename):
    return send_from_directory(PLOT_FOLDER, filename)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)