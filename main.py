from fastapi import FastAPI
from fastapi import Body
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
import gc
import joblib
import tensorflow as tf
import pathlib
import sys
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI()


@app.get("/neuro/get-class")
async def root(plainText: str = Body(..., embed=True)):
    # Функиця для загрузки модели
    def load_model(model_file):
        model = tf.keras.models.load_model(model_file)
        return model

    # Функция для предобработки текста
    def preprocess_text(text):
        # Токенизация текста
        tokens = nltk.word_tokenize(text.lower())
        # Удаление стоп-слов и пунктуации, а также приведение слов к начальной форме
        tokens = [morph.parse(word)[0].normal_form for word in tokens if word.isalnum() and word not in stop_words]
        transformed_data = loaded_vectorizer.transform([' '.join(tokens)])
        return transformed_data

    classes = {
        0: 'Разрешенный контент',
        1: 'Мосгорсуд',
        2: 'Суд',
        3: 'Роскомнадзор',
        4: 'Роспотребнадзор',
        5: 'Генеральная прокуратура',
        6: 'МВД',
        8: 'ФНС',
        9: 'ФСКН',
        10: 'Росалкогольрегулирование',
        11: 'Россельхознадзор',
        12: 'Росздравнадзор',
    }

    model = load_model(pathlib.Path('./kaggle/working/fnn_model.h5'))  # Заменить на путь до модели
    loaded_vectorizer = joblib.load('./tfidf_vectorizer.pkl')  # Заменить на путь до векторизатора

    nltk.download('punkt')
    # Загрузка стоп-слов
    nltk.download('stopwords')
    stop_words = set(stopwords.words('russian'))
    morph = pymorphy2.MorphAnalyzer()

    text = preprocess_text(plainText)  # То что в скобках, заменить на любой текст
    model_predict = model.predict(text)
    pred_fnn = np.argmax(model_predict, axis=-1)
    text_class = classes[pred_fnn[0]]
    print(f'Текст относится к классу: {text_class}')
    return {"class": str(text_class)}
