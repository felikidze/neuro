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
import re
import requests
import torch
from transformers import BertTokenizer, BertModel
import pathlib
import sys
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI()


@app.get("/neuro/get-class")
async def root(plainText: str = Body(..., embed=True)):
    # Загрузка предварительно обученной модели BERT
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')

    # Функция для преобразования текста в эмбеддинги BERT
    def get_bert_embeddings(text):
        inputs = tokenizer_bert(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()

    # Функиця для загрузки модели
    def load_model(model_file):
        model = tf.keras.models.load_model(model_file)
        return model

    # Функция для предобработки текста
    def preprocess_text(text):
        text = re.sub(r'\d', '', str(text))  # Удаление цифр
        text = re.sub(r'[^\w\s]', '', text)  # Удаление символов, кроме букв и пробелов
        tokens = nltk.word_tokenize(text.lower())
        stop_words_ru = set(stopwords.words('russian'))
        stop_words_en = set(stopwords.words('english'))
        text = ' '.join(tokens)  # Объединение токенов в строку
        if any(char.isalpha() for char in text):  # Проверка на наличие букв в тексте
            stop_words = stop_words_ru if all(char.isalpha() or char.isspace() for char in text) else stop_words_en
            tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]
        else:  # Если текст состоит только из символов, пропускаем его без предобработки
            tokens = []
        print(tokens)
        return np.vstack(get_bert_embeddings(' '.join(tokens)))

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

    model = load_model(pathlib.Path('./kaggle/working/bertv2_fnn_model.h5'))

    nltk.download('punkt')
    # Инициализация стоп-слов

    morph = pymorphy2.MorphAnalyzer()
    text = preprocess_text(plainText)
    pred_fnn = np.argmax(model.predict(text), axis=-1)
    print(f'Текст относится к классу: {classes[pred_fnn[0]]}')

    return {"class": str(classes[pred_fnn[0]])}
