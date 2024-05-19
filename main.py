import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, BertTokenizer, BertForSequenceClassification
import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage
import speech_recognition as sr
import torch
import re

#Загрузка данных из Excel без заголовков столбцов
df = pd.read_excel('your_path_to_the_training_sample', header=None) #в репозитории доступен файл norm.csv.xlsx, который можно скачать и использовать для дообучения BERT
transcriptions = df.iloc[:, :-1].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()
labels = df.iloc[:, -1].astype(int).tolist()

#преобразование в формат Dataset
data = Dataset.from_dict({'text': transcriptions, 'label': labels})

#Токенизация данных
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
tokenized_data = data.map(tokenize_function, batched=True)

#Разделение данных на обучающую и тестовую выборки
train_dataset = tokenized_data.shuffle().select(range(int(0.8 * len(tokenized_data))))
test_dataset = tokenized_data.shuffle().select(range(int(0.8 * len(tokenized_data)), len(tokenized_data)))

#шаг 2: Обучение модели
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
training_args = TrainingArguments(
    output_dir='/results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/logs',
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()

# Инициализация распознавателя речи
recognizer = sr.Recognizer()

def transcribe_google(audio_path):
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ru-RU')
    return text

def classify_transcription(transcription, tokenizer, model):
    inputs = tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=3072)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(predictions, dim=-1).item()

def analyze_transcription(transcription):
  violations = []

  CRITERIA = {
       1: [
           {"pattern": r'\bпозывной\b', "meaning": "Отсутствие упоминания позывного"},
           {"pattern": r'\bсокращение\b', "meaning": "Использование нестандартных сокращений не допускается"},
           {"pattern": r'\bсветофор\b', "meaning": "Не переданы показания светофоров по маршруту следования"}
       ],
       2: [
           {"pattern": r'\bпередача команды\b', "meaning": "Передача команды не лаконично"},
           {"pattern": r'\bздравствуйте|добрый день|пожалуйста|хорошо|до свидания|спасибо\b', "meaning": "Использование лишних слов"},
           {"pattern": r'\bверно\b', "meaning": "Отсутствие подтверждения правильности восприятия команды"}
       ],
       3: [
           {"pattern": r'\bрасстоянии до сцепления\b', "meaning": "Не передано сообщение о расстоянии до сцепления с вагонами"},
           {"pattern": r'\bизъятие тормозных башмаков\b', "meaning": "Передача разрешения на изъятие тормозных башмаков без доклада от машиниста"}
       ]
   }

  # Проверка на отсутствие фамилии
  if not re.search(r'\b[А-Я][а-я]*[ов|ев|ин|ский|цкий|ая|яя|ой|ий]+\b', transcription):
    violations.append((1, "Отсутствие фамилии"))

  for degree, criteria_list in CRITERIA.items():
       for criterion in criteria_list:
           if re.search(criterion['pattern'], transcription, re.IGNORECASE):
               violations.append((degree, criterion['meaning']))
  return violations

def analyze_audio(path):
    transcription = transcribe_google(path)
    classification = classify_transcription(transcription, tokenizer_bert, model_bert)
    violations = analyze_transcription(transcription)
    result = f"Транскрипция: {transcription}\n"
    #result += f"Классификация: {'Нарушение' if classification == 0 else 'Соответствие'}\n"
    if violations:
        result += "Нарушения:\n"
        for degree, meaning in violations:
            result += f"Степень {degree}: {meaning}\n"
    else:
        result += "Транскрипция соответствует стандартам разговора."
    return result

# Загрузка предобученной модели BERT для классификации текста
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

def browse_file():
    global audio_path, file_label
    audio_path = filedialog.askopenfilename()
    if audio_path:
        file_label.config(text="Выбран файл: " + audio_path.split('/')[-1])

def process_audio():
    if audio_path:
        result = analyze_audio(audio_path)
        text_output.delete('1.0', tk.END)
        text_output.insert(tk.END, result)

root = tk.Tk()
root.title("Анализатор аудиозаписей")

# Загрузка изображения
background_image = PhotoImage(file="your_path_to_image")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

root.configure(background='#f7f7f7')
root.option_add("*Font", "Helvetica 16")

file_label = tk.Label(root, text="Нет выбранных файлов", bg='#f7f7f7')
file_label.pack(pady=10)

browse_button = tk.Button(root, text="Выбрать аудиофайлы", command=browse_file, bg='#c8c8c8', borderwidth=1, height=1, width=20)
browse_button.pack(pady=13)

analyze_button = tk.Button(root, text="Анализировать", command=process_audio, bg='#4caf50', fg='white', borderwidth=1, height=1, width=13)
analyze_button.pack(pady=13)

text_output = tk.Text(root, height=13, width=67, borderwidth=3)
text_output.pack(pady=27)

root.mainloop()
