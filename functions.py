import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization

df = pd.read_csv("book_data.csv")
df.dropna(subset=["genres", "book_desc"], inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

max_length = 200  # Define the maximum length of sequences

vectorizer = TextVectorization(
    max_tokens=10000,
    output_sequence_length=max_length,
    output_mode="int"
    )

text_data = df['book_desc'].values  # Assuming 'summary' column contains the book summaries
vectorizer.adapt(text_data)

def test_vectorizer(text):
    a = clean_text(text)
    a = vectorizer(a)
    a = np.reshape(a, (1, max_length))
    return a

def reviewBook(model,text):
    labels = ['fiction', 'nonfiction']
    a = clean_text(text)
    a = vectorizer(a)
    a = np.reshape(a, (1, max_length))
    output = model.predict(a, batch_size=1)
    score = (output>0.5)*1
    pred = score.item()
    return labels[pred]
