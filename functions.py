import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from langdetect import detect

df = pd.read_csv("cleaned_data_checkpoint.csv")

def standardize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

max_length = 200  # Define the maximum length of sequences

vectorizer = TextVectorization(
    max_tokens=20000,
    output_sequence_length=max_length,
    output_mode="int"
    )

text_data = df['desc'].values  
vectorizer.adapt(text_data)

def reformat(text):
    a = standardize_text(text)
    a = vectorizer(a)
    a = np.reshape(a, (1, max_length))
    return a

def predict_genre(model,text):
    a = reformat(text)
    a = np.reshape(a, (1, max_length))
    output = model.predict(a, batch_size=1)
    if output < 0.5:
      return "fiction"
    else:
      return "non-fiction"

