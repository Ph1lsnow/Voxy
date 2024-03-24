import streamlit as st
import time
# from inference import predict, hesitator


st.title('Модель для добавления хезитаций в текст')

import torch
import torch.nn as nn
from IPython.display import Audio
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, VitsModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# Архитектура модели

class hesitator(nn.Module):
    def __init__(self, freeze_bert=False, lstm_dim=-1):
        super(hesitator, self).__init__()
        self.output_dim = 2
        self.bert_layer = BertModel.from_pretrained('ai-forever/ruBert-base')
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = 768
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=self.output_dim)

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        # (B, N, E) -> (B, E, N)
        x = x.permute(0, 2, 1)
        return x


# device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
device = torch.device('cpu')
# print('device:', device)
tokenizer = BertTokenizer.from_pretrained('ai-forever/ruBert-base')


# !cp '/content/drive/MyDrive/sirius_summer/model.pt' 'model_sirius_bert_lstm.pt'
# model = hesitator()
# st.write('Loading model!')
bert_loc = 'model.pt'
if not os.path.isfile(bert_loc):
    torch.hub.download_url_to_file('https://huggingface.co/Diedmen/voxy/resolve/main/model.pt?download=true',
                                   bert_loc)
model = torch.load(bert_loc, map_location=device)
model.eval()


torch.set_num_threads(4)
local_file = 'model_tts.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://huggingface.co/Diedmen/voxy/resolve/main/model_tts.pt?download=true',
                                   local_file)

model_tts = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model_tts.to(device)

sample_rate = 48000
put_accent=True
put_yo=True
speaker='kseniya'

def predict(text):

    for i in ['.', ',']:
        text = re.sub(rf'([^0-9{i}])(\{i})', r'\1 \2', text)
        text = re.sub(rf'(\{i})([^0-9{i}])', r'\1 \2', text)

    for i in list('!?()-'):
        text = re.sub(rf'([^{i} ])(\{i})', r'\1 \2', text)
        text = re.sub(rf'(\{i})([^{i} ])', r'\1 \2', text)

    for i in list('.,!?()'):
        while re.search(rf'(\{i}) (\{i})', text):
            text = re.sub(rf'(\{i}) (\{i})', r'\1\2', text)


    hesitations = ['ну..', 'эм..,','эээ..,', 'ааа..,']

    def get_hesitation(hesitations):
      return np.random.choice(hesitations)

    max_length = 512

    def convert_sent_to_dummy_dataset(sent, max_length):
      x = [101]
      y_mask = [1]
      attention_mask = [1]
      words = text.split()

      for word in words:
          ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))


          if len(ids) > 0:
              x.append(ids[0])
              y_mask.append(1)
              attention_mask.append(1)

          for i in range(1, len(ids)):
              x.append(ids[i])
              y_mask.append(0)
              attention_mask.append(1)


      x = x[:max_length-1]
      y_mask = y_mask[:max_length-1]
      attention_mask = attention_mask[:max_length-1]


      x.append(102)
      y_mask.append(0)
      attention_mask.append(1)

      return x, y_mask, attention_mask


    x, y_mask, attention_mask = convert_sent_to_dummy_dataset(text, max_length)
    threshold = 1e-3

    with torch.no_grad():
        outputs = model(torch.tensor(x).to(device), torch.tensor(attention_mask).reshape(1, -1).to(device))
    # predictions = torch.argmax(outputs, dim=1)[0]#[0, :-1]
    predictions = torch.nn.Softmax(dim=1)(outputs[0])[1]
    predictions = predictions[torch.tensor(y_mask) == 1]
    words = text.split()
    new_words = []
    new_words_for_tts = []
    count_hesitations = 0

    is_hesitations = [0]

    if predictions[0] > threshold and np.random.uniform() > 0.75:
      count_hesitations += 1
      hes = get_hesitation(hesitations[1:])
      hes = hes[0].upper() + hes[1:] + ','
      new_words.append(hes)
      if hes in ['эээ', 'ааа']:
          new_words_for_tts.append(f'< prosody rate = "x-slow" > {hes[0]} < / prosody >')
      else:
          new_words_for_tts.append(f'< prosody rate = "x-slow" > {hes} < / prosody >')
      is_hesitations.append(1)
    else:
      is_hesitations.append(0)

    for i, (word, pred) in enumerate(zip(words, predictions[1:])):
      if is_hesitations[-1]:
          if not word.isalnum():
              new_words = new_words[:-1]
              new_words_for_tts = new_words_for_tts[:-1]
          else:
              word = word[0].lower() + word[1:]
      new_words.append(word)
      new_words_for_tts.append(word)

      if pred >= threshold and i < len(words) - 1 and count_hesitations < 2 and is_hesitations[-2] == is_hesitations[-1] == 0:
        count_hesitations += 1
        hes = get_hesitation(hesitations[1:])
        if word[-1].isalnum():
          hes = ', ' + hes
        hes += ', '
        new_words.append(hes)
        if hes in ['эээ', 'ааа']:
            new_words_for_tts.append(f'< prosody rate = "x-slow" > {hes[0]} < / prosody >')
        else:
            new_words_for_tts.append(f'< prosody rate = "x-slow" > {hes} < / prosody >')
        is_hesitations.append(1)
      else:
        is_hesitations.append(0)


    text = ' '.join(new_words)
    # text_hes = ' '.join(new_words_for_tts)
    text_hes = text


    with torch.no_grad():
        output = model_tts.apply_tts(text=text,
                    speaker=speaker,
                    sample_rate=sample_rate,
                     put_accent=put_accent,
                     put_yo=put_yo
                                     )



    return text, output

#
# if __name__ == "__main__":
                #     text = 'Добрый день, с вашего счёта спишется 500 рублей'
#
#     res = predict(text)
#
#     print(f"РЕЗУЛЬТАТ: {res}")


text = st.text_input("Введите текст:")
start_time = time.time()
if st.button("Добавить хезитации"):
    text, audio = predict(text)
    text1 = text.replace('.,', '')
    text1 = text1.replace('.', '')
    st.write(f"РЕЗУЛЬТАТ: \n{text1}")
    st.audio(audio.numpy(), sample_rate =sample_rate)
    end_time = time.time()
    st.write(f"Время выполнения: {round(end_time-start_time, 1)}s")
