

import onnxruntime as ort
import os
import gradio as gr
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F





BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("ПУТЬ:  ", BASE_DIR)

# Корень проекта — на два уровня выше, т.к. __file__ в gradio_projects/projects
# ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Добавляем корень проекта в sys.path, если его там нет
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# print("ПУТЬ:  ", ROOT_DIR)
# Теперь импортируем модуль как абсолютный из корня проекта
from projects.common.session import  get_device


default_text = "In Queen of The Damned,Akasha(Aaliyah) was more sexy and had a bigger,demanding presence, she just caught your eye and attention. now the movie did have faults, like the lack of explaining Akasha's past. What i also Did not like was the that the movie didn't really explain or show more of what the relationship between Lestat and Akasha was/ or was like.Akasha's (Aaliyah's) role was sort of limited in the movie and she didn't appear until the 2nd half of the movie and then to top it off, her(Akasha's) death came 2 quickly.But i liked how Akasha fought back when the ancients tried to kill her, because in the book the last fight between Akasha and The ancients was rather boring (they killed Akasha in like 2 secs).Akasha's head got knocked off in 1 sec and Lestat turned into the biggest punk in the world.<br /><br />Aaliyah played Akasha very well and Stuart was perfect as Lestat, they could not have picked a better Akasha or Lestat."



path_vocab = os.path.join(BASE_DIR, 'files/imdb.vocab')
path_model = os.path.join(BASE_DIR, "files/lstm_model.pth")





# гиперпараметры модели
input_size = 40000 
hidden_size = 256
num_classes = 8


#Определение класса модели
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)       
        self.dropout = nn.Dropout(p=0.2)  # Добавление слоя Dropout
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm1(embedded)
        output, _ = self.lstm2(output)
        #output = self.dropout(output)  # Применение Dropout к выходу LSTM
        #output = self.fc(output[:, -1, :])  # Используется только последний выход LSTM
        
        output = output[-1, :]  # Получение последнего выхода LSTM
        output = self.fc(output)  # Применение полносвязного слоя
        return output


##### функция преобразования текста в числа

def transform_text(text1, path_vocab=path_vocab, len_text=280):
	fl = False 
	with open(path_vocab, "r") as f1:
		vocab = f1.read().splitlines()
	vocab = vocab[:39998]  
	mass = [] 
	str1 =''    
	for ch in text1:
		if len(mass)>len_text-1: # если слов больше чем нужно, выходим
			data_tensor = torch.tensor(mass)
			return data_tensor
		if ch !=' ':
			str1 = str1 + ch
		if ch ==' ':
			if str1 !='':                      
				fl = False                                                            
				for i in range(len(vocab)):
					if str1.lower() == vocab[i].lower():
						fl = True
						mass.append(i+2)
						str1 =''
						break                                                 
				if fl == False: #  если слово не найдено, заменяем нулями
					mass.append(0)
					str1 =''        
	for i in range(len(vocab)): # Проверяем последнее слово
		if str1.lower() == vocab[i].lower():
			fl = True
			mass.append(i+2)
			str1 =''
			break                                                 
	if fl == False: #  если слово не найдено, заменяем нулями
		mass.append(0)
		str1 =''

	if len(mass)<len_text:  #  если слов меньше чем нужно, добавляем нулями.
		while len(mass)<len_text:
			mass.insert(0, 0)
			str1 =''       
	data_tensor = torch.tensor(mass)    
	return data_tensor




def imdb_classification(text_input):
        # Создание и загрузка модели
    model = SentimentClassifier(input_size, hidden_size, num_classes).to('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()
    model = model.to(device)


    text2 = transform_text(text_input)
    text2 = text2.to(device)
    output = model(text2)
    probs = F.softmax(output, dim=0)
    predicted_class = torch.argmax(probs, dim=0)
    prediction = predicted_class.item()

    if prediction < 4:
        prediction += 1
        text_out = "Отзыв негативный"
    if prediction > 3:
        prediction += 3
        text_out = "Отзыв положительный"
    label_out =  prediction
    return text_out, label_out



def get_imdb_classification_tab():

        gr.Markdown("## Модель Классификации отзывов IMDb LSTM")
        gr.Markdown("---")
        with gr.Row():
            # Левая колонка со слайдерами (общие для обеих вкладок)
            with gr.Column(scale=1):
                # Добавляем метку устройства
                device_label = gr.Label(value=get_device(), label="Работаем на устройстве")
                text_output = gr.Label(value="", label="Классификация отзыва")
                label_output = gr.Label(value="", label="Рейтинг")

                
                generate_btn = gr.Button("Присвоить рейтинг")
    
            with gr.Column(scale=2): 
				
                text_input = gr.Textbox(value=default_text, label="Текст отзыва", lines=15)

        generate_btn.click(imdb_classification, inputs=[text_input], outputs=[text_output, label_output])
		


