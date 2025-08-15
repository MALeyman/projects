import os 
import pathlib 
import numpy as np
from glob import glob
import pickle
import shutil



# функция преобразования файла текста в файл с числами

def transform_text_1(file1, path_vocab):
    # Читаем словарь и создаём маппинг слова -> индекс
    with open(path_vocab, "r") as f_vocab:
        lines = f_vocab.read().splitlines()
    vocab_dict = {word.lower(): idx+1 for idx, word in enumerate(lines)}

    mass = []
    with open(file1, "r") as f_text:
        text = f_text.read()

    # Разбиваем весь текст на слова по пробелу
    words = text.split()

    for word in words:
        word_lower = word.lower()
        if word_lower in vocab_dict:
            mass.append(vocab_dict[word_lower])
        else:
            mass.append(0)  # если слова нет в словаре

    return mass



import os
import random
import pickle

def transform_text_2(text1, vocab, len_text, rand=False):
    mass = [] 
    str1 =''    
    fl = False
    list_1 = [0, 1]

    for ch in text1:
        if len(mass) > len_text - 1:
            return mass
        if ch != ' ':
            str1 = str1 + ch
        if ch == ' ':
            if str1 != '':
                fl = False
                for i in range(len(vocab)):
                    if str1.lower() == vocab[i].lower():
                        fl = True
                        random_choice = random.choice(list_1)
                        if rand and random_choice == 0:
                            mass.append(1)
                            break
                        mass.append(i + 2)
                        str1 = ''
                        break
                if not fl:
                    mass.append(0)
                    str1 = ''
    # Проверка последнего слова
    if str1 != '':
        fl = False
        for i in range(len(vocab)):
            if str1.lower() == vocab[i].lower():
                fl = True
                mass.append(i + 2)
                break
        if not fl:
            mass.append(0)

    while len(mass) < len_text:
        mass.insert(0, 0)
    return mass


def create_balanced_and_test_datasets(base_path, vocab, len_text, rand=False):
    classes = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    class_counts = {}

    # Подсчёт количества файлов в каждом классе
    for cl in classes:
        folder_path = os.path.join(base_path, cl)
        count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        class_counts[cl] = count

    min_count = min(class_counts.values())
    print(f"Минимальное количество примеров на класс: {min_count}")

    balanced_data = []
    balanced_labels = []
    test_data = []
    test_labels = []

    for cl in classes:
        folder_path = os.path.join(base_path, cl)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        random.shuffle(files)

        # Выбираем min_count файлов для сбалансированного датасета
        balanced_files = files[:min_count]
        # Остальные — для тестового датасета
        test_files = files[min_count:]

        # Обработка balanced
        for file_name in balanced_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            vec = transform_text_2(text, vocab, len_text, rand=rand)
            balanced_data.append(vec)
            balanced_labels.append(int(cl))

        # Обработка test
        for file_name in test_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            vec = transform_text_2(text, vocab, len_text, rand=rand)
            test_data.append(vec)
            test_labels.append(int(cl))

    return balanced_data, balanced_labels, test_data, test_labels



# функция получения имени файла из пути
def str_name(str1):
    ''' 
    функция получения имени файла из пути
    '''
    p = os.path.basename(str1)
    return os.path.splitext(p)[0]

# функция получения рейтинга отзыва и номера отзыва по имени файла
def inv1(str1):
    ''' 
    функция получения рейтинга отзыва и номера отзыва по имени файла
    '''
    l = len(str1)
    ss = '444'
    kk = ''
    for i in range(l):
        s = str1[i]
        ss = ss + s
        if s !='_':
            continue
        if s =='_':
            i = i+1
            while i < l:
                ss = ss + str1[i]
                kk = str1[i]
                i = i+1
            break
    return ss, kk  



# функция вывода текста  отзыва
def read_file(file1):
    ''' 
    функция вывода текста  отзыва
    '''
    with open(file1, "r") as f1:
        text = f1.read()
        print(text)
        f1.close
    return text


# функция разбивает исходный датасет по классам (по рейтингу)
def preprocess_text(mas, cls):   
    ''' 
    функция разбивает исходный датасет по классам (по рейтингу)
    '''
    for i in range(len(mas)):
        str_n = mas[i]
        str_1 = str_name(str_n)
        
        name_faile, num_class = inv1(str_1)
        name_faile1 = name_faile + '.txt'
        
        if num_class == '0':
            path_new = cls[0]
        elif num_class == '1':
            path_new = cls[1]          
        elif num_class == '2':
            path_new = cls[2]
        elif num_class == '3':
            path_new = cls[3]
        elif num_class == '4':
            path_new = cls[4]
        elif num_class == '5':
            path_new = None             
        elif num_class == '6':
            path_new = None
        elif num_class == '7':
            path_new = cls[7]  
        elif num_class == '8':
            path_new = cls[8]
        elif num_class == '9':
            path_new = cls[9]                    

                  
        shutil.copy2(str_n, os.path.join(path_new, name_faile1))


# функция загрузки словаря для кодировки слов из файла
def vocab_read(path_vocab):
    ''' 
    функция загрузки словаря для кодировки слов из файла
    '''
    with open(path_vocab, "r") as f1:
        line = f1.read().splitlines()
    return line




