
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr
import sys
import cv2
import torch


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
from projects.files.utils import  morph_video_1
from projects.common.session import  get_device


default1 = os.path.join(BASE_DIR, "files/1.jpg")
default2 = os.path.join(BASE_DIR, "files/2.jpg")


def get_interpolated_tab():

        gr.Markdown("## Модель морфинга между двумя изображениями VAE-GAN")
        gr.Markdown("---")
        with gr.Row():
            # Левая колонка со слайдерами (общие для обеих вкладок)
            with gr.Column(scale=1):
                # Добавляем метку устройства
                device_label = gr.Label(value=get_device(), label="Работаем на устройстве")
                
                image1 = gr.Image(label="Изображение 1", value=default1, type="pil", show_label=True, elem_id="img1_small")
                image2 = gr.Image(label="Изображение 2", value=default2, type="pil", show_label=True, elem_id="img2_small")
                generate_btn = gr.Button("Создать переход")
    
            with gr.Column(scale=2):  # Правая колонка - 2/3 ширины
                output_video = gr.Video(label="Морф-видео", format="mp4", autoplay=True, height=512, width=512)

        generate_btn.click(morph_video_1, inputs=[image1, image2], outputs=output_video)