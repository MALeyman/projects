
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr



from projects.segmentation_1 import get_segmentation_tab
from projects.home_tab import home_tab
from projects.detection_1 import get_detection_tab_1
from projects.detection_2 import get_detection_tab_2
from projects.interpolated import get_interpolated_tab
from projects.imdb_classification import get_imdb_classification_tab

def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Главная"):
                home_tab()
            with gr.TabItem("Сегментация дорожных сцен"):
                get_segmentation_tab()
            with gr.TabItem("Детекция лиц"):
                get_detection_tab_1()
            with gr.TabItem("Детекция с БПЛА"):
                get_detection_tab_2()
            with gr.TabItem("Интерполяция изображений"):
                get_interpolated_tab()
            with gr.TabItem("Классификация отзывов"):
                get_imdb_classification_tab()

    demo.launch(debug=True)


if __name__ == "__main__":
    main()

