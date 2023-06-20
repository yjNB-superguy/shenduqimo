import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from PIL import Image, ImageEnhance
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
from model import NeuralStyleTransferModel
import settings
import utils

st.set_page_config(page_title="图像编辑器", page_icon=":eyeglasses:")

st.title("图像编辑器")

# 上传一个图像
uploaded_file = st.file_uploader("上传一个图像", type=["png", "jpg", "jpeg"])
if not uploaded_file:
    st.warning("请上传一张图像。")
    st.stop()

original_image = Image.open(uploaded_file)

# 显示原始图像
st.image(original_image, use_column_width=True, caption="原始图像")



def restore_image(image_array):
    # 使用 Mask R-CNN 进行图像修复

        # 将图像转化为 OpenCV 格式
        image = np.array(image_array)
        image = image[:, :, ::-1].copy()

      

restored_image = restore_image(original_image)

# 滤镜效果

filtered_image = restored_image

# 风格转换
style = st.sidebar.selectbox("风格转换", ["原图", "梵高"])

if style == "梵高":
    # 加载预训练的梵高风格转换模型
    model_file = "models/vgg19_starry_night.h5"
    model = tf.keras.models.load_model(model_file, compile=False)

    # 将图像转化为 Numpy 数组
    input_img = np.array(filtered_image)

    # 对图像进行预处理
    img = tf.keras.preprocessing.image.array_to_img(input_img)
    img = img.resize((model.input_shape[1], model.input_shape[2]))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg19.preprocess_input(x)

    # 使用预训练模型进行风格转换
    y = model.predict(x)

    # 将输出转化为 PIL.Image 对象并显示
    output_img = y.reshape((model.output_shape[1:]))
    output_img = np.clip(output_img, 0, 255)
    output_img = output_img.astype("uint8")
    output_img = Image.fromarray(output_img)
else:
    output_img = filtered_image