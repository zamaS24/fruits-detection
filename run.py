import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests 




model = load_model('C:/Users/msihamza/Desktop/projet/Veg.h5')
labels = {0: 'betterave',
        1: 'carotte',
        2: 'courgette',
        3: 'haricot',
        4: 'obergine',
        5: 'oignon',
        6: 'poivron',
        7: 'poivre',
        8: 'radis',
        9: 'pomme de terre'}


def processed_img(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("Vegetable Recognition System")
    st.write("By OMARI Hamza and hamzaoui thamr")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            st.success("**Predicted: " + result + '**')


run()
