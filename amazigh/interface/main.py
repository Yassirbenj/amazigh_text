import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageChops


def load_model():
    #model_path='/Users/yassir2/code/Yassirbenj/amazigh_text/models/amazighmodel3.h5'
    model=tf.keras.models.load_model("amazigh/interface/amazighmodel3.h5")
    return model

def predict(model,image):
    yhat=model.predict(np.expand_dims(image/255,0))
    labels=['ya','yab','yach','yad','yadd','yae','yaf','yag',
            'yagh','yagw','yah','yahh','yaj','yak','yakw','yal',
            'yam','yan','yaq','yar','yarr','yas','yass','yat',
            'yatt','yaw','yax','yay','yaz','yazz','yey','yi','yu']
    result=labels[np.argmax(yhat)]
    return result

def trim(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    box = diff.getbbox()
    if box:
        image=image.crop(box)
    return image

with st.form("input_form"):
    st.write("<h3>Upload your image for the magic ✨</h3>", unsafe_allow_html=True)
    input_img = st.file_uploader('character image',type=['png', 'jpg','jpeg'])
    if st.form_submit_button("Predict"):
        if input_img:
            image = Image.open(input_img)
            image=trim(image)
            new_image=image.resize((64,64))
            img_array = np.array(new_image)
            loaded_model = load_model()
            prediction = predict(loaded_model,img_array)
            st.image(new_image)
            st.write(f"<h3>The prediction is: {prediction} </h3>", unsafe_allow_html=True)
