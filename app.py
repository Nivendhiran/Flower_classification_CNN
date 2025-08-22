import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

#header of the page

st.header('Flower Classification Using CNN Model')
flower_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model=load_model('flower_class.h5')
#use function
def classify_images(image_path):
    input_images=tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array=tf.keras.utils.img_to_array(input_images)
    #our model trained the data by batch wise so we give input also in batch wise
    input_image_exp_dim=tf.expand_dims(input_image_array,0)


    predictions=model.predict(input_image_exp_dim)
    score=tf.nn.softmax(predictions[0])
    outcome='the image belongs to '+ flower_names[np.argmax(score)] +' With the score of ' +str(np.max(score)*100)
    return outcome


uploaded_file=st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload_imgs',uploaded_file.name,),'wb')as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, width=200)
st.markdown(classify_images(uploaded_file))


     