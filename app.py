import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
# @st.cache(allow_output_mutation=True)
def load_models():
    model=tf.keras.models.load_model('main.h5')
    return model
model=load_models()
st.title('AI Detector')
st.subheader('Differentiate between AI generated and Real faces')
st.write('made by Gautam Sahu:heart:')
file=st.file_uploader(label='Upload an image',type=['jpg','jpeg','png'])
#preparation through evaluation
# classes = ['fake','real']


 
if file is not None:
   
    image = load_img(file)
    
    st.image(image,caption='Uploaded Image',width=400)
    # Convert the image to a NumPy array
    image_array = img_to_array(image)
    

    # Preprocess the image
    image_array = tf.image.resize(image_array, (128, 128))
    image_array = image_array / 255.0

    # Make the image a batch of 1
    image_array = tf.expand_dims(image_array, axis=0)
   
    classes = ['Fake','Real']
    with st.spinner('Please wait...'):
        prediction=model.predict(image_array)
        predictLabel = classes[np.argmax(prediction)]
        prediction_conf=max(prediction[0])
        if predictLabel=='Real':
            labelString = str(f'The image is {predictLabel} with a confidence rating of {str(int(1000*prediction_conf)/10)}%')
            st.subheader(labelString)
        else:
            labelString = str(f'The image is {predictLabel} with a confidence rating of {str(int(1000*prediction_conf)/10)}%')
            st.subheader(labelString)
