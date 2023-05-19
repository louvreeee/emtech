import streamlit as st
import tensorflow as tf
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('finals11.hdf5')
  return model
model=load_model()

st.markdown(
"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 450px;
           max-width: 450px;
 }
 """,unsafe_allow_html=True,)  
  
# The side bar that contains radio buttons for selection of charts
with st.sidebar:
    st.header('Select the image that you would like to display')
    chart = st.radio("Sample Image", ('Lion', 'Cheetah','Conclusion'))
    
    if chart == 'Lion': 
      image1 = Image.open('Lion.jpg')
      st.image(image1)
    if chart == 'Cheetah': 
      image1 = Image.open('Cheetah.jpg')
      st.image(image1)
    if chart == 'Conclusion':
      st.title("In this task, I successfully deployed a deep learning model in the cloud. Specifically, I utilized a Convolutional Neural Network (CNN) 
                selected the Lion Vs. Cheetah dataset. To accomplish the deployment, I needed a GitHub account, which I used to sign in to a Streamlit website. 
               It can be denoted that to successfully deploy the model, necessary libraries should be installed such as the tensorflow, streamlit, numpy, etc.
               During the activity, I encountered an error stating no module named TensorFlow, even though I had already installed it. Upon researching the issue, 
               I discovered that the error was due to the file type I used. Initially, I used the H5 file format and attempted to load the model using .load_weights(). 
               After realizing this was the source of the error, I switched to using the Hdf5 file format and saved the model using the command `model.save()`. 
               While any file format can be used to deploy the model, I recommend using the .save() method instead of just saving the weights.")
 
    

st.title("Lion or Cheetah Classifier")
st.info("An image classifying project that differentiates between two very similar-looking wild cats: Cheetahs and Lion using Python and TensorFlow")

image = Image.open('LionCheetah.png')
st.image(image, caption='Lion vs Cheetah')

st.info("""To view and display a sample image of a cheetah or lion, please select the desired option from the sidebar menu.""")    
# This container will be displayed below the text above
        
#st.write("""
# Lion or Cheetah Classification""")
file=st.file_uploader("Choose photo from computer, must be a lion or cheetah",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Lion','Cheetah']
    string="The picture shown above is a  "+class_names[np.argmax(prediction)]
    st.success(string)
