import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('finals11.hdf5')
  return model

model = load_model()

# Create a function to collect user feedback and save it to a CSV file
def collect_user_feedback(image_name, classification, user_feedback):
    feedback_data = pd.DataFrame({'Image': [image_name], 'Classification': [classification], 'UserFeedback': [user_feedback]})
    
    # Append feedback to an existing CSV file or create a new one
    feedback_data.to_csv('user_feedback.csv', mode='a', header=not st.session_state.feedback_data_exists, index=False)
    
    # Set a session state flag to indicate that feedback data exists
    st.session_state.feedback_data_exists = True

# Check if the feedback data exists
if 'feedback_data_exists' not in st.session_state:
    st.session_state.feedback_data_exists = False

st.markdown(
"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 450px;
           max-width: 450px;
    }
 """, unsafe_allow_html=True)

# The side bar that contains radio buttons for selection of charts
with st.sidebar:
    st.header('Select the image that you would like to display')
    chart = st.radio("Sample Image", ('Lion', 'Cheetah'))
    
    if chart == 'Lion':
        image1 = Image.open('LION3.jpg')
        st.image(image1)
    if chart == 'Cheetah':
        image1 = Image.open('Cheetah.jpg')
        st.image(image1)
    st.header('ABOUT')
    chart1 = st.radio("More Info", ('Conclusion', 'About the student'))
    if chart1 == 'About the student':
        st.info("""Maria Lourdes Camenforte Gesite, 3rd Year BSCpE Student""")
        image1 = Image.open('MariaLourdesGesite.JPG')
        st.image(image1)
    if chart1 == 'Conclusion':
        st.info("""In this activity, I successfully deployed a deep learning model in the cloud. Specifically, I used a Convolutional Neural Network (CNN) 
                selected the Lion Vs. Cheetah dataset. To accomplish the deployment, I needed a GitHub account, which I used to sign in to a Streamlit website. 
                It can be denoted that to successfully deploy the model, necessary libraries should be installed such as TensorFlow, Streamlit, NumPy, etc.
                During the activity, I encountered an error stating no module named TensorFlow, even though I had already installed it. Upon researching the issue, 
                I discovered that the error was due to how I saved the model. Initially, I used the H5 file format and attempted to save the model using .load_weights(). 
                After realizing this was the source of the error, I switched to using the Hdf5 file format and saved the model using the command `model.save()`. 
                While any file format can be used to deploy the model, I recommend using the .save() method instead of just using the weights in saving the model.""")
    st.header('LINKS')
    chart2 = st.radio('', ('Google Colab Link', 'Github Repository Link'))
    if chart2 == 'Google Colab Link':
        st.info("""https://colab.research.google.com/drive/17ovVa1jHejGPrDaT_S8-0--y7vUmyHRM?usp=sharing""")
    if chart2 == 'Github Repository Link':
        st.info("""https://github.com/louvre11/emtech/tree/a16b6c7fcea5647349d089336994a45a6a8fe1cd""")    

st.title("Lion or Cheetah Classifier")
st.info("An image classifying project that differentiates between two very similar-looking wild cats: Cheetahs and Lion using Python and TensorFlow")

image = Image.open('LionCheetah.png')
st.image(image, caption='Lion vs Cheetah')

st.info("""To view and display a sample image of a cheetah or lion, please select the desired option from the sidebar menu. The conclusion for this activity is also provided there.""")

file = st.file_uploader("Choose a photo from your computer, it must be a lion or cheetah", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Lion', 'Cheetah']
    string = "The picture shown above is a  " + class_names[np.argmax(prediction)]
    st.success(string)

# User feedback section
st.header("User Feedback")
user_feedback = st.checkbox("Was this classification correct?")

if st.button("Submit Feedback"):
    collect_user_feedback(file.name, class_names[np.argmax(prediction)], user_feedback)
    st.success("Feedback submitted successfully!")

# Display statistics or insights based on user interactions and collected feedback
if st.session_state.feedback_data_exists:
    feedback_data = pd.read_csv('user_feedback.csv')

    total_feedback = len(feedback_data)
    correct_feedback = len(feedback_data[feedback_data['UserFeedback']])

    st.subheader("Feedback Statistics")
    st.write(f"Total Feedback Submitted: {total_feedback}")
    st.write(f"Correct Feedback: {correct_feedback}")
    st.write(f"Feedback Accuracy: {correct_feedback / total_feedback:.2%}")
