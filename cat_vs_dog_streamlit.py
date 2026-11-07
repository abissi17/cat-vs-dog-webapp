import streamlit as st
from fastai.vision.all import *
import pathlib
import platform

# this is how to make the title yellow
st.markdown("<h1 style='color: yellow;'>Cat or Dog Classifier</h1>", unsafe_allow_html=True)
st.text("Created by Abigail")

# this is how to make a file uploader
uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])

# labeling mechanism / function (YOU NEED THIS FOR THE LOAD LEARNER)
def cat_or_dog(file_name):
    if file_name[0].isupper():
        return "CAT"
    else:
        return "NOT CAT"

# upload the pkl file (import fastai library)
model = load_learner("cat_vs_dog_model_fastai2_7_19.pkl")

# this is how to display the image after uploading
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    resized_img = img.resize((224, 224), Image.NEAREST)
    prediction = model.predict(resized_img)

    # do this line first to show them the list --> st.text(prediction)
    # it should appear as something like this --> ('CAT', tensor(0), tensor([1.0000e+00, 3.3088e-12]))
    # then do this line next --> st.text(prediction[0])

    # then get the confidence score
    index = int(prediction[1]) # this is getting the index of the confidence score
    # this is getting the confidence score and multiplying by 100
    confident_level = prediction[2][index] * 100

    # ('CAT', tensor(0), tensor([1.0000e+00, 3.3088e-12]))
    # (CAT or NOT CAT, index of whatever the model predicted, [CAT confidence score, NOT CAT confidence score])

    # .2f is suppose to 91.128123123 --> 91.13
    if confident_level > 90:
        label = f"I am {confident_level:.2f} % sure that it is a {prediction[0]}"
    else:
        label = f"WARNING. It am {confident_level:.2f} % sure that it is a {prediction[0]}"

    st.text(label)
    st.image(resized_img)


