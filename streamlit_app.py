import streamlit as st
from img_pred import Imagepredict
import os


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # st.write(bytes_data)
    st.image(bytes_data, width=300)
    
    image = Imagepredict.read_image(bytes_data)
    image = Imagepredict.preprocess(image)
    predictions = Imagepredict.predict(image)
    
    st.write()
    
    
    txt = st.text_area("Prediciton Output: \n", predictions)
    # st.write('Sentiment:', run_sentiment_analysis(txt))
    
    
def add_bg_from_url():
    st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700360787.jpg");
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

add_bg_from_url() 
