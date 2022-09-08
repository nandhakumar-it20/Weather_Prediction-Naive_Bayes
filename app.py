import streamlit as st
import pandas as pd
import time

st.title('Weather Prediction using Python')
st.caption('Predicting Weather using the uploaded csv file')
st.info("Developed by NANDHAKUMAR S, SUJITH V, MOHAMED RAFEEK S, DHIVAKAR S [Daisi Hackathon]")
st.snow()
with st.spinner('Loading...'):
    time.sleep(3)
uploaded_files = st.file_uploader("Choose a CSV file to process", type='csv')
if uploaded_files is not None:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)

    


