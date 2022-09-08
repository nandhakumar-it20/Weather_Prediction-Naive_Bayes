import streamlit as st
import pandas as pd
import time

st.title('Weather Prediction using Python')
st.caption('Predicting Weather using the uploaded csv file')
st.info("Developed by NANDHAKUMAR S, SUJITH V, MOHAMED RAFEEK S, DHIVAKAR S [Daisi Hackathon]")
st.snow()
with st.spinner('Loading...'):
    time.sleep(3)
uploaded_files = st.file_uploader("Choose a CSV file to process", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data) 
my_bar = st.progress(0)
for percent_complete in range(100):
     time.sleep(0.1)
     my_bar.progress(percent_complete + 1)
    


