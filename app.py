import streamlit as st
import pandas as pd
import time
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import fetch_openml

st.title('Weather Prediction using Python')
st.caption('Predicting Weather using the uploaded csv file')
st.info("Developed by NANDHAKUMAR S, SUJITH V, MOHAMED RAFEEK S, DHIVAKAR S [Daisi Hackathon]")
st.snow()
with st.spinner('Loading...'):
    time.sleep(1)
    
weather = datasets.fetch_openml(name='weather', version=2)
st.write('Features:',   weather.feature_names)
st.write('Target(s):',  weather.target_names)
    


