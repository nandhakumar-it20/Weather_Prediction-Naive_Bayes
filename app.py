import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import fetch_openml
import seaborn as sns; 
import matplotlib.pyplot as plt
import threading
import concurrent.futures

from sklearn.metrics import confusion_matrix

from threading import Thread

def request_search_terms(*args):
    """weather_prediction_naive_bayes.add_bg_from_url().value"""
    pass

...
searchTerms = ["python", "machine learning", "artificial intelligence"]
threads = []
for st in searchTerms:
    threads.append (Thread (target=request_search_terms, args=(st,)))
    threads[-1].start()

for t in threads:
    t.join();


def add_bg_from_url():
    st.markdown("""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/white-background-with-blue-tech-hexagon_1017-19366.jpg?w=740&t=st=1662727120~exp=1662727720~hmac=57c063f03cda9f1849119e18ee1332ac5b0f589ff37afc9b28ff000a4c5e8dc7");
             background-attachment: fixed;
             background-size: cover}}
             </style>""",unsafe_allow_html=True)

st.title('Weather Prediction using Python')
st.caption('Predicting Weather using the dataset fetched from openml')
st.info("Developed by NANDHAKUMAR S", icon="©")
st.subheader('Prediction')
st.snow()  
def st_ui():
    
    with st.sidebar:
        st.image("W.jpg")
        st.header("A PROJECT BY NANDHAKUMAR S")
        st.write("")  
weather = datasets.fetch_openml(name='weather', version=2)
st.write('Features:',   weather.feature_names)
st.write('Target(s):',  weather.target_names)


df = pd.DataFrame( np.c_[weather.data, weather.target],columns=np.append(weather.feature_names, weather.target_names) )
df


st.write('Values:')
st.write(df['outlook'])
st.write('\nFrequencies:')
st.write( df['outlook'].value_counts() )
st.write('\nFrequencies grouped by target:')
st.write( df.groupby(['play','outlook']).size() )


st.write('Values:')
st.write(df['windy'])
st.write('\nFrequencies:')
st.write(df['windy'].value_counts())
st.write('\nFrequencies grouped by target:')
st.write( df.groupby(['play','windy']).size() )


st.write('Values:')
st.write(df['play'])
st.write('\nFrequencies:')
st.write(df['play'].value_counts())


st.write( 'Instances where play=yes:')
st.write( df.query('play == "yes"') )
meanYtemp = df.query('play == "yes"').temperature.mean()
stdYtemp  = df.query('play == "yes"').temperature.std()
st.write( '\nmean temperature (play=yes) ={:10.6f}'.format(meanYtemp) )
st.write( ' std temperature (play=yes) ={:10.6f}'.format( stdYtemp) )
st.write( '\nInstances where play=no:')
st.write( df.query('play == "no"') )
meanNtemp = df.query('play == "no"').temperature.mean()
stdNtemp  = df.query('play == "no"').temperature.std()
st.write( '\nmean temperature (play=no)  ={:10.6f}'.format(meanNtemp) )
st.write( ' std temperature (play=no)  ={:10.6f}'.format( stdNtemp) )


st.write( 'Instances where play=yes:')
st.write( df.query('play == "yes"') )
meanYhumd = df.query('play == "yes"').humidity.mean()
stdYhumd  = df.query('play == "yes"').humidity.std()
st.write( '\nmean humidity (play=yes) ={:10.6f}'.format(meanYhumd) )
st.write( ' std humidity (play=yes) ={:10.6f}'.format( stdYhumd) )
st.write( '\nInstances where play=no:')
st.write( df.query('play == "no"') )
meanNhumd = df.query('play == "no"').humidity.mean()
stdNhumd  = df.query('play == "no"').humidity.std()
st.write( '\nmean humidity (play=no)  ={:10.6f}'.format(meanNhumd) )
st.write( ' std humidity (play=no)  ={:10.6f}'.format( stdNhumd) )


category_columns = ['outlook', 'windy', 'play']
df[ category_columns] = df[ category_columns].apply(lambda x: pd.factorize(x)[0])
df


model = GaussianNB()
x = df.loc[:,:'play']
y = df.loc[:,'play']
model.fit(x, y)
st.write( 'Model parameters:', model.get_params() )


expected = y
predicted = model.predict(x)
st.write('Actual:',    expected)
st.write('Predicted:', predicted)

def pdFunc(power, mean, std, val):
    a = 1/(np.sqrt(2*np.pi)*std)
    diff = np.abs(np.power(val-mean, power))
    b = np.exp(-(diff)/(2*std*std))
    return a*b

likelihoodYes = 2/9 * pdFunc(2,meanYtemp,stdYtemp,77) * pdFunc(2,meanYhumd,stdYhumd,43) * 9/9
st.write( 'Likelihood(play=yes|E) = {:.8f}'.format(likelihoodYes) )


likelihoodNo = 3/5 * pdFunc(2,meanNtemp,stdNtemp,77) * pdFunc(2,meanNhumd,stdNhumd,43) * 5/5
st.write( 'Likelihood(play=no|E) = {:.8f}'.format(likelihoodNo) )


st.write( 'Probability(play=yes|E) = {:.8f}'.format(likelihoodYes/(likelihoodYes+likelihoodNo)) )

st.write( 'Probability(play=no|E) = {:.8f}'.format(likelihoodNo/(likelihoodYes+likelihoodNo)) )
    
st.write('Accuracy =', metrics.accuracy_score(expected, predicted))
st.write('Cohen kappa =', metrics.cohen_kappa_score(expected, predicted))
st.write('Precision =', metrics.precision_score(expected, predicted, average=None))
st.write('Recall =', metrics.recall_score(expected, predicted, average=None))
st.write()
st.write('Metrics =', metrics.precision_recall_fscore_support(expected, predicted, average=None))
st.write('\nPERFORMANCE REPORT:\n')
st.write(metrics.classification_report(expected, predicted))
print('CONFUSION MATRIX:\n')
mat = confusion_matrix(expected, predicted)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['no','yes'], yticklabels=['no','yes'])
plt.xlabel('predicted label')
plt.ylabel('actual label'); 
st.pyplot(plt)
    
    
            
if __name__ == "__main__":
    st_ui()
    
    
    
    
    
    
    


