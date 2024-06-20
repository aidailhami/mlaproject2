import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.title('Schzophrenia Proneness Predictor')

st.sidebar.header('User Input Parameters')

def user_input_features1():
    age = st.sidebar.slider('Age',0,100,50,key = "age")
    gender = st.sidebar.radio('Gender',["Male","Female"],key = "gender")
    if(gender=="Male"):gender=0
    else:gender=1
    marital_status = st.sidebar.radio('Gender',["Single","Married","Divorced","Widowed"],key = "marital_status")
    if(marital_status=="Single"):marital_status=0
    elif(marital_status=="Married"):marital_status=1
    elif(marital_status=="Divorced"):marital_status=2
    else:marital_status=3
    fatigue = st.sidebar.slider('Fatigue',-1.0,2.0,0.0,key = "fatigue")
    slowing = st.sidebar.slider('Slowing',-1.0,2.0,0.0,key = "slowing")
    pain = st.sidebar.slider('Pain',-1.0,2.0,0.0,key = "pain")
    hygiene = st.sidebar.slider('Hygiene',-1.0,2.0,0.0,key = "hygiene")
    movement = st.sidebar.slider('Movement',-1.0,2.0,0.0,key = "movement")
    data={'age':age,
          'gender':gender,
          'marital_status':marital_status,
          'fatigue':fatigue,
          'slowing':slowing,
          'pain':pain,
          'hygiene':hygiene,
          'movement':movement}
    features=pd.DataFrame(data, index=[0])
    return features
df=user_input_features1()


def user_input_features2():
    age = st.sidebar.slider('Age',0,100,50,key = "age2")
    gender = st.sidebar.radio('Gender',["Male","Female"],key = "gender2")
    if(gender=="Male"):gender=0
    else:gender=1
    marital_status = st.sidebar.radio('Gender',["Single","Married","Divorced","Widowed"],key = "marital_status2")
    if(marital_status=="Single"):marital_status=0
    elif(marital_status=="Married"):marital_status=1
    elif(marital_status=="Divorced"):marital_status=2
    else:marital_status=3
    fatigue = st.sidebar.slider('Fatigue',-1.0,2.0,0.0,key = "fatigue2")
    slowing = st.sidebar.slider('Slowing',-1.0,2.0,0.0,key = "slowing2")
    pain = st.sidebar.slider('Pain',-1.0,2.0,0.0,key = "pain2")
    hygiene = st.sidebar.slider('Hygiene',-1.0,2.0,0.0,key = "hygiene2")
    movement = st.sidebar.slider('Movement',-1.0,2.0,0.0,key = "movement2")
    data={'age':age,
          'gender':gender,
          'marital_status':marital_status,
          'fatigue':fatigue,
          'slowing':slowing,
          'pain':pain,
          'hygiene':hygiene,
          'movement':movement}
    features=pd.DataFrame(data, index=[0])
    return features
input_df=user_input_features2()

st.subheader('User Input Parameters')
st.write(df)

ds_raw=pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/BCI3333 MLA/Final Assessment/schizophrenia/SchizophreniaSymptomnsData.csv")
ds=ds_raw.drop(columns=['Name','Schizophrenia'])
df=pd.concat([input_df,ds],axis=0)

encode=['age','gender','marital_status','fatigue','slowing','pain','hygiene','movement']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
df=df[:1]


#read built classification model
load_clf=pickle.load(open('built_model.pkl','rb'))

#apply model to make predictions
prediction=load_clf.predict(df)
prediction_proba=load_clf.prediction_proba(df)

st.subheader('Prediction')
schizo_proneness=np.array('Low','Moderate','Elevated','High','Very High')
st.write(schizo_proneness[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)