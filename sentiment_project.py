import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

def mycleaning(doc):
    return re.sub("[^a-zA-Z ]","",doc).lower()

model = joblib.load("sentiment_model.pkl")

import streamlit as st

st.markdown(
    "<div style='"
    "background-image: url(https://images.unsplash.com/photo-1566073771259-6a8506099945);"
    "background-size: cover;"
    "background-position: center;"
    "padding: 40px;"
    "border-radius: 12px;"
    "text-align: center;"
    "color: white;'>"
    "<h1>Sentiment Analysis</h1>"
    "</div>",
    unsafe_allow_html=True
)




st.sidebar.image(
    "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa",
    use_container_width=True
)

st.sidebar.title("Sentiment Analysis")

st.sidebar.title("About project")
st.sidebar.write("prediction of sentiment Neg or pos for a food review ")

st.sidebar.title("contact us📱")
st.sidebar.write("09999991")

st.sidebar.title("About u🙋‍♂️")
st.sidebar.write("we are a group in AI Engineers at Ducat")

st.write("\n")
st.write("##### Enter Review")
sample = st.text_input("")



if st.button("Predict"):
    pred = model.predict([sample])
    prob = model.predict_proba([sample])

    if pred[0] == 0:
        st.write("Negative👎")
        st.write(f"Confidence Score: {prob[0][0]:.2f}")
        st.balloons()
    else:
        st.write("Positive👍")
        st.write(f"Confidence Score: {prob[0][1]:.2f}")
        st.balloons()

st.write("#### Bulk Prediction")
file=st.file_uploader("select file",type=["csv","txt"])
if file:
    df=pd.read_csv(file,names=["Review"])
    placeholder=st.empty()
    placeholder.dataframe(df)
    if st.button("Predict",key="b2"):
        corpus=df.Review
        pred=model.predict(corpus)
        prob=np.max(model.predict_proba(corpus),axis=1)
        df['Sentiemnt']=pred
        df['Confidance']=prob
        df['Sentiemnt']=df['Sentiemnt'].map({0:'Neg 👎',1:'Pos 👍'})
        placeholder.dataframe(df)


