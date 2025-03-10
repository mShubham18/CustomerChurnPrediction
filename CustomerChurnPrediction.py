import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import tensorflow as tf

label_encoder_gender_path = "models/label_encoder_gender.pkl"
one_hot_encoder_geography_path = "models/one_hot_encoder_geography.pkl"
standard_scaler_path = "models/std_scaler.pkl"
model_path = "models/model.h5"

# Loading the Training model
model = tf.keras.models.load_model(model_path)

with open(label_encoder_gender_path,"rb") as file:
    label_encoder_gender = pickle.load(file)

with open(one_hot_encoder_geography_path,"rb") as file:
    one_hot_encoder_geography = pickle.load(file)

with open(standard_scaler_path,"rb") as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")


#user input
CreditScore = st.number_input("Credit Score")
Geography = st.selectbox("Geography",one_hot_encoder_geography.categories_[0])
Gender = st.selectbox("Gender",label_encoder_gender.classes_)
Age = st.slider("Age",18,92)
Tenure = st.slider("Tenure",0,10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("Number of Products",1,4)
HasCrCard=st.selectbox("Has Credit Card",[0,1])
IsActiveMember = st.selectbox("Is Active Member",[0,1])
EstimatedSalary = st.number_input("Estimated Salary")

df = pd.DataFrame({
    "CreditScore":[CreditScore],
    "Gender":[label_encoder_gender.transform([Gender])[0]],
    "Age":[Age],
    "Tenure":[Tenure],
    "Balance":[Balance],
    "NumOfProducts":[NumOfProducts],
    "HasCrCard":[HasCrCard],
    "IsActiveMember":[IsActiveMember],
    "EstimatedSalary":[EstimatedSalary]
})
geo_encoded = one_hot_encoder_geography.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder_geography.get_feature_names_out(["Geography"]))

df = pd.concat([df.reset_index(drop=True),geo_encoded_df],axis=1)

# Scaling the data
df_scaled = scaler.transform(df)
#Predict the value

prediction = model.predict(df_scaled)
prediction_probability = prediction[0][0]
if st.button("Predict Churn"):
    st.write(f'Churn Probability: {prediction_probability:.2f}')

    if prediction_probability>0.5:
        st.write("The Customer is likely to churn")
    else:
        st.write('The Customer is unlikely to churn.')

if st.button("Reset"):
    st.experimental_rerun()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)