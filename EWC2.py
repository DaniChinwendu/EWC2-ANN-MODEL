import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import streamlit as st 
#collect the model object file
filename ='EWC_2.pkl' 
model = pickle.load(open(filename,'rb'))

def welcome():
    return "Welcome All"
def inverse_transform(y_pred, min_max_values):
  min_value = min_max_values[0]
  max_value = min_max_values[1]
  return (y_pred * (max_value - min_value)) + min_value

def prediction_LOGEC3(log_DPRA_mean, log_hCLAT_MIT,log_KSIma, scaler, model):
    # Scale the input
    scaled_input = scaler.transform([[log_DPRA_mean, log_hCLAT_MIT,log_KSIma]])
    prediction = model.predict(scaled_input)
    return prediction

def main():
    st.title("'EDELWEISS CONNECT ITS SKIN SENSITIZATION SOLUTION'")
    st.markdown('An Artificial Neural Network Regression model Utilizing invitro and inchemo(h-CLAT,DPRA,KSIma) Descriptors for predicting skin Sensitization')
    html_temp = """
    EWC_1 SKIN SENSITIZATION PREDICTION App 
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    log_DPRA_mean = st.number_input("DPRA",min_value=None, max_value=None, value=0.0, step=None)
    log_hCLAT_MIT = st.number_input("hCLAT",min_value=None, max_value=None, value=0.0, step=None)
    log_KSIma=st.number_input("KSIma",min_value=None, max_value=None, value=0.0, step=None)
    
    prediction_type = st.selectbox("Select prediction type:", ["Two-class", "Three-class"])
    if st.button("Predict"):
    # Scale the input
      scaler = MinMaxScaler()
      scaler.fit([[log_DPRA_mean, log_hCLAT_MIT,log_KSIma]])
    # Call the prediction function
      result = prediction_LOGEC3(log_DPRA_mean, log_hCLAT_MIT,log_KSIma, scaler, model)
    # Convert the prediction back to the original scale
      min_max_values=(0,1)
      result = inverse_transform(result,min_max_values)#scaler.inverse_transform(result,min_max_values)
      #result=result.reshape(1,1)
      if result is not None:
        if prediction_type == "Three-class":
            if float(result) < (-1):
                result = 'Strong'
            elif float(result) >= (-1) and float(result) < 0:
                result = 'Strong'
            elif float(result) >= 0 and float(result) < 1:
                result = 'Moderate'
            elif float(result) >= 1:
                result = 'Moderate'
            else:
                result = 'Non'
        else:
            if float(result) < (-1):
                result = 'Positive'
            elif float(result) >= (-1) and float(result) < 0:
                result = 'Positive'
            elif float(result) >= 0 and float(result) < 1:
                result = 'Positive'
            elif float(result) >= 1:
                result = 'Positive'
            else:
                result = 'Negative'
        st.success(f'The chemical Potency is {result}')
    else:
        st.warning("Prediction failed, please check your inputs and try again.")
    
if __name__=='__main__':
    main()
