import numpy as np
import pickle 
import streamlit as st 


load_model = pickle.load(open("C:/Users/rahul/Desktop/webapp/trained_model.sav", 'rb'))

def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshape = input_data_as_np_array.reshape(1,-1)
    prediction = load_model.predict(input_data_reshape)
    print(prediction)
    if (prediction[0]==0):
       return 'The person is not diabetic'
    else:
       return "The person is diabetic"
 
def main():
    st.title("Diabetes Prediction Web App")
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness= st.text_input("Skin Thickness Level")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Level")
    DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function Level")
    Age = st.text_input("Age of the Person")
    
    
    diagnosis = ""
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)
    
if __name__ == '__main__' :
    main()  
