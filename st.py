import streamlit as st
import numpy as np
import joblib
import time

# Load the model
loaded_model = joblib.load(open('bigmart_model.pkl', 'rb'))

# Prediction function
def sales_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    st.title('Sales Prediction Web Application (Streamlit)')

    # Get user inputs and convert to numeric types
    try:
        Item_Weight = float(st.text_input('Item Weight(ex:0.6)'))
        Item_Fat_Content = float(st.text_input('Item_Fat_Content(ex:1)'))
        Item_Visibility = float(st.text_input('Item_Visibility(ex:0.8)'))
        Item_Type = float(st.text_input('Item_Type(ex:2)'))
        Item_MRP = float(st.text_input('Item_MRP(ex:3)'))
        Outlet_Size = float(st.text_input('Outlet_Size(ex:5)'))
        Outlet_Location_Type = float(st.text_input('Outlet_Location_Type(ex:25)'))
        Outlet_Type = float(st.text_input('Outlet_Type(ex:5)'))
    except ValueError:
        st.write("")

    if st.button('Click here..'):
        with st.spinner('Please wait...'):
            time.sleep(2)
            diagnosis = sales_prediction([Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP,Outlet_Size, Outlet_Location_Type, Outlet_Type])
            st.success(f"Predicted Sales: {diagnosis}")

if __name__ == '__main__':
    main()