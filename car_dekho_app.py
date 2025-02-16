#import the required packages
import pandas as pd
import numpy as np 
import streamlit as st
from annotated_text import annotated_text
import pickle


#reading the dataframes
raw = pd.read_csv('car_details.csv')#raw datas for option settings
data = pd.read_csv('cleaned_details.csv')

data.dropna(inplace=True)

#UI starts here
#creating a fixed heading 
st.markdown(
    """
    <style>
    .fixed-header {
        max-height:250px;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #AB29F7; 
        color: white; 
        text-align: center;
        padding: 50px;
        padding-bottom: 20px;
        font-size: 20px;
        z-index: 9999;
        border-bottom: 2px solid #ccc;
        display: flex;
        align-items: center;
        justify-content: center;
         
    }
    
    .fixed-header img {
        height: 90px;
        width: 90px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#heading tag
st.markdown(
    '''
    <div class="fixed-header">
    <img src="https://purepng.com/public/uploads/large/purepng.com-lamborghini-huracan-front-view-carcarvehicletransportlamborghini-961524661745jdw6l.png" >
        <h1>Car Dheko - Used Car Price Prediction</h1>
    </div>
    ''', 
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
st.write(" ")
st.write(" ")
st.write('This app allows user to predict price of cars based on various features selected by the user!')

st.markdown(f"""
<div style="background-color: #AB29F7; padding-top: 5px; border-radius: 8px; display: inline-block;">
<h6>Use Filters Tab For Feature Selection!.</h6>
</div>""", unsafe_allow_html=True)

st.write(" ")



## sidebar and features

st.sidebar.header('Car Features')
with st.sidebar.expander('Filters and Options', expanded=True):
    loc_op = list(raw['location'].unique())     
    selected_location = st.selectbox("Location", options=loc_op)
    
    oem_op = list(raw['oem'].unique())    
    selected_oem = st.selectbox("Manufacturer", options=oem_op)

    selected_insurance = st.selectbox("Insurance Validity", options=['Third party insurance', 'Comprehensive', 'Zero Dep', 'Not Available'])
    
    body_options = list(raw['body_type'].unique())    
    selected_body_type = st.selectbox("Body Type", options=body_options)
    
    color_options = list(raw['Color'].unique())    
    selected_color = st.selectbox('Color', options=color_options)
    
    #fuel_options = list(raw['fuel_type'].unique())    
    #fuel_type = st.selectbox("Fuel Type", options=fuel_options)
    
    owner_op = list(raw['ownerNo'].unique())    
    owner_no_ = st.selectbox("Owner No", options=owner_op)
    
    gear_op = list(data['Gear Box'].unique())
    gear_ = st.selectbox("Gear Box", options=gear_op)
    
    steering_op = list(raw['Steering Type'].dropna().unique())
    selected_steering = st.selectbox("Steering Type", options=steering_op)

    selected_fuel = st.selectbox("Fuel Type", options=list(raw['fuel_type'].dropna().unique()))
    
    modelyear_ = st.number_input("Model Year", min_value=min(data['modelYear']), max_value=max(data['modelYear']))

    doorno = st.selectbox("No of Doors", options=data['No Door Numbers'].unique())
    
    cylinder_op = list(data['No of Cylinder'].unique())
    cylinders_ = st.selectbox("No of Cylinders", options=cylinder_op)
    
    kilomerters_ = st.slider("Kms Driven", min_value=(min(data['kilometers_driven'])), max_value=float(max(data['kilometers_driven'])), value=20000.0)
    
    length_ = st.slider("Length", min_value=(min(data['Length'])), max_value=(max(data['Length'])), value=3990.0)
    
    width_ = st.slider("Width", min_value=(min(data['Width'])), max_value=(max(data['Width'])), value=1750.0)
    
    height_ = st.slider("Height", min_value=(min(data['Height'])), max_value=(max(data['Height'])), value=1550.0)
    
    wheel_ = st.slider("Wheel Base", min_value=(min(data['Wheel Base'])), max_value=(max(data['Wheel Base'])), value=2500.0)
    
    kerb_ = st.slider("Kerb Weight", min_value=(min(data['Kerb Weight'])), max_value=(max(data['Kerb Weight'])), value=900.0)
    
    turning_ = st.slider("Turning Radius", min_value=min(data['Turning Radius']), max_value=max(data['Turning Radius']), value=5.3)
    
    cargo_ = st.slider('Cargo Volume', min_value=min(data['Cargo Volumn']), max_value=max(data['Cargo Volumn']), value=392.0)
    
    centralvar_ = st.slider('Central Varient Id', min_value=min(data['centralVariantId']), max_value=max(data['centralVariantId']), value=8654)

    selected_turbo = st.selectbox('Turbo Charger', options=['Yes', 'No'])

    drive_op = list(raw['Drive Type'].sort_values().unique())
    selected_drive = st.selectbox('Drive Type', options=drive_op)
    
    engine_options = list(raw['Engine Type'].unique())
    selected_engine = st.selectbox("Engine Type", options=engine_options)
    
    displacement_ = st.slider("Displacement", min_value=(min(data['Engine Displacement'])), max_value=(max(data['Engine Displacement'])), value=data['Engine Displacement'].mean())


    
    

## HANDLING INPUTS ## 

loc_unique = list(raw['location'].sort_values().unique())

loc_ = loc_unique.index(selected_location)


# manufacture type
oem_unique = list(raw['oem'].sort_values().unique())

oem_ = oem_unique.index(selected_oem)

## body type
body_type_unique = list(raw['body_type'].sort_values().unique())

body_type_ = body_type_unique.index(selected_body_type)

## Engine Type 

engine_type_unique = list(raw['Engine Type'].sort_values().unique())

engine_ = engine_type_unique.index(selected_engine)

## color 

color_unique = list(raw['Color'].sort_values().unique())
color_ = color_unique.index(selected_color)


## insurance Validity

insurance_unique = ['Comprehensive', 'Not Available', 'Third party insurance', 'Zero Dep']
insurance_ = insurance_unique.index(selected_insurance)

## turbo charger

if selected_turbo == 'Yes':
    turbo_ = 1
else:
    turbo_ = 0

## drive type 

drive_unique = list(raw['Drive Type'].sort_values().unique())
drive_type = drive_unique.index(selected_drive)

## Steering Type 
steering_unique = list(raw['Steering Type'].dropna().sort_values().unique())
steering_ = steering_unique.index(selected_steering)

## fuel Type 

fuel_type_unique = list(raw['fuel_type'].dropna().sort_values().unique())
fuel_type = fuel_type_unique.index(selected_fuel)

st.markdown("""<i>Your Filtered Features are Here:</i>""", unsafe_allow_html=True)

## ANNOTATE FILTERS
annotated_text(("Location: ", selected_location), "  ", ('Manufacturer: ', selected_oem), "  ", ('Body Type: ', selected_body_type), " ", ('Owner No: ', str(owner_no_)), " ", ('Model Year: ', str(modelyear_)), " ", ('No of Cylinders: ', str(cylinders_)), " ", ('Kms Driven: ', str(kilomerters_)), " ", ('Width: ', str(width_)), " ", ('Length: ', str(length_)), " ", ('Height: ', str(height_)), " ", ('Wheel Base: ', str(wheel_)), " ", ('Kerb Weight: ', str(kerb_)))

## Features selected

features = [[0,kilomerters_,owner_no_,modelyear_,centralvar_,insurance_,displacement_,color_,engine_,cylinders_,turbo_,length_,width_,height_,wheel_,kerb_,gear_,drive_type,steering_, doorno, cargo_, turning_,fuel_type, body_type_,oem_,loc_,]]

## LOADING MODEL

with open('model.pkl' , 'rb') as file:
    model = pickle.load(file)

result = round(model.predict(features)[0], 2)

st.markdown("---", unsafe_allow_html=True)


st.markdown(f"""
<div style="background-color: #AB29F7; padding-top: 5px; border-radius: 8px; display: inline-block;">
<h6>Approximate Car Price Prediction based on your Feature Selection: </h6>
</div>""", unsafe_allow_html=True)


st.markdown(f'''<h2 align="center">₹ {result} Lakhs</h2>''', unsafe_allow_html=True)
st.markdown(f'<h6>Note: The Prediction is <b>66%</b> accurate</h6>', unsafe_allow_html=True)

st.markdown("""---""", unsafe_allow_html=True)