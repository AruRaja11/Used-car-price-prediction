#import the required packages
import pandas as pd
import numpy as np 
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import streamlit as st
from annotated_text import annotated_text
import training_

## INITIALIZING VARIABLES ##
owner_no_ = 1
modelyear_ = 1999
loc_ = 0
seats_ = 5
kilomerters_ = 20000
displacement_ = 1420
cylinders_ = 3
length_ = 3990
width_ = 1750
height_ = 1750
wheel_ = 2500
kerb_ = 900
gear_ = 7
door_no_ = 5
speed_ = 100
Convertibles, Coupe, Hatchback, Hybrids, MUV, Minivans,Pickup_Trucks, SUV, Sedan, Wagon = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
location = 1
transmission_ = 0
fuel_type_Cng, fuel_type_Diesel, fuel_type_Electric, fuel_type_Lpg, fuel_type_Petrol = 0, 0, 0, 0, 0

Audi, BMW, Chevrolet, Citroen, Datsun, Fiat, Ford, Hindustan_Motors, Honda, Hyundai, Isuzu, Jaguar, Jeep, Kia, Land_Rover, Lexus, MG, Mahindra, Mahindra_Renault, Mahindra_Ssangyong, Maruti, Mercedes_Benz, Mini, Mitsubishi, Nissan, Opel, Porsche, Renault, Skoda, Tata, Toyota, Volkswagen, Volvo = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
result = "Feeding Inputs.."

#reading the dataframes
raw = pd.read_csv('car_details.csv')#raw datas for option settings
data = pd.read_csv('cleaned_details.csv')

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
    location = st.selectbox("Location", options=loc_op)
    oem_op = list(raw['oem'].unique())
    oem = st.selectbox("Manufacturer", options=oem_op)
    body_options = list(raw['body_type'].unique())
    body_type = st.selectbox("Body Type", options=body_options)
    fuel_options = list(raw['fuel_type'].unique())
    fuel_type = st.selectbox("Fuel Type", options=fuel_options)
    owner_op = list(raw['ownerNo'].unique())
    owner_no_ = st.selectbox("Owner No", options=owner_op)
    gear_op = list(data['Gear Box'].unique())
    gear_ = st.selectbox("Gear Box", options=gear_op)
    transmis_op = list(data['Transmission'].unique())
    transmission_ = st.selectbox("Transmission ", options=transmis_op)
    seat_op = list(data['Seats'].unique())
    seats_ = st.selectbox("No of Seats", options=seat_op)
    door_op = list(data['No Door Numbers'].unique())
    door_no_ = st.selectbox("No of Doors", options=door_op)

    modelyear_ = st.number_input("Model Year", min_value=min(data['modelYear']), max_value=max(data['modelYear']))
    cylinder_op = list(data['No of Cylinder'].unique())
    cylinders_ = st.selectbox("No of Cylinders", options=cylinder_op)
    speed_ = st.slider("Top Speed in Kms", min_value=(min(data['Top Speed'])), max_value=(max(data['Top Speed'])), value=100.0)
    kilomerters_ = st.slider("Kms Driven", min_value=(min(data['Kms Driven'])), max_value=(max(data['Kms Driven'])), value=20000)
    length_ = st.slider("Length", min_value=(min(data['Length'])), max_value=(max(data['Length'])), value=3990.0)
    width_ = st.slider("Width", min_value=(min(data['Width'])), max_value=(max(data['Width'])), value=1750.0)
    height_ = st.slider("Height", min_value=(min(data['Height'])), max_value=(max(data['Height'])), value=1550.0)
    wheel_ = st.slider("Wheel Base", min_value=(min(data['Wheel Base'])), max_value=(max(data['Wheel Base'])), value=2500.0)
    kerb_ = st.slider("Kerb Weight", min_value=(min(data['Kerb Weight'])), max_value=(max(data['Kerb Weight'])), value=900.0)
    displacement_ = st.slider("Displacement", min_value=(min(data['Displacement'])), max_value=(max(data['Displacement'])), value=data['Displacement'].mean())


    
    

## HANDLING INPUTS ## 

if location == 'chennai':
    loc_ = 1
elif location == 'bangalore':
    loc_ = 0
elif location == 'delhi':
    loc_ = 2
elif location == 'hydrabad':
    loc_ = 3
elif location == 'jaipur':
    loc_ = 4
else: 
    loc_ = 5

# fuel type
if fuel_type == "Petrol":
    fuel_type_Petrol = 1
elif fuel_type == "Diesel":
    fuel_type_Diesel = 1
elif fuel_type == "Electric":
    fuel_type_Electric = 1
elif fuel_type == "Cng":
    fuel_type_Cng = 1
else: 
    fuel_type_Lpg = 1

# manufacture type
if oem == "Audi":
    Audi = 1
elif oem == "BMW":
    BMW = 1
elif oem == "Chevrolet":
    Chevrolet = 1
elif oem == "Citroen":
    Citroen = 1
elif oem == "Datsun":
    Datsun = 1
elif oem == "Fiat":
    Fiat = 1
elif oem == "Ford":
    Ford = 1
elif oem == "Hindustan Motors":
    Hindustan_Motors = 1
elif oem == "Honda":
    Honda = 1
elif oem == "Hyundai":
    Hyundai = 1
elif oem == "Isuzu":
    Isuzu = 1
elif oem == "Jaguar":
    Jaguar = 1
elif oem == "Jeep":
    Jeep = 1
elif oem == "Kia":
    Kia = 1
elif oem == "Land Rover":
    Land_Rover = 1
elif oem == "Lexus":
    Lexus = 1
elif oem == "MG":
    MG = 1
elif oem == "Mahindra":
    Mahindra = 1
elif oem == "Mahindra Renault":
    Mahindra_Renault = 1
elif oem == "Mahindra Ssangyong":
    Mahindra_Ssangyong = 1
elif oem == "Maruti":
    Maruti = 1
elif oem == "Mercedes Benz":
    Mercedes_Benz = 1
elif oem == "Mini":
    Mini = 1
elif oem == "Mitsubishi":
    Mitsubishi = 1
elif oem == "Maruti":
    Maruti = 1
elif oem == "Nissan":
    Nissan = 1
elif oem == "Opel":
    Opel = 1
elif oem == "Porsche":
    Porsche = 1
elif oem == "Renault":
    Renault = 1
elif oem == "Skoda":
    Skoda = 1
elif oem == "Tata":
    Tata = 1
elif oem == "Toyota":
    Toyota = 1
elif oem == "Volkswagen":
    Volkswagen = 1
else: 
    Volvo = 1

## body type ##Convertibles, Coupe, Hatchback, Hybrids, MUV, Minivans,Pickup_Trucks, SUV, Sedan, Wagon
if body_type == "Convertibles":
    Convertibles = 1
elif body_type == "Coupe":
    Coupe = 1
elif body_type == "Hatchback":
    Hatchback = 1
elif body_type == "Hybrids":
    Hybrids = 1
elif body_type == "MUV":
    MUV = 1
elif body_type == "Minivans":
    Minivans = 1
elif body_type == "Pickup Trucks":
    Pickup_Trucks = 1
elif body_type == "SUV":
    SUV = 1
elif body_type == "Sedan":
    Sedan = 1
else: 
    Wagon = 1
    
st.markdown("""<i>Your Filtered Features are Here:</i>""", unsafe_allow_html=True)

## ANNOTATE FILTERS
annotated_text(("Location: ", location), "  ", ('Manufacturer: ', oem), "  ", ('Body Type: ', body_type), " ", ('Owner No: ', str(owner_no_)), " ", ('Seats: ', str(seats_)), " ", ('No of Doors: ', str(door_no_)), " ", ('Model Year: ', str(modelyear_)), " ", ('Top Speed', str(speed_)), " ", ('No of Cylinders: ', str(cylinders_)), " ", ('Kms Driven: ', str(kilomerters_)), " ", ('Width: ', str(width_)), " ", ('Length: ', str(length_)), " ", ('Height: ', str(height_)), " ", ('Wheel Base: ', str(wheel_)), " ", ('Kerb Weight: ', str(kerb_)))

#adding button
if st.button("Apply Filter"):
    features = [[owner_no_, modelyear_, seats_, kilomerters_, displacement_ , cylinders_, length_, width_, height_, wheel_, kerb_, gear_, door_no_, speed_, Convertibles, Coupe, Hatchback, Hybrids, MUV, Minivans,Pickup_Trucks, SUV, Sedan, Wagon, loc_, transmission_, fuel_type_Cng, fuel_type_Diesel, fuel_type_Electric, fuel_type_Lpg, fuel_type_Petrol, Audi, BMW, Chevrolet, Citroen, Datsun, Fiat, Ford, Hindustan_Motors, Honda, Hyundai, Isuzu, Jaguar, Jeep, Kia, Land_Rover, Lexus, MG, Mahindra, Mahindra_Renault, Mahindra_Ssangyong, Maruti, Mercedes_Benz, Mini, Mitsubishi, Nissan, Opel, Porsche, Renault, Skoda, Tata, Toyota, Volkswagen, Volvo]]
    
    predicted = training_.predict_price(features)[0] # predict function from training file
    result = str(round(predicted, 2))
else:
    pass
    

st.markdown("---", unsafe_allow_html=True)


st.markdown(f"""
<div style="background-color: #AB29F7; padding-top: 5px; border-radius: 8px; display: inline-block;">
<h6>Approximate Car Price Prediction based on your Feature Selection: </h6>
</div>""", unsafe_allow_html=True)


acc = training_.get_acc() #getting accuracy from training py file
if result == 'Feeding Inputs..':
    st.markdown(f'''<h2 align="center">{result}</h2>''', unsafe_allow_html=True)
    st.markdown(f'''<p align="center">Click Apply Filters to see the results!.</p>''', unsafe_allow_html=True)
    st.markdown(f'<h6>Note: The Prediction is <b>{round(acc*100, 2)}%</b> accuracy</h6>', unsafe_allow_html=True)
else:
    st.markdown(f'''<h2 align="center">₹ {result} Lakhs</h2>''', unsafe_allow_html=True)
    st.markdown(f'<h6>Note: The Prediction is <b>{round(acc*100, 2)}%</b> accuracy</h6>', unsafe_allow_html=True)

st.markdown("""---""", unsafe_allow_html=True)