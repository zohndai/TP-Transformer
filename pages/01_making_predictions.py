import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem import Draw
#import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Welcome to DP-Transformer",    
    page_icon="💧",        
    layout="wide",                
    initial_sidebar_state="auto"
)


st.subheader('Please select the ROSs that drive the pollutant degradation')
select=st.selectbox('What ROSs?', ('OH radical', 'SO4- radical', "Unkown"))
st.write('You selected:', select)
#select = st.radio("Please specify the property or activity you want to predict", ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv' ))
st.subheader('Please input the precursors of the ROSs')
st.text_input("Please offer the SMILES of precursors, e.g.'OO.[Fe+2]' for the fenton reagent H2O2/Fe2+ ", "OO.[Fe+2]")

st.subheader("Please select the method for extertal energy input for the ROSs generation", "UV")
st.selectbox("what method?",("UV", "Heat", "Visible light", "Micro wave", "Electricity", "Ultrasound", "Sunlight"))

st.subheader('Please input the reaction pH for pollutant degradation')
st.text_input("keep two decimal places","7.00")

st.subheader('Please input the SMILES of pollutants')
st.text_input("Please offer Chemical name, CAS number, or SMILES of the pollutant, e.g. 'c1ccccc1' for benzene", "c1ccccc1")



st.subheader('About the model')
if select=='OH radical':
	st.text('ML algorithm: CatBoost algorithm \nHyperparameters: \niterations=999 \nbagging_temperature=116.85 \ndepth=6, \nl2_leaf_reg=0.166, \nrandom_strength=43.40 \nCount-based Morgan finerprint: radius = 0, length = 3764')
if select=='Koc':
	st.text('ML algorithm: Ridge regression algorithm \nHyperparameters: \nalpha=9.619 \nCount-based Morgan finerprint: radius = 1, length = 4385') 
if select=='SO4- radical':
	st.text('ML algorithm: CatBoost algorithm \nHyperparameters: \niterations=645 \nbagging_temperature=20.21 \ndepth=5, \nl2_leaf_reg=0.078, \nrandom_strength=54.95 \nCount-based Morgan finerprint: radius = 0, length = 190') 
if select=='Solubility':
	st.text('ML algorithm: CatBoost algorithm \nHyperparameters: \niterations=948 \nbagging_temperature=88.88 \ndepth=6, \nl2_leaf_reg=0.097, \nrandom_strength=52.58 \nCount-based Morgan finerprint: radius = 0, length = 3878')
if select=='pKd':
	st.text('ML algorithm: CatBoost algorithm \nHyperparameters: \niterations=962 \nbagging_temperature=161.825 \ndepth=6, \nl2_leaf_reg=9.057, \nrandom_strength=189.01 \nCount-based Morgan finerprint: radius = 4, length = 3085') 
if select=='pIC50':
	st.text('ML algorithm: CatBoost algorithm \nHyperparameters: \niterations=929 \nbagging_temperature=121.744 \ndepth=6, \nl2_leaf_reg=0.051, \nrandom_strength=5.250 \nCount-based Morgan finerprint: radius = 2, length = 4840') 
if select=='CCSM_H':
	st.text('ML algorithm: Ridge regression algorithm \nHyperparameters: \nalpha=10.949 \nCount-based Morgan finerprint: radius = 1, length = 2843') 
if select=='CCSM_Na':
	st.text('ML algorithm: Ridge regression algorithm \nHyperparameters: \nalpha=11.012 \nCount-based Morgan finerprint: radius = 2, length = 4754') 
if select=='Lipo':
	st.text('ML algorithm: Ridge regression algorithm \nHyperparameters: \nalpha=14.811 \nCount-based Morgan finerprint: radius = 2, length = 4891') 
if select=='FreeSolv':
	st.text('ML algorithm: Ridge regression algorithm \nHyperparameters: \nalpha=3.097 \nCount-based Morgan finerprint: radius = 1, length = 3318') 
