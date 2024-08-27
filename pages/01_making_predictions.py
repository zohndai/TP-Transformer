import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import cirpy
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


st.subheader('What is contaminant?')
s = st.text_input("Please offer Chemical name, CAS number, or SMILES of the pollutant, e.g. 'c1ccccc1' for benzene", "c1ccccc1")

if s =='':
	st.warning('You should at least provide one chemical')
	st.stop()
smile = cirpy.resolve(s, 'smiles')

if smile is None:
	st.warning('Invalid chemical name or CAS number, please recheck it again or you can directly type the SMILES')
	st.stop()

with st.expander("Show how to get SMILES of chemicals"):
	st.write('You can get SMILES of any molecules from PubChem https://pubchem.ncbi.nlm.nih.gov/ by typing Chemical name or ACS number')



if select=='OH radical':
	ros_mis = '[OH]'
	col1, col2, col3= st.columns([1,1,1])
	
	if "click1" not in st.session_state:
		st.session_state.click1=False 
	if col1.button('Get the prediction') or st.session_state.click1:
		##checking if it is within AD
		ms1 = Chem.MolFromSmiles(smile)
		fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 3764)
		xx = []
	if "click2" not in st.session_state:
		st.session_state.click2=False 
