import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Welcome to DP-Transformer",    
    page_icon="💧",        
    layout="wide",                
    initial_sidebar_state="auto"
)


st.subheader('Please specify the property or activity you want to predict')
select=st.selectbox('What property?', ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv'))
st.write('You selected:', select)
#select = st.radio("Please specify the property or activity you want to predict", ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv' ))
