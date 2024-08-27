import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import cirpy
import torch
import models
from models import gdown_model.download
#from rdkit.Chem import Draw
#import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Welcome to DP-Transformer",    
    page_icon="log.ico",        
    layout="wide",                
    initial_sidebar_state="auto"
)

ros_name = ["HO∙", "SO₄∙⁻","O₃", "¹O₂",  "Fe(VI)", "O₂∙⁻", "MnO₄⁻", "ClO⁻","HClO", "Cl₂","Cl∙","CO₃∙⁻","Cl₂∙⁻","C₂H₃O₃∙", \
             "Cu(III)","Fe(V)",  "NO₂∙", "Mn(V)", "HSO₄∙", "O₂", "BrO⁻","NO∙", "ClO∙","Fe(IV)","Br∙", "IO⁻","C₂H₃O₂∙",\
             "HSO₅⁻", "ClO₂∙", "Br₂","HOBr","HO₂⁻","I∙", "NO₃∙", "IO₃∙⁻", \
           "Fe(III)", "S₂O₈∙⁻","HCO₃∙", "SO₃∙⁻","Unkown"]
ros_smi = ['[OH]', '[O]S(=O)(=O)[O-]', 'O=[O+][O-]','OO1', 'O=[Fe](=O)([O-])[O-]', '[O][O-]', 'O=[Mn](=O)(=O)[O-]','[O-]Cl','OCl', 'ClCl', '[Cl]', '[O]C(=O)[O-]','Cl[Cl-]', 'CC(=O)O[O]',\
	    '[Cu+3]','O=[Fe]([O-])([O-])[O-]', '[O]N=O','[O-][Mn]([O-])([O-])=O', '[O]S(=O)(=O)O', 'O=O', '[O-]Br','[N]=O', '[O]Cl','[O-][Fe]([O-])([O-])[O-]','[Br]','[O-]I','CC([O])=O',\
	    'O=S(=O)([O-])OO', '[O][Cl+][O-]','BrBr', 'OBr', '[O-]O', '[I]', '[O][N+](=O)[O-]', '[O-][I+2]([O-])[O-]',\
	   '[Fe+3]', '[O]S(=O)(=O)OOS(=O)(=O)[O-]','[O]C(=O)O', '[O]S(=O)[O-]', '']

acti_methd=["UV", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight", "No"]
methd_token=["UV", "heat", "VL", "MW", "E", "US", "SL", ""]

st.subheader('Please select the ROSs that drive the pollutant degradation')
ros_selct=st.selectbox('What ROSs?', ( "HO∙", "SO₄∙⁻","O₃", "¹O₂",  "Fe(VI)", "O₂∙⁻", "MnO₄⁻", "ClO⁻","HClO", "Cl₂","Cl∙","CO₃∙⁻","Cl₂∙⁻","C₂H₃O₃∙", \
             "Cu(III)","Fe(V)",  "NO₂∙", "Mn(V)", "HSO₄∙", "O₂", "BrO⁻","NO∙", "ClO∙","Fe(IV)","Br∙", "IO⁻","C₂H₃O₂∙",\
             "HSO₅⁻", "ClO₂∙", "Br₂","HOBr","HO₂⁻","I∙", "NO₃∙", "IO₃∙⁻", \
           "Fe(III)", "S₂O₈∙⁻","HCO₃∙", "SO₃∙⁻", "Unkown"))
#st.write('You selected:', ros_selct)
#select = st.radio("Please specify the property or activity you want to predict", ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv' ))
st.subheader('Please input the precursors of the ROSs')
st.text_input("Please offer the SMILES of precursors, e.g.'OO.[Fe+2]' for the fenton reagent H2O2/Fe2+ ", "OO.[Fe+2]")

st.subheader("Please select the method for extertal energy input for the ROSs generation", "UV")
methd_selct=st.selectbox("what method?",("UV", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight"))

st.subheader('Please input the reaction pH for pollutant degradation')
st.text_input("Keep two decimal places","7.00")


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

def run():
	model_path = download()
