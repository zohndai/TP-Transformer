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


st.subheader('What is contaminant?')
    s = st.text_input('Please offer Chemical name, CAS number, or SMILES of contaminat', 'CCCCC')
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
		fp = morgan_fp(0, 3764)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3= st.columns([1,1,1])
		
		if "click1" not in st.session_state:
			st.session_state.click1=False 
		if col1.button('Get the prediction') or st.session_state.click1:
			##checking if it is within AD
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 3764)
			xx = []
			for j in range(len(train_OH)):
				ms2 = Chem.MolFromSmiles(train_OH['smiles'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 0, 3764)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.01:
				output = model_OH.predict(x_input)
				output= sc_oh.inverse_transform(output.reshape(-1,1))
				st.success(f'The predicted log second-rate constants toward OH radical is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()
		if "click2" not in st.session_state:
			st.session_state.click2=False 
		if col2.button('Get local interpretation') or st.session_state.click2:
			st.session_state['click2']=True
			#st.success(f'The predicted log second-rate constants toward OH radical is {st.session_state.output}')
			explainer = shap.TreeExplainer(model_OH)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col31, _= st.columns([1,1,1])
			col31.image('h.png')
			#st.session_state.shaplot = 'h.png'
		if "click3" not in st.session_state:
			st.session_state.click3=False 
		if col3.button('Get atom group') or st.session_state.click3:
			st.session_state['click3']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 0, 3764)
			#st.success(f'The predicted log second-rate constants toward OH radical is {st.session_state.output}')
			_, col31, _= st.columns([1,1,1])
			#col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 0, 3764, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = 
