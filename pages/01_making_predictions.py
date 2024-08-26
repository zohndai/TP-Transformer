import streamlit as st
import pandas as pd
import numpy as np
#import dill
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from catboost import CatBoostRegressor
import cirpy
import shap
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

@st.cache_data(ttl=24*3600)
def load_data(file_name):
	file_type= str(file_name).split('.')[1]
	if file_type=='xlsx':
		data = pd.read_excel(file_name)
	elif file_type=='csv':
		data = pd.read_csv(file_name)
	else:
		st.warning('File format is not supported. Please upload the CSV or Excel file')
		st.stop()
	return data
@st.cache_data(ttl=24*3600)
def load_model(file_name):
    model = dill.load(open(file_name, 'rb'))
    return model
@st.cache_data(ttl=24*3600)
def load_sc(file_name):
    sc = dill.load(open(file_name, 'rb'))
    return sc

@st.cache_data(ttl=24*3600)
def convert_df(df):
     return df.to_csv().encode('utf-8')
@st.cache_data(ttl=24*3600)
def return_s(s):
  return cirpy.resolve(str(s), 'smiles')

class morgan_fp:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetHashedMorganFingerprint(mol, self.radius, self.length)
        npfp = np.array(list(fp)).astype('float32')
        return npfp

training_data_OH = load_data('data/OH.xlsx')
training_data_SO4 = load_data('data/SO4.xlsx')
training_data_KOC = load_data('data/KOC.xlsx')
training_data_SOL= load_data('data/SOL.xlsx')
training_data_pkd = load_data('data/pkd.xlsx')
training_data_pic50 = load_data('data/IC50.xlsx')
training_data_ccsmh = load_data('data/MH_1425.csv')
training_data_ccsmna= load_data('data/MNa_720.csv')
training_data_lipo = load_data('data/Lipophilicity.csv')
training_data_freesolv = load_data('data/freesolv.csv')

train_OH = load_data('data/OH_train_2.csv')
train_SO4 = load_data('data/SO4_train_2.csv')
train_KOC = load_data('data/koc_train_2.csv')
train_SOL= load_data('data/Sol_train_2.csv')
train_pkd = load_data('data/pka_train_2.csv')
train_pic50 = load_data('data/pic50_train_2.csv')
train_ccsmh = load_data('data/ccsmh_train_2.csv')
train_ccsmna= load_data('data/ccsmna_train_2.csv')
train_lipo = load_data('data/lipo_train_2.csv')
train_freesolv= load_data('data/freesolv_train_2.csv')

model_OH = load_model('model/OH_model.model')
sc_oh = load_sc('data/OH_sc.sc')
model_SO4 = load_model('model/SO4_model.model')
sc_so4 = load_sc('data/SO4_sc.sc')
model_KOC = load_model('model/KOC_model.model')
model_SOL= load_model('model/SOL_model.model')
model_pkd = load_model('model/pkd.model')
model_pic50= load_model('model/pIC50_model.model')
model_ccsmh = load_model('model/ccsmh_model.model')
model_ccsmna= load_model('model/ccsmna_model.model')
model_lipo = load_model('model/lipo_model.model')
model_freesolv= load_model('model/freesolv_model.model')

	
st.subheader('Raw training data')
if st.checkbox('Show raw training data'):
	st.session_state['select']=False
	if select=='OH radical':
		st.write(training_data_OH)
		st.text(f'OH radical dataset containin {len(training_data_OH)} samples' )
	if select=='SO4- radical':
		st.write(training_data_SO4)
		st.text(f'SO4 radical dataset containin {len(training_data_SO4)} samples' )
	if select=='Koc':
		st.write(training_data_KOC)
		st.text(f'KOC dataset containin {len(training_data_KOC)} samples' )
	if select=='Solubility':
		st.write(training_data_SOL)
		st.text(f'SOL dataset containin {len(training_data_SOL)} samples' )
	if select=='pKd':
		st.write(training_data_pkd)
		st.text(f'pKd dataset containin {len(training_data_pkd)} samples' )
	if select=='pIC50':
		st.write(training_data_pic50)
		st.text(f'pIC50 dataset containin {len(training_data_pic50)} samples' )
	if select=='CCSM_H':
		st.write(training_data_ccsmh)
		st.text(f'CCS_MH dataset containin {len(training_data_ccsmh)} samples' )
	if select=='CCSM_Na':
		st.write(training_data_ccsmna)
		st.text(f'CCS_MNa dataset containin {len(training_data_ccsmna)} samples' )
	if select=='Lipo':
		st.write(training_data_lipo)
		st.text(f'Lipo dataset containin {len(training_data_lipo)} samples' )
	if select=='FreeSolv':
		st.write(training_data_freesolv)
		st.text(f'FreeSolv dataset containin {len(training_data_freesolv)} samples' )
		
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
def run():
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
			
	if select=='Koc':
		fp = morgan_fp(1, 4385)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3= st.columns([1,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 1, 4385)
			xx = []
			for j in range(len(train_KOC)):
				ms2 = Chem.MolFromSmiles(train_KOC['new_smile'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 1, 4385)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.035:
				output = model_KOC.predict(x_input)
				st.success(f'The predicted Koc is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted Koc is {st.session_state.output}')
                        
			train_KOC['fp'] = train_KOC['new_smile'].apply(fp)
			x_fp = np.array(list(train_KOC['fp']))
			explainer = shap.LinearExplainer(model_KOC, x_fp)
                        
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,1,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 1, 4385)
			st.success(f'The predicted Koc is {st.session_state.output}')
			_, col31, _= st.columns([1,1,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 1, 4385, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
			
			
	if select=='SO4- radical':
		fp = morgan_fp(0, 190)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3= st.columns([1,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 190)
			xx = []
			for j in range(len(train_SO4)):
				ms2 = Chem.MolFromSmiles(train_SO4['smiles'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 0, 190)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.04:
				output = model_SO4.predict(x_input)
				output= sc_so4.inverse_transform(output.reshape(-1,1))
				st.success(f'The predicted log second-rate constants toward SO4- radical is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted second-rate constants toward SO4- radical is {st.session_state.output}')
			explainer = shap.TreeExplainer(model_SO4)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,1,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 0, 190)
			st.success(f'The predicted log second-rate constants toward SO4- radical is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 0, 190, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
			
	if select=='Solubility':
		fp = morgan_fp(0, 3878)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 3878)
			xx = []
			for j in range(len(train_SOL)):
				ms2 = Chem.MolFromSmiles(train_SOL['smile'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 0, 3878)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.06:
				output = model_SOL.predict(x_input)
				st.success(f'The predicted Solubility is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted Solubility is {st.session_state.output}')
			explainer = shap.TreeExplainer(model_SOL)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 0, 3878)
			st.success(f'The predicted Solubility is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 0, 3878, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
			
	if select=='pKd':
		fp = morgan_fp(4, 3085)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 4, 3085)
			xx = []
			for j in range(len(train_pkd)):
				ms2 = Chem.MolFromSmiles(train_pkd['Neutralized SMILES code'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 4, 3085)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.176:
				output = model_pkd.predict(x_input)
				st.success(f'The predicted pKd is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted pKd is {st.session_state.output}')
			explainer = shap.TreeExplainer(model_pkd)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 4, 3085)
			st.success(f'The predicted pKd is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 4, 3085, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
					
	if select=='pIC50':
		fp = morgan_fp(2, 4840)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 2, 4840)
			xx = []
			for j in range(len(train_pic50)):
				ms2 = Chem.MolFromSmiles(train_pic50['Neutralized SMILES code'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 2, 4840)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.133:
				output = model_pic50.predict(x_input)
				st.success(f'The predicted pIC50 is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted pIC50 is {st.session_state.output}')
			explainer = shap.TreeExplainer(model_pic50)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 2, 4840)
			st.success(f'The predicted pIC50 is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 2, 4840, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
					
			
	if select=='CCSM_H':
		fp = morgan_fp(1, 2843)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 1, 2843)
			xx = []
			for j in range(len(train_ccsmh)):
				ms2 = Chem.MolFromSmiles(train_ccsmh['CanonicalSMILES'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 1, 2843)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.093:
				output = model_ccsmh.predict(x_input)
				st.success(f'The predicted CCSM_H is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted CCSM_H is {st.session_state.output}')
			
			train_ccsmh['fp'] = train_ccsmh['CanonicalSMILES'].apply(fp)
			x_fp = np.array(list(train_ccsmh['fp']))
			explainer = shap.LinearExplainer(model_ccsmh, x_fp)
			
			#explainer = shap.TreeExplainer(model_ccsmh)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 1, 2843)
			st.success(f'The predicted CCSM_H is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 1, 2843, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
					
	if select=='CCSM_Na':
		fp = morgan_fp(2, 4754)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 2, 4754)
			xx = []
			for j in range(len(train_ccsmna)):
				ms2 = Chem.MolFromSmiles(train_ccsmna['CanonicalSMILES'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 2, 4754)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.08:
				output = model_ccsmna.predict(x_input)
				st.success(f'The predicted CCSM_Na is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted CCSM_Na is {st.session_state.output}')
			
			train_ccsmna['fp'] = train_ccsmna['CanonicalSMILES'].apply(fp)
			x_fp = np.array(list(train_ccsmna['fp']))
			explainer = shap.LinearExplainer(model_ccsmna, x_fp)
			
			#explainer = shap.TreeExplainer(model_ccsmna)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 2, 4754)
			st.success(f'The predicted CCSM_Na is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 2, 4754, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
					
	if select=='Lipo':
		fp = morgan_fp(2, 4891)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 2, 4891)
			xx = []
			for j in range(len(train_lipo)):
				ms2 = Chem.MolFromSmiles(train_lipo['smile'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 2, 4891)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.139:
				output = model_lipo.predict(x_input)
				st.success(f'The predicted Lipo is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted Lipo is {st.session_state.output}')
			explainer = shap.TreeExplainer(model_lipo)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 2, 4891)
			st.success(f'The predicted Lipo is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 2, 4891, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
					
	if select=='FreeSolv':
		fp = morgan_fp(1, 3318)
		x_input =fp(smile).reshape(1, -1)
		col1, col2, col3, col4= st.columns([2,2,1,1])
		if col1.button('Get the prediction'):
			ms1 = Chem.MolFromSmiles(smile)
			fp_test= AllChem.GetHashedMorganFingerprint(ms1, 1, 3318)
			xx = []
			for j in range(len(train_freesolv)):
				ms2 = Chem.MolFromSmiles(train_freesolv['smiles'][j])
				fp2= AllChem.GetHashedMorganFingerprint(ms2, 1, 3318)
				s= DataStructs.DiceSimilarity(fp_test, fp2)
				xx.append(s)
			if np.mean(xx)>0.031:
				output = model_freesolv.predict(x_input)
				st.success(f'The predicted Solubility is {output[0]}')
				st.session_state.output=output[0]
			else:
				st.warning("The query chemical is outside the model's AD, please try aonther one")
				st.stop()

		if col2.button('Get local interpretation'):
			st.success(f'The predicted FreeSolv is {st.session_state.output}')
			
			train_freesolv['fp'] = train_freesolv['smiles'].apply(fp)
			x_fp = np.array(list(train_freesolv['fp']))
			explainer = shap.LinearExplainer(model_freesolv, x_fp)
			
			#explainer = shap.TreeExplainer(model_freesolv)
			shap_value_train = explainer(x_input)
			plot = shap.plots.waterfall(shap_value_train[0], show=False)
			plt.xticks(fontsize= 18)
			plt.yticks( fontsize = 18)
			plt.tight_layout()
			plt.savefig('h.png')
			_, col3, _= st.columns([1,2,1])
			col3.image('h.png')
			st.session_state.shaplot = 'h.png'
		if "click" not in st.session_state:
			st.session_state.click=False 
		if col3.button('Get atom group') or st.session_state.click:
			st.session_state['click']=True
			ms1 = Chem.MolFromSmiles(smile)
			fp1= AllChem.GetHashedMorganFingerprint(ms1, 1, 3318)
			st.success(f'The predicted FreeSolv is {st.session_state.output}')
			_, col31, _= st.columns([1,2,1])
			col31.image(st.session_state.shaplot)
			st.success(f'The feature # containing atom groups are {list(fp1.GetNonzeroElements().keys())}')
			st.session_state.sf =st.selectbox('Which Feature?', list(fp1.GetNonzeroElements().keys()))
			bi = {}
			fp_env = AllChem.GetMorganFingerprintAsBitVect(ms1, 1, 3318, bitInfo=bi)
			y=Draw.DrawMorganBit(ms1, int(st.session_state.sf), bitInfo= bi)
			_, col41, _= st.columns([1,2,1])
			col41.image(y)
			#st.session_state.atomgroup = y
					
			
	st.subheader('If you want to make predicitons for many contaminants')
	upload_file = st.file_uploader("Choose a csv or excel file containing molecules")
	sample_data = load_data('data/example.xlsx')
	if st.checkbox('Show an example of csv or excel file'):
		st.write(sample_data)
	if upload_file is not None:
		file_type=str(upload_file.name).split('.')[1]
		if file_type=='xlsx':
			up_data = pd.read_excel(upload_file)
		elif file_type=='csv':
			up_data = pd.read_csv(upload_file)
		else:
			st.warning('File format is not supported. Please upload the CSV or Excel file')
			st.stop()
		with st.expander("Show your data"):
			st.write(up_data)
	if st.button('Get the predictions'):
		if upload_file is None:
			st.warning('You should first upload your files')
			st.stop()
		else:
			st.write('Preparing SMILES')
			up_data['smile']=up_data[up_data.columns[0]].apply(return_s)
			st.write('Finished preparing SMILES')
			#st.write(up_data)
			non_smile = up_data[up_data['smile'].isnull()==True]
			
			if len(non_smile)>0:
				st.warning(f'You uploaded {len(up_data)} chemicals, in which {len(non_smile)} of them are failed to obtain their SMILES. They are chemicals with index of {list(non_smile.index)}')
				up_data.drop(index = non_smile.index, inplace=True)
				
			fp = morgan_fp(0, 100)
			ind=[]
			for i in up_data.index:
				try:
					fp(up_data['smile'][i])
				except:
					ind.append(i)
					
			if len(ind)>0:
				st.warning(f'You uploaded {len(up_data)} chemicals, in which {len(ind)} of them are failed to convert to C-MF. They are chemicals with index of {ind}')
				up_data.drop(index = ind, inplace=True)
			
			if select=='OH radical':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 3764)
					xx = []
					for j in range(len(train_OH)):
						ms2 = Chem.MolFromSmiles(train_OH['smiles'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 0, 3764)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.01])} chemicals outside the AD')
				up_data = up_data.loc[h>0.01]
				fp = morgan_fp(0, 3764) 
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_OH.predict(x_input)
				output= sc_oh.inverse_transform(output.reshape(-1,1))
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)

			if select=='Koc':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 1, 4385)
					xx = []
					for j in range(len(train_KOC)):
						ms2 = Chem.MolFromSmiles(train_KOC['smiles'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 1, 4385)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.039])} chemicals outside the AD')
				up_data = up_data.loc[h>0.039]
				
				fp = morgan_fp(1, 4385) 
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_KOC.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)

			if select=='SO4- radical':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 190)
					xx = []
					for j in range(len(train_SO4)):
						ms2 = Chem.MolFromSmiles(train_SO4['smiles'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 0, 190)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.04])} chemicals outside the AD')
				up_data = up_data.loc[h>0.04]
				
				fp = morgan_fp(0, 190) 
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_SO4.predict(x_input)
				output= sc_so4.inverse_transform(output.reshape(-1,1))
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)

			if select=='Solubility':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 0, 3878)
					xx = []
					for j in range(len(train_SOL)):
						ms2 = Chem.MolFromSmiles(train_SOL['smiles'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 0, 3878)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.035])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.035]
				fp = morgan_fp(0, 3878)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_SOL.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
				
			if select=='pKd':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 4, 3085)
					xx = []
					for j in range(len(train_pkd)):
						ms2 = Chem.MolFromSmiles(train_pkd['Neutralized SMILES code'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 4, 3085)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.176])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.176]
				fp = morgan_fp(4, 3085)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_pkd.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
				
			if select=='pIC50':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 2, 4840)
					xx = []
					for j in range(len(train_pic50)):
						ms2 = Chem.MolFromSmiles(train_pic50['Neutralized SMILES code'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 2, 4840)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.133])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.133]
				fp = morgan_fp(2, 4840)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_pic50.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
				
			if select=='CCSM_H':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 1, 2843)
					xx = []
					for j in range(len(train_ccsmh)):
						ms2 = Chem.MolFromSmiles(train_ccsmh['CanonicalSMILES'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 1, 2843)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.093])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.093]
				fp = morgan_fp(1, 2843)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_ccsmh.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
				
			if select=='CCSM_Na':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 2, 4754)
					xx = []
					for j in range(len(train_ccsmna)):
						ms2 = Chem.MolFromSmiles(train_ccsmna['CanonicalSMILES'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 2, 4754)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.008])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.008]
				fp = morgan_fp(2, 4754)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_ccsmna.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
				
			if select=='Lipo':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 2, 4891)
					xx = []
					for j in range(len(train_lipo)):
						ms2 = Chem.MolFromSmiles(train_lipo['smile'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 2, 4891)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.139])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.192]
				fp = morgan_fp(2, 4891)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_lipo.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
			if select=='FreeSolv':
				##CHECKING AD
				tot = []
				for i in up_data.index:
					ms1 = Chem.MolFromSmiles(up_data['smile'][i])
					fp_test= AllChem.GetHashedMorganFingerprint(ms1, 1, 3318)
					xx = []
					for j in range(len(train_freesolv)):
						ms2 = Chem.MolFromSmiles(train_freesolv['smiles'][j])
						fp2= AllChem.GetHashedMorganFingerprint(ms2, 1, 3318)
						s= DataStructs.DiceSimilarity(fp_test, fp2)
						xx.append(s)
					tot.append(np.mean(xx))
				h = np.array(tot)
				st.warning(f'There are {len(h[h<0.031])} chemicals outside the AD')
				
				up_data = up_data.loc[h>0.031]
				fp = morgan_fp(1, 3318)
				up_data['fp'] = up_data['smile'].apply(fp)
				x_input=np.array(list(up_data['fp']))
				output = model_freesolv.predict(x_input)
				up_data['pred']=output
				up_data.drop(columns=['fp'],inplace=True)
			
			st.success('Finished prediction')
			st.write(up_data)
			st.balloons()
			up_csv = convert_df(up_data)
			st.download_button(label="Download data as CSV",
					   data=up_csv,
					   file_name='my_dataset_prediciton.csv',
					   mime='text/csv')

if __name__ == '__main__':
    run()

