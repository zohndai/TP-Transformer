import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import cirpy
import torch
from rdkit.Chem import Draw
#import matplotlib.pyplot as plt
import os
import gdown
import time
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab, _load_vocab, DatasetLazyIter, OrderedIterator
from functools import partial
from multiprocessing import Pool
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab
import onmt.translate
import pandas as pd
import onmt.model_builder
import onmt.translate
from onmt.utils.misc import split_corpus
import re

def smi_tokenize(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|UV|MW|VL|hv|E|US|heat|<|_)"
    compiled_pt = re.compile(pattern)
    tokens = [token for token in compiled_pt.findall(smi)]
    #assert smi == ''.join(tokens)
    return " ".join(tokens)

def load_model(fd,model_name):
    file_id = fd
    model_path = model_name
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, model_path, quiet=True)
#if col1.button('Get the prediction')
def download():
	name = '47700_step'
	destination_dir = 'models'
	os.makedirs(destination_dir, exist_ok=True)
	message_container = st.empty()
	message_container.text("Downloading the models... Please wait.")
	fd_dict = {'1IkvRvlpiLcETHrJahtLFuxiQtx0SBkdU':f'{name}_2024_0826',}
	for fd in fd_dict.keys():
		fd_file = fd
		model_name = fd_dict[fd]
		model_path = os.path.join(destination_dir, model_name)
		load_model(fd_file, model_path)
		time.sleep(4)
		current_file_path = f'models/{model_name}'
		new_file_path = f'models/{model_name}.pt'
		if not os.path.exists(new_file_path):
		    os.rename(current_file_path, new_file_path)
	message_container.text("Model is ready!")
	return new_file_path
def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])#加载模型参数
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):#False
        fields = inputters.load_old_vocab(vocab, opt.data_type, dynamic_dict=model_opt.copy_attn)
    else:
        fields = vocab
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, opt.gpu)
    if opt.fp32:#False
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt

def build_translator(opt, report_score, logger=None, out_file=None):
    log_probs_out_file = None
    target_score_out_file = None
    if out_file is None:#True
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
        
    if opt.log_probs:#False
        log_prob_oout_file = codecs.open(opt.ouput + '_log_probs', 'w+', 'utf-8')
        target_score_out_file = codecs.open(opt.output + '_gold_score', 'w+', 'utf-8')
        
    load_test_model = onmt.decoders.ensemble.load_test_model if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)
    
    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    
    translator = onmt.translate.Translator.from_opt(
        model, fields, opt, model_opt, global_scorer=scorer, out_file=out_file, report_align=opt.report_align, \
        report_score=report_score, logger=logger, log_probs_out_file=log_probs_out_file, target_score_out_file=target_score_out_file,
    )
    return translator

st.set_page_config(
    page_title="Welcome to DP-Transformer",    
    page_icon="log.ico",        
    layout="wide",                
    initial_sidebar_state="auto"
)





if True:
	ros_name = ["HO∙", "SO₄∙⁻","O₃", "¹O₂",  "Fe(VI)", "O₂∙⁻", "MnO₄⁻", "ClO⁻","HClO", "Cl₂","Cl∙","CO₃∙⁻","Cl₂∙⁻","C₂H₃O₃∙", \
             "Cu(III)","Fe(V)",  "NO₂∙", "Mn(V)", "HSO₄∙", "O₂", "BrO⁻","NO∙", "ClO∙","Fe(IV)","Br∙", "IO⁻","C₂H₃O₂∙",\
             "HSO₅⁻", "ClO₂∙", "Br₂","HOBr","HO₂⁻","I∙", "NO₃∙", "IO₃∙⁻", \
           "Fe(III)", "S₂O₈∙⁻","HCO₃∙", "SO₃∙⁻","Unkown"]
	ros_smis = ['[OH]','[O]S(=O)(=O)[O-]','O=[O+][O-]','OO1','O=[Fe](=O)([O-])[O-]','[O][O-]','O=[Mn](=O)(=O)[O-]','[O-]Cl','OCl','ClCl','[Cl]','[O]C(=O)[O-]','Cl[Cl-]',\
	 'CC(=O)O[O]','[Cu+3]','O=[Fe]([O-])([O-])[O-]','[O]N=O','O=[Mn]([O-])([O-])[O-]','[O]S(=O)(=O)O','O=O','[O-]Br','[N]=O','[O]Cl','[O-][Fe]([O-])([O-])[O-]','[Br]',\
	 '[O-]I','CC([O])=O','O=S(=O)([O-])OO','[O][Cl+][O-]','BrBr','OBr','[O-]O','[I]','[O][N+](=O)[O-]','[O-][I+2]([O-])[O-]','[Fe+3]','[O]S(=O)(=O)OOS(=O)(=O)[O-]',\
	 '[O]C(=O)O','[O]S(=O)[O-]','']
	
	
	acti_methd=["UV", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight", "No"]
	methd_tokens=["UV", "heat", "VL", "MW", "E", "US", "SL", ""]
	
	st.subheader('Please select the ROSs that drive the pollutant degradation')
	ros_selct=st.selectbox('What ROSs?', ( "HO∙", "SO₄∙⁻","O₃", "¹O₂",  "Fe(VI)", "O₂∙⁻", "MnO₄⁻", "ClO⁻","HClO", "Cl₂","Cl∙","CO₃∙⁻","Cl₂∙⁻","C₂H₃O₃∙", \
	             "Cu(III)","Fe(V)",  "NO₂∙", "Mn(V)", "HSO₄∙", "O₂", "BrO⁻","NO∙", "ClO∙","Fe(IV)","Br∙", "IO⁻","C₂H₃O₂∙",\
	             "HSO₅⁻", "ClO₂∙", "Br₂","HOBr","HO₂⁻","I∙", "NO₃∙", "IO₃∙⁻", \
	           "Fe(III)", "S₂O₈∙⁻","HCO₃∙", "SO₃∙⁻", "Unkown"))
	#st.write('You selected:', ros_selct)
	#select = st.radio("Please specify the property or activity you want to predict", ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv' ))
	st.subheader('Please input the precursors of the ROSs')
	prec = st.text_input("Please offer the Chemical name, CAS number or SMILES of precursors, e.g.'OO.[Fe+2]' for the fenton reagent H2O2/Fe2+ ", "OO.[Fe+2]")
	if prec !='':
		prec_smile = cirpy.resolve(prec, 'smiles')
		if prec_smile is None:
			st.warning('Invalid chemical name or CAS number of precursors, please check it again or imput SMILES')
			st.stop()
	with st.expander("Show how to get SMILES of chemicals"):
		st.write('You can get SMILES of any molecules from PubChem https://pubchem.ncbi.nlm.nih.gov/ by typing Chemical name or ACS number')
	
	st.subheader("Please select the method for extertal energy input for the ROSs generation", "UV")
	methd_selct=st.selectbox("What activation method?",("UV", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight", "No"))
	
	st.subheader('Please input the reaction pH for pollutant degradation')
	pH_value = st.text_input("Keep two decimal places","7.00")
	
	st.subheader('What pollutant?')
	poll = st.text_input("Please offer the Chemical name, CAS number or SMILES of the pollutant, e.g. 'c1ccccc1' for benzene", "c1ccccc1")
	if poll =='':
		st.warning('You should at least provide one chemical')
		st.stop()
	else:
		pol_smile = cirpy.resolve(poll, 'smiles')
		if pol_smile is None:
			st.warning('Invalid chemical name or CAS number or SMILES, please check it again or input SMILES')
			st.stop()

	col1, col2, col3, col4= st.columns([2,2,1,1])
	ros_smi = ros_smis[ros_name.index(ros_selct)]
	methd_token = methd_tokens[acti_methd.index(methd_selct)]
	pH = "".join(pH_value.split("."))
	try:
		cano_prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec))
	except:
		st.warning("invalid precursors's SMILES, please check it again")
		st.stop()
	
	try:
		cano_pollu = Chem.MolToSmiles(Chem.MolFromSmiles(poll))
	except:
		st.warning("invalid pollutant SMILES, please check it again")
		st.stop()
	reactant = cano_pollu + "." + ros_smi
	
	src = reactant+">"+cano_prec+"<"+methd_token+"_"+pH
	input = smi_tokenize(src)
	with open("src.txt", "w") as file:
		file.write(input)
	
	if col1.button('Get the prediction'):
		model_path = download()
		message_container = st.empty()
		message_container.text("model version:DP-Transformer-1.0.20240826")
	
		parser_tsl = ArgumentParser(description="translate.py")
		opts.config_opts(parser_tsl)
		opts.translate_opts(parser_tsl)
		args_tsl = ['-model', model_path, \
			    '-src', 'src.txt', \
			    '-output', 'predictions.txt', \
			    '-n_best', '10', \
			    '-beam_size', '10', \
			    '-max_length', '3000', \
			    '-batch_size', '64']
		opt_tsl = parser_tsl.parse_args(args_tsl)
		ArgumentParser.validate_translate_opts(opt_tsl)
		#logg1er = init_logger(opt_tsl.log_file)
	#model_path = opt_tsl.models[0]
		checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
		vocab = checkpoint['vocab']
		translator = build_translator(opt_tsl, report_score=True)
		src_shards_tsl = split_corpus(opt_tsl.src, opt_tsl.shard_size) # list(islice(f, shard_size))
		tgt_shards_tsl = split_corpus(opt_tsl.tgt, opt_tsl.shard_size)
		shard_pairs_tsl = zip(src_shards_tsl, tgt_shards_tsl)
		for i, (src_shard_tsl, tgt_shard_tsl) in enumerate(shard_pairs_tsl): # 0, ([src], None)
			#logger.info("Translating shard %d." % i) # 只有0
			translator.translate(
			    src=src_shard_tsl,
			    tgt=tgt_shard_tsl,
			    src_dir=opt_tsl.src_dir, #''
			    batch_size=opt_tsl.batch_size, # 64
			    batch_type=opt_tsl.batch_type, # sents
			    attn_debug=opt_tsl.attn_debug, # False
			    align_debug=opt_tsl.align_debug # False
			)
		
		dp_smis = pd.read_csv(opt_tsl.output,header=None)
		smis_li=["".join(dp_smi.split(" ")) for dp_smi in dp_smis[0]]
		message_container = st.empty()
		message_container.text(f"top1:{smis_li[0]},top2:{smis_li[1]},top3:{smis_li[2]},top4:{smis_li[3]},top5:{smis_li[4]},top6:{smis_li[5]},top7:{smis_li[6]},top8:{smis_li[7]},top9:{smis_li[8]},top10:{smis_li[9]}")
