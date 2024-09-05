import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
st.set_page_config(
    page_title="Welcome to DP-Transformer",    
    page_icon="💧",        
    layout="wide",                
    initial_sidebar_state="auto"  
)

GA_TRACKING_ID = "G-6MJ8FDZ7GH"

GA_TRACKING_CODE = f"""
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());

gtag('config', '{GA_TRACKING_ID}');
</script>
"""

components.html(GA_TRACKING_CODE)


if "show_animation" not in st.session_state:
    st.session_state.show_animation = True
st.header('Welcome to DP-Transformer!')
st.markdown(f'{TEXT1}', unsafe_allow_html=True)
#st.image(Image.open('Fig1.jpg'), caption = 'Figure 1. The comparison between binary MF (B-MF) and count-based MF (C-MF) when representing 1-Decanol, 1-Nonanol and 1-Ocatal')
#col1= st.columns([1])
st.image(Image.open('predic.jpg'), width=800, caption = 'Figure 1. The workflow that DP-Transformer makes predictions')
#col2.image(Image.open('Fig2.jpg'), caption = 'Figure 2. The performance enhancment C-MF brings for each dataset')
if "has_snowed" not in st.session_state:
    st.snow()
    st.session_state["has_snowed"] = True

if 'visitor_count' not in st.session_state:
	st.session_state.visitor_count = 0
if 'session_initialized' not in st.session_state:
	st.session_state.visitor_count += 1
	st.session_state.session_initialized = True
st.write(f'Visitor Number: {st.session_state.visitor_count}')
