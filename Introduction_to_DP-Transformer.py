import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#ffffff",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""

st.set_page_config(
    page_title="Welcome to DP-Transformer",    
    page_icon="💧",        
    layout="wide",                
    initial_sidebar_state="auto"  
)

TEXT1 = """
        <body style='text-align: justify; color: black;'>
        <p> DP-Transformer platform is backend by advanced machine learning models to service users for predicting the degradation products of aqueous organic pollutants in chemical oxidation processes. 
		DP-Transformer can now predict the degradation products and degradation pathways of organic pollutants. DP-Tramsformer uses SMILES to represent chemicals.   
        </p>DP-Transformer is based on a similar model architecture as the Molecular Transformer but is distinguished by the utilized dataset. The well-trained DP-Transformer accepts pollutant SMILES, oxidative species,
	and reaction conditions (e.g., pH) as inputs and outputs the SMILES of the degradation products. This model is capable of predicting not only the degradation intermediates but also the complete degradation pathways. 
 The prediction of degradation pathways is accomplished through an iterative process, where a predicted degradation product made by DP-Transformer is used as input for subsequent prediction. This process continues until the 
 model predicts CO2 or when the predicted chemicals remain unchanged (i.e., non-degradable), indicating the formation of the final degradation products (Figure 1). 
	  <p> 
	  </p>
        </body>         
        """
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
if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)
