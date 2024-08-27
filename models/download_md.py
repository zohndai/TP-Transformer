import os
import gdown
import time
import steamlit as st
def load_model(fd,model_name):
    file_id = fd
    model_path = model_name
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, model_path, quiet=True)
#if col1.button('Get the prediction')
def download():
    name = '47700_step'
    destination_dir = 'pt_models'
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
    message_container.tex("Model is ready!")
  return new_file_path
