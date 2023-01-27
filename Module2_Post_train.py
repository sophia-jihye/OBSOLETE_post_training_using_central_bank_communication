from glob import glob
import pandas as pd
import os, time
from datetime import timedelta
from transformers_helper import load_tokenizer_and_model
import post_training_helper

post_filepath = os.path.join('pt_data', 'post_central_bank_communication.txt') 

save_dir_format = '/data/jihye_data/post_training_using_central_bank_communication/model_pt_{}'
    
def record_elasped_time(start, save_filepath):
    end = time.time()
    content = "Time elapsed: {}".format(timedelta(seconds=end-start))
    print(content)
    with open(save_filepath, "w") as f:
        f.write(content)    
    
def start_post_train(model_name_or_dir, post_filepath, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='masking')
    dataset = post_training_helper.LazyLineByLineTextDataset(tokenizer=tokenizer, file_path=post_filepath)
    post_training_helper.train_mlm(tokenizer, model, dataset, save_dir)

if __name__ == '__main__':
    
    for model_name_or_dir in ['nlpaueb/sec-bert-base', 'yiyanghkust/finbert-pretrain', 'ProsusAI/finbert', 'bert-base-uncased']:
        start = time.time()
        save_dir = save_dir_format.format(os.path.basename(model_name_or_dir))
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        start_post_train(model_name_or_dir, post_filepath, save_dir)
        record_elasped_time(start, os.path.join(save_dir, 'elapsed-time.log'))