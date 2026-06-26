import pandas as pd
import json
from datasets import load_dataset,concatenate_datasets

df=pd.read_json("/results/HateXplain.jsonl",lines=True)
ds = load_dataset("hatexplain", trust_remote_code=True)
            
train_data = concatenate_datasets([ds['train'], ds['validation']])
test_data = ds['test']
print(len(df),len(train_data),len(test_data))
