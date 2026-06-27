import pandas as pd
import json
from datasets import load_dataset,concatenate_datasets
from collections import Counter
LABEL2ID = {"hatespeech": 1, "hate speech": 1, "hate": 1,
            "offensive": 2,
            "normal": 0}
def majority_label(annotators):
    labels = []
    for lab in annotators["label"]:
        if int(lab)==1 or int(lab)==2:
            return 1
       
    return 0



df=pd.read_json("results/HateXplain.jsonl",lines=True)
ds = load_dataset("hatexplain", trust_remote_code=True)
            
train_data = concatenate_datasets([ds['train'], ds['validation']])
test_data = ds['test']
print(df['label'].value_counts())
og_df=train_data.to_pandas()

texts=[]
labels=[]
for ex in train_data:
    text = " ".join(ex["post_tokens"])
    label = majority_label(ex["annotators"])
    if label is None:
        continue
    texts.append(text)
    labels.append(label)
    if label==0:
        print(ex)

new_df=pd.DataFrame({"text":texts,"label_binary":labels})
merge_df=df.merge(new_df,on='text',how='left')
print(len(df),len(new_df),len(merge_df),merge_df.columns)
print(merge_df['label_binary'].value_counts())