##extracting non toxic and test triplets
import os
import re
import math
import torch
import argparse
from tqdm import tqdm
from promptsv3 import REASONPROMPTV3, TRIPLETPROMPTV4
from promptsv2 import REASONPROMPTV2, TRIPLETPROMPTV2, FILTERPROMPTV2
from transformers import AutoTokenizer, AutoModelForCausalLM
#from prompts import REASONPROMPT, TRIPLETPROMPT, FILTERPROMPT
import json
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from transformers import AutoTokenizer, AutoModel
import numpy as np
from utils import save_jsonl, dataset_preprocess, load_jsonl, load_llm, load_bert, setup_logger

logger = setup_logger("Extract")

def _batch_inference(llm, tokenizer, batch_data, prompt, key,ending):
    raw_inputs = []
    for ex in batch_data:
        if key:
            if isinstance(key, str):
                to_format = ex[key].strip("<|endoftext|>")
                raw_inputs.append(prompt.format(context = to_format))
            else:
                raw_inputs.append(prompt.format(
                    context=ex['text'].strip("<|endoftext|>"),
                    analysis=ex['reason'].strip("<|endoftext|>")
                ))
        else:
            raw_inputs.append(prompt.format(context = ex.strip("<|endoftext|>")))
    inputs = tokenizer(raw_inputs, return_tensors = "pt",
                       padding = True, truncation = True).to(llm.device)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens = 128)
    return tokenizer.batch_decode(outputs)

def _rulebase_triplet_extract(text):
    # Use a regular expression to find all occurrences of text within parentheses
    pattern = r'(\(.*?\))'
    matches = re.findall(pattern, text)
    res = "[" + ",".join(list(set(matches))) + "]"
    return res

def preprocess(dataset_folder, dataset_names, output_folder):
    '''
    Step 0: load dataset and save to jsonl

    saving at the output_folder with the name format of dataset_name.jsonl
    '''
    for dataset_name in dataset_names:
        normal_data,toxic_data = dataset_preprocess(dataset_folder, dataset_name)
        logger.info(f"Sample data before saving: {normal_data[0]}")
        save_jsonl(normal_data, os.path.join(output_folder, f"non_hs_worationale_{dataset_name}.jsonl"))
        save_jsonl(toxic_data, os.path.join(output_folder, f"hs_worationale_{dataset_name}.jsonl"))
        logger.info(f"Step 0: Dataset {dataset_name} preprocessed and saved to {output_folder}")
    logger.info(f"Step 0: All datasets: {' '.join(dataset_names)} preprocessed and saved to {output_folder} with name format of dataset_name.jsonl, and test_dataset_name.jsonl and balanced_test_dataset_name.jsonl for evaluation.")

def reasoning(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version):
    '''
    Step 1: reasoning using LLM

    loading from the output_folder with the name format of dataset_name.jsonl

    saving at the output_folder with the name format of rationale_dataset_name.jsonl
    '''
    # load the llm and tokenizer first.
    tokenizer = AutoTokenizer.from_pretrained(llm_name,padding_side="left")

    llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto"
    )

    llm.eval()  
    for dataset_name in dataset_names:
        # load the dataset.
        data = load_jsonl(os.path.join(output_folder, f"non_hs_worationale_{dataset_name}.jsonl"))
        length = len(data)
        logger.info(f"Step 1: Dataset {dataset_name} loaded, total {length} samples.")

        iterate_times = math.ceil(length/batch_size)
        with torch.inference_mode():    
            for i in tqdm(range(resume_inference, iterate_times)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, length)
                
                if end_idx - start_idx < batch_size and end_idx - start_idx > 0:
                    batch_data = data[start_idx:end_idx]
                    logger.info(f"Processing last batch with {len(batch_data)} samples")
                else:
                    batch_data = data[start_idx:end_idx]
            
                if not batch_data:
                    continue
                # batch inference.

                match prompt_version:
                    case "v1":
                        prompt = REASONPROMPT
                    case "v2":
                        prompt = REASONPROMPTV2
                    case "v3":
                        prompt = REASONPROMPTV3 #non-toxic or toxic
                    case _:
                        raise NotImplementedError
                texts = _batch_inference(llm, tokenizer, batch_data, prompt, "text", "\nAnalysis: ")

                # save the results each batch.
                for text, ex in zip(texts, batch_data):
                    ex["reason"] = text.split("<ASSISTANT>: Analysis:")[1].split("\nContext")[0].strip("<|endoftext|>")
                save_jsonl(batch_data, os.path.join(output_folder, f"rationale_non_hs_worationale_{dataset_name}.jsonl"))
        # done.
        logger.info(f"Step 1: Dataset {dataset_name} processed, total {iterate_times} batches. saving at {output_folder}/rationale_non_hs_{dataset_name}.jsonl")

def triplet_extracting(dataset_names, dataset_folder,output_folder, llm_name, device, batch_size, resume_inference, prompt_version):
    '''
    Step 2: extract triplets using LLM

    loading from the output_folder with the name format of rationale_dataset_name.jsonl

    saving at the output_folder with the name format of triplets_dataset_name.jsonl
    '''
    # load the llm and tokenizer first.
    llm, tokenizer = load_llm(llm_name, device)
    llm.eval()
    for dataset_name in dataset_names:
        # load the dataset.
        data = load_jsonl(os.path.join(dataset_folder, f"{dataset_name}.jsonl"))
        length = len(data)
        logger.info(f"Step 2: Dataset {dataset_name} loaded, total {length} samples.")
        iterate_times = math.ceil(length/batch_size)
        with torch.inference_mode():
            for i in tqdm(range(resume_inference, iterate_times)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, length)
            
                if end_idx - start_idx < batch_size and end_idx - start_idx > 0:
                    batch_data = data[start_idx:end_idx]
                    logger.info(f"Processing last batch with {len(batch_data)} samples")
                else:
                    batch_data = data[start_idx:end_idx]
            
                if not batch_data:
                    continue
                # batch inference.
                inference_prompt = ""
                match prompt_version:
                    case "v1":
                        inference_prompt = TRIPLETPROMPT
                        key = "reason"
                    case "v2":
                        inference_prompt = TRIPLETPROMPTV2
                        key = "reason"
                    case "v3":
                        inference_prompt = TRIPLETPROMPTV4
                        key = "text"
                    case _:
                        raise NotImplementedError
                    
                texts = _batch_inference(llm, tokenizer, batch_data, inference_prompt, key, "\nOutput: ")
                # using such a complex ending to avoid the model repeat and generate more examples deviate the original inputs.
                for text, ex in zip(texts, batch_data):
                    ex["triplets"] = text.split("<ASSISTANT>: Output:")[1].strip("<|endoftext|>").split("\n<USER>")[0]
                    ex["triplets"] = _rulebase_triplet_extract(ex["triplets"])
                save_jsonl(batch_data, os.path.join(output_folder, f"triplets_{dataset_name}.jsonl"))
        # done.
        logger.info(f"Step 2: Dataset {dataset_name} processed, total {iterate_times} batches. saving at {output_folder}/triplets_{dataset_name}.jsonl")


        
def load_triplets(dataset):
    
    pattern = r'(\(.*?\))'
    matches = re.findall(pattern, dataset)
    triplets = []
    for match in matches:
        elements = match.split(',')
        if len(elements) == 3:
            triplet = tuple(elem.strip().strip(")").strip("(") for elem in elements)
            triplets.append(triplet)
    return triplets

def build_graph(post_triplets):
    G=nx.MultiDiGraph()
    for h,r,t in post_triplets:
        if not h or not t:
            continue
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h,t,relation=r)
    return G

def get_embeddings(nodes, tokenizer, model,device):
    embeddings = []
    for node in nodes:
        inputs = tokenizer(node, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

def graph_to_data(graph,embedder,tokenizer,device,label):
    nodes=list(graph.nodes())
    node2idx={n:i for i,n in enumerate(nodes)}
    if len(nodes)==0:
        return 0
    node_emb=get_embeddings(nodes,tokenizer,embedder,device)
    src,des,rel=[],[],[]

    for s,o,r in graph.edges(data=True):
        relation=r.get("relation","")
        src.append(node2idx[s])
        des.append(node2idx[o])
        rel.append(relation)
        src.append(node2idx[o])
        des.append(node2idx[s])
        rel.append(relation)

    edge_index=torch.tensor([src,des],dtype=torch.long)
    edge_attr=get_embeddings(rel,tokenizer,embedder,device)

    data=Data(x=node_emb,edge_index=edge_index,edge_attr=edge_attr)
    if not torch.is_tensor(data.x):
        data.x = torch.tensor(data.x, dtype=torch.float)
    if hasattr(data, "edge_attr") and data.edge_attr is not None and not torch.is_tensor(data.edge_attr):
        data.edge_attr = torch.tensor(data.edge_attr, dtype=torch.float)
    if label=="a":
        data.y=torch.tensor([1],dtype=torch.long)
    else:
        data.y=torch.tensor([0],dtype=torch.long)
    return data

def build_dataset(dataset_name, output_folder, bert_name, device):
    
    posts = []
    embedder,tokenizer=load_bert(model_path=bert_name,device=device)
    posts=load_jsonl(output_folder + f"/triplets_{dataset_name}.jsonl")
    logger.info(f"Loaded triplet data {dataset_name} successfully.")
    dataset=[]
    for post in tqdm(posts,desc="Getting graphs of posts: "):
        triplets=load_triplets(post['triplets'])
        if len(triplets)==0: #remove the instance with no triplets
            print(post)
            continue
        G=build_graph(triplets)
        graph_data=graph_to_data(G,embedder,tokenizer,device,label=post['label'])
        if graph_data==0:
            print(f"graph data is zero {triplets}")
            continue
        graph_data.text=post.get("text")
        graph_data.source=post.get("source")
        dataset.append(graph_data)

    print(dataset[0])
    logger.info(f"graph data is built {len(dataset)}")
    if dataset:
        avg_nodes = sum(d.num_nodes for d in dataset) / len(dataset)
        avg_edges = sum(d.edge_index.size(1) for d in dataset) / len(dataset)
        print(f"avg nodes/graph: {avg_nodes:.1f}, avg edges/graph: {avg_edges:.1f}")

    torch.save(dataset, f"results/{dataset_name}_graphs.pt")
    print(f"saved -> results/{dataset_name}_graphs.pt") #tensor transform 필요

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/media/volume/boot-vol-k-project/Fairness_graph/Code/results")
    parser.add_argument("--dataset_name", type=str, default="balanced_test_HateXplain")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_folder", type=str, default="/media/volume/boot-vol-k-project/Fairness_graph/Code/enhancement/results")
    parser.add_argument("--step_by_step", type=str, default="True")

    parser.add_argument("--resume_step", type=int, default=0,
                        help="the step to do. once a step.\
        0 for the beginning. Then preprocess.\
        1 is loading the dataset from the processed data. Then analyze the reason of hate.\
        2 is loading the analyzed dataset from the processed data. Then extract triplets.\
        3 is loading the triplet file from the processed data in output folder. Then filter.")
    
    parser.add_argument("--resume_inference", type=int, default=0,
        help = "To avoid the crash during the inference, we use this parameter to resume from the last processed sample. 0 means from the beginning. U can change it according to your log and batchsize. Usually, the last tqdm number * batchsize is a good choice.")
    
    parser.add_argument("--prompt_version", type=str, default="v3",
        help = "The version of the prompts.")
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    dataset_folder = args.dataset_folder
    output_folder = args.output_folder
    step_by_step = args.step_by_step
    llm_name = args.llm_name
    step_by_step = step_by_step.lower() == "true"
    resume_step = args.resume_step
    resume_inference = args.resume_inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    bert_name=args.bert_name
    prompt_version = args.prompt_version
    if not step_by_step:
        logger.warning(f"Step by step mode is False. It may takes a long time to run all steps. Please consider using step by step mode.")
        preprocess(dataset_folder, dataset_name, output_folder)
        reasoning(dataset_name, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
        triplet_extracting(dataset_name, dataset_folder,output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
        build_dataset(dataset_name, output_folder, bert_name, device)
    else:
        match resume_step:
            case 0:
                preprocess(dataset_folder, dataset_name, output_folder)
            case 1:
                reasoning(dataset_name, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
            case 2:
                triplet_extracting(dataset_name, dataset_folder,output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
            case 3:
                build_dataset(dataset_name, output_folder, bert_name, device)
            case _:
                raise ValueError(f"Step {resume_step} not implemented")



if __name__ == "__main__":
    main()
