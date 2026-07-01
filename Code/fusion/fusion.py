import torch
import argparse
import random
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool,GATConv
from torch_geometric.utils import softmax
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from utils import load_bert

class RelationAwareGATLayer(MessagePassing):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_r = nn.Linear(3 * dim, dim)      
        self.W_Q = nn.Linear(dim, dim, bias=False)   
        self.W_K = nn.Linear(dim, dim, bias=False)  
        self.W_V = nn.Linear(dim, dim, bias=False)  
 
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        edge_attr = self.W_r(torch.cat([x[src], edge_attr, x[dst]], dim=-1))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        print(f"updated event node shape: {out.shape}")
        return out, edge_attr
 
    def message(self, x_i, edge_attr, index, ptr, size_i):
        q = self.W_Q(x_i)
        k = self.W_K(edge_attr)
        print(edge_attr.shape,k.shape)
        score = (q * k).sum(dim=-1) / (self.dim ** 0.5)
        alpha = softmax(score, index, ptr, size_i)
        print(f"alpha shape {alpha.shape}")
        v = self.W_V(edge_attr)
        return alpha.unsqueeze(-1) * v
    
class GraphRelationEncoder(nn.Module):
    def __init__(self, in_dim=768, hid_dim=128, num_layers=2):
        super().__init__()
        self.node_proj=nn.Linear(in_dim, hid_dim)
        self.rel_proj=nn.Linear(in_dim, hid_dim)
        self.layers =nn.ModuleList([
            RelationAwareGATLayer(hid_dim) for _ in range(num_layers)
        ])
 
    def forward(self, data):
        x = self.node_proj(data.x)
        r = self.rel_proj(data.edge_attr)
        for layer in self.layers:
            x_new,r=layer(x,data.edge_index,r)
            x=F.relu(x_new)+x
                                     
        g = global_mean_pool(x, data.batch)            
        return g          

class GraphEncoder(nn.Module):
    def __init__(self,in_dim=768,hid_dim=128):
        super().__init__()
        self.node_proj=nn.Linear(in_dim,hid_dim)
        self.gat = GATConv(hid_dim, hid_dim, heads=4, concat=False,dropout=0.3)  

    def forward(self,data):
        x=self.node_proj(data.x)
        x=F.relu(self.gat(x,data.edge_index))
        g=global_mean_pool(x,data.batch)

        return g

class Gatedfusion(nn.Module):
    def __init__(self,hid_dim,text_dim):
        super().__init__()
        self.graph_proj=nn.Linear(hid_dim,hid_dim)
        self.text_proj=nn.Linear(text_dim,hid_dim)
        self.gate=nn.Linear(2*hid_dim,hid_dim)

    def forward(self,g,t):
        g=self.graph_proj(g)
        t=self.text_proj(t)
        out=self.gate(torch.cat([g, t], dim=-1)) #concat to the Euclidean space
        #print(f"gate output shape: {out.shape}")
        
        return out 

class TwopillerModel(nn.Module):
    def __init__(self,graph_encoder,text_dim,hid_dim,num_class,device):
        super().__init__()
        self.graph_encoder=graph_encoder
        self.fusion=Gatedfusion(hid_dim,text_dim)
        self.sequence=nn.Sequential(nn.ReLU(),nn.Dropout(0.3),nn.Linear(hid_dim,num_class))
        self.device=device

    def forward(self,data):
        g=self.graph_encoder(data)
        #print(len(data.text_emb),len(data.text_emb[0]))
        t=torch.tensor(data.text_emb,dtype=torch.float).view(g.size(0), -1).to(self.device)
        #print(g.shape,t.shape)
        out=self.fusion(g, t)
        #print(out.shape)
        out=self.sequence(out)
        return out   
    
def run(model,data_loader,opt,device,train=True):
    total_loss=0
    preds,labels=[],[]
    if train:
        model.train()
        for data in data_loader:
            data=data.to(device)
            out=model(data)
            loss=F.cross_entropy(out,data.y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss+=loss.item()
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(data.y.cpu().tolist())
    else:
        with torch.no_grad():
            model.eval()
            for data in data_loader:
                data=data.to(device)
                out=model(data)
                loss=F.cross_entropy(out,data.y)
                total_loss+=loss.item()
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(data.y.cpu().tolist())
    return total_loss,preds,labels

def get_textembeddings(data, tokenizer, model,device):
    embeddings = []
    for d in tqdm(data, desc="Getting texts' embeddings"):
        inputs = tokenizer(d.text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        d.text_emb=cls_embedding
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/media/volume/boot-vol-k-project/Fairness_graph/Code/fusion/results/")
    parser.add_argument("--dataset_name", type=str, default="train_HateXplain_graphs")
    parser.add_argument("--test_dataset_name", type=str, default="balanced_test_HateXplain_graphs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--output_folder", type=str, default="/media/volume/boot-vol-k-project/Fairness_graph/Code/fusion/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment", type=str, default="relationaware")
    parser.add_argument("--relation", type=bool, default=False)
    args=parser.parse_args()


    device=args.device
    random.seed(42)
    train_dataset=torch.load(os.path.join(args.dataset_folder,f"{args.dataset_name}.pt"))
    test_dataset=torch.load(os.path.join(args.dataset_folder,f"{args.test_dataset_name}.pt"))
    pos=[d for d in train_dataset if int(d.y.item())==1]
    neg=[d for d in train_dataset if int(d.y.item())==0]
    print(f"before {len(train_dataset)} hate: {len(pos)}, non-hate: {len(neg)}")
    print(f"test dataset {len(test_dataset)}")
    n=min(len(pos),len(neg))
    balanced=random.sample(pos,n)+random.sample(neg,n)
    random.shuffle(balanced)

    train_dataset=balanced
    print(f"after: {len(train_dataset)}, each class: {n}")
    embedder,tokenizer=load_bert(model_path='GroNLP/hateBERT',device=device)
    #train_dataset=get_textembeddings(train_dataset, tokenizer, embedder,device)
    #test_dataset=get_textembeddings(test_dataset,tokenizer,embedder,device)
    #torch.save(train_dataset, "fusion/results/train_HateXplain_graphs.pt") 
    #torch.save(test_dataset, "fusion/results/balanced_test_HateXplain_graphs.pt") 
    print(train_dataset[0],test_dataset[0])

    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size*2, shuffle=False)

    if args.relation:
        graph_encoder=GraphRelationEncoder(in_dim=768,hid_dim=128,num_layers=2)
    else:
        graph_encoder=GraphEncoder(in_dim=768,hid_dim=128)
    model= TwopillerModel(graph_encoder, text_dim=768,hid_dim=128,num_class=2, device=device).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        total_loss,_,_=run(model,train_dataloader,optim,device,train=True)
        te_loss, preds, te_labels = run(model, test_dataloader, optim, device, train=False)
        f1 = f1_score(te_labels, preds, average="macro")
        print(f"epoch {epoch:02d}  train_loss {total_loss/len(train_dataloader):.4f} test_loss {te_loss/len(test_dataloader):.4f}  val macro-F1 {f1:.4f}")
        if epoch==0:
            best_loss=te_loss/len(test_dataloader)
            best_f1=f1
        if best_f1<f1:
            best_f1=f1
            best_loss=te_loss/len(test_dataloader)
            best_preds=preds


    report=classification_report(te_labels, best_preds,target_names=["not hate","hate"], digits=2)
    
    with open(f"{args.output_folder}/{args.experiment}_{args.dataset_name}_result.txt",'a') as f:
        f.write(f"\nEpoch: {args.epoch}, learning_rate: {args.lr}, batch_size: {args.batch_size}, training: {len(train_dataloader)}, valid: {len(test_dataloader)}\n{report}")
if __name__=="__main__":
    main()