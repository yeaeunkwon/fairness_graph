from __future__ import annotations
import argparse
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModel, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--threshold", type=float, default=0.95)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--val_frac", type=float, default=0.15)
ap.add_argument("--experiment", type=str,default="wasi")
ap.add_argument("--extraction",type=str, default="spacy")
args = ap.parse_args()
LABEL2ID = {"hatespeech": 0, "hate speech": 0, "hate": 0, "offensive": 1, "normal": 2}


def majority_label(annotators):
    labels = []
    for lab in annotators["label"]:
        lab = str(lab).lower()
        if lab in LABEL2ID:
            labels.append(LABEL2ID[lab])
        elif lab.isdigit():
            labels.append(int(lab))
    return Counter(labels).most_common(1)[0][0] if labels else None

def load_hatexplain(split="train", limit=None):
    from datasets import load_dataset
    ds = load_dataset("hatexplain", split=split, trust_remote_code=True)
    rows = []
    for ex in ds:
        lab = majority_label(ex["annotators"])
        if lab is None:
            continue
        rows.append({"text": " ".join(ex["post_tokens"]), "label": lab})
        if limit and len(rows) >= limit:
            break
    return rows

@torch.no_grad()
def embed_sentences(texts, device, name="Hate-speech-CNERG/bert-base-uncased-hatexplain", bs=64):
    tok = AutoTokenizer.from_pretrained(name)
    bert = AutoModel.from_pretrained(name).to(device).eval()
    feats = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(device)
        out = bert(**enc).last_hidden_state        
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        feats.append(pooled.cpu())

    return torch.cat(feats, dim=0) 

def build_similarity_edges(X, threshold=0.725, block=2048):
    Xn = F.normalize(X, dim=1)                          
    N = Xn.size(0)
    srcs, dsts = [], []
    for i in range(0, N, block):                      
        sim = Xn[i:i + block] @ Xn.t()#[block,N] cosine
        rows, cols = (sim > threshold).nonzero(as_tuple=True)
        rows = rows + i
        mask = rows != cols                             
        srcs.append(rows[mask])
        dsts.append(cols[mask])
    ei = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)
    print(f"edge_num: {ei.shape}")
    return ei                                          

class XGHSINode(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=3, heads=4, dropout=0.3):
        super().__init__()
        self.drop = dropout
        self.lin_in = nn.Linear(in_dim, hidden)         
        self.sage = SAGEConv(hidden, hidden, aggr="mean") 
        self.gcn = GCNConv(hidden, hidden)               
        self.gat = GATConv(2 * hidden, hidden, heads=heads, concat=False, dropout=dropout)  
        self.lin_out = nn.Linear(2 * hidden, num_classes)  
 
    def forward(self, x, edge_index):
        x1 = self.lin_in(x)                              
        x2 = self.sage(x1, edge_index)                
        x3 = self.gcn(x1, edge_index)                   
        x23 = torch.cat([x2, x3], dim=-1)               
        x4 = self.gat(x23, edge_index)                   
        xc = torch.cat([x1, x4], dim=-1)               
        return self.lin_out(xc)   

class CustomNode(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=3, heads=4, dropout=0.3):
        super().__init__()
        self.w1=nn.Linear(hidden,hidden)
        self.w2=nn.Linear( hidden,hidden)
        self.lin_in = nn.Linear(in_dim,  hidden)                   
        self.gat = GATConv(2 * hidden, hidden, heads=heads, concat=False, dropout=dropout)  
        self.lin_out = nn.Linear(2 * hidden, num_classes)  
    
    def encoder(self, x, edge_index):
        row,col=edge_index

        neighbor=torch.zeros_like(x).scatter_add(0,row.unsqueeze(-1).expand(-1,x.size(-1)),x[col])
        print(f"neighbor shape: {neighbor.shape}")
        deg=torch.bincount(row,minlength=x.size(0)).clamp(min=1).unsqueeze(-1)
        mean_nei=neighbor/deg

        x2=self.w1(x)+self.w2(mean_nei)
        x3=self.w1(x)+self.w2(neighbor)

        return x2,x3
 
    def forward(self, x, edge_index):
        x1 = self.lin_in(x)                              
        x2,x3 = self.encoder(x1, edge_index)                               
        x23 = torch.cat([x2, x3], dim=-1)               
        x4 = self.gat(x23, edge_index)                   
        xc = torch.cat([x1, x4], dim=-1)               
        return self.lin_out(xc)                              
 
 
def main():
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
 
    rows = load_hatexplain(limit=args.limit)
    texts = [r["text"] for r in rows]
    y = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    print(f"{len(rows)} utterances (nodes)")
    
    N = len(rows)
    perm = torch.randperm(N)
    n_val = int(N * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    print("embedding sentences ...")
    X = embed_sentences(texts, device)

    X_train=X[train_idx]
    X_val=X[val_idx]

    y_train=y[train_idx]
    y_val=y[val_idx]


    print("building similarity edges ...")
    edge_index_train = build_similarity_edges(X_train, threshold=args.threshold)
    edge_index_val = build_similarity_edges(X_val, threshold=args.threshold)
    print(f"edges: {edge_index_train.shape}, {edge_index_train.size(1)} (avg degree {edge_index_train.size(1)/len(rows):.1f})")
    print(f"edges: {edge_index_val.shape}, {edge_index_val.size(1)} (avg degree {edge_index_val.size(1)/len(rows):.1f})")
    
    
    X_train, edge_index_train, y_train = X_train.to(device), edge_index_train.to(device), y_train.to(device)
    X_val, edge_index_val, y_val = X_val.to(device), edge_index_val.to(device), y_val.to(device)

 
    model = CustomNode(in_dim=X.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
 
    for ep in range(args.epochs):
        model.train()
        logits = model(X_train, edge_index_train)                    
        loss = F.cross_entropy(logits, y_train)   
        opt.zero_grad()
        loss.backward()
        opt.step()
 
        if ep % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits=model(X_val, edge_index_val)
                pred =val_logits.argmax(-1)
                f1 = f1_score(y_val.cpu(), pred.cpu(), average="macro")
            print(f"epoch {ep:03d}  loss {loss.item():.4f}  val macro-F1 {f1:.4f}")
 
    report=classification_report(
        y_val.cpu(), pred.cpu(),
        target_names=["hate", "offensive", "normal"], digits=3)

    with open(f"results/{args.experiment}_{args.extraction}_result.txt",'a') as f:
        f.write(f"\nEpoch: {args.epochs}, learning_rate: {args.lr}, training: {len(X_train)}, valid: {len(X_val)}, Threshold: {args.threshold} \n{report}")
    return model
 
 
if __name__ == "__main__":
    main()
 