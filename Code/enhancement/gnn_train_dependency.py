import argparse, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--graphs", type=str, default="hatexplain_spacy_graphs.pkl")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--n_val", type=float, default=0.15)
ap.add_argument("--experiment", type=str,default="wasi")
ap.add_argument("--extraction",type=str, default="spacy")
args = ap.parse_args()


class NodeEmbedding:
    def __init__(self,model_name="Hate-speech-CNERG/bert-base-uncased-hatexplain", device="cuda"):
        self.device=device
        self.tok=AutoTokenizer.from_pretrained(model_name)
        self.bert=AutoModel.from_pretrained(model_name).to(self.device)
        self.dim=self.bert.config.hidden_size

    @torch.no_grad()
    def embed(self,texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to(self.device)
        out = self.bert(**enc).last_hidden_state          
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9) 

def attach_features(graphs, embedder, batch_nodes=512):
    
    all_texts, spans = [], []
    for g in graphs:
        start = len(all_texts)
        all_texts.extend(g.node_texts)
        spans.append((start, len(all_texts)))
 
    feats = []
    for i in range(0, len(all_texts), batch_nodes):
        feats.append(embedder.embed(all_texts[i:i + batch_nodes]).cpu())
    feats = torch.cat(feats, dim=0)
 
    for g, (s, e) in zip(graphs, spans):
        g.x = feats[s:e]                                   
    return graphs

class XGHSIEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=3, heads=4, dropout=0.3):
        super().__init__()
        self.drop = dropout
        self.lin_in = nn.Linear(in_dim, hidden)            
        self.sage = SAGEConv(hidden, hidden, aggr="mean") 
        self.gcn = GCNConv(hidden, hidden)                
        self.gat = GATConv(2 * hidden, hidden, heads=heads, concat=False, dropout=dropout) 
        self.lin_out = nn.Linear(2 * hidden, num_classes)  
 
    def forward(self, x, edge_index, batch):
        
        x1=self.lin_in(x)
        x2 = self.sage(x1, edge_index)
        x3 = self.gcn(x1, edge_index)
        x23 = torch.cat([x2, x3], dim=-1)        
        x4 =self.gat(x23, edge_index)
        xc = torch.cat([x1, x4], dim=-1)                  
        graph_repr = global_mean_pool(xc, batch)        
        return self.lin_out(graph_repr)
   
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
 
    def forward(self, x, edge_index, batch):
        x1 = self.lin_in(x)                              
        x2,x3 = self.encoder(x1, edge_index)                               
        x23 = torch.cat([x2, x3], dim=-1)               
        x4 = self.gat(x23, edge_index)                   
        xc = torch.cat([x1, x4], dim=-1)
        graph_repr = global_mean_pool(xc, batch)                
        return self.lin_out(graph_repr)        

def run(train_g, val_g, in_dim, device, epochs=20, lr=1e-5, bs=32):
    model = CustomNode(in_dim=in_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tl = DataLoader(train_g, batch_size=bs, shuffle=True)
    vl = DataLoader(val_g, batch_size=bs)
 
    for ep in range(epochs):
        model.train()
        tot = 0
        for b in tl:
            b = b.to(device)
            logits = model(b.x, b.edge_index, b.batch)
            loss = F.cross_entropy(logits, b.y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
 
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                p = model(b.x, b.edge_index, b.batch).argmax(-1)
                preds += p.cpu().tolist()
                gts += b.y.view(-1).cpu().tolist()
        f1 = f1_score(gts, preds, average="macro")
        print(f"epoch {ep:02d}  loss {tot/len(tl):.4f}  val macro-F1 {f1:.4f}")
    report=classification_report(gts, preds,
          target_names=["hate", "offensive", "normal"], digits=3)
    with open(f"results/{args.experiment}_{args.extraction}_result.txt",'a') as f:
        f.write(f"\nEpoch: {epochs}, learning_rate: {lr}, batch_size: {bs}, training: {len(train_g)}, valid: {len(val_g)}\n{report}")
    return model


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")

    graphs=pickle.load(open(args.graphs,"rb"))
    print(f"loaded {len(graphs)} graphs")

    embedding=NodeEmbedding(device=device)
    graphs=attach_features(graphs,embedding)

    n_val=int(len(graphs)*args.n_val)
    val_g,train_g=graphs[:n_val],graphs[n_val:]

    print(f"train {len(train_g)}  val {len(val_g)}")

    run(train_g, val_g, in_dim=embedding.dim, device=device, epochs=args.epochs)

if __name__ == "__main__":
    
    main()