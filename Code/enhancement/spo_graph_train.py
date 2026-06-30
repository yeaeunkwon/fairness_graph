import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import argparse
import random

class BaselineGNN(nn.Module):
    def __init__(self,in_dim=768,hidden=128,num_classes=2,heads=4,dropout=0.3):
        super().__init__()
        self.w1=nn.Linear(hidden,hidden)
        self.w2=nn.Linear( hidden,hidden)
        self.lin_in = nn.Linear(in_dim,  hidden)                   
        self.gat = GATConv(2 * hidden, hidden, heads=heads, concat=False, dropout=dropout)  
        self.lin_out = nn.Linear(2 * hidden, num_classes)  
    
    def encoder(self, x, edge_index):
        row,col=edge_index

        neighbor=torch.zeros_like(x).scatter_add(0,row.unsqueeze(-1).expand(-1,x.size(-1)),x[col])
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

def run(model,data_loader,opt,device,train=True):
    total_loss=0
    preds,labels=[],[]
    if train:
        model.train()
        for data in data_loader:
            data=data.to(device)
            out=model(data.x,data.edge_index,data.batch)
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
                out=model(data.x,data.edge_index,data.batch)
                loss=F.cross_entropy(out,data.y)
                total_loss+=loss.item()
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(data.y.cpu().tolist())
    return total_loss,preds,labels
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/media/volume/boot-vol-k-project/Fairness_graph/Data")
    parser.add_argument("--dataset_name", type=str, default="train_HateXplain_graphs")
    parser.add_argument("--test_dataset_name", type=str, default="balanced_test_HateXplain_graphs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--output_folder", type=str, default="/media/volume/boot-vol-k-project/Fairness_graph/Code/enhancement/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment", type=str, default="balanced")
    args=parser.parse_args()
    device=args.device
    random.seed(42)
    train_dataset=torch.load(f"results/{args.dataset_name}.pt")
    test_dataset=torch.load(f"results/{args.test_dataset_name}.pt")
    pos=[d for d in train_dataset if int(d.y.item())==1]
    neg=[d for d in train_dataset if int(d.y.item())==0]
    print(f"before {len(train_dataset)} hate: {len(pos)}, non-hate: {len(neg)}")
    print(f"test dataset {len(test_dataset)}")
    n=min(len(pos),len(neg))
    balanced=random.sample(pos,n)+random.sample(neg,n)
    random.shuffle(balanced)

    train_dataset=balanced
    print(f"after: {len(train_dataset)}, each class: {n}")
    print(train_dataset[0],train_dataset[1])
    

    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size*2, shuffle=False)


    model=BaselineGNN(in_dim=768,hidden=128,num_classes=2,heads=4,dropout=0.3).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)


    for epoch in range(args.epoch):
        total_loss,_,_=run(model,train_dataloader,opt,device,train=True)
        te_loss, preds, te_labels = run(model, test_dataloader, opt, device, train=False)
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
    
    with open(f"results/{args.experiment}_{args.dataset_name}_result.txt",'a') as f:
        f.write(f"\nEpoch: {args.epoch}, learning_rate: {args.lr}, batch_size: {args.batch_size}, training: {len(train_dataloader)}, valid: {len(test_dataloader)}\n{report}")

    

if __name__=="__main__":
    main()
    