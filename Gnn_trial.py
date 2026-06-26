import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
import numpy as np

class Config:
    
    dataset_name   = "tweet_eval"
    dataset_subset = "hate"

    text_model     = "roberta-base"
    hidden_dim     = 256
    num_heads      = 4
    dropout        = 0.3
    num_classes    = 2
    
    sim_threshold = 0.99
    
    lr             = 1e-3
    weight_decay   = 1e-4
    max_epochs     = 50
    batch_size     = 32
    num_workers    = 2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()


class TweetDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length=128):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        encoding=self.tokenizer(self.texts[idx],padding='max_length',truncation=True,max_length=self.max_length,return_tensors="pt")

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }
class TextEmbedder:
    def __init__(self,model_name=cfg.text_model,device=cfg.device):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModel.from_pretrained(model_name).to(device)
        self.device=device
        self.model.eval()

    @torch.no_grad()
    def embed(self,texts,labels):
        
        dataset=TweetDataset(texts,labels,self.tokenizer)
        dataloader=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)

        all_embeddings=[]
        all_labels=[]

        for i,batch in enumerate(dataloader):
            input_ids=batch['input_ids'].to(self.device)
            attention_mask=batch['attention_mask'].to(self.device)

            output=self.model(input_ids=input_ids,attention_mask=attention_mask)

            #embeddings=output.last_hidden_state[:,0,:]
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch['label'])

            if i%10 ==0:
                print(f"{i}/{len(dataloader)}")

        return torch.cat(all_embeddings,dim=0), torch.cat(all_labels,dim=0)

def build_graph(embeddings, labels, threshold=0.99):
    embeddings = embeddings.detach().cpu().float()
    norm_emb = F.normalize(embeddings, dim=1)
    print(f"norm_emb device: {norm_emb.device}")
    # 메모리 효율을 위해 배치로 계산
    batch_size = 100
    n = len(embeddings)
    src, dst = [], []
    
    print("그래프 구성 중...")
    for i in range(0, n, batch_size):
        batch = norm_emb[i:i+batch_size]
        sim   = torch.mm(batch, norm_emb.T)  # cosine similarity
        
        rows, cols = torch.where(sim > threshold)
        rows = rows + i
        
        # 자기 자신 제외
        mask = rows != cols
        src.extend(rows[mask].tolist())
        dst.extend(cols[mask].tolist())
        
        if i % 1000 == 0:
            print(f"  {i}/{n}")
        del batch,sim
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=embeddings, edge_index=edge_index, y=labels)
    
    print(f"\n=== Graph 구조 ===")
    print(f"노드 수:     {data.num_nodes}")
    print(f"엣지 수:     {data.num_edges}")
    print(f"평균 degree: {data.num_edges/data.num_nodes:.1f}")
    
    return data

class HatespeechGNN(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 768

        self.conv1 = GATv2Conv(
            in_channels,
            cfg.hidden_dim,
            heads=cfg.num_heads,
            dropout=cfg.dropout,
            concat=True
        )
        self.conv2 = GATv2Conv(
            cfg.hidden_dim * cfg.num_heads,
            cfg.hidden_dim,
            heads=1,
            dropout=cfg.dropout,
            concat=False
        )
        # why 2 layers?
        self.bn1 = nn.BatchNorm1d(cfg.hidden_dim * cfg.num_heads)
        self.bn2 = nn.BatchNorm1d(cfg.hidden_dim) #why bath norm?

        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.num_classes)
        )


    def forward(self,x,edge_index):

        x = self.conv1(x, edge_index) #[2* num_edges]
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=cfg.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)

        return self.classifier(x)


class GNNLightning(L.LightningModule):

    def __init__(self,data,train_mask,test_mask):
        super().__init__()
        self.model=HatespeechGNN()
        self.data=data
        self.train_mask=train_mask
        self.test_mask=test_mask

        self.train_f1=torchmetrics.F1Score(task='binary',average='macro')
        self.test_f1=torchmetrics.F1Score(task='binary',average='macro')
        self.test_acc=torchmetrics.Accuracy(task='binary')


    def forward(self,x,edge_index):
        return self.model(x,edge_index) 

    def training_step(self,batch,batch_idx):
        x=self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        y= self.data.y.to(self.device)
        out  = self(x, edge_index)
        loss = F.cross_entropy(
        out[self.train_mask], #mask가 의미하는 건 뭘까
        y[self.train_mask])
        pred = out[self.train_mask].argmax(dim=1)
            
        self.train_f1(pred, y[self.train_mask])
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1',   self.train_f1, prog_bar=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        x          = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        y          = self.data.y.to(self.device)
            
        out  = self(x, edge_index)
        pred = out[self.test_mask].argmax(dim=1)
            
        self.test_f1(pred,  y[self.test_mask])
        self.test_acc(pred, y[self.test_mask])
        self.log('test_f1',  self.test_f1,  prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.max_epochs
            )
        return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        
    def train_dataloader(self): #GNN? or lightning?
        return DataLoader([0], batch_size=1)
        
    def test_dataloader(self):
        return DataLoader([0], batch_size=1)

def main():
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_subset)
    n_train = 1200  # 원래 ~10000
    n_test  = 400  # 원래 ~2970
    
    train_texts  = list(dataset['train']['text'])[:n_train]
    train_labels = list(dataset['train']['label'])[:n_train]
    test_texts   = list(dataset['test']['text'])[:n_test]
    test_labels  = list(dataset['test']['label'])[:n_test]

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    print(f"Hate 비율 (train): {sum(train_labels)/len(train_labels):.2%}")
    print(f"Hate 비율 (test): {sum(test_labels)/len(test_labels):.2%}")

    embedder = TextEmbedder()

    print("Train embedding...")
    train_emb, train_y = embedder.embed(train_texts, train_labels)
    print("Test embedding...")
    test_emb,  test_y  = embedder.embed(test_texts,  test_labels)

    all_emb    = torch.cat([train_emb, test_emb], dim=0)
    all_labels = torch.cat([train_y,   test_y],   dim=0)
    
    data = build_graph(all_emb, all_labels, threshold=cfg.sim_threshold)
    n_train = len(train_texts)
    n_total = len(train_texts) + len(test_texts)
    
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask  = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    test_mask[n_train:]  = True

    model = GNNLightning(data, train_mask, test_mask)

    trainer = L.Trainer(
        max_epochs      = cfg.max_epochs,
        accelerator     = cfg.device,
        log_every_n_steps = 1,
        enable_progress_bar = True,
        callbacks = [
            L.pytorch.callbacks.EarlyStopping(
                monitor='train_f1',
                patience=5,
                mode='max'
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='train_f1',
                mode='max',
                save_top_k=1,
                filename='best-{epoch}-{train_f1:.3f}'
            )
        ]
    )
    trainer.fit(model)

    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out[test_mask].argmax(dim=1).cpu()
        true = data.y[test_mask].cpu()
    
    print("\nClassification Report:")
    print(classification_report(
        true, pred,
        target_names=['non-hate', 'hate']
    ))

if __name__ == "__main__":
    main()