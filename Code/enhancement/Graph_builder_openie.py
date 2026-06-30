from __future__ import annotations
import argparse, json, pickle, os
from collections import Counter
from datasets import load_dataset 
import torch
from torch_geometric.data import Data
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None, help="only process first N posts")
ap.add_argument("--out", type=str, default="hatexplain_openie_graphs.pkl")
ap.add_argument("--extraction", type=str, default="spacy")
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--n_process", type=int, default=4)    
ap.add_argument("--checkpoint", type=int, default=2000)
args = ap.parse_args()
LABEL2ID = {"hatespeech": 0, "hate speech": 0, "hate": 0,
            "offensive": 1,
            "normal": 2}

def majority_label(annotators):
    labels = []
    for lab in annotators["label"]:
        lab = str(lab).lower()
        if lab in LABEL2ID:
            labels.append(LABEL2ID[lab])
        elif lab.isdigit():            
            labels.append(int(lab))
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def load_hatexplain(limit: int | None = None):
   
    from datasets import load_dataset
    ds = load_dataset("hatexplain", split="train", trust_remote_code=True)
    print(len(ds))
    n = 0
    for ex in ds:
        text = " ".join(ex["post_tokens"])
        label = majority_label(ex["annotators"])
        if label is None:
            continue
        yield {"id": ex["id"], "text": text, "label": label}
        n += 1
        if limit and n >= limit:
            break

def triples_to_graph(text: str, triples: list[dict]):
   
    if not triples:
        toks = text.split()
        edges = [(i, i + 1) for i in range(len(toks) - 1)]
        return toks, edges, True
 
    node_texts: list[str] = []
    index: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
 
    def nid(span: str) -> int:
        key = span.strip().lower()
        if key == "":
            key = "<empty>"
        if key not in index:
            index[key] = len(node_texts)
            node_texts.append(key)
        return index[key]
 
    for t in triples:
        s, r, o = nid(t["subject"]), nid(t["relation"]), nid(t["object"])
        edges.append((s, r))
        edges.append((r, o))
    return node_texts, edges, False


def to_pyg(node_texts, edges, label) -> Data:
   
    num_nodes = len(node_texts)
    if len(edges) == 0:
        edges = [(i, i) for i in range(num_nodes)]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ei = torch.cat([ei, ei.flip(0)], dim=1)               # undirected
    data = Data(edge_index=ei, num_nodes=num_nodes)
    data.node_texts = node_texts                          # list[str], embed later
    data.y = torch.tensor([label], dtype=torch.long)
    return data

class agent:
    _client = None
    _nlp = None  
    @classmethod
    def get(cls):
        if args.extraction=="openie":
            if cls._client is None:
                from openie import StanfordOpenIE
                cls._client = StanfordOpenIE(properties={"openie.affinity_probability_cap": 1 / 3})
            return cls._client
        elif args.extraction=="spacy":
            if cls._nlp is None:
                import spacy
                cls._nlp=spacy.load("en_core_web_sm",disable=["ner","lemmatizer"])
            return cls._nlp
 
    @classmethod
    def triples(cls, text: str) -> list[dict]:
        try:
            return cls.get().annotate(text)
        except Exception:
            return []

def dependency_extract(utterance: str):
   
    doc = agent.get()(utterance)
 
    node_texts = [tok.text for tok in doc]          # node i = i-th token
    edges, edge_labels = [], []
 
    for tok in doc:
        head_idx = tok.head.i                        # index of the token it depends on
        if head_idx == tok.i:                        # ROOT -> points to itself
            edges.append((tok.i, tok.i))
            edge_labels.append("root")
        else:
            edges.append((tok.i, head_idx))          # child -> head
            edge_labels.append(tok.dep_)             # e.g. 'nsubj', 'det', 'amod'
 
    return node_texts, edges, edge_labels

def build_dependency_graph(nodes, edges, edg_labels, label: int) -> Data:
 
    if len(node_texts) == 0:                          # empty string guard
        node_texts, edges, edge_labels = ["<empty>"], [(0, 0)], ["root"]
 
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ei = torch.cat([ei, ei.flip(0)], dim=1)           # undirected (both directions)
 
    data = Data(edge_index=ei, num_nodes=len(node_texts))
    data.node_texts = node_texts                      # list[str], embed later with HateBERT
    # edge labels duplicated for the flipped half so they line up with edge_index columns
    data.edge_type = edge_labels + edge_labels        # optional: for RGCN typed edges
    data.y = torch.tensor([label], dtype=torch.long)
    return data

def main():

    rows=load_dataset(split="train",limit=args.limit)

    print(f"The length of data: {len(rows)}")

    nlp=spacy.load("en_core_web_sm",disable=["ner","lemmatizer"])

    texts=[r["text"] for r in rows]
    graphs, stats = [], Counter()
    for i, ex in enumerate(load_hatexplain(limit=args.limit)):
        if args.extraction=="openie":
            triples = agent.triples(ex["text"])
            node_texts, edges, fb = triples_to_graph(ex["text"], triples)
            g = to_pyg(node_texts, edges, ex["label"])
            g.post_id = ex["id"]
            graphs.append(g)
    
            stats["total"] += 1
            stats["fallback" if fb else "openie"] += 1
            stats["nodes"] += len(node_texts)
            if len(triples)>2:
                print(g)                    # Data(edge_index=[2, ...], num_nodes=..., y=[1], ...)
                print(g.node_texts)          # ['no', 'liberal', ...]
                print(g.edge_index)          # tensor([[0,1,...],[1,2,...]])
                print(g.y)                   # tensor([라벨])
            if i < 5:   
                print(f"\n[{ex['id']}] label={ex['label']}  text={ex['text'][:80]}")
                print(f"   triples={len(triples)}  nodes={len(node_texts)}  fallback={fb}")
                print(f"   node_texts={node_texts[:8]}{' ...' if len(node_texts) > 8 else ''}")
        elif args.extraction=="spacy":
            nodes, edges, labels=dependency_extract(ex["text"])
            print("tokens (nodes):", nodes)
            print("\ndependency edges (child -> head : relation):")
            for (c, h), lab in zip(edges, labels):
                print(f"  {nodes[c]:>12} -> {nodes[h]:<12} : {lab}")
            g = build_dependency_graph(nodes, edges, labels, ex["label"])
            g.post_id = ex["id"]
            graphs.append(g)
            stats["total"] += 1
            stats["nodes"] += len(nodes)
 
    # save
    with open(args.out, "wb") as f:
        pickle.dump(graphs, f)
    n = max(stats["total"], 1)
    print("\n==== summary ====")
    print(f"posts processed : {stats['total']}")
    print(f"avg nodes/graph : {stats['nodes']/n:.1f}")
    print(f"saved -> {os.path.abspath(args.out)}")
    if args.extraction=="openie":
        print(f"OpenIE success  : {stats['openie']} ({100*stats['openie']/n:.1f}%)")
        print(f"fallback (chain): {stats['fallback']} ({100*stats['fallback']/n:.1f}%)")
    
 
if __name__ == "__main__":
    main()
 