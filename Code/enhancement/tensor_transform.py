import torch

dataset = torch.load("results/train_HateXplain_graphs.pt")

for data in dataset:
    if not torch.is_tensor(data.x):
        data.x = torch.tensor(data.x, dtype=torch.float)
    # 혹시 edge_attr나 다른 것도 리스트면 같이 변환
    if hasattr(data, "edge_attr") and data.edge_attr is not None and not torch.is_tensor(data.edge_attr):
        data.edge_attr = torch.tensor(data.edge_attr, dtype=torch.float)

torch.save(dataset, "results/train_HateXplain_graphs.pt")   # 덮어쓰기
print("converted and saved")