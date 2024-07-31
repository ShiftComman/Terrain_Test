import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def train_multitask_gat(model, X_train, y_train, edge_index, epochs=200, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    x = torch.tensor(X_train.values, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    y_tl = torch.tensor(y_train['TL'].values, dtype=torch.long).to(device)
    y_yl = torch.tensor(y_train['YL'].values, dtype=torch.long).to(device)
    y_ts = torch.tensor(y_train['TS'].values, dtype=torch.long).to(device)
    y_tz = torch.tensor(y_train['TZ'].values, dtype=torch.long).to(device)
    
    data = Data(x=x, edge_index=edge_index, y_tl=y_tl, y_yl=y_yl, y_ts=y_ts, y_tz=y_tz)
    loader = DataLoader([data], batch_size=1)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            out_tl, out_yl, out_ts, out_tz = model(batch.x, batch.edge_index)
            loss = (F.nll_loss(out_tl, batch.y_tl) + 
                    F.nll_loss(out_yl, batch.y_yl) + 
                    F.nll_loss(out_ts, batch.y_ts) + 
                    F.nll_loss(out_tz, batch.y_tz))
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

# 使用示例
# model = train_multitask_gat(model, X_train, y_train, edge_index)