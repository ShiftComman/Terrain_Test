import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MultiTaskGATSoilClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=8, dropout=0.6):
        super(MultiTaskGATSoilClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        
        # 为每个分类任务创建单独的输出层
        self.tl_out = GATConv(hidden_channels * heads, num_classes['TL'], heads=1, concat=False, dropout=dropout)
        self.yl_out = GATConv(hidden_channels * heads, num_classes['YL'], heads=1, concat=False, dropout=dropout)
        self.ts_out = GATConv(hidden_channels * heads, num_classes['TS'], heads=1, concat=False, dropout=dropout)
        self.tz_out = GATConv(hidden_channels * heads, num_classes['TZ'], heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 对每个任务应用单独的输出层
        tl_out = F.log_softmax(self.tl_out(x, edge_index), dim=1)
        yl_out = F.log_softmax(self.yl_out(x, edge_index), dim=1)
        ts_out = F.log_softmax(self.ts_out(x, edge_index), dim=1)
        tz_out = F.log_softmax(self.tz_out(x, edge_index), dim=1)
        
        return tl_out, yl_out, ts_out, tz_out

# 使用示例
# in_channels = X_train.shape[1]
# hidden_channels = 64
# num_classes = {
#     'TL': len(y_train['TL'].unique()),
#     'YL': len(y_train['YL'].unique()),
#     'TS': len(y_train['TS'].unique()),
#     'TZ': len(y_train['TZ'].unique())
# }
# model = MultiTaskGATSoilClassifier(in_channels, hidden_channels, num_classes)