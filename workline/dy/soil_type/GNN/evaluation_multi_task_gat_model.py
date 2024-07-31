from sklearn.metrics import accuracy_score, classification_report

def evaluate_multitask_gat(model, X_test, y_test, edge_index, label_encoders):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    x = torch.tensor(X_test.values, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    
    with torch.no_grad():
        out_tl, out_yl, out_ts, out_tz = model(x, edge_index)
    
    pred_tl = out_tl.max(1)[1].cpu().numpy()
    pred_yl = out_yl.max(1)[1].cpu().numpy()
    pred_ts = out_ts.max(1)[1].cpu().numpy()
    pred_tz = out_tz.max(1)[1].cpu().numpy()
    
    for task, pred in zip(['TL', 'YL', 'TS', 'TZ'], [pred_tl, pred_yl, pred_ts, pred_tz]):
        print(f"{task} Classification Report:")
        true_labels = label_encoders[task].inverse_transform(y_test[task])
        pred_labels = label_encoders[task].inverse_transform(pred)
        print(classification_report(true_labels, pred_labels))
        
        acc = accuracy_score(y_test[task], pred)
        print(f"{task} Accuracy: {acc:.4f}\n")

# 使用示例
# evaluate_multitask_gat(model, X_test, y_test, edge_index_test, label_encoders)