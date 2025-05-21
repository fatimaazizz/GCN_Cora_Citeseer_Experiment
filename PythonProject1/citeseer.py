import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from torch import nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

device = torch.device("cpu")
dataset = Planetoid(root='./data', name='Citeseer')
data = dataset[0].to(device)
os.makedirs("plots_citeseer", exist_ok=True)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

hidden_units_list = [16, 32]
dropout_list = [0.3, 0.5]
learning_rates = [0.01, 0.005]
epochs = 10
results = []
best_model_data = {}

for hidden in hidden_units_list:
    for dropout in dropout_list:
        for lr in learning_rates:
            model = GCN(dataset.num_node_features, hidden, dataset.num_classes, dropout).to(device)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            losses, accuracies = [], []

            for epoch in range(epochs):
                model.train()
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                model.eval()
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                correct = pred[data.test_mask] == data.y[data.test_mask]
                acc = int(correct.sum()) / int(data.test_mask.sum())
                losses.append(loss.item())
                accuracies.append(acc)

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(losses, marker='o')
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.subplot(1, 2, 2)
            plt.plot(accuracies, marker='o', color='green')
            plt.title("Test Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plot_path = f"plots_citeseer/h{hidden}_d{dropout}_lr{lr}.png"
            plt.savefig(plot_path)
            plt.close()

            results.append({
                "Hidden Units": hidden,
                "Dropout": dropout,
                "Learning Rate": lr,
                "Final Accuracy": accuracies[-1],
                "Final Loss": losses[-1],
                "Plot": plot_path
            })

            if len(best_model_data) == 0 or accuracies[-1] > best_model_data["acc"]:
                best_model_data = {
                    "model": model,
                    "pred": pred,
                    "true": data.y,
                    "acc": accuracies[-1]
                }

results_df = pd.DataFrame(results)
results_df.to_csv("citeseer_experiment_results.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Hidden Units", y="Final Accuracy", hue="Dropout", palette="Set2")
plt.title("Accuracy by Hidden Units and Dropout (Citeseer)")
plt.savefig("citeseer_barplot_accuracy_comparison.png")
plt.close()

cm = confusion_matrix(best_model_data["true"][data.test_mask].cpu(), best_model_data["pred"][data.test_mask].cpu())
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Best Model (Citeseer)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("citeseer_best_model_confusion_matrix.png")
plt.close()

test_mask = data.test_mask
y_true_test = best_model_data["true"][test_mask].cpu()
y_pred_test = best_model_data["pred"][test_mask].cpu()
precision, recall, f1, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average=None)
macro = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro')
micro = precision_recall_fscore_support(y_true_test, y_pred_test, average='micro')
weighted = precision_recall_fscore_support(y_true_test, y_pred_test, average='weighted')

metrics_df = pd.DataFrame({
    "Class": list(range(len(precision))),
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})
metrics_df.loc[len(metrics_df)] = ["Macro Avg", *macro[:3]]
metrics_df.loc[len(metrics_df)] = ["Micro Avg", *micro[:3]]
metrics_df.loc[len(metrics_df)] = ["Weighted Avg", *weighted[:3]]
metrics_df.to_csv("citeseer_best_model_metrics.csv", index=False)
