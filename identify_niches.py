import pandas as pd
import numpy as np
import torch




import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

import plotly.offline as pyo
import optuna.visualization as ov







# Function to load data (modify as per your file structure)
def load_data(coord_file, gene_file):
    spatial_coords = pd.read_csv(coord_file).drop(['spot_ids'], axis=1).values
    gene_expression = pd.read_csv(gene_file)
    spot_ids = gene_expression['spot_ids'].values
    gene_expression = gene_expression.drop(['spot_ids'], axis=1).values
    return spatial_coords, gene_expression, spot_ids

# Load and concatenate data for all samples
coords1, expr1, ids1 = load_data("/N/project/cytassist/Classwork/s1_coords.csv", "/N/project/cytassist/deniche/s1_counts.csv")
coords2, expr2, ids2 = load_data("/N/project/cytassist/Classwork/s2_coords.csv", "/N/project/cytassist/deniche/s2_counts.csv")


all_coords = np.concatenate((coords1, coords2), axis=0)
all_expr = np.concatenate((expr1, expr2), axis=0)
all_spot_ids = np.concatenate((ids1, ids2), axis=0)

# Convert to tensors
spatial_coords_tensor = torch.tensor(all_coords, dtype=torch.float)


from sklearn.preprocessing import StandardScaler

# Assume all_expr is the concatenated gene expression data from all samples
scaler = StandardScaler()
scaled_all_expr = scaler.fit_transform(all_expr)

# Convert the scaled data back to a tensor
gene_expression_tensor = torch.tensor(scaled_all_expr, dtype=torch.float)




def create_sample_specific_edge_index(spatial_coords, k=5, start_idx=0):
    """
    Create an edge index for a specific sample, adjusting indices based on the starting index.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    edge_pairs = []
    for i in range(indices.shape[0]):
        for j in indices[i, :]:
            edge_pairs.append([start_idx + i, start_idx + j])  # Adjust index by start_idx

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return edge_index

# Create edge indices for each sample and concatenate them
edge_index_1 = create_sample_specific_edge_index(coords1, k=5, start_idx=0)
edge_index_2 = create_sample_specific_edge_index(coords2, k=5, start_idx=len(coords1))
edge_index_3 = create_sample_specific_edge_index(coords3, k=5, start_idx=len(coords1) + len(coords2))
edge_index_4 = create_sample_specific_edge_index(coords3, k=5, start_idx=len(coords1) + len(coords2) + len(coords3) )


combined_edge_index = torch.cat((edge_index_1, edge_index_2, edge_index_3, edge_index_4), dim=1)





import torch
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors



# Example usage
# spatial_coords = np.array([[x1, y1], [x2, y2], ...])
edge_index = combined_edge_index 



class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.decoder = torch.nn.Linear(output_dim, num_features)  # Decoder layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        embeddings = self.conv2(x, edge_index)
        reconstructed_x = self.decoder(embeddings)  # Decode the embeddings
        return embeddings, reconstructed_x

def reconstruction_loss(reconstructed_data, original_data):
    return F.mse_loss(reconstructed_data, original_data)


num_features = gene_expression_tensor.shape[1]

import optuna
from torch_geometric.data import Data
graph_data = Data(x=gene_expression_tensor, edge_index=edge_index)


num_nodes = graph_data.x.size(0)  # Total number of nodes
val_ratio = 0.2  # 20% of nodes for validation
num_val_nodes = int(num_nodes * val_ratio)

# Randomly select validation nodes
val_nodes = np.random.choice(num_nodes, num_val_nodes, replace=False)
train_nodes = [node for node in range(num_nodes) if node not in val_nodes]


train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_nodes] = True

val_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask[val_nodes] = True

graph_data.train_mask = train_mask
graph_data.val_mask = val_mask


def calculate_validation_loss(model, graph_data):
    model.eval()
    with torch.no_grad():
        _, reconstructed_data = model(graph_data)
        val_loss = reconstruction_loss(reconstructed_data[graph_data.val_mask], graph_data.x[graph_data.val_mask])
    return val_loss.item()

def objective(trial):
    # Hyperparameters to be optimized by Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    output_dim = trial.suggest_categorical('output_dim', [32, 64, 128, 256])

    # Create and train your model using these hyperparameters
    model = GCN(num_features, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        _, reconstructed_data = model(graph_data)
        loss = reconstruction_loss(reconstructed_data[graph_data.train_mask], graph_data.x[graph_data.train_mask])
        loss.backward()
        optimizer.step()

    # Return the metric of interest, e.g., validation loss
    val_loss = calculate_validation_loss(model, graph_data)
    return val_loss

# Running the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Adjust the number of trials as needed

print("Best hyperparameters: ", study.best_params)



best_value = study.best_value
print('Best value:', best_value)



# Plot the optimization history
fig = ov.plot_optimization_history(study)

# Save the figure as an HTML file
pyo.plot(fig, filename='optimization_history_min_ST_2samples.html', auto_open=False)


fig = ov.plot_parallel_coordinate(study)

pyo.plot(fig, filename='parallel_coordinate_min_ST_2samples.html', auto_open=False)



best_trial = study.best_trial







# Retrieve best hyperparameters
best_params = study.best_params
best_learning_rate = best_params["learning_rate"]
best_hidden_dim = best_params["hidden_dim"]
best_output_dim = best_params["output_dim"]

# Create final model with best hyperparameters
model = GCN(num_features=num_features, hidden_dim=best_hidden_dim, output_dim=best_output_dim)
model.eval()
with torch.no_grad():
    graph_data = Data(x=gene_expression_tensor, edge_index=edge_index)
    embeddings, _ = model(graph_data)
    embeddings = embeddings.detach().numpy()

# KMeans clustering
kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as per your data
clusters = kmeans.fit_predict(embeddings)


spot_ids_1 = ids1.tolist()
spot_ids_2 = ids2.tolist()


all_spot_ids = spot_ids_1 + spot_ids_2
df = pd.DataFrame({
    'Spot_ID': all_spot_ids,
    'Cluster': clusters
})

df.to_csv('combined_spot_clusters_2_samples.csv', index=False)
