import os
import networkx as nx
import torch
from torch_geometric.nn import Node2Vec
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib

class NodeEmbeddingPredictor:
    def __init__(self, kg_path, icd_matrix_path, variant_matrix_path):
        self.kg = nx.read_gml(kg_path)
        self.filtered_icd_df = pd.read_csv(icd_matrix_path)
        self.variant_df = pd.read_csv(variant_matrix_path, index_col=0)

    def create_node_mapping(self):
        self.node_list = list(self.kg.nodes())
        self.node_to_id = {node: idx for idx, node in enumerate(self.node_list)}
        self.id_to_node = {idx: node for node, idx in self.node_to_id.items()}

    def prepare_edge_index(self):
        edge_index = []
        for edge in self.kg.edges():
            edge_index.append([self.node_to_id[edge[0]], self.node_to_id[edge[1]]])
            edge_index.append([self.node_to_id[edge[1]], self.node_to_id[edge[0]]])
        self.edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def train_node_embeddings(self, embedding_dim=128):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node2vec = Node2Vec(
            self.edge_index_tensor,
            embedding_dim=embedding_dim,
            walk_length=50,
            context_size=20,
            walks_per_node=40,
            p=0.5,
            q=2,
            sparse=True
        ).to(device)

        loader = node2vec.loader(batch_size=64, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.005)

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()

        for epoch in range(1, 51):
            node2vec.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.embeddings = node2vec.embedding.weight.data.cpu().numpy()
        np.save("./node_embeddings.npy", self.embeddings)

    def generate_features_and_labels(self):
        patient_ids = self.filtered_icd_df['PMBB_ID'].tolist()
        gene_protein_nodes = [
            node for node, attrs in self.kg.nodes(data=True)
            if attrs.get('type') == 'gene/protein'
        ]
        patient_indices = [self.node_to_id[pid] for pid in patient_ids]
        gene_indices = [self.node_to_id[gene] for gene in gene_protein_nodes]

        features = []
        labels = []

        for p_idx, patient_id in zip(patient_indices, patient_ids):
            for g_idx, gene in zip(gene_indices, gene_protein_nodes):
                feature = np.concatenate([self.embeddings[p_idx], self.embeddings[g_idx]])
                features.append(feature)
                label = 1 if self.variant_df.loc[patient_id, gene] != 0 else 0
                labels.append(label)

        self.features = np.array(features)
        self.labels = np.array(labels)
        np.save("./features.npy", self.features)
        np.save("./labels.npy", self.labels)

    def train_binary_classifier(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        classifier = LogisticRegression(max_iter=1000, solver='saga')
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, "./logistic_regression_model.pkl")
        y_pred = classifier.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

    def main():
        kg_path = 'C:/bmin520/inference_engine/updated_knowledge_graph.gml'
        icd_matrix_path = 'C:/bmin520/patients_filtered_new/filtered_icd10_matrix.csv'
        variant_matrix_path = 'C:/bmin520/patients_filtered_new/pathogenic_variant_matrix.csv'
        predictor = NodeEmbeddingPredictor(kg_path, icd_matrix_path, variant_matrix_path)
        predictor.create_node_mapping()
        predictor.prepare_edge_index()
        predictor.train_node_embeddings()
        predictor.generate_features_and_labels()
        predictor.train_binary_classifier()

if __name__ == "__main__":
    NodeEmbeddingPredictor.main()
