import pandas as pd
import networkx as nx
import logging
import os
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import multiprocessing
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer


# Data paths (This was programmed on a Windows machine, adjust as needed)
kg_path = "C:/bmin520/inference_engine/knowledge_graph.gml"
patients_folder = "C:/bmin520/patients_filtered_new"
ie_path = "C:/bmin520/inference_engine"
filtered_icd_matrix_path = os.path.join(patients_folder, "filtered_icd10_matrix.csv")
pathogenic_variant_matrix_path = os.path.join(patients_folder, "pathogenic_variant_matrix.csv")
sampled_patient_ids_path = os.path.join(patients_folder, "sampled_patient_ids.csv")

# Output paths
filtered_icd_output_path = os.path.join(patients_folder, "filtered_icd_df.csv")
selected_patients_icd_output_path = os.path.join(patients_folder, "selected_patients_icd_df.csv")
associated_genes_output_path = os.path.join(patients_folder, "associated_genes.json")
ranked_genes_output_path = os.path.join(ie_path, "ranked_genes_to_check.csv")
best_model_path = os.path.join(ie_path, "best_model.pth")
metrics_plot_path = os.path.join(ie_path, "training_metrics.png")

# Logger for good luck
log_path = os.path.join(patients_folder, "patient_processing.log")
logging.basicConfig(
    filename=log_path,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Load the knowledge graph
kg = nx.read_gml(kg_path)
logging.info(f"Loaded KG with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges")
node_types = nx.get_node_attributes(kg, 'type')
gene_protein_nodes = [node for node, typ in node_types.items() if typ and typ.lower() == 'gene/protein']
disease_nodes = [node for node, typ in node_types.items() if typ and typ.lower() == 'disease']
logging.info(f"KG contains {len(gene_protein_nodes)} gene/protein nodes and {len(disease_nodes)} disease nodes.")

# Extract and clean ICD-10 codes from the KG
kg_icd10_codes = [
    node for node, attrs in kg.nodes(data=True)
    if attrs.get('type') == 'disease' and node[0].isalpha()
]
kg_icd10_codes_clean = [code.replace('.', '') for code in kg_icd10_codes]
icd10_mapping = {code.replace('.', ''): code for code in kg_icd10_codes}
logging.info(f"Extracted {len(kg_icd10_codes_clean)} ICD10 codes from KG")

try:
    icd_df = pd.read_csv(filtered_icd_matrix_path)
    logging.info(f"Loaded Filtered ICD Matrix with shape {icd_df.shape}.")
except Exception as e:
    logging.error(f"Error loading Filtered ICD Matrix: {e}")
    raise e

try:
    variant_df = pd.read_csv(pathogenic_variant_matrix_path)
    logging.info(f"Loaded Pathogenic Variant Matrix shape {variant_df.shape}.")
    logging.debug(f"Pathogenic Variant Matrix columns: {list(variant_df.columns[:5])} ...")
    logging.debug(f"Sample Variant data:\n{variant_df.head()}")
except Exception as e:
    logging.error(f"Error loading Pathogenic Variant Matrix: {e}")
    raise e

try:
    sampled_ids_df = pd.read_csv(sampled_patient_ids_path)
    patient_ids = sampled_ids_df['PMBB_ID'].dropna().unique().tolist()
    logging.info(f"Loaded {len(patient_ids)} sampled patient IDs.")
    logging.debug(f"Sampled Patient IDs: {patient_ids[:5]} ...")
except Exception as e:
    logging.error(f"Error loading sampled patient IDs: {e}")
    raise e

# Remove periods from ICD-10 codes in the DataFrame columns
icd_df.columns = [col.replace('.', '') for col in icd_df.columns]

# Filter ICD-10 codes to include only those present in the KG
filtered_icd10_codes = [icd10_mapping[code] for code in kg_icd10_codes_clean if code in icd_df.columns]
filtered_icd_df = icd_df[['PMBB_ID'] + filtered_icd10_codes]

output_path = 'C:/bmin520/patients_filtered_new/filtered_icd10_kg.csv'
filtered_icd_df.to_csv(output_path, index=False)
logging.info(f"Saved filtered ICD-10 matrix with KG codes to {output_path}.")