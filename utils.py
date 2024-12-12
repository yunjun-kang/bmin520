import igraph as ig
import os
from glob import glob
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import networkx as nx
import icd10
from sentence_transformers import SentenceTransformer


# Load nodes from PrimeKG.
def load_nodes(primekg_dir: str) -> pd.DataFrame:
    nodes_path = os.path.join(primekg_dir, "nodes.csv")
    return pd.read_csv(nodes_path, low_memory=False)


# Load edges from PrimeKG.
def load_edges(primekg_dir: str) -> pd.DataFrame:
    edges_path = os.path.join(primekg_dir, "edges.csv")
    return pd.read_csv(edges_path, low_memory=False)


# Select nodes that cointain the given keywords in their node_name.
def filter_nodes_by_keywords(nodes: pd.DataFrame, keywords: list) -> pd.DataFrame:
    mask = nodes['node_name'].str.contains('|'.join(keywords), case=False, na=False)
    return nodes[mask]


# Build ig from nodes and edges dataframes
def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> ig.Graph:
    nodes = nodes.reset_index(drop=True)
    node_mapping = {original_index: new_index for new_index, original_index in enumerate(nodes['node_index'])}
    edges = edges.copy()
    edges['x_index'] = edges['x_index'].map(node_mapping)
    edges['y_index'] = edges['y_index'].map(node_mapping)
    edges = edges.dropna(subset=['x_index', 'y_index'])
    edges['x_index'] = edges['x_index'].astype(int)
    edges['y_index'] = edges['y_index'].astype(int)

    g = ig.Graph()
    g.add_vertices(len(nodes))
    edge_tuples = list(zip(edges['x_index'], edges['y_index']))
    g.add_edges(edge_tuples)

    return g


# Get nodes within given distance from the start_nodes in the graph
def get_nodes_within_distance(g: ig.Graph, start_nodes: list, max_distance: int) -> set:
    distances = g.shortest_paths(source=start_nodes, target=None)
    nodes_to_keep = set()
    for dist_list in distances:
        for idx, d in enumerate(dist_list):
            if d <= max_distance and d != float('inf'):
                nodes_to_keep.add(idx)
    return nodes_to_keep


# Subset the graph to given set of nodes_to_keep
def subset_graph(nodes: pd.DataFrame, edges: pd.DataFrame, nodes_to_keep: set) -> (pd.DataFrame, pd.DataFrame):
    sub_nodes = nodes[nodes['node_index'].isin(nodes_to_keep)]
    sub_edges = edges[(edges['x_index'].isin(nodes_to_keep)) & (edges['y_index'].isin(nodes_to_keep))]
    return sub_nodes, sub_edges


# Save subgraph nodes and edges
def save_subgraph(sub_nodes: pd.DataFrame, sub_edges: pd.DataFrame, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sub_nodes.to_csv(os.path.join(out_dir, "subgraph_nodes.csv"), index=False)
    sub_edges.to_csv(os.path.join(out_dir, "subgraph_edges.csv"), index=False)


# Create a new edge in the dataframe
def create_new_edge(x_id: str, x_type: str, x_name: str, x_source: str, y_id: str, y_type: str, y_name: str, y_source: str, relation: str, display_relation: str) -> pd.DataFrame:
    data = {
        'relation': [relation],
        'display_relation': [display_relation],
        'x_id': [x_id],
        'x_type': [x_type],
        'x_name': [x_name],
        'x_source': [x_source],
        'y_id': [y_id],
        'y_type': [y_type],
        'y_name': [y_name],
        'y_source': [y_source]
    }
    return pd.DataFrame(data)


# Filter PMBB datasets by patients with non-zero value in given ICD10 code, excludes Q prefix by default
def filter_pmbb_by_icd10(icd10_code, pmbb_dir, output_dir, num_patients=500, seed=None, exclude_prefix='Q'):
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    icd10_file = os.path.join(pmbb_dir, 'PMBB-Release-2020-2.3_phenotype_icd-10-matrix.txt')
    icd10_df = pd.read_csv(icd10_file, sep='\t', low_memory=False)
    icd10_df.fillna(0, inplace=True)  # Replace NaNs with 0

    # Exclude ICD codes starting with Q
    icd10_columns = [col for col in icd10_df.columns if col != 'PMBB_ID' and not col.startswith(exclude_prefix)]
    if icd10_code in icd10_columns:
        selected_patients = icd10_df[icd10_df[icd10_code] > 0]['PMBB_ID'].unique().tolist()
    else:
        selected_patients = []

    if len(selected_patients) < num_patients:
        print(f"Only {len(selected_patients)} eligible patients available")
        sampled_ids = selected_patients
    else:
        sampled_ids = random.sample(selected_patients, num_patients)

    sampled_ids_file = os.path.join(output_dir, 'sampled_patient_ids.csv')
    pd.DataFrame(sampled_ids, columns=['PMBB_ID']).to_csv(sampled_ids_file, index=False)

    # Filter all datasets using sampled patient IDs
    files = glob(os.path.join(pmbb_dir, '*'))
    unmatched_counts = {}

    for file_path in files:
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            try:
                data = pd.read_csv(file_path, sep='\t', low_memory=False)
                if 'PMBB_ID' in data.columns:
                    filtered_data = data[data['PMBB_ID'].isin(sampled_ids)]
                    if filtered_data.empty:
                        unmatched_counts[file_name] = len(sampled_ids)
                    else:
                        unmatched_counts[file_name] = len(sampled_ids) - len(filtered_data)
                    filtered_data.to_csv(os.path.join(output_dir, file_name), sep='\t', index=False)
            except Exception as e:
                print(f"Error filtering: {file_name}: {e}")

    return sampled_ids_file


def filter_dataset_by_ids(dataset_path, patient_ids, output_path):
    try:
        data = pd.read_csv(dataset_path, sep='\t', low_memory=False)

        if 'PMBB_ID' in data.columns:
            filtered_data = data[data['PMBB_ID'].isin(patient_ids)]
            filtered_data.to_csv(output_path, sep='\t', index=False)
        else:
            print(f"No PMBB_ID column in {dataset_path}.")

    except Exception as e:
        print(f"Error filtering {dataset_path}: {e}")



def get_non_zero_icd10_codes(icd10_file_path, selected_patient_ids, exclude_prefix='Q') -> dict:
    icd10_data = pd.read_csv(icd10_file_path, sep='\t', low_memory=False)
    icd10_data.fillna(0, inplace=True)  # Replace NaNs with 0

    # Identify ICD-10 code column (excluding 'Q' codes
    icd10_columns = [col for col in icd10_data.columns if col != 'PMBB_ID' and not col.startswith(exclude_prefix)]

    # Filter data for selected patients
    filtered_data = icd10_data[icd10_data['PMBB_ID'].isin(selected_patient_ids)]

    patient_icd_mapping = {}
    for patient_id in selected_patient_ids:
        if patient_id in filtered_data['PMBB_ID'].values:
            patient_row = filtered_data[filtered_data['PMBB_ID'] == patient_id].iloc[0]
            # Get ICD-10 codes with non-zero entries
            non_zero_icds = patient_row[icd10_columns][patient_row[icd10_columns] > 0].index.tolist()
            patient_icd_mapping[patient_id] = list(set(non_zero_icds))  # Remove duplicates
        else:
            patient_icd_mapping[patient_id] = []

    return patient_icd_mapping


def get_icd_descriptions(codes):
    descriptions = []
    for code in codes:
        icd = icd10.find(code)
        if icd:
            descriptions.append({
                "icd_code": code,
                "description": icd.description
            })
    return pd.DataFrame(descriptions)


def sort_icd_codes(icd_codes):
    icd9 = sorted([code for code in icd_codes if '.' in code and code[0].isdigit() and int(code.split('.')[0]) < 10])
    icd10 = sorted([code for code in icd_codes if code not in icd9])
    return icd9 + icd10


def process_matrix_file(file_path, patient_ids):
    data = pd.read_csv(file_path, sep='\t', low_memory=False)
    filtered_data = data[data['PMBB_ID'].isin(patient_ids)]
    return filtered_data.set_index('PMBB_ID').to_dict(orient='index')


def process_phenotype_file(file_path, patient_ids):
    data = pd.read_csv(file_path, sep='\t', low_memory=False)
    return data[data['PMBB_ID'].isin(patient_ids)]


def extract_patient_data(pmbb_dir, selected_patient_ids):
    patient_data = {}

    matrix_files = [
        "PMBB-Release-2020-2.3_phenotype_icd-9-matrix.txt",
        "PMBB-Release-2020-2.3_phenotype_icd-10-matrix.txt",
        "PMBB-Release-2020-2.3_phenotype_PheCode-matrix.txt",
    ]
    for matrix_file in matrix_files:
        file_path = os.path.join(pmbb_dir, matrix_file)
        observations = process_matrix_file(file_path, selected_patient_ids)
        patient_data[matrix_file] = observations

    phenotype_files = [
        "PMBB-Release-2020-2.3_phenotype_labs-A1C.txt",
        "PMBB-Release-2020-2.3_phenotype_medications.txt",
        "PMBB-Release-2020-2.3_phenotype_vitals-BP.txt",
    ]
    for phenotype_file in phenotype_files:
        file_path = os.path.join(pmbb_dir, phenotype_file)
        observations = process_phenotype_file(file_path, selected_patient_ids)
        patient_data[phenotype_file] = observations

    return patient_data


def combine_patient_data(patient_data):
    combined_data = {}
    for file_name, data in patient_data.items():
        if isinstance(data, pd.DataFrame):
            for patient_id in data['PMBB_ID'].unique():
                if patient_id not in combined_data:
                    combined_data[patient_id] = {}
                combined_data[patient_id][file_name] = data[data['PMBB_ID'] == patient_id].to_dict(orient='records')
        elif isinstance(data, dict):
            for patient_id, observations in data.items():
                if patient_id not in combined_data:
                    combined_data[patient_id] = {}
                combined_data[patient_id][file_name] = observations
    return combined_data

# def enforce_uniform_sampling(pmbb_dir, sampled_ids, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     files = glob(os.path.join(pmbb_dir, '*'))
#     unmatched_counts = {}
#
#     for file_path in files:
#         if os.path.isfile(file_path):
#             file_name = os.path.basename(file_path)
#             output_path = os.path.join(output_dir, file_name)
#
#             try:
#                 data = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#                 if 'PMBB_ID' in data.columns:
#                     filtered_data = data[data['PMBB_ID'].isin(sampled_ids)]
#                     if filtered_data.empty:
#                         unmatched_counts[file_name] = len(sampled_ids)
#                     else:
#                         unmatched_counts[file_name] = len(sampled_ids) - len(filtered_data)
#
#                     filtered_data.to_csv(output_path, sep='\t', index=False)
#                 else:
#                     unmatched_counts[file_name] = len(sampled_ids)
#
#             except Exception as e:
#                 print(f"Error processing {file_name}: {e}")


# # Go through pmbb_dir and find patients with specified ICD codes and return a list of patient IDs and filtered icd-9 and icd-10 files
# def filter_pmbb_by_icd(icd9, icd10, pmbb_dir, output_dir, num_patients=500, seed=None):
#     if seed is not None:
#         random.seed(seed)
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     icd9_file = os.path.join(pmbb_dir, 'PMBB-Release-2020-2.3_phenotype_icd-9-matrix.txt')
#     icd10_file = os.path.join(pmbb_dir, 'PMBB-Release-2020-2.3_phenotype_icd-10-matrix.txt')
#     icd9_df = pd.read_csv(icd9_file, sep='\t', low_memory=False)
#     icd10_df = pd.read_csv(icd10_file, sep='\t', low_memory=False)
#     icd10_df.columns = icd10_df.columns.astype(str)
#
#     selected_patients = set()
#     if icd9 in icd9_df.columns:
#         selected_patients.update(icd9_df.loc[icd9_df[icd9] > 0, 'PMBB_ID'].unique())
#     if icd10 in icd10_df.columns:
#         selected_patients.update(icd10_df.loc[icd10_df[icd10] > 0, 'PMBB_ID'].unique())
#
#     selected_patients = list(selected_patients)
#     print(f"Found {len(selected_patients)} patients with specified ICD codes.")
#
#     if len(selected_patients) < num_patients:
#         print(f"Only {len(selected_patients)} eligible patients available")
#         sampled_ids = selected_patients
#         sampled_ids = selected_patients
#     else:
#         sampled_ids = random.sample(selected_patients, num_patients)
#
#     sampled_ids_file = os.path.join(output_dir, 'sampled_patient_ids.csv')
#     pd.DataFrame(sampled_ids, columns=['PMBB_ID']).to_csv(sampled_ids_file, index=False)
#
#     # Filter all datasets using the sampled patient IDs
#     files = glob(os.path.join(pmbb_dir, '*'))
#     unmatched_counts = {}
#
#     for file_path in files:
#         if os.path.isfile(file_path):
#             file_name = os.path.basename(file_path)
#
#             try:
#                 data = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#                 if 'PMBB_ID' in data.columns:
#                     filtered_data = data[data['PMBB_ID'].isin(sampled_ids)]
#                     if filtered_data.empty:
#                         unmatched_counts[file_name] = len(sampled_ids)
#                     else:
#                         unmatched_counts[file_name] = len(sampled_ids) - len(filtered_data)
#
#                     filtered_data.to_csv(os.path.join(output_dir, file_name), sep='\t', index=False)


# def filter_pmbb_by_icd(icd9, icd10, pmbb_dir, output_dir, num_patients=500, seed=None):
#     if seed is not None:
#         random.seed(seed)
#     os.makedirs(output_dir, exist_ok=True)
#     icd9_file = os.path.join(pmbb_dir, 'PMBB-Release-2020-2.3_phenotype_icd-9-matrix.txt')
#     icd10_file = os.path.join(pmbb_dir, 'PMBB-Release-2020-2.3_phenotype_icd-10-matrix.txt')
#     icd9_df = pd.read_csv(icd9_file, sep='\t', low_memory=False)
#     icd10_df = pd.read_csv(icd10_file, sep='\t', low_memory=False)
#     icd10_df.columns = icd10_df.columns.astype(str)
#     selected_patients = set()
#     if icd9 in icd9_df.columns:
#         selected_patients.update(icd9_df.loc[icd9_df[icd9] > 0, 'PMBB_ID'].unique())
#     if icd10 in icd10_df.columns:
#         selected_patients.update(icd10_df.loc[icd10_df[icd10] > 0, 'PMBB_ID'].unique())
#
#     selected_patients = list(selected_patients)
#     if len(selected_patients) < num_patients:
#         print(f"Warning: Only {len(selected_patients)} eligible patients available. Returning all.")
#         sampled_ids = selected_patients
#     else:
#         sampled_ids = random.sample(selected_patients, num_patients)
#
#     sampled_ids_file = os.path.join(output_dir, 'sampled_patient_ids.csv')
#     pd.DataFrame(sampled_ids, columns=['PMBB_ID']).to_csv(sampled_ids_file, index=False)
#     files = glob(os.path.join(pmbb_dir, '*'))
#     unmatched_counts = {}
#
#     for file_path in files:
#         if os.path.isfile(file_path):
#             file_name = os.path.basename(file_path)
#
#             try:
#                 data = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#                 if 'PMBB_ID' in data.columns:
#                     filtered_data = data[data['PMBB_ID'].isin(sampled_ids)]
#                     if filtered_data.empty:
#                         unmatched_counts[file_name] = len(sampled_ids)
#                     else:
#                         unmatched_counts[file_name] = len(sampled_ids) - len(filtered_data)
#
#                     # Save filtered data
#                     filtered_data.to_csv(os.path.join(output_dir, file_name), sep='\t', index=False)
#
#             except Exception as e:
#                 print(f"Error {e}")
#     return sampled_ids_file