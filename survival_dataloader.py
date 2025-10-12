import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import gc
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd

# Task sampler for survival data
class SimpleSurvivalMetaDataset:
    
    def __init__(self, drug_response_file, patient2ind_file, genotype_files, 
                 class_ind=1, time_col=2, event_col=3):
        """
        Args:
            drug_response_file: CSV with [patient_id, drug, time, event, ...]
            patient2ind_file: CSV mapping patient_id to index
            genotype_files: Dict {'mutation': 'path/to/mutations.txt', 'cna': 'path/to/cna.txt', 'expression': 'path/to/expression.txt'}
            class_ind: Column index for grouping (e.g., drug, cancer subtype, etc.)
            time_col: Column index for survival time
            event_col: Column index for event indicator
        """
        # Load drug response data
        self.drug_response = pd.read_csv(drug_response_file, header=None)
        self.class_ind = class_ind
        self.time_col = time_col
        self.event_col = event_col
        
        # Get unique classes (drugs/treatments)
        self.classes = list(self.drug_response.iloc[:, class_ind].unique())
        self.drug_response_df_grouped = self.drug_response.groupby(by=class_ind)
        
        # Load patient to index mapping
        patient2ind_df = pd.read_csv(patient2ind_file, header=None)
        self.patient2ind = {str(patient): idx for idx, patient in patient2ind_df.itertuples(False)}
        
        # Load genomic data files
        self.genotype_data = {}
        for genotype_type, file_path in genotype_files.items():
            self.genotype_data[genotype_type] = pd.read_csv(file_path, header=None).astype('int32')
        
        # Subset for genes that have all genotype types
        # NOTE: Might not be necessary, but keeps consistency across types 
        common_genes = set(self.genotype_data[next(iter(genotype_files))].columns)
        for genotype_type in genotype_files.keys():
            self.genotype_data[genotype_type] = self.genotype_data[genotype_type][list(common_genes)]
        
        self.genotype_types = list(genotype_files.keys())

    def __len__(self):
        return self.drug_response.shape[0]

    def __getitem__(self, index):
        """Get a single sample with genomic features and survival data"""
        row = self.drug_response.iloc[index]
        
        # Extract basic information
        patient_id = str(row.iloc[0])
        class_type = row.iloc[self.class_ind]
        time = float(row.iloc[self.time_col])
        event = float(row.iloc[self.event_col])
        
        # Get patient index
        patient_ind = self.patient2ind.get(patient_id, 0)
        
        # Extract genomic features for this patient
        genotype_features = {}
        for genotype_type in self.genotype_types:
            # Get binary feature vector for this patient
            features = self.genotype_data[genotype_type].iloc[patient_ind].values
            genotype_features[genotype_type] = torch.tensor(features, dtype=torch.float32)
        
        return {
            'patient_id': patient_id,
            'genotype': genotype_features,
            'class': class_type,
            'time': torch.tensor(time, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32)
        }

class SimpleSurvivalCollator:
    """Simple collator that flattens genomic features"""
    
    def __init__(self, genotype_types):
        self.genotype_types = genotype_types
    
    def __call__(self, batch):
        """Convert batch of samples to MLP-friendly format"""
        if not batch:
            return {}
        
        # Collect survival data
        times = torch.stack([sample['time'] for sample in batch])
        events = torch.stack([sample['event'] for sample in batch])
        
        # Flatten and concatenate genomic features
        all_features = []
        for genotype_type in self.genotype_types:
            # Stack features for this genotype type across batch
            genotype_features = torch.stack([sample['genotype'][genotype_type] for sample in batch])
            all_features.append(genotype_features)
        
        # Concatenate all genomic feature types
        combined_features = torch.cat(all_features, dim=1) if all_features else torch.zeros(len(batch), 100)
        
        return {
            'features': combined_features,
            'time': times,
            'event': events,
            'classes': [sample['class'] for sample in batch],
            'patient_ids': [sample['patient_id'] for sample in batch]
        }

class SimpleSurvivalTaskSampler:
    """Simplified task sampler"""
    
    def __init__(self, dataset, collator, num_classes_per_task, num_support, num_query, rank, world_size, test_class):
        self.dataset = dataset
        self.collator = collator
        self.num_classes_per_task = num_classes_per_task
        self.num_support = num_support
        self.num_query = num_query
        self.rank = rank
        self.world_size = world_size
        self.test_class = test_class
        self.all_classes = list(set(self.dataset.classes) - {self.test_class})
        
        self._shuffle_classes()

    def _shuffle_classes(self):
        """Assign classes to this rank"""
        random.shuffle(self.all_classes)
        
        if self.world_size > 1:
            classes_per_rank = len(self.all_classes) // self.world_size
            start_idx = self.rank * classes_per_rank
            end_idx = start_idx + classes_per_rank
            self.assigned_classes = self.all_classes[start_idx:end_idx]
        else:
            self.assigned_classes = self.all_classes

    def get_n_class_samples(self, class_label, n_samples):
        """Get n samples from a specific class"""
        if class_label not in self.dataset.drug_response_df_grouped.groups:
            return []
            
        indices = self.dataset.drug_response_df_grouped.groups[class_label].tolist()
        random.shuffle(indices)
        indices = indices[:n_samples]
        return [self.dataset[i] for i in indices]

    def sample_task(self):
        """Sample support and query sets"""
        available_classes = [cls for cls in self.assigned_classes 
                           if cls in self.dataset.drug_response_df_grouped.groups]
        
        if len(available_classes) < self.num_classes_per_task:
            selected_classes = available_classes
        else:
            selected_classes = random.sample(available_classes, self.num_classes_per_task)

        support_set = []
        query_set = []

        for class_label in selected_classes:
            total_needed = self.num_support + self.num_query
            class_samples = self.get_n_class_samples(class_label, total_needed)
            
            if len(class_samples) < total_needed:
                split_point = len(class_samples) // 2
                support_samples = class_samples[:split_point]
                query_samples = class_samples[split_point:]
            else:
                support_samples = class_samples[:self.num_support]
                query_samples = class_samples[self.num_support:total_needed]

            support_set.extend(support_samples)
            query_set.extend(query_samples)

        # Convert to batches
        support = self.collator(support_set) if support_set else {}
        query = self.collator(query_set) if query_set else {}

        return support, query

    def get_few_shot(self, class_label, n_samples):
        """Get few-shot data for specific class"""
        class_label = self.test_class  # Ensure we only get data for the test class
        if class_label not in self.dataset.drug_response_df_grouped.groups:
            return {}, {}
            
        indices = self.dataset.drug_response_df_grouped.groups[class_label].tolist()
        
        if len(indices) <= n_samples:
            k_indices = indices
            test_indices = []
        else:
            k_indices = indices[:n_samples]
            test_indices = indices[n_samples:]

        k_data = self.collator([self.dataset[i] for i in k_indices]) if k_indices else {}
        test_data = self.collator([self.dataset[i] for i in test_indices]) if test_indices else {}
        
        return k_data, test_data

# class DistributedMetaTaskSampler:
#     def __init__(self, dataset: PatientResponseDataset, collator: PatientResponseCollator, num_classes_per_task, num_support, num_query, rank, world_size):
#         self.dataset = dataset
#         # TODO: Instantiate default Pytorch dataloader/collator classes (can try just calling dataloader?)
#         self.collator = collator
#         self.num_classes_per_task = num_classes_per_task
#         self.num_support = num_support
#         self.num_query = num_query
#         self.rank = rank
#         self.world_size = world_size
#         self.all_classes = list(self.dataset.classes)

#         # Initial assignment of classes per rank
#         self._shuffle_classes()

#     def _shuffle_classes(self):
#         # Shuffle the classes to provide variability each epoch
#         random.shuffle(self.all_classes)

#         # Split classes into chunks assigned to each GPU
#         self.classes_per_rank = len(self.all_classes) // self.world_size
#         start_idx = self.rank * self.classes_per_rank
#         end_idx = start_idx + self.classes_per_rank
#         self.assigned_classes = self.all_classes[start_idx:end_idx]

#     def get_n_class_samples(self, class_label, n_samples):
#         indices = self.dataset.drug_response_df_grouped.groups[class_label].tolist()
#         random.shuffle(indices)
#         indices = indices[:n_samples]
#         return [self.dataset[i] for i in indices]

#     def sample_task(self):
#         # Randomly sample a subset of classes assigned to this GPU
#         selected_classes = random.sample(self.assigned_classes, self.num_classes_per_task)

#         support_set = []
#         query_set = []

#         for class_label in selected_classes:
#             # Get all samples for the selected class
#             class_samples = self.get_n_class_samples(class_label, self.num_support+self.num_query)
#             # Shuffle the class samples
#             random.shuffle(class_samples)

#             # Split samples into support and query sets
#             support_samples = class_samples[:self.num_support]
#             query_samples = class_samples[self.num_support:self.num_support + self.num_query]

#             support_set.extend(support_samples)
#             query_set.extend(query_samples)

#         # Convert support and query sets to tensors
#         support = self.collator(support_set)
#         query = self.collator(query_set)

#         return support, query

#     def get_few_shot(self, class_label, n_samples):
#         indices = self.dataset.drug_response_df_grouped.groups[class_label].tolist()
#         #random.shuffle(indices)
#         k_indices = indices[:n_samples]
#         test_indices = indices[n_samples:]

#         k_data = self.collator([self.dataset[i] for i in k_indices])
#         test_data = self.collator([self.dataset[i] for i in test_indices])
#         return k_data, test_data