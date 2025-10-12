import pandas as pd
import torch
import numpy as np
import argparse
from torch.utils.data.dataloader import DataLoader

import torch
import torchtuples as tt
import torch.nn.parallel
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from survival_dataloader import SimpleSurvivalMetaDataset, SimpleSurvivalCollator, SimpleSurvivalTaskSampler
from survival_MAML_trainer import SurvivalMAMLTrainer, SurvivalMLP

def create_simple_survival_setup(
    drug_response_file="/path/to/survival_data.csv",
    patient2ind_file="/path/to/patient2ind.txt", 
    genotype_files={
        'mutation': "/path/to/mutations.txt",
        'cna': "/path/to/cna.txt"
    }
):
    """
    Create training setup with just raw files
    
    Expected formats:
    - drug_response_file: [patient_id, drug, time, event]
    - patient2ind_file: [patient_id, index] (tab-separated)
    - genotype files: binary matrices [patients x features]
    """
    
    print("Creating simplified survival dataset...")
    dataset = SimpleSurvivalMetaDataset(
        drug_response_file=drug_response_file,
        patient2ind_file=patient2ind_file,
        genotype_files=genotype_files,
        class_ind=1,  # Class column
        time_col=2,   # Survival time column  
        event_col=3   # Event indicator column
    )
    
    print("Creating collator...")
    collator = SimpleSurvivalCollator(
        genotype_types=list(genotype_files.keys())
    )
    
    return dataset, collator

# Example of what your data files should look like:
print("\nExpected data format:")
print("drug_response_file (CSV):")
print("  TCGA-123,cisplatin,24.5,1")
print("  TCGA-124,cisplatin,15.2,0") 
print("  TCGA-125,gemcitabine,8.7,1")

print("\npatient2ind_file (tab-separated):")
print("  TCGA-123\t0")
print("  TCGA-124\t1")
print("  TCGA-125\t2")

print("\ngenotype files (binary matrices):")
print("  mutations.txt: [patients x genes] with 0/1 values")
print("  cna.txt: [patients x genes] with 0/1 values")

print("\nUsage:")
print("python simple_survival_maml.py \\")
print("  --drug_response_file data/survival.csv \\")
print("  --patient2ind_file data/patient2ind.txt \\") 
print("  --mutation_file data/mutations.txt \\")
print("  --cna_file data/cna.txt \\")
print("  --epochs 50")

# Model hyperparameters
parser = argparse.ArgumentParser()
# parser.add_argument('--feature_dim', type=int, default=2000, help='Input feature dimension')
parser.add_argument('--hidden_dim', type=int, default=5, help='Number of hidden units of NN for single task')
parser.add_argument('--layers', type=int, default=1, help='Number of layers of NN for single task')

# MAML hyperparameters
parser.add_argument('--inner_lr', type=float, default=0.001)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--inner_loop_steps', type=int, default=5)
parser.add_argument('--meta_batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)

# Training data files
parser.add_argument('--drug_response_file', type=str, required=True,
                    help='CSV: patient_id,drug,time,event')
parser.add_argument('--patient2ind_file', type=str, required=True,
                    help='Tab-separated: patient_id index') 
parser.add_argument('--mutation_file', type=str,
                    help='Binary mutation matrix')
parser.add_argument('--cna_file', type=str,
                    help='Binary CNA matrix')
parser.add_argument('--expression_file', type=str, 
                    help='Expression matrix')

# Validation (holdout) data files
parser.add_argument('--valid_drug_response_file', type=str, required=True,
                    help='CSV: patient_id,drug,time,event')
parser.add_argument('--valid_patient2ind_file', type=str, required=True,
                    help='Tab-separated: patient_id index') 
parser.add_argument('--valid_mutation_file', type=str,
                    help='Binary mutation matrix')
parser.add_argument('--valid_cna_file', type=str,
                    help='Binary CNA matrix')
parser.add_argument('--valid_expression_file', type=str, 
                    help='Expression matrix')


# Task sampling
parser.add_argument('--num_classes_per_task', type=int, default=2, help='Number of classes per task')
parser.add_argument('--num_support', type=int, default=8, help='Support set size per drug')
parser.add_argument('--num_query', type=int, default=8, help='Query set size per drug')

# Training setup
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--val_step', type=int, default=10)
parser.add_argument('--use_pycox', action='store_true', help='Use pycox for Cox loss')
parser.add_argument('--output_path', type=str, default='survival_maml_model')

# Distributed training (optional)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)

args = parser.parse_args()

# Setup device
if args.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

print(f"Using device: {device}")

genotype_files = {}
if args.mutation_file:
    genotype_files['mutation'] = args.mutation_file
if args.cna_file:
    genotype_files['cna'] = args.cna_file
if args.expression_file:
    genotype_files['expression'] = args.expression_file

# Create training setup
dataset, collator = create_simple_survival_setup(
    drug_response_file=args.drug_response_file,
    patient2ind_file=args.patient2ind_file,
    genotype_files=genotype_files
)

dataloader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=32,
    num_workers=args.jobs,
    collate_fn=collator
)

print(f"Training dataset created with {len(dataset)} samples")
print(f"Available drugs/treatments in training: {dataset.classes}")

# Test single sample to get feature dimensions
sample = dataset[0] 
batch = collator([sample])
feature_dim = batch['features'].shape[1]

print(f"Feature dimension: {feature_dim}")

# Create task sampler
task_sampler = SimpleSurvivalTaskSampler(
    dataset=dataset,
    collator=collator,
    num_classes_per_task=args.num_classes_per_task,
    num_support=args.num_support,
    num_query=args.num_query,
    rank=args.rank,
    world_size=args.world_size
)

print(f"Task sampler created with {len(task_sampler.assigned_classes)} assigned classes")

# Create model
model = SurvivalMLP(
    feature_dim=feature_dim,
    layer=args.layers,
    hidden=args.hidden_dim
)

# Instantiate a Hazard Tranform Cox-Time Model
# func = MLPVanillaCoxTime(self.feature_dim, self.hidden_dim, dropout=self.transform_dropout)
# model = CoxTime(func, self.optimizer, labtrans=self.labtrans, device=self.device)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Add validation dataloader
valid_genotype_files = {}
if args.valid_mutation_file:
    valid_genotype_files['mutation'] = args.valid_mutation_file
if args.valid_cna_file:
    valid_genotype_files['cna'] = args.valid_cna_file
if args.valid_expression_file:
    valid_genotype_files['expression'] = args.valid_expression_file

# Create validation setup
validation_dataset, validation_collator = create_simple_survival_setup(
    drug_response_file=args.valid_drug_response_file,
    patient2ind_file=args.valid_patient2ind_file,
    genotype_files=valid_genotype_files
)

validation_dataloader = DataLoader(
    validation_dataset,
    shuffle=True,
    batch_size=32,
    num_workers=args.jobs,
    collate_fn=validation_collator
)

print(f"Validation dataset created with {len(dataset)} samples")
print(f"Available drugs/treatments in validation: {dataset.classes}")

# Create trainer
trainer = SurvivalMAMLTrainer(
    survival_model=model,
    device=device,
    args=args,
    task_sampler=task_sampler,
    validation_dataloader= validation_dataloader  # Could add validation here
)

# Update Cox loss to use pycox if requested
# if args.use_pycox:
#     trainer.cox_loss = CoxLoss(use_pycox=True)
#     print("Using pycox backend for Cox loss")

print("Starting meta-training...")

# Train the model
trainer.train(epochs=args.epochs, output_path=args.output_path)

# Example of few-shot adaptation
# if len(dataset.classes) > 1:
#     print("\nTesting few-shot adaptation...")
#     test_drug = dataset.classes[0]  # Use first drug for testing
    
#     k_data, test_data = task_sampler.get_few_shot(test_drug, n_samples=5)
    
#     if k_data and 'features' in k_data:
#         print(f"Few-shot adaptation on {test_drug} with {k_data['features'].size(0)} samples")
        
#         # Adapt model
#         adapted_model = trainer.few_shot_adaptation(k_data, n_adaptation_steps=10)
        
#         # Test adapted model (you could evaluate C-index here)
#         if test_data and 'features' in test_data:
#             with torch.no_grad():
#                 test_features = test_data['features'].to(device)
#                 risk_scores, _ = adapted_model(test_features)
#                 print(f"Generated risk scores for {len(risk_scores)} test patients")

# def main():
#     parser = argparse.ArgumentParser()
    

#     # Model parameters
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--inner_lr', type=float, default=0.01)
#     parser.add_argument('--outer_lr', type=float, default=0.001)
#     parser.add_argument('--device', type=str, default='auto')
    
#     args = parser.parse_args()
    
#     # Setup device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     # Create dataset and sampler
#     genotype_files = {
#         'mutation': args.mutation_file,
#         'cna': args.cna_file,
#         'expression': args.expression_file
#     }
    
#     dataset, collator = create_simple_survival_setup(
#         drug_response_file=args.drug_response_file,
#         patient2ind_file=args.patient2ind_file,
#         genotype_files=genotype_files
#     )
    
#     print(f"Dataset: {len(dataset)} samples")
#     print(f"Drugs: {dataset.classes}")
    
#     # Test single sample to get feature dimensions
#     sample = dataset[0] 
#     batch = collator([sample])
#     feature_dim = batch['features'].shape[1]
    
#     print(f"Feature dimension: {feature_dim}")
    
#     # Create sampler
#     sampler = SimpleSurvivalTaskSampler(
#         dataset=dataset,
#         collator=collator,
#         num_classes_per_task=2,
#         num_support=8,
#         num_query=8
#     )
    
#     # Create model
#     model = SurvivalMLP(
#         feature_dim=feature_dim,
#         layer=2,
#         hidden=128
#     )
    
#     # Create trainer
#     trainer = SurvivalMAMLTrainer(
#         survival_model=model,
#         device=device,
#         args=args,
#         task_sampler=sampler
#     )
    
#     print("Starting training...")
#     trainer.train(epochs=args.epochs, output_path="simple_survival_maml")
    
#     print("Training complete!")

# if __name__ == "__main__":
#     # Example with mock data paths (replace with your actual paths)
#     mock_args = [
#         "--drug_response_file", "/path/to/tcga_survival.csv",
#         "--cell2ind_file", "/path/to/patient2ind.txt", 
#         "--mutations_file", "/path/to/patient2mutation.txt",
#         "--cna_file", "/path/to/patient2cna.txt",
#         "--ontology_file", "/path/to/ontology.txt",
#         "--gene2ind_file", "/path/to/gene2ind.txt",
#         "--epochs", "50",
#         "--use_pycox"
#     ]
    
#     print("Survival MAML Training Example")
#     print("==============================")
#     print("Command line arguments:")
#     for i in range(0, len(mock_args), 2):
#         if i+1 < len(mock_args):
#             print(f"  {mock_args[i]} {mock_args[i+1]}")
#         else:
#             print(f"  {mock_args[i]}")
    
#     print("\nTo run with actual data, use:")
#     print("python survival_maml_training.py " + " ".join(mock_args))
    
    # Uncomment to run with mock args:
    # import sys
    # sys.argv = ["script_name"] + mock_args
    # main()