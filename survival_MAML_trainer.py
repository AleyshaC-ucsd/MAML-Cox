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
import pickle
from pycox.models import CoxTime
from pycox.evaluation import EvalSurv
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models.loss import cox_cc_loss_single_ctrl

from survival_dataloader import *

# Custom layers for manual forward pass with weights (for MAML)
def linear(input, weight, bias=None):
    """Manual linear layer implementation for MAML"""
    return F.linear(input, weight, bias)

def relu(input):
    """ReLU activation"""
    return F.relu(input)

class SurvivalMLP(nn.Module):
    """MLP that outputs risk scores for survival analysis"""

    def __init__(self, feature_dim, layer, hidden):
        super(SurvivalMLP, self).__init__()

        self.add_module('linear1', nn.Linear(feature_dim, hidden))
        
        if layer == 2:
            self.add_module('linear2', nn.Linear(hidden, hidden))
        
        # Output single risk score (higher = higher risk of event)
        self.add_module('linear3', nn.Linear(hidden, 1))

        self.layer = layer
        self._init_weights()

    def forward(self, x, weights=None):
        if weights is None:
            # Standard forward pass
            hidden = F.relu(self._modules['linear1'](x))
            
            if self.layer == 2:
                hidden = F.relu(self._modules['linear2'](hidden))
            
            # Risk score output (no activation - can be any real number)
            out = self._modules['linear3'](hidden)
            
        else:
            # MAML forward pass with custom weights
            hidden = relu(linear(x, weights['linear1.weight'], weights['linear1.bias']))
            
            if self.layer == 2:
                hidden = relu(linear(hidden, weights['linear2.weight'], weights['linear2.bias']))
            
            out = linear(hidden, weights['linear3.weight'], weights['linear3.bias'])

        return out, hidden

    def copy_weights(self, net):
        """Copy weights from another network"""
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def net_forward(self, x, weights=None):
        return self.forward(x, weights)

    def _init_weights(self):
        torch.manual_seed(1337)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())

# class CoxLoss(nn.Module):
#     """Cox Partial Likelihood Loss with option to use pycox backend"""
    
#     def __init__(self, use_pycox=False):
#         super(CoxLoss, self).__init__()
#         self.use_pycox = use_pycox
        
#         if use_pycox:
#             try:
#                 self.pycox_loss = cox_cc_loss_single_ctrl()
#             except ImportError:
#                 print("Warning: pycox not available, falling back to custom implementation")
#                 self.use_pycox = False
    
#     def forward(self, risk_scores, times, events):
#         """
#         Compute Cox partial likelihood loss
        
#         Args:
#             risk_scores: Tensor of shape (batch_size,) - log hazard ratios
#             times: Tensor of shape (batch_size,) - survival times  
#             events: Tensor of shape (batch_size,) - event indicators (1=event, 0=censored)
#         """
#         if self.use_pycox:
#             return self._pycox_cox_loss(risk_scores, times, events)
#         else:
#             return self._custom_cox_loss(risk_scores, times, events)
    
#     def _pycox_cox_loss(self, risk_scores, times, events):
#         """Use pycox's optimized Cox loss computation"""
#         try:
#             # Convert to format expected by pycox
#             risk_scores = risk_scores.squeeze()
#             times = times.squeeze()
#             events = events.squeeze()
            
#             # Create target in pycox format: (durations, events)
#             target = (times, events)
            
#             # Compute loss using pycox
#             loss = self.pycox_loss(risk_scores, target)
#             return loss
            
#         except Exception as e:
#             print(f"Pycox loss computation failed: {e}, falling back to custom")
#             return self._custom_cox_loss(risk_scores, times, events)
    
#     def _custom_cox_loss(self, risk_scores, times, events):
#         """Custom Cox loss implementation (fallback)"""
#         # Ensure proper shapes
#         risk_scores = risk_scores.squeeze()
#         times = times.squeeze()
#         events = events.squeeze()
        
#         # Handle edge cases
#         if len(risk_scores) <= 1:
#             return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
#         # Sort by survival time (descending order)
#         sorted_indices = torch.argsort(times, descending=True)
#         sorted_times = times[sorted_indices]
#         sorted_events = events[sorted_indices]
#         sorted_risk_scores = risk_scores[sorted_indices]
        
#         # Compute partial likelihood
#         log_likelihood = 0.0
#         n_events = 0
        
#         # For numerical stability
#         max_risk = torch.max(sorted_risk_scores)
#         sorted_risk_scores = sorted_risk_scores - max_risk
        
#         for i in range(len(sorted_times)):
#             if sorted_events[i] == 1:  # Event occurred
#                 # Risk set: all individuals with time >= current time
#                 at_risk_scores = sorted_risk_scores[i:]
                
#                 if len(at_risk_scores) > 1:
#                     # Partial likelihood contribution
#                     numerator = sorted_risk_scores[i]
#                     denominator = torch.logsumexp(at_risk_scores, dim=0)
#                     log_likelihood += numerator - denominator
#                     n_events += 1
        
#         # Return negative log likelihood (for minimization)
#         if n_events > 0:
#             return -log_likelihood / n_events
#         else:
#             return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

class SurvivalMAMLTrainer(object):
    """MAML trainer for survival prediction using Cox loss"""

    def __init__(self, survival_model, device, args, task_sampler, validation_dataloader=None):
        self.device = device
        self.survival_model = survival_model.to(self.device)
        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        self.inner_loop_steps = args.inner_loop_steps
        self.meta_batch_size = args.meta_batch_size
        self.task_sampler = task_sampler
        self.meta_optimizer = optim.Adam(self.survival_model.parameters(), lr=self.outer_lr)
        self.cox_loss = cox_cc_loss_single_ctrl
        self.validation_dataloader = validation_dataloader
        self.args = args

    def _create_case_control_pairs(self, risk_scores, times, events):
        """Create case-control pairs from risk scores and survival data, to use with CoxTime loss function"""
        batch_size = len(risk_scores)
        
        if batch_size <= 1:
            # Handle single sample case
            g_case = risk_scores
            g_control = torch.zeros_like(risk_scores)
            return g_case, g_control
        
        # Strategy: Use survival time and event status to create meaningful pairs
        times = times.squeeze()
        events = events.squeeze()
        
        # Find patients who died (events=1) vs survived/censored (events=0)
        event_mask = events == 1
        
        if event_mask.sum() > 0 and (~event_mask).sum() > 0:
            # Case: patients who experienced event
            # Control: patients who were censored or survived longer
            g_case = risk_scores[event_mask]
            g_control = risk_scores[~event_mask]
        else:
            # Fallback: split by survival time
            median_time = times.median()
            early_death_mask = times <= median_time
            
            if early_death_mask.sum() > 0 and (~early_death_mask).sum() > 0:
                g_case = risk_scores[early_death_mask]    # Shorter survival
                g_control = risk_scores[~early_death_mask] # Longer survival
            else:
                # Final fallback: random split
                perm = torch.randperm(batch_size)
                mid = batch_size // 2
                g_case = risk_scores[perm[:mid]] if mid > 0 else risk_scores
                g_control = risk_scores[perm[mid:]] if mid < batch_size else risk_scores
        
        return g_case, g_control

    def train(self, epochs, output_path=None):
        self.best_model = self.survival_model
        best_performance = 0
        # Initialize dictionary to store metrics
        self.performance_metrics = {
            'epochs': [],
            'harrell': [],
            'cox_loss': []
        }
        epoch_iterator = tqdm(range(epochs))
        
        for epoch in epoch_iterator:
            support_loss, query_loss = self.iter_tasks(self.survival_model, self.task_sampler)
            
            epoch_iterator.set_description(
                f"Train epoch: {epoch}, Support loss {support_loss:.3f}, Query loss: {query_loss:.3f}"
            )
            
            gc.collect()
            torch.cuda.empty_cache()

            # if (epoch % self.args.val_step) == 0 and self.validation_dataloader is not None:
            #     c_index = self.evaluate(self.survival_model, self.validation_dataloader, epoch + 1)
            #     if c_index > best_performance:
            #         best_performance = c_index
            #         if output_path:
            #             torch.save({
            #                 "arguments": self.args, 
            #                 "state_dict": self.survival_model.state_dict()
            #             }, output_path + f".best")

            # if output_path and (epoch % 50 == 0):
            #     torch.save({
            #         "arguments": self.args, 
            #         "state_dict": self.survival_model.state_dict()
            #     }, output_path + f".{epoch}")

            if (epoch % self.args.val_step) == 0:
                if self.validation_dataloader is not None:
                   harrell, cox_loss = self.evaluate(self.survival_model, self.validation_dataloader, epoch + 1,
                                               name="Validation")
                   # Store metrics
                   self.performance_metrics['epochs'].append(epoch)
                   self.performance_metrics['harrell'].append(float(harrell))
                   self.performance_metrics['cox_loss'].append(float(cox_loss))

                   if harrell > best_performance:
                       best_performance = harrell
                       self.best_model = copy.deepcopy(self.survival_model).to('cpu')
                   torch.cuda.empty_cache()
                   gc.collect()
                   # TODO: Instead of creating separate sampler for few-shot, use specific class from task sampler
                # if self.few_shot_sampler is not None:
                self.few_shot_learning(self.survival_model, self.task_sampler, self.k)
                if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                                 and self.args.rank % torch.cuda.device_count() == 0):
                    if output_path:
                        output_path_epoch = output_path + ".%d" % epoch
                        print("Save to...", output_path_epoch)
                        if self.args.multiprocessing_distributed:
                            torch.save(
                                {"arguments": self.args, "state_dict": self.survival_model.module.state_dict()},
                                output_path_epoch)
                        else:
                            torch.save({"arguments": self.args, "state_dict": self.survival_model.state_dict()},
                                       output_path_epoch)
                            
        out = output_path.split("/")
        folder = out[0]
        fold = out[1].split("_")[2]
        with open(folder + '/val_performance_' + fold + '.pkl', 'wb') as handle:
            pickle.dump(self.performance_metrics, handle)

    # def train_pycox_CT(self, dataloader, batch_size):
        
    #     for i, batch in enumerate(dataloader):
    #         batch = move_to(batch, self.device)
        
    
    #     time = batch['time'].detach().cpu().numpy()
    #     event = batch['event'].detach().cpu().numpy()

    #     # Create DF for training hazard transform model
    #     df_train = pd.DataFrame({'AUC': auc, 'Age': age, 'Sex': sex, 'Time': time, 'Event': event}) 
    #     df_test = df_train.sample(frac=0.2)
    #     df_train = df_train.drop(df_test.index)

    #     # Standardize the features for train and test
    #     standards = [(['AUC'], StandardScaler()), (['Age'], StandardScaler())]
    #     binarized = [('Sex', None)]
    #     x_mapper = DataFrameMapper(standards + binarized)
    #     x_train = x_mapper.fit_transform(df_train).astype('float32')
    #     x_test = x_mapper.transform(df_test).astype('float32')

    #     # Standardize the time labels
    #     get_target = lambda df: (df['Time'].values, df['Event'].values)
    #     y_train = self.labtrans.fit_transform(*get_target(df_train))
    #     y_test = self.labtrans.transform(*get_target(df_test))
    #     durations_test, events_test = get_target(df_test)
    #     val = tt.tuplefy(x_test, y_test)

    #     # Instantiate a Hazard Tranform Cox-Time Model
    #     func = MLPVanillaCoxTime(self.num_features, self.hidden_layers, dropout=self.transform_dropout)
    #     model = CoxTime(func, self.optimizer, labtrans=self.labtrans, device=self.device)

    #     # Train the Hazard Transform Model
    #     callbacks = [tt.callbacks.EarlyStopping()]
    #     log = model.fit(x_train, y_train, 
    #                     batch_size, 
    #                     verbose=True, 
    #                     epochs=self.epochs, 
    #                     callbacks=callbacks,
    #                     val_data=val.repeat(10).cat())


    def iter_tasks(self, meta_model, task_sampler):
        """MAML training iteration with Cox loss"""
        meta_model.train()
        mean_support_loss = 0.
        mean_query_loss = 0.

        task_model = copy.deepcopy(meta_model).to(self.device)
        task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.

        for i in range(self.meta_batch_size):
            support, query = task_sampler.sample_task()
            
            # Inner loop - adapt on support set
            for _ in range(self.inner_loop_steps):
                support_batch = self.move_to_device(support)
                
                # Get risk scores from model
                # surv = task_model.predict_surv(support_batch['features'].cpu().numpy())
                risk_scores, _ = task_model(support_batch['features'])
                risk_scores = risk_scores.squeeze()
                # surv = surv.squeeze()
                
                # Compute Cox loss
                # support_loss = self.cox_loss(
                #     risk_scores, 
                #     support_batch['time'], 
                #     support_batch['event']
                # )
                # Create case-control pairs for support set
                g_case, g_control = self._create_case_control_pairs(
                    risk_scores, 
                    support_batch['time'], 
                    support_batch['event']
                )
                
                # Compute Cox case-control loss
                support_loss = self.cox_loss(g_case, g_control, shrink=0.0)
                
                task_optimizer.zero_grad()
                support_loss.backward()
                task_optimizer.step()
                
                mean_support_loss += float(support_loss)

            # Outer loop - evaluate on query set
            query_batch = self.move_to_device(query)
            risk_scores, _ = task_model(query_batch['features'])
            risk_scores = risk_scores.squeeze()
            # surv = task_model.predict_surv(query_batch['features'].cpu().numpy())
            # surv = surv.squeeze()

            # Create case-control pairs for query set
            g_case, g_control = self._create_case_control_pairs(
                risk_scores,
                query_batch['time'], 
                query_batch['event']
            )
            
            query_loss = self.cox_loss(g_case, g_control, shrink=0.0)
                
                # query_loss = self.cox_loss(
                #     surv,
                #     query_batch['time'], 
                #     query_batch['event']
                # )
            
            meta_loss += query_loss

        # Meta-update
        meta_loss = meta_loss / self.meta_batch_size
        meta_loss.backward()
        self.meta_optimizer.step()

        del task_model
        
        return (mean_support_loss / (self.meta_batch_size * self.inner_loop_steps), 
                float(meta_loss))
    
    def compute_c_index(self, risk_scores, times, events):
        """Compute Harrell's C-index"""
        risk_scores = risk_scores.cpu().numpy()
        times = times.cpu().numpy()
        events = events.cpu().numpy()
        
        n = len(risk_scores)
        concordant = 0
        comparable = 0
        
        for i in range(n):
            if events[i] == 1:  # Event occurred
                for j in range(n):
                    if times[j] > times[i]:  # Patient j survived longer
                        comparable += 1
                        if risk_scores[i] > risk_scores[j]:  # Higher risk had earlier event
                            concordant += 1
        
        return concordant / comparable if comparable > 0 else 0.5

    def evaluate(self, model, dataloader, epoch, name="Validation"):
        model.eval()

        all_risk_scores = []
        all_times = []
        all_events = []
       
        # # Get single batch containing all test data
        # batch = next(iter(dataloader))
        # batch = self.move_to_device(batch)

        # # Get predictions
        # _ = model.compute_baseline_hazards()
        # surv = model.predict_surv_df(batch['features'].cpu().numpy())
        
        # # Get true values
        # times = batch['time'].cpu().numpy()
        # events = batch['event'].cpu().numpy()
        
        # # Evaluate using pycox metrics
        # ev = EvalSurv(surv, times, events, censor_surv='km')
        
        # # Calculate metrics
        # time_grid = np.linspace(times.min(), times.max(), 100)
        # antolini = ev.concordance_td(method='antolini')
        # brier_score = ev.integrated_brier_score(time_grid)
        # nbll = ev.integrated_nbll(time_grid)
        
        # print(f"\nEvaluation Metrics:")
        # print(f"Antolini Concordance Index: {antolini:.4f}")
        # print(f"Integrated Brier Score: {brier_score:.4f}")
        # print(f"Integrated NBLL: {nbll:.4f}")

        # return antolini, brier_score, nbll
        with torch.no_grad():
            for batch in dataloader:
                batch = self.move_to_device(batch)
                
                # Get risk scores from model
                risk_scores, _ = model(batch['features'])
                risk_scores = risk_scores.squeeze()
                
                all_risk_scores.append(risk_scores.cpu())
                all_times.append(batch['time'].cpu())
                all_events.append(batch['event'].cpu())
    
        # Concatenate all results
        risk_scores = torch.cat(all_risk_scores)
        times = torch.cat(all_times)
        events = torch.cat(all_events)
        
        # Compute Cox loss
        g_case, g_control = self._create_case_control_pairs(risk_scores, times, events)
        cox_loss = float(self.cox_loss(g_case, g_control, shrink=0.0))
        
        # Compute C-index
        c_index = self.compute_c_index(risk_scores, times, events)
        
        print(f"\nEvaluation Metrics:")
        print(f"Cox Loss: {cox_loss:.4f}")
        print(f"C-index: {c_index:.4f}")
        
        return c_index, cox_loss
        
        # all_features = []
        # all_times = []
        # all_events = []
        
        # dataloader_with_tqdm = tqdm(dataloader)
        
        # with torch.no_grad():
        #     for batch in dataloader_with_tqdm:
        #         batch = self.move_to_device(batch)
                
        #         # Collect features and targets
        #         all_features.append(batch['features'].cpu())
        #         all_times.append(batch['time'].cpu())
        #         all_events.append(batch['event'].cpu())
                
        #         dataloader_with_tqdm.set_description(f"{name} epoch: {epoch}")

        # # Concatenate all batches
        # features = torch.cat(all_features, dim=0).numpy()
        # durations = torch.cat(all_times, dim=0).numpy()
        # events = torch.cat(all_events, dim=0).numpy()
        
        # # Get survival predictions
        # _ = model.compute_baseline_hazards()
        # surv = model.predict_surv_df(features)
        
        # # Evaluate using pycox metrics
        # ev = EvalSurv(surv, durations, events, censor_surv='km')
        
        # # Calculate metrics
        # time_grid = np.linspace(durations.min(), durations.max(), 100)
        # antolini = ev.concordance_td(method='antolini')
        # brier_score = ev.integrated_brier_score(time_grid)
        # nbll = ev.integrated_nbll(time_grid)
        
        # print(f"\nEvaluation Metrics:")
        # print(f"Antolini Concordance Index: {antolini:.4f}")
        # print(f"Integrated Brier Score: {brier_score:.4f}")
        # print(f"Integrated Negative Binomial Log-Likelihood: {nbll:.4f}")
        
        # return antolini  # Return concordance as primary metric

    # def compute_c_index(self, risk_scores, times, events):
    #     """Compute Harrell's C-index"""
    #     n = len(risk_scores)
    #     concordant = 0
    #     comparable = 0
        
    #     for i in range(n):
    #         if events[i] == 1:  # Event occurred for patient i
    #             for j in range(n):
    #                 if times[j] > times[i]:  # Patient j survived longer
    #                     comparable += 1
    #                     if risk_scores[i] > risk_scores[j]:  # Higher risk had earlier event
    #                         concordant += 1
        
    #     if comparable > 0:
    #         return concordant / comparable
    #     else:
    #         return 0.5  # Random performance

    def move_to_device(self, data):
        """Move data to device"""
        if isinstance(data, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data
    
    def few_shot_learning(self, meta_model, few_shot_sampler, k):
        meta_model.eval()
        # NOTE: Removed separate few-shot sampler, now using specific class from task sampler for few-shot learning
        # for tissue_type in few_shot_sampler.all_classes:

        few_shot_model = copy.deepcopy(meta_model).to(self.device)
        k_data, rest_data = few_shot_sampler.get_few_shot(n_samples=k)
        with torch.no_grad():
            k_batch = self.move_to_device(k_data, self.device)
            rest_batch = self.move_to_device(rest_data, self.device)
            # drug_response_predicted = meta_model.predict_surv(rest_batch['features'].cpu().numpy())
            # drug_response_predicted = meta_model(rest_batch['genotype'], rest_batch['drug'],
            #                                 self.nested_subtrees_forward, self.nested_subtrees_backward,
            #                                 self.gene2system_mask, self.system2gene_mask,
            #                                 sys2cell=self.args.sys2cell,
            #                                 cell2sys=self.args.cell2sys,
            #                                 sys2gene=self.args.sys2gene,
            #                                 gene2drug=self.args.gene2drug,
            #                                 mut2gene=self.args.mut2gene,
            #                                 with_indices=self.args.with_indices)
            
            # Get true values
            # times = rest_batch['time'].cpu().numpy()
            # events = rest_batch['event'].cpu().numpy()
            
            # Evaluate using pycox metrics
            # ev = EvalSurv(drug_response_predicted, times, events, censor_surv='km')
            
            # Calculate metrics
            # time_grid = np.linspace(times.min(), times.max(), 100)
            # antolini = ev.concordance_td(method='antolini')
            # brier_score = ev.integrated_brier_score(time_grid)
            # nbll = ev.integrated_nbll(time_grid)
            # # trues = (rest_batch['response_mean'] + rest_batch['response_residual']).detach().cpu().numpy()
            # # results = drug_response_predicted.detach().cpu().numpy()
            # # pearson = pearsonr(trues, results)
            # if self.args.rank==0:
            #     print("%s Antolini C-index before k-shot %f"%(tissue_type, antolini))

            # Get risk scores from model (not predict_surv!)
            risk_scores, _ = meta_model(rest_batch['features'])
            risk_scores = risk_scores.squeeze()
            
            # Create case-control pairs for evaluation
            g_case, g_control = self._create_case_control_pairs(
                risk_scores,
                rest_batch['time'], 
                rest_batch['event']
            )

            # Compute loss as proxy for performance
            initial_loss = self.cox_loss(g_case, g_control, shrink=0.0)
            
            if self.args.rank == 0:
                print(f"{tissue_type} initial Cox loss before k-shot: {float(initial_loss):.4f}")

        few_shot_optimizer = optim.SGD(few_shot_model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_loop_steps):
            # drug_response_predicted_few_shot = few_shot_model.predict_surv(k_batch['features'].cpu().numpy())
            # drug_response_predicted_few_shot = few_shot_model(k_batch['genotype'], k_batch['drug'],
            #                                                    self.nested_subtrees_forward,
            #                                                    self.nested_subtrees_backward,
            #                                                    self.gene2system_mask, self.system2gene_mask,
            #                                                    sys2cell=self.args.sys2cell,
            #                                                    cell2sys=self.args.cell2sys,
            #                                                    sys2gene=self.args.sys2gene,
            #                                                    gene2drug=self.args.gene2drug,
            #                                                    mut2gene=self.args.mut2gene,
            #                                                    with_indices=self.args.with_indices)
            # few_shot_loss = self.cox_loss(
            #     drug_response_predicted_few_shot, 
            #     k_batch['time'], 
            #     k_batch['event']
            # )
            # few_shot_loss = self.drug_response_loss(drug_response_predicted_few_shot[:, 0],
            #                                              (k_batch['response_mean'] + k_batch['response_residual']).to(
            #                                                  torch.float32))

            # Get risk scores from few-shot model
            risk_scores, _ = few_shot_model(k_batch['features'])
            risk_scores = risk_scores.squeeze()
            
            # Create case-control pairs for adaptation
            g_case, g_control = self._create_case_control_pairs(
                risk_scores,
                k_batch['time'], 
                k_batch['event']
            )
            
            # Compute loss
            few_shot_loss = self.cox_loss(g_case, g_control, shrink=0.0)

            few_shot_optimizer.zero_grad()
            few_shot_loss.backward()
            few_shot_optimizer.step()
            del few_shot_loss
        with torch.no_grad():
            few_shot_model.eval()
            # drug_response_predicted = few_shot_model.predict_surv(rest_batch['features'].cpu().numpy())
            # drug_response_predicted = few_shot_model(rest_batch['genotype'], rest_batch['drug'],
            #                                 self.nested_subtrees_forward, self.nested_subtrees_backward,
            #                                 self.gene2system_mask, self.system2gene_mask,
            #                                 sys2cell=self.args.sys2cell,
            #                                 cell2sys=self.args.cell2sys,
            #                                 sys2gene=self.args.sys2gene,
            #                                 gene2drug=self.args.gene2drug,
            #                                 mut2gene=self.args.mut2gene,
            #                                 with_indices=self.args.with_indices)
            # Get true values
            # times = rest_batch['time'].cpu().numpy()
            # events = rest_batch['event'].cpu().numpy()
            
            # Evaluate using pycox metrics
            # ev = EvalSurv(drug_response_predicted, times, events, censor_surv='km')
            
            # # Calculate metrics
            # time_grid = np.linspace(times.min(), times.max(), 100)
            # antolini = ev.concordance_td(method='antolini')
            # brier_score = ev.integrated_brier_score(time_grid)
            # nbll = ev.integrated_nbll(time_grid)
            # if self.args.rank==0:
            #     print("%s Antolini C-index after k-shot %f"%(tissue_type, antolini))

            # Get risk scores from adapted model
            risk_scores, _ = few_shot_model(rest_batch['features'])
            risk_scores = risk_scores.squeeze()
            
            # Create case-control pairs for final evaluation
            g_case, g_control = self._create_case_control_pairs(
                risk_scores,
                rest_batch['time'], 
                rest_batch['event']
            )
            
            # Compute final loss
            final_loss = self.cox_loss(g_case, g_control, shrink=0.0)
            
            if self.args.rank == 0:
                print(f"{tissue_type} final Cox loss after k-shot: {float(final_loss):.4f}")
                improvement = float(initial_loss) - float(final_loss)
                print(f"{tissue_type} improvement: {improvement:.4f}")

        del few_shot_optimizer
        del few_shot_model
    # def few_shot_adaptation(self, support_data, n_adaptation_steps=10):
    #     """Adapt model to new task (e.g., new treatment)"""
    #     adapted_model = copy.deepcopy(self.survival_model)
    #     adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
    #     support_data = self.move_to_device(support_data)
        
    #     for step in range(n_adaptation_steps):
    #         risk_scores, _ = adapted_model(support_data['features'])
    #         risk_scores = risk_scores.squeeze()
            
    #         loss = self.cox_loss(
    #             risk_scores, 
    #             support_data['time'], 
    #             support_data['event']
    #         )
            
    #         adapted_optimizer.zero_grad()
    #         loss.backward()
    #         adapted_optimizer.step()
        
    #     return adapted_model



# class SurvivalTaskSampler:
#     """Sample tasks for survival MAML training"""
    
#     def __init__(self, datasets_dict, support_size=16, query_size=16):
#         self.datasets_dict = datasets_dict
#         self.task_names = list(datasets_dict.keys())
#         self.support_size = support_size
#         self.query_size = query_size
    
#     def sample_task(self):
#         """Sample a task and return support/query split"""
#         task_name = random.choice(self.task_names)
#         dataset = self.datasets_dict[task_name]
        
#         # Ensure sufficient samples
#         if len(dataset) < (self.support_size + self.query_size):
#             total_size = len(dataset)
#             support_size = max(1, total_size // 2)
#             query_size = total_size - support_size
#         else:
#             support_size = self.support_size
#             query_size = self.query_size
        
#         # Random sample
#         all_indices = list(range(len(dataset)))
#         random.shuffle(all_indices)
        
#         support_indices = all_indices[:support_size]
#         query_indices = all_indices[support_size:support_size + query_size]
        
#         support_batch = self._create_batch([dataset[i] for i in support_indices])
#         query_batch = self._create_batch([dataset[i] for i in query_indices])
        
#         return support_batch, query_batch
    
#     def _create_batch(self, samples):
#         """Convert samples to batch format"""
#         batch = {}
        
#         if samples:
#             batch['features'] = torch.stack([s['features'] for s in samples])
#             batch['time'] = torch.stack([s['time'] for s in samples])
#             batch['event'] = torch.stack([s['event'] for s in samples])
            
#             # Optional fields
#             if 'age' in samples[0]:
#                 batch['age'] = torch.stack([s['age'] for s in samples])
#             if 'sex' in samples[0]:
#                 batch['sex'] = torch.stack([s['sex'] for s in samples])
        
#         return batch


# Example usage
def create_example_args():
    """Create example arguments for training"""
    class Args:
        def __init__(self):
            self.inner_lr = 0.01
            self.outer_lr = 0.001
            self.inner_loop_steps = 5
            self.meta_batch_size = 4
            self.val_step = 10
    
    return Args()


# Integration with your existing data pipeline
def adapt_genomic_data_to_mlp(batch, max_features=1000):
    """
    Convert complex genomic data to MLP input format
    This function needs to be customized based on your data structure
    """
    features = []
    
    # Process different types of genomic data
    if 'genotype' in batch and isinstance(batch['genotype'], dict):
        for genotype_type, genotype_data in batch['genotype'].items():
            if isinstance(genotype_data, dict) and 'mut' in genotype_data:
                # Convert mutation indices to binary vector
                batch_size = genotype_data['mut'].size(0)
                mut_binary = torch.zeros(batch_size, max_features)
                
                for i in range(batch_size):
                    mut_indices = genotype_data['mut'][i]
                    # Remove padding (assuming -1 or large numbers are padding)
                    valid_indices = mut_indices[mut_indices < max_features]
                    valid_indices = valid_indices[valid_indices >= 0]
                    if len(valid_indices) > 0:
                        mut_binary[i, valid_indices] = 1
                
                features.append(mut_binary)
    
    # Combine all features
    if features:
        combined_features = torch.cat(features, dim=1)
    else:
        # Fallback
        combined_features = torch.randn(batch.get('time', torch.tensor([1])).size(0), max_features)
    
    return {
        'features': combined_features,
        'time': batch.get('time', torch.zeros(combined_features.size(0))),
        'event': batch.get('event', torch.ones(combined_features.size(0))),
        'age': batch.get('age', torch.full((combined_features.size(0),), 65.0)),
        'sex': batch.get('sex', torch.randint(0, 2, (combined_features.size(0),)).float())
    }


# if __name__ == "__main__":

#     # Example setup
#     feature_dim = 1000
#     hidden_dim = 128
    
#     # Create model
#     model = SurvivalMLP(feature_dim=feature_dim, layer=2, hidden=hidden_dim)
    
#     # Create mock task sampler
#     mock_datasets = {}  # You would populate this with your actual datasets
#     task_sampler = SurvivalTaskSampler(mock_datasets)
    
#     # Create trainer
#     args = create_example_args()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     trainer = SurvivalMAMLTrainer(
#         survival_model=model,
#         device=device,
#         args=args,
#         task_sampler=task_sampler
#     )
    
#     print(f"Initialized trainer with device: {device}")
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
#     print("Ready for training!")