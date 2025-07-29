"""
Utility functions for data processing and dataloader creation.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
import numpy as np
from datasets import DatasetDict

class BookClassificationDataset(Dataset):
    """Custom dataset for book sentence classification."""
    
    def __init__(self, dataset, tokenizer, max_length: int = 512, task: str = "classification"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        if self.task == "classification":
            return self._prepare_classification_item(example)
        elif self.task == "mlm":
            return self._prepare_mlm_item(example)
        elif self.task == "nsp":
            return self._prepare_nsp_item(example)
        elif self.task == "similarity":
            return self._prepare_similarity_item(example)
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
    def _prepare_classification_item(self, example):
        """Prepare item for classification task."""
        sentence = example['sentence']
        
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example['label'], dtype=torch.long)
        }
        
    def _prepare_mlm_item(self, example):
        """Prepare item for masked language modeling."""
        original_sentence = example['original_sentence']
        masked_sentence = example['masked_sentence']
        masked_word = example['masked_word']
        
        # Tokenize original sentence
        original_encoding = self.tokenizer(
            original_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize masked sentence
        masked_encoding = self.tokenizer(
            masked_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create MLM labels
        mlm_labels = original_encoding['input_ids'].clone()
        mlm_labels[mlm_labels == self.tokenizer.pad_token_id] = -100
        
        # Mask the target word
        target_tokens = self.tokenizer.tokenize(masked_word)
        mask_positions = (masked_encoding['input_ids'] == self.tokenizer.mask_token_id)
        
        return {
            'input_ids': masked_encoding['input_ids'].squeeze(0),
            'attention_mask': masked_encoding['attention_mask'].squeeze(0),
            'mlm_labels': mlm_labels.squeeze(0)
        }
        
    def _prepare_nsp_item(self, example):
        """Prepare item for next sentence prediction."""
        sentence_a = example['sentence_a']
        sentence_b = example['sentence_b']
        is_next = example['is_next']
        
        # Tokenize sentence pair
        encoding = self.tokenizer(
            sentence_a,
            sentence_b,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'nsp_labels': torch.tensor(1 if is_next else 0, dtype=torch.long)
        }
        
    def _prepare_similarity_item(self, example):
        """Prepare item for sentence similarity."""
        sentence_1 = example['sentence_1']
        sentence_2 = example['sentence_2']
        similarity = example['similarity']
        
        # Tokenize both sentences
        encoding_1 = self.tokenizer(
            sentence_1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding_2 = self.tokenizer(
            sentence_2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoding_1['input_ids'].squeeze(0),
            'attention_mask_1': encoding_1['attention_mask'].squeeze(0),
            'input_ids_2': encoding_2['input_ids'].squeeze(0),
            'attention_mask_2': encoding_2['attention_mask'].squeeze(0),
            'similarity_labels': torch.tensor(similarity, dtype=torch.float)
        }

def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, 
                    max_length: int = 512, task: str = "classification") -> DatasetDict:
    """Tokenize a dataset."""
    tokenized_datasets = {}
    
    for split in dataset.keys():
        tokenized_datasets[split] = BookClassificationDataset(
            dataset[split], tokenizer, max_length, task
        )
        
    return tokenized_datasets

def create_dataloaders(datasets: Union[DatasetDict, Dict], 
                      batch_size: int = 16,
                      num_workers: int = 4,
                      shuffle: bool = True) -> Dict[str, DataLoader]:
    """Create dataloaders from datasets."""
    dataloaders = {}
    
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if split == 'train' else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=split == 'train'
        )
        
    return dataloaders

def collate_fn_classification(batch):
    """Collate function for classification task."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def collate_fn_mlm(batch):
    """Collate function for MLM task."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    mlm_labels = torch.stack([item['mlm_labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'mlm_labels': mlm_labels
    }

def collate_fn_nsp(batch):
    """Collate function for NSP task."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    nsp_labels = torch.stack([item['nsp_labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'nsp_labels': nsp_labels
    }

def collate_fn_similarity(batch):
    """Collate function for similarity task."""
    input_ids_1 = torch.stack([item['input_ids_1'] for item in batch])
    attention_mask_1 = torch.stack([item['attention_mask_1'] for item in batch])
    input_ids_2 = torch.stack([item['input_ids_2'] for item in batch])
    attention_mask_2 = torch.stack([item['attention_mask_2'] for item in batch])
    similarity_labels = torch.stack([item['similarity_labels'] for item in batch])
    
    return {
        'input_ids_1': input_ids_1,
        'attention_mask_1': attention_mask_1,
        'input_ids_2': input_ids_2,
        'attention_mask_2': attention_mask_2,
        'similarity_labels': similarity_labels
    }

def create_task_specific_dataloaders(datasets: Dict, task: str, 
                                   batch_size: int = 16,
                                   num_workers: int = 4) -> Dict[str, DataLoader]:
    """Create task-specific dataloaders."""
    collate_fn_map = {
        'classification': collate_fn_classification,
        'mlm': collate_fn_mlm,
        'nsp': collate_fn_nsp,
        'similarity': collate_fn_similarity
    }
    
    dataloaders = {}
    
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_map.get(task, collate_fn_classification),
            drop_last=split == 'train'
        )
        
    return dataloaders

def get_dataset_statistics(dataset: DatasetDict) -> Dict:
    """Get statistics about the dataset."""
    stats = {}
    
    for split in dataset.keys():
        split_data = dataset[split]
        
        # Count samples per class
        if 'label' in split_data.column_names:
            labels = split_data['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            stats[split] = {
                'total_samples': len(split_data),
                'num_classes': len(unique_labels),
                'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist()))
            }
            
            # Average sentence length
            if 'sentence' in split_data.column_names:
                sentence_lengths = [len(sentence.split()) for sentence in split_data['sentence']]
                stats[split]['avg_sentence_length'] = np.mean(sentence_lengths)
                stats[split]['min_sentence_length'] = np.min(sentence_lengths)
                stats[split]['max_sentence_length'] = np.max(sentence_lengths)
                
    return stats

def balance_dataset(dataset: DatasetDict, method: str = "undersample") -> DatasetDict:
    """Balance dataset by class distribution."""
    if method == "undersample":
        return _undersample_dataset(dataset)
    elif method == "oversample":
        return _oversample_dataset(dataset)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
        
def _undersample_dataset(dataset: DatasetDict) -> DatasetDict:
    """Undersample majority classes to balance dataset."""
    balanced_datasets = {}
    
    for split in dataset.keys():
        split_data = dataset[split]
        
        if 'label' not in split_data.column_names:
            balanced_datasets[split] = split_data
            continue
            
        # Get class distribution
        labels = split_data['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)
        
        # Sample equal number from each class
        balanced_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(label_indices, min_count, replace=False)
            balanced_indices.extend(sampled_indices)
            
        # Create balanced dataset
        balanced_datasets[split] = split_data.select(balanced_indices)
        
    return DatasetDict(balanced_datasets)

def _oversample_dataset(dataset: DatasetDict) -> DatasetDict:
    """Oversample minority classes to balance dataset."""
    balanced_datasets = {}
    
    for split in dataset.keys():
        split_data = dataset[split]
        
        if 'label' not in split_data.column_names:
            balanced_datasets[split] = split_data
            continue
            
        # Get class distribution
        labels = split_data['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)
        
        # Oversample each class to max_count
        balanced_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            
            if len(label_indices) < max_count:
                # Oversample with replacement
                sampled_indices = np.random.choice(label_indices, max_count, replace=True)
            else:
                sampled_indices = label_indices
                
            balanced_indices.extend(sampled_indices)
            
        # Create balanced dataset
        balanced_datasets[split] = split_data.select(balanced_indices)
        
    return DatasetDict(balanced_datasets) 