"""
Constructive Learning Model for Book Sentence Classification.
Implements incremental knowledge building through multiple learning phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

class ConstructiveLearningModel(nn.Module):
    """
    Constructive Learning Model that builds knowledge incrementally.
    
    Phases:
    1. Basic Understanding: MLM and NSP tasks
    2. Book Patterns: Sentence similarity and book classification
    3. Cross-book Knowledge: All tasks with cross-book learning
    4. Fine-tuning: Final classification optimization
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_config = self.config['model']
        self.encoder_config = self.model_config['encoder']
        
        # Initialize encoder
        self.encoder = AutoModel.from_pretrained(self.encoder_config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_config['model_name'])
        
        # Add special tokens if needed
        special_tokens = ['[BOOK_SEP]', '[SENTENCE_SEP]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Constructive learning components
        self._init_constructive_components()
        
        # Self-supervised heads
        self._init_self_supervised_heads()
        
        # Classification head
        self._init_classification_head()
        
        # Phase tracking
        self.current_phase = 0
        self.phase_knowledge = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_constructive_components(self):
        """Initialize constructive learning components."""
        hidden_size = self.encoder_config['hidden_size']
        
        # Knowledge accumulation layers
        self.knowledge_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ])
        
        # Knowledge gates (control information flow)
        self.knowledge_gates = nn.ModuleList([
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Sigmoid()
        ])
        
        # Knowledge memory (store accumulated knowledge)
        self.knowledge_memory = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size)),
            nn.Parameter(torch.randn(hidden_size)),
            nn.Parameter(torch.randn(hidden_size)),
            nn.Parameter(torch.randn(hidden_size))
        ])
        
        # Phase-specific adapters
        self.phase_adapters = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ])
        
    def _init_self_supervised_heads(self):
        """Initialize self-supervised learning heads."""
        hidden_size = self.encoder_config['hidden_size']
        
        # Masked Language Modeling head
        self.mlm_head = nn.Linear(hidden_size, self.tokenizer.vocab_size)
        
        # Next Sentence Prediction head
        self.nsp_head = nn.Linear(hidden_size, 2)
        
        # Sentence Similarity head
        self.similarity_head = nn.Linear(hidden_size * 2, 1)
        
        # Contrastive learning projection
        self.contrastive_projection = nn.Linear(hidden_size, hidden_size)
        
    def _init_classification_head(self):
        """Initialize classification head."""
        hidden_size = self.encoder_config['hidden_size']
        num_classes = self.config['data']['selected_books'].__len__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.encoder_config['dropout']),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def set_phase(self, phase: int):
        """Set the current learning phase."""
        self.current_phase = phase
        self.logger.info(f"Switching to phase {phase}")
        
    def _apply_constructive_knowledge(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply constructive knowledge building."""
        current_knowledge = hidden_states
        
        for phase in range(self.current_phase + 1):
            # Apply knowledge layer
            knowledge_output = self.knowledge_layers[phase](current_knowledge)
            
            # Apply knowledge gate
            gate_output = self.knowledge_gates[phase](knowledge_output)
            
            # Combine with stored knowledge
            stored_knowledge = self.knowledge_memory[phase].unsqueeze(0).expand_as(current_knowledge)
            combined_knowledge = gate_output * current_knowledge + (1 - gate_output) * stored_knowledge
            
            # Apply phase adapter
            adapted_knowledge = self.phase_adapters[phase](combined_knowledge)
            
            # Update current knowledge
            current_knowledge = adapted_knowledge
            
        return current_knowledge
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                task: str = "classification",
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with constructive learning.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            task: Task type ("classification", "mlm", "nsp", "similarity")
            **kwargs: Additional arguments
        """
        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get [CLS] representation
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        # Apply constructive knowledge building
        enhanced_output = self._apply_constructive_knowledge(cls_output)
        
        outputs = {}
        
        if task == "classification":
            logits = self.classification_head(enhanced_output)
            outputs['logits'] = logits
            
        elif task == "mlm":
            # Masked language modeling
            mlm_logits = self.mlm_head(enhanced_output)
            outputs['mlm_logits'] = mlm_logits
            
        elif task == "nsp":
            # Next sentence prediction
            nsp_logits = self.nsp_head(enhanced_output)
            outputs['nsp_logits'] = nsp_logits
            
        elif task == "similarity":
            # Sentence similarity
            if 'sentence_2_encoded' in kwargs:
                sentence_2_output = kwargs['sentence_2_encoded']
                combined = torch.cat([enhanced_output, sentence_2_output], dim=-1)
                similarity_score = self.similarity_head(combined)
                outputs['similarity_score'] = similarity_score
            else:
                # Contrastive learning
                projected = self.contrastive_projection(enhanced_output)
                outputs['projected'] = projected
                
        elif task == "all":
            # All tasks simultaneously
            logits = self.classification_head(enhanced_output)
            mlm_logits = self.mlm_head(enhanced_output)
            nsp_logits = self.nsp_head(enhanced_output)
            
            outputs.update({
                'logits': logits,
                'mlm_logits': mlm_logits,
                'nsp_logits': nsp_logits
            })
            
        return outputs
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    labels: Optional[torch.Tensor] = None,
                    task: str = "classification",
                    **kwargs) -> torch.Tensor:
        """Compute loss for different tasks."""
        
        if task == "classification":
            return F.cross_entropy(outputs['logits'], labels)
            
        elif task == "mlm":
            mlm_labels = kwargs.get('mlm_labels')
            return F.cross_entropy(outputs['mlm_logits'].view(-1, self.tokenizer.vocab_size), 
                                 mlm_labels.view(-1))
                                 
        elif task == "nsp":
            nsp_labels = kwargs.get('nsp_labels')
            return F.cross_entropy(outputs['nsp_logits'], nsp_labels)
            
        elif task == "similarity":
            similarity_labels = kwargs.get('similarity_labels')
            return F.binary_cross_entropy_with_logits(outputs['similarity_score'], similarity_labels)
            
        elif task == "all":
            # Combined loss
            total_loss = 0
            
            if 'labels' in kwargs:
                cls_loss = F.cross_entropy(outputs['logits'], kwargs['labels'])
                total_loss += cls_loss
                
            if 'mlm_labels' in kwargs:
                mlm_loss = F.cross_entropy(outputs['mlm_logits'].view(-1, self.tokenizer.vocab_size),
                                         kwargs['mlm_labels'].view(-1))
                total_loss += mlm_loss
                
            if 'nsp_labels' in kwargs:
                nsp_loss = F.cross_entropy(outputs['nsp_logits'], kwargs['nsp_labels'])
                total_loss += nsp_loss
                
            return total_loss
            
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def update_knowledge_memory(self, phase: int, knowledge: torch.Tensor):
        """Update knowledge memory for a specific phase."""
        with torch.no_grad():
            self.knowledge_memory[phase].data = knowledge.mean(dim=0)
            
    def get_phase_knowledge(self, phase: int) -> torch.Tensor:
        """Get accumulated knowledge for a specific phase."""
        return self.knowledge_memory[phase].data.clone()
        
    def save_phase_knowledge(self, phase: int, save_path: str):
        """Save phase-specific knowledge."""
        knowledge = self.get_phase_knowledge(phase)
        torch.save(knowledge, save_path)
        self.logger.info(f"Saved phase {phase} knowledge to {save_path}")
        
    def load_phase_knowledge(self, phase: int, load_path: str):
        """Load phase-specific knowledge."""
        knowledge = torch.load(load_path)
        with torch.no_grad():
            self.knowledge_memory[phase].data = knowledge
        self.logger.info(f"Loaded phase {phase} knowledge from {load_path}")

class ConstructiveTrainer:
    """Trainer for constructive learning model."""
    
    def __init__(self, model: ConstructiveLearningModel, config_path: str = "configs/config.yaml"):
        self.model = model
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.training_config = self.config['training']
        self.constructive_config = self.config['model']['constructive_learning']
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train_phase(self, phase: int, dataloader, optimizer, scheduler=None):
        """Train a specific phase of constructive learning."""
        self.model.set_phase(phase)
        phase_config = self.constructive_config['phases'][phase]
        
        self.logger.info(f"Training phase {phase}: {phase_config['name']}")
        
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Phase {phase}")):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task=phase_config['tasks'][0] if len(phase_config['tasks']) == 1 else "all"
            )
            
            # Compute loss
            loss = self.model.compute_loss(outputs, **batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.training_config['max_grad_norm'])
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                
            total_loss += loss.item()
            
            # Update knowledge memory periodically
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    cls_output = self.model.encoder(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    self.model.update_knowledge_memory(phase, cls_output)
                    
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"Phase {phase} average loss: {avg_loss:.4f}")
        
        return avg_loss
        
    def train_all_phases(self, dataloaders, optimizers, schedulers=None):
        """Train all phases of constructive learning."""
        phase_losses = []
        
        for phase in range(len(self.constructive_config['phases'])):
            phase_config = self.constructive_config['phases'][phase]
            
            # Create optimizer for this phase
            if isinstance(optimizers, list):
                optimizer = optimizers[phase]
            else:
                optimizer = optimizers
                
            # Create scheduler for this phase
            scheduler = None
            if schedulers and isinstance(schedulers, list):
                scheduler = schedulers[phase]
            elif schedulers:
                scheduler = schedulers
                
            # Train phase
            loss = self.train_phase(phase, dataloaders[phase], optimizer, scheduler)
            phase_losses.append(loss)
            
            # Save phase knowledge
            save_path = f"experiments/checkpoints/phase_{phase}_knowledge.pt"
            self.model.save_phase_knowledge(phase, save_path)
            
        return phase_losses 