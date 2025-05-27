import torch
import torch.nn as nn
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import json
import os
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PBTConfig:
    """Configurazione per Population Based Training"""
    def __init__(self):
        # Parametri della popolazione
        self.population_size = 2
        self.exploit_interval = 1000  # ogni quanti step fare exploit/explore
        self.truncation_threshold = 0.2  # frazione bottom da rimpiazzare
        
        # Spazio degli iperparametri
        self.hyperparameter_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 2e-4],
            'weight_decay': [0.0, 0.01, 0.1],
            'warmup_ratio': [0.06, 0.1, 0.2],
            'dropout': [0.1, 0.2, 0.3]
        }
        
        # Parametri di perturbazione
        self.perturb_factors = [0.8, 1.2]  # fattori moltiplicativi per perturbazione

class Individual:
    """Rappresenta un individuo nella popolazione PBT"""
    def __init__(self, model, optimizer, scheduler, hyperparams: Dict, individual_id: int):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hyperparams = hyperparams.copy()
        self.id = individual_id
        
        # Metriche di performance
        self.performance_history = []
        self.current_performance = 0.0
        self.step = 0
        
    def get_performance(self) -> float:
        """Ritorna la performance corrente (es. accuracy di validazione)"""
        return self.current_performance
    
    def update_performance(self, performance: float):
        """Aggiorna la performance corrente"""
        self.current_performance = performance
        self.performance_history.append(performance)
    
    def copy_weights_from(self, other_individual):
        """Copia i pesi da un altro individuo"""
        self.model.load_state_dict(other_individual.model.state_dict())
    
    def save_checkpoint(self, path: str):
        """Salva checkpoint dell'individuo"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparams': self.hyperparams,
            'performance_history': self.performance_history,
            'current_performance': self.current_performance,
            'step': self.step
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Carica checkpoint dell'individuo"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.hyperparams = checkpoint['hyperparams']
        self.performance_history = checkpoint['performance_history']
        self.current_performance = checkpoint['current_performance']
        self.step = checkpoint['step']

class PBTTrainer:
    """Trainer principale per Population Based Training"""
    
    def __init__(self, config: PBTConfig, num_labels: int = 5, model_name: str = "distilroberta-base"):
        self.config = config
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Inizializza popolazione
        self.population = []
        self._initialize_population()
        
        # Statistiche
        self.generation = 0
        self.total_steps = 0
        
    def _initialize_population(self):
        """Inizializza la popolazione con iperparametri casuali"""
        for i in range(self.config.population_size):
            # Campiona iperparametri casuali
            hyperparams = self._sample_hyperparameters()
            
            # Crea modello
            model = RobertaForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels,
                hidden_dropout_prob=hyperparams['dropout']
            )
            
            # Crea optimizer
            optimizer = AdamW(
                model.parameters(),
                lr=hyperparams['learning_rate'],
                weight_decay=hyperparams['weight_decay']
            )
            
            # Placeholder per scheduler (sarà inizializzato con il numero totale di step)
            scheduler = None
            
            # Crea individuo
            individual = Individual(model, optimizer, scheduler, hyperparams, i)
            self.population.append(individual)
    
    def _sample_hyperparameters(self) -> Dict:
        """Campiona iperparametri casuali dallo spazio definito"""
        hyperparams = {}
        for param, values in self.config.hyperparameter_space.items():
            hyperparams[param] = random.choice(values)
        return hyperparams
    
    def _perturb_hyperparameters(self, hyperparams: Dict) -> Dict:
        """Perturba gli iperparametri esistenti"""
        new_hyperparams = hyperparams.copy()
        
        for param in new_hyperparams:
            if random.random() < 0.3:  # 30% probabilità di perturbazione
                if param in ['learning_rate', 'weight_decay', 'warmup_ratio']:
                    # Perturbazione moltiplicativa per parametri continui
                    factor = random.choice(self.config.perturb_factors)
                    new_hyperparams[param] *= factor
                    
                    # Clamp ai valori validi
                    if param == 'learning_rate':
                        new_hyperparams[param] = max(1e-6, min(1e-3, new_hyperparams[param]))
                    elif param == 'weight_decay':
                        new_hyperparams[param] = max(0.0, min(0.5, new_hyperparams[param]))
                    elif param == 'warmup_ratio':
                        new_hyperparams[param] = max(0.0, min(0.5, new_hyperparams[param]))
                
                elif param == 'dropout':
                    # Campiona nuovo valore per dropout
                    new_hyperparams[param] = random.choice(self.config.hyperparameter_space[param])
        
        return new_hyperparams
    
    def _exploit_and_explore(self):
        """Esegue fase di exploit e explore della popolazione"""
        # Ordina popolazione per performance
        self.population.sort(key=lambda x: x.get_performance(), reverse=True)
        
        # Calcola soglie
        top_k = int(len(self.population) * (1 - self.config.truncation_threshold))
        bottom_k = len(self.population) - top_k
        
        logger.info(f"Exploit & Explore - Generation {self.generation}")
        logger.info(f"Top performers: {[f'ID:{ind.id}({ind.get_performance():.4f})' for ind in self.population[:3]]}")
        
        # Exploit: rimpiazza bottom performers con top performers
        for i in range(bottom_k):
            bottom_idx = len(self.population) - 1 - i
            top_idx = random.randint(0, top_k - 1)
            
            bottom_individual = self.population[bottom_idx]
            top_individual = self.population[top_idx]
            
            logger.info(f"Replacing individual {bottom_individual.id} (perf: {bottom_individual.get_performance():.4f}) "
                       f"with copy of {top_individual.id} (perf: {top_individual.get_performance():.4f})")
            
            # Copia pesi
            bottom_individual.copy_weights_from(top_individual)
            
            # Explore: perturba iperparametri
            new_hyperparams = self._perturb_hyperparameters(top_individual.hyperparams)
            bottom_individual.hyperparams = new_hyperparams
            
            # Aggiorna optimizer con nuovi iperparametri
            self._update_optimizer_hyperparams(bottom_individual)
        
        self.generation += 1
    
    def _update_optimizer_hyperparams(self, individual: Individual):
        """Aggiorna gli iperparametri dell'optimizer"""
        # Crea nuovo optimizer con nuovi iperparametri
        individual.optimizer = AdamW(
            individual.model.parameters(),
            lr=individual.hyperparams['learning_rate'],
            weight_decay=individual.hyperparams['weight_decay']
        )
    
    def train_step(self, individual: Individual, batch: Dict, device: torch.device) -> float:
        """Esegue un singolo step di training per un individuo"""
        individual.model.train()
        individual.model.to(device)

        import sys
        print(f"[DEBUG] Step {individual.step}, Indiv {individual.id}")
        print(" > input_ids shape:", batch["input_ids"].shape)
        print(" > attention_mask shape:", batch["attention_mask"].shape)
        print(" > labels:", batch["labels"])
        sys.stdout.flush()

        # Forward pass
        outputs = individual.model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )

        loss = outputs.loss
        print(f"[DEBUG] Loss computed: {loss.item()}")
        sys.stdout.flush()

        # Backward pass
        individual.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(individual.model.parameters(), 1.0)

        individual.optimizer.step()
        if individual.scheduler:
            individual.scheduler.step()

        individual.step += 1

        return loss.item()
    
    def evaluate(self, individual: Individual, dataloader: DataLoader, device: torch.device) -> float:
        """Valuta un individuo sul validation set"""
        individual.model.eval()
        individual.model.to(device)
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = individual.model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_correct += (predictions == batch['labels'].to(device)).sum().item()
                total_samples += batch['labels'].size(0)
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              num_epochs: int, device: torch.device, save_dir: str = None):
        """Training loop principale"""

        import time
        start_time = time.time()

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pbt_checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        # Inizializza schedulers
        total_steps = len(train_dataloader) * num_epochs
        for individual in self.population:
            individual.scheduler = get_linear_schedule_with_warmup(
                individual.optimizer,
                num_warmup_steps=int(total_steps * individual.hyperparams['warmup_ratio']),
                num_training_steps=total_steps
            )

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(train_dataloader):

                # Train step per ogni individuo
                for individual in self.population:
                    loss = self.train_step(individual, batch, device)

                self.total_steps += 1

                # Exploit & Explore periodicamente
                if self.total_steps % self.config.exploit_interval == 0:
                    # Valuta popolazione
                    for individual in self.population:
                        performance = self.evaluate(individual, val_dataloader, device)
                        individual.update_performance(performance)

                    # Exploit & Explore
                    self._exploit_and_explore()

                    # Salva checkpoint
                    self._save_population_checkpoint(save_dir)

                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}, Step {self.total_steps}")

        # Valutazione finale
        final_performances = []
        for individual in self.population:
            performance = self.evaluate(individual, val_dataloader, device)
            individual.update_performance(performance)
            final_performances.append(performance)

        # Salva risultati finali
        self._save_final_results(save_dir, final_performances)

        # # Confusion Matrix per il best individual
        # from sklearn.metrics import confusion_matrix, classification_report
        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # best_individual = max(self.population, key=lambda x: x.get_performance())
        # best_individual.model.eval()
        # best_individual.model.to(device)

        # all_preds = []
        # all_labels = []

        # with torch.no_grad():
        #     for batch in val_dataloader:
        #         input_ids = batch["input_ids"].to(device)
        #         attention_mask = batch["attention_mask"].to(device)
        #         labels = batch["labels"].to(device)

        #         outputs = best_individual.model(input_ids=input_ids, attention_mask=attention_mask)
        #         preds = torch.argmax(outputs.logits, dim=1)

        #         all_preds.extend(preds.cpu().numpy())
        #         all_labels.extend(labels.cpu().numpy())

        # # Confusion Matrix
        # cm = confusion_matrix(all_labels, all_preds)
        # print("Confusion Matrix:")
        # print(cm)

        # # Classification Report
        # print("\nClassification Report:")
        # print(classification_report(all_labels, all_preds))

        # # Visualizzazione Confusion Matrix
        # plt.figure(figsize=(8,6))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.title("Confusion Matrix")
        # plt.show()

        # end_time = time.time()
        # total_time = end_time - start_time
        # print(f"Tempo totale di training: {total_time:.2f} secondi")

        return self.population

    def print_confusion_matrix(self, dataloader: DataLoader, device: torch.device):
        """Stampa confusion matrix e classification report per il best individual"""
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json
        import os

        best_individual = max(self.population, key=lambda x: x.get_performance())
        best_individual.model.eval()
        best_individual.model.to(device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = best_individual.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Salva y_true e y_pred come int in un file JSON nella directory pbt_checkpoints accanto a questo file
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pbt_checkpoints", "confusion_matrix_predictions.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"y_true": [int(x) for x in all_labels], "y_pred": [int(x) for x in all_preds]}, f)

        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    
    def _save_population_checkpoint(self, save_dir: str):
        """Salva checkpoint dell'intera popolazione"""
        for individual in self.population:
            checkpoint_path = os.path.join(save_dir, f"individual_{individual.id}_gen_{self.generation}.pt")
            individual.save_checkpoint(checkpoint_path)
    
    def _save_final_results(self, save_dir: str, performances: List[float]):
        """Salva risultati finali"""
        results = {
            'generation': self.generation,
            'total_steps': self.total_steps,
            'final_performances': performances,
            'best_performance': max(performances),
            'population_hyperparams': [ind.hyperparams for ind in self.population],
            'performance_histories': [ind.performance_history for ind in self.population]
        }
        
        with open(os.path.join(save_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Salva il miglior modello
        best_individual = max(self.population, key=lambda x: x.get_performance())
        best_model_path = os.path.join(save_dir, 'best_model')
        best_individual.model.save_pretrained(best_model_path)
        # Salva anche il tokenizer nella stessa cartella
        self.tokenizer.save_pretrained(best_model_path)
        
        logger.info(f"Best performance: {max(performances):.4f}")
        logger.info(f"Best hyperparams: {best_individual.hyperparams}")

# Caricamento dataset JSON con coppie utterance-hat indipendenti
def load_utterance_hat_dataset(json_file_path: str, tokenizer, max_length: int = 512):
    """Carica dataset JSON con coppie indipendenti utterance/hat"""
    
    class UtteranceHatDataset(Dataset):
        def __init__(self, json_file_path, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.texts = []
            self.labels = []
            
            # Carica il file JSON
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Processa ogni coppia utterance-hat
            for item in data:
                if 'utterance' in item and 'hat' in item:
                    # Usa l'utterance come input
                    utterance = item['utterance']
                    
                    # Usa hat come label (devi definire la logica di mappatura)
                    hat_value = item['hat']
                    label = self._process_hat_to_label(hat_value)
                    
                    self.texts.append(utterance)
                    self.labels.append(label)
            
            logger.info(f"Caricati {len(self.texts)} esempi da {json_file_path}")
            logger.info(f"Distribuzione labels: {np.bincount(self.labels)}")
        
        def _process_hat_to_label(self, hat_value):
            """
            Converte il valore 'hat' in una label numerica
            MODIFICA QUESTA FUNZIONE SECONDO IL TUO TASK
            """
            # Esempio 1: Se hat è già un numero (0, 1, 2, etc.)
            if isinstance(hat_value, int):
                return hat_value
            
            # Esempio 2: Se hat è una stringa da mappare
            if isinstance(hat_value, str):
                # Definisci la tua mappatura qui
                mapping = {
                    "positive": 1,
                    "negative": 0,
                    # aggiungi altre mappature secondo necessità
                }
                return mapping.get(hat_value.lower(), 0)  # default a 0
            
            # Esempio 3: Se hat è un dizionario con informazioni strutturate
            if isinstance(hat_value, dict):
                # Estrai la label dal dizionario secondo la tua struttura
                return hat_value.get('label', 0)  # esempio
            
            # Default: ritorna 0
            return 0
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            # Tokenizza solo l'utterance
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    return UtteranceHatDataset(json_file_path, tokenizer, max_length)

def load_utterance_hat_with_custom_mapping(json_file_path: str, tokenizer, label_mapping_func, max_length: int = 512):
    """
    Versione più flessibile dove puoi passare una funzione custom per mappare hat -> label
    
    Args:
        json_file_path: path al file JSON
        tokenizer: tokenizer del modello  
        label_mapping_func: funzione che prende hat_value e ritorna label numerica
        max_length: lunghezza massima sequenza
    """
    
    class CustomUtteranceHatDataset(Dataset):
        def __init__(self, json_file_path, tokenizer, label_mapping_func, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.label_mapping_func = label_mapping_func
            self.texts = []
            self.labels = []
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if 'utterance' in item and 'hat' in item:
                    utterance = item['utterance']
                    hat_value = item['hat']
                    
                    # Usa la funzione custom per mappare hat -> label
                    try:
                        label = self.label_mapping_func(hat_value)
                        self.texts.append(utterance)
                        self.labels.append(label)
                    except Exception as e:
                        logger.warning(f"Errore nel mappare hat_value {hat_value}: {e}")
                        continue
            
            logger.info(f"Dataset creato con {len(self.texts)} esempi")
            unique_labels = np.unique(self.labels)
            logger.info(f"Labels uniche: {unique_labels}")
            for label in unique_labels:
                count = np.sum(np.array(self.labels) == label)
                logger.info(f"Label {label}: {count} esempi")
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    return CustomUtteranceHatDataset(json_file_path, tokenizer, label_mapping_func, max_length)

def install_requirements(directory: str):
    """Installa i requisiti dal file requirements.txt"""
    import subprocess
    import sys
    file = directory+"/distilroberta_requirements.txt"
    if not os.path.exists(file):
        raise FileNotFoundError(f"File requirements non trovato: {file}")
    print(f"Installing requirements from {file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file])

if __name__ == "__main__":
    this_file_dir = str(os.path.dirname(os.path.abspath(__file__)))
    # from this_file_dir go to the parent directory
    dataset_dir = os.path.dirname(this_file_dir)
    
    # Installa i requisiti se necessario
    # install_requirements(this_file_dir)
    
    # Configurazione
    config = PBTConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inizializza trainer
    trainer = PBTTrainer(config, num_labels=5)
       
    # Carica i tuoi dataset JSON (coppie utterance-hat indipendenti)
    train_dataset = load_utterance_hat_dataset(
        # "./Models Fine Tuning/eda_train_dataset.json", 
        dataset_dir+"/eda_train_dataset.json",
        trainer.tokenizer, 
        max_length=512
    )
    test_dataset = load_utterance_hat_dataset(
        # "./Models Fine Tuning/test_dataset.json",
        dataset_dir+"/synthetic_test_dataset.json", 
        trainer.tokenizer, 
        max_length=512
    )
    
    labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
    print(f"Min: {min(labels)}, Max: {max(labels)}")
    
    # Oppure usa la versione con mapping personalizzato:
    # def my_hat_to_label(hat_value):
    #     # La tua logica personalizzata
    #     if hat_value == 1:
    #         return 1
    #     else:
    #         return 0
    # 
    # train_dataset = load_utterance_hat_with_custom_mapping(
    #     "train_dataset.json", 
    #     trainer.tokenizer, 
    #     my_hat_to_label,
    #     max_length=512
    # )
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Misura il tempo di esecuzione del main
    import time
    start_time = time.time()
    
    # Avvia training
    trained_population = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=5,
        device=device
    ) 
    
    end_time = time.time()
    print(f"Tempo totale esecuzione main: {end_time - start_time:.2f} secondi")
    
    print("Training completato!")
    print(f"Miglior performance: {max([ind.get_performance() for ind in trained_population]):.4f}")

    # Stampa la confusion matrix per il best individual
    trainer.print_confusion_matrix(val_dataloader, device)
    