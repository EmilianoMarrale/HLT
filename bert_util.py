import random

import pandas
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sympy.physics.units import momentum
from transformers import BertTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime
from typing import Optional
from transformers.modeling_utils import SpecificPreTrainedModelType
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, PeftModel
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import get_peft_model
from peft import LoraConfig
from torch.optim import AdamW
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from typing import Union
from torch.optim import SGD
from transformers import Adafactor
import random
import numpy as np

hat_map = {
    0: "red",
    1: "white",
    2: "black",
    3: "yellow",
    4: "green",
}

def scoring_fn(model, val_dataloader, test_labels):
    # You need to implement or import this function.
    # It should return a scalar metric (e.g., accuracy, F1 score, etc.)
    # Example: return accuracy_score(y_true, y_pred)
    preds, confs = model_predict(model, val_dataloader)
    labels_flat = test_labels.flatten()

    preds_array = np.array(preds)
    cl_rep = classification_report(labels_flat, preds_array, target_names=list(hat_map.values()), output_dict=True)
    print(classification_report(labels_flat, preds_array, target_names=list(hat_map.values())))
    return cl_rep["accuracy"]  # or any other metric you want to return


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (slower but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def randomized_cv_rasearch(model_fn, tokenizer, train_data, train_labels, test_data, test_labels, num_samples=10, use_lora=True):
    """
    Performs a randomized search over hyperparameters with simple holdout validation.

    Parameters:
    - model_class: Callable that returns a fresh instance of your model
    - tokenizer: HuggingFace tokenizer
    - train_data: Training data (e.g., pandas.Series of texts)
    - train_labels: Training labels
    - configs: Dictionary of hyperparameter options
    - num_samples: Number of configurations to try
    - scoring_fn: Function to evaluate the model (takes model and val_dataloader)
    - use_lora: Whether to apply LoRA

    Returns:
    - best_model: Model with the best score
    - best_config: Configuration dictionary for the best model
    - best_score: Best evaluation score
    """
    set_seed(seed=42)
    configs = {
        "epochs": [10, 15, 20, 30],
        "model_dropout": [0.1, 0.2, 0.3],
        "optimizer_lr": [1e-5, 2e-5, 3e-5, 5e-5],
        "scheduler_warmup_steps": [0, 100, 200],
        "lora_r": [4, 8, 16],
        "lora_alpha": [16, 32, 64],
        "lora_dropout": [0.05, 0.1, 0.2],
        "tokenizer_max_length": [16, 32, 64, 128],
        "dataloader_batch_size": [8, 16, 32],
        "clip_grad_norm": [1.0, 2.0, 5.0],
        "early_stopping_patience": [3, 5, 10],
        "early_stopping_delta": [0.01, 0.05, 0.1],
        "scheduler_type": ["linear", "cosine", "constant"],
        "optimizer_type": ["AdamW", "Adafactor", "SGD"],
        "weight_decay": [0.01, 0.001, 0.0001],
    }

    best_score = -np.inf
    best_model = None
    best_config = None

    for i in range(num_samples):

        config = sample_config(configs)
        print(f"Trying configuration {i + 1}/{num_samples}: {config}")
        model = model_fn()
        model.config.hidden_dropout_prob = config["model_dropout"]
        model.config.attention_probs_dropout_prob = config["model_dropout"]

        # Train the model with current configuration
        trained_model = train_pipeline(model, tokenizer, train_data, train_labels, config=config, use_lora=use_lora)

        # Tokenize and prepare val_dataloader again to evaluate
        tids, amids = bert_tokenize_data(tokenizer, pd.Series(test_data), max_length=config["tokenizer_max_length"])
        val_dataloader = get_data_loader(tids, amids, batch_size=config["dataloader_batch_size"], shuffle=False)

        score = scoring_fn(trained_model, val_dataloader, test_labels)

        print(f"Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = trained_model
            best_config = config

    return best_model, best_config, best_score


def sample_config(config_parameters):
    return {key: random.choice(values) for key, values in config_parameters.items()}

def train_pipeline(model :SpecificPreTrainedModelType, tokenizer : PreTrainedTokenizer, train_data :pandas.Series, train_labels :pandas.Series, config=None, use_lora=True):

    token_ids, attention_masks = bert_tokenize_data(tokenizer, train_data, max_length=config["tokenizer_max_length"])
    train_dataloader, val_dataloader = tensor_train_test_split(torch.tensor(train_labels), token_ids, attention_masks, test_size=0.2)

    epochs = config["epochs"]

    if use_lora:
        # Load model and apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            inference_mode=False
        )
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, lora_config)

    optimizer = None
    if config["optimizer_type"] == "AdamW":
        optimizer =  AdamW(model.parameters(), lr=config["optimizer_lr"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "Adafactor":
        optimizer = Adafactor(model.parameters(),  lr=config["optimizer_lr"], weight_decay=config["weight_decay"], scale_parameter=True, relative_step=False)
    elif config["optimizer_type"] == "SGD":
        optimizer = SGD(model.parameters(),  lr=config["optimizer_lr"], weight_decay=config["weight_decay"])


    num_training_steps = epochs * len(train_dataloader)
    scheduler = None
    if config["scheduler_type"] == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, config["scheduler_warmup_steps"], num_training_steps)
    elif config["scheduler_type"] == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, config["scheduler_warmup_steps"], num_training_steps)
    elif config["scheduler_type"] == "constant":
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(optimizer, config["scheduler_warmup_steps"])

    model = train_bert_model(model, optimizer, scheduler, train_dataloader, val_dataloader, config)
    return model


def train_bert_model(model: Union[SpecificPreTrainedModelType, PeftModel],
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    config  # Minimum change to qualify as improvement
                    ) -> SpecificPreTrainedModelType:

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_dict = {
        'epoch': [],
        'average training loss': [],
        'average validation loss': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    t0_train = datetime.now()

    for epoch in range(config["epochs"]):
        model.train()
        training_loss = 0
        t0_epoch = datetime.now()

        print(f'\n{"-" * 20} Epoch {epoch + 1} {"-" * 20}')
        print('Training:\n---------')
        print(f'Start Time:       {t0_epoch}')

        for batch in train_dataloader:
            batch_token_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            model.zero_grad()
            loss, logits = model(
                batch_token_ids,
                token_type_ids=None,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                return_dict=False
            )
            training_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()
            scheduler.step()

        average_train_loss = training_loss / len(train_dataloader)
        time_epoch = datetime.now() - t0_epoch
        print(f'Average Training Loss: {average_train_loss}')
        print(f'Time Taken:            {time_epoch}')

        model.eval()
        val_loss = 0
        val_accuracy = 0
        t0_val = datetime.now()
        print('\nValidation:\n-----------')
        print(f'Start Time:       {t0_val}')

        for batch in val_dataloader:
            batch_token_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                loss, logits = model(
                    batch_token_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    token_type_ids=None,
                    return_dict=False
                )
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(logits, label_ids)

        average_val_loss = val_loss / len(val_dataloader)
        average_val_accuracy = val_accuracy / len(val_dataloader)
        time_val = datetime.now() - t0_val

        print(f'Average Validation Loss:     {average_val_loss}')
        print(f'Average Validation Accuracy: {average_val_accuracy}')
        print(f'Time Taken:                  {time_val}')

        # Store losses
        loss_dict['epoch'].append(epoch + 1)
        loss_dict['average training loss'].append(average_train_loss)
        loss_dict['average validation loss'].append(average_val_loss)

        # Early stopping check
        if average_val_loss < best_val_loss - config["early_stopping_delta"]:
            best_val_loss = average_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save best model state
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epoch(s).')
            if patience_counter >= config["early_stopping_patience"]:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
                model.load_state_dict(best_model_state)  # Load best model
                break

    print(f'\nTotal training time: {datetime.now() - t0_train}')
    return model



def bert_tokenize_data(tokenizer :PreTrainedTokenizer,  series: pd.Series, max_length: int=128, truncation: bool=True, padding :str='max_length') -> (torch.Tensor, torch.Tensor):
    """
    Tokenizes the data using BERT tokenizer.
    :param tokenizer: The BERT tokenizer to use.
    :param series: The data to be tokenized.
    :param max_length: The maximum length of the tokenized sequence.
    :param truncation: Truncate the sequence if it is longer than max_length.
    :param padding: Padding strategy to use. Can be 'max_length' or 'longest'.
    :return: Tuple of tokenized data and attention masks.
    """
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    token_ids = []
    attention_masks = []

    for data in series:
        batch_encoder = tokenizer.encode_plus(data, return_tensors='pt', max_length=max_length, truncation=truncation,padding=padding)
        token_ids.append(batch_encoder['input_ids'])
        attention_masks.append(batch_encoder['attention_mask'])

    token_ids = torch.cat(token_ids)
    attention_masks = torch.cat(attention_masks)
    return token_ids, attention_masks

def tensor_train_test_split(
        labels: torch.Tensor,
        token_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        test_size: float=0.1,
        sampler: Optional[WeightedRandomSampler] = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Splits the data into training and testing sets.
    :param labels: The labels for the data.
    :param token_ids: The tokenized data.
    :param attention_masks: The attention masks for the data.
    :param test_size: The size of the test set.
    :return: Tuple of training and testing dataloader.
    """
    train_ids, test_ids = train_test_split(token_ids, test_size=test_size, shuffle=False)
    train_masks, test_masks = train_test_split(attention_masks, test_size=test_size, shuffle=False)
    train_labels, test_labels = train_test_split(labels, test_size=test_size, shuffle=False)

    train_dataloader = get_data_loader(train_ids, train_masks, train_labels, sampler=sampler)
    val_dataloader = get_data_loader(test_ids, test_masks, test_labels, sampler=sampler)

    return train_dataloader, val_dataloader

def get_data_loader(
        token_ids: torch.Tensor,
        token_masks: torch.Tensor,
        token_labels: Optional[torch.Tensor] = None,
        batch_size: int = 8,
        sampler: Optional[WeightedRandomSampler] = None,
        shuffle: bool = True) -> DataLoader:
    """
    Creates a DataLoader for the tokenized data.

    Args:
        token_ids (torch.Tensor): Tokenized input IDs.
        token_masks (torch.Tensor): Attention masks.
        token_labels (Optional[torch.Tensor]): Optional labels.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    if token_labels is not None:
        tensor_data = TensorDataset(token_ids, token_masks, token_labels)
    else:
        tensor_data = TensorDataset(token_ids, token_masks)

    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle, batch_sampler=sampler)

    return dataloader

def calculate_accuracy(preds, labels):
    """ Calculate the accuracy of model predictions against true labels.

    Parameters:
        preds (np.array): The predicted label from the model
        labels (np.array): The true label

    Returns:
        accuracy (float): The accuracy as a percentage of the correct
            predictions.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

    return accuracy

def model_predict(model: SpecificPreTrainedModelType, dataloader: DataLoader) -> (list, list):
        """
        Run the model in evaluation mode and return predictions for all batches.

        Args:
            model (PreTrainedModel): The transformer model.
            dataloader (DataLoader): DataLoader for the dataset to predict on.

        Returns:
            List[int]: List of predicted class labels and the confidence of the predictions.
        """
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        model.eval()

        all_preds = []
        all_confidence = []

        with torch.no_grad():
            for batch in dataloader:
                batch_token_ids = batch[0].to(device)
                batch_attention_mask = batch[1].to(device)

                outputs = model(
                    input_ids=batch_token_ids,
                    attention_mask=batch_attention_mask,
                    token_type_ids=None,
                    return_dict=False
                )

                logits = outputs[0]
                predictions = torch.argmax(logits, dim=1)
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidences = torch.max(probs, dim=1).values
                all_preds.extend(predictions.cpu().numpy().tolist())
                all_confidence.extend(confidences.cpu().numpy().tolist())

        return all_preds, all_confidence


