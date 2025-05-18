import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime
from typing import Optional
from transformers.modeling_utils import SpecificPreTrainedModelType

def bert_tokenize_data(series: pd.Series, max_length: int=128, truncation: bool=True, padding :str='max_length') -> (torch.Tensor, torch.Tensor):
    """
    Tokenizes the data using BERT tokenizer.
    :param series: The data to be tokenized.
    :param max_length: The maximum length of the tokenized sequence.
    :param truncation: Truncate the sequence if it is longer than max_length.
    :param padding: Padding strategy to use. Can be 'max_length' or 'longest'.
    :return: Tuple of tokenized data and attention masks.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        test_size: float=0.1) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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

    train_dataloader = get_data_loader(train_ids, train_masks, train_labels)
    val_dataloader = get_data_loader(test_ids, test_masks, test_labels)

    return train_dataloader, val_dataloader

def get_data_loader(
    token_ids: torch.Tensor,
    token_masks: torch.Tensor,
    token_labels: Optional[torch.Tensor] = None,
    batch_size: int = 8,
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

    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def train_bert_model(model: SpecificPreTrainedModelType,
                    optimizer: torch.optim,
                    scheduler: torch.optim.lr_scheduler,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    epochs: int=4) -> SpecificPreTrainedModelType:

    # Check if GPU is available for faster training time
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    loss_dict = {
        'epoch': [i + 1 for i in range(epochs)],
        'average training loss': [],
        'average validation loss': []
    }
    t0_train = datetime.now()

    for epoch in range(0, epochs):

        model.train()
        training_loss = 0
        t0_epoch = datetime.now()

        print(f'{"-" * 20} Epoch {epoch + 1} {"-" * 20}')
        print('\nTraining:\n---------')
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
                return_dict=False)

            training_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        average_train_loss = training_loss / len(train_dataloader)
        time_epoch = datetime.now() - t0_epoch
        print(f'Average Loss:     {average_train_loss}')
        print(f'Time Taken:       {time_epoch}')

        model.eval()
        val_loss = 0
        val_accuracy = 0

        t0_val = datetime.now()
        print('\nValidation:\n---------')
        print(f'Start Time:       {t0_val}')

        for batch in val_dataloader:
            batch_token_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                (loss, logits) = model(
                    batch_token_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    token_type_ids=None,
                    return_dict=False)

            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(logits, label_ids)

        average_val_accuracy = val_accuracy / len(val_dataloader)
        average_val_loss = val_loss / len(val_dataloader)
        time_val = datetime.now() - t0_val

        print(f'Average Loss:     {average_val_loss}')
        print(f'Average Accuracy: {average_val_accuracy}')
        print(f'Time Taken:       {time_val}\n')

        loss_dict['average training loss'].append(average_train_loss)
        loss_dict['average validation loss'].append(average_val_loss)

    print(f'Total training time: {datetime.now() - t0_train}')
    return model

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