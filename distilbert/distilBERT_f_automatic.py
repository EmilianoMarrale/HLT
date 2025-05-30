import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class HatDataset(Dataset):
    """
    Custom PyTorch Dataset for the Hat classification task.
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_hat_datasets(
    train_json_path: str,
    test_json_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_len: int = 128,
    pretrained_model: str = 'distilbert-base-uncased'
):
    """
    Load and preprocess train and test data, encode labels, split train/validation,
    and return PyTorch datasets along with tokenizer and label metadata.

    Returns a dict with:
      - train_dataset: HatDataset for training
      - val_dataset: HatDataset for validation
      - test_dataset: HatDataset for testing
      - tokenizer: DistilBertTokenizerFast instance
      - label_encoder: fitted LabelEncoder
      - class_names: list of label classes
      - num_labels: number of distinct classes
    """
    # 1. Load training data from JSON file
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_df = pd.DataFrame(train_data).rename(columns={'utterance': 'text'})

    # 2. Encode labels to integers
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['hat'])
    class_names = le.classes_.tolist()
    num_labels = len(class_names)

    # 3. Split into training and validation sets
    train_df, val_df = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df['label']
    )

    # 4. Load test data from JSON file
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_df = pd.DataFrame(test_data).rename(columns={'utterance': 'text'})
    test_df['label'] = le.transform(test_df['hat'])

    # 5. Initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model)

    # 6. Create PyTorch datasets for train, validation, and test
    train_dataset = HatDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    val_dataset = HatDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    test_dataset = HatDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'tokenizer': tokenizer,
        'label_encoder': le,
        'class_names': class_names,
        'num_labels': num_labels
    }


def prepare_trainer(
    train_dataset,
    val_dataset,
    num_labels: int,
    class_names: list,
    output_dir: str = './results',
    pretrained_model: str = 'distilbert-base-uncased',
    num_train_epochs: int = 100,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    eval_strategy: str = 'epoch',
    save_strategy: str = 'epoch',
    metric_for_best_model: str = 'accuracy',
    logging_steps: int = 50,
    logging_dir: str = './logs',
    save_total_limit: int = 3,
    early_stopping_patience: int = 20,
):
    """
    Initialize the model, define training arguments, metrics, early stopping,
    and return a configured Hugging Face Trainer.
    """
    # 7. Set device and load pre-trained model for sequence classification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels
    ).to(device)

    # 8. Define evaluation metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        f1_weighted = f1_score(labels, preds, average='weighted')
        cm = confusion_matrix(labels, preds, labels=list(range(num_labels)))
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        metrics = {'accuracy': accuracy, 'f1_weighted': f1_weighted}
        for idx, cls in enumerate(class_names):
            metrics[f'accuracy_{cls}'] = float(per_class_acc[idx])
        return metrics

    # 9. Configure training arguments and early stopping
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        logging_steps=logging_steps,
        logging_dir=logging_dir,
        save_total_limit=save_total_limit,
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stop]
    )

    return trainer, model, training_args, device


def evaluate_and_save(
    trainer,
    test_dataset,
    class_names: list,
    label_encoder,
    tokenizer,
    save_dir: str = './hat_classifier_model'
):
    """
    Evaluate the trained model on the test set, print a detailed classification report,
    and save the model artifacts (model, tokenizer, and label encoder) to disk.
    """
    # 11. Run predictions on the test dataset
    test_output = trainer.predict(test_dataset)
    y_true = test_output.label_ids
    y_pred = np.argmax(test_output.predictions, axis=-1)

    # 12. Compute confusion matrix and per-class metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)

    precision = np.divide(tp, pred_counts, out=np.zeros_like(tp, dtype=float), where=pred_counts!=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support!=0)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=float),
        where=(precision + recall)!=0
    )

    total = support.sum()
    accuracy = tp.sum() / total
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1_scores.mean()
    weighted_p = np.dot(precision, support) / total
    weighted_r = np.dot(recall, support) / total
    weighted_f1 = np.dot(f1_scores, support) / total

    # 13. Print classification report
    print("\n=== Classification Report (test set) ===")
    header = f"{'class':<12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"


if __name__ == '__main__':
    # Example use case
    paths = {
        'train_json_path': r"C:\Users\t00fe\Documents\HLT\Prototype\normal_train_dataset.json",
        'test_json_path':  r"C:\Users\t00fe\Downloads\test_dataset.json"
    }

    # Steps 1-6: Prepare datasets
    data = prepare_hat_datasets(
        train_json_path=paths['train_json_path'],
        test_json_path=paths['test_json_path']
    )

    # Steps 7-9: Prepare trainer
    trainer, model, training_args, device = prepare_trainer(
        train_dataset=data['train_dataset'],
        val_dataset=data['val_dataset'],
        num_labels=data['num_labels'],
        class_names=data['class_names'],
        output_dir='./results',
        logging_dir='./logs'
    )

    # Train the model
    trainer.train()

    # Steps 11-13: Evaluate and save
    evaluate_and_save(
        trainer=trainer,
        test_dataset=data['test_dataset'],
        class_names=data['class_names'],
        label_encoder=data['label_encoder'],
        tokenizer=data['tokenizer'],
        save_dir='./hat_classifier_model'
    )
