import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


class HatDataset(Dataset):

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
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_hat_datasets(
    raw_json_path: str,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
    max_len: int = 128,
    pretrained_model: str = 'distilbert-base-uncased'
):
    """
    1. Load raw JSON of dialogs, flatten utterances, balance classes via undersampling,
    2. Encode labels, split into train/validation/test,
    3. Tokenize and wrap into PyTorch datasets.

    Returns dict with:
      - train_dataset, val_dataset, test_dataset
      - tokenizer, label_encoder, class_names, num_labels
    """
    # Load and flatten raw dataset
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        dialogs = json.load(f)

    examples = []
    for dialog in dialogs:
        for utt in dialog.get('utterances', []):
            examples.append({'text': utt['utterance'], 'hat': utt['hat']})
    df = pd.DataFrame(examples)

    # Balance classes by undersampling to minimum count
    min_count = df['hat'].value_counts().min()
    df_bal = (
        df.groupby('hat', group_keys=False)
          .apply(lambda grp: grp.sample(min_count, random_state=random_state))
          .reset_index(drop=True)
    )

    # Encode labels
    le = LabelEncoder()
    df_bal['label'] = le.fit_transform(df_bal['hat'])
    class_names = le.classes_.tolist()
    num_labels = len(class_names)

    # Split into train+val and holdout
    train_val, holdout = train_test_split(
        df_bal,
        test_size=test_size,
        stratify=df_bal['label'],
        random_state=random_state
    )
    # Split holdout into validation and test
    val_df, test_df = train_test_split(
        holdout,
        test_size=val_size,
        stratify=holdout['label'],
        random_state=random_state
    )

    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model)

    # Wrap into PyTorch datasets
    train_dataset = HatDataset(
        texts=train_val['text'].tolist(),
        labels=train_val['label'].tolist(),
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
    num_epochs: int = 100,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    logging_dir: str = './logs',
    early_stop_patience: int = 3,
):
    """
    Initialize DistilBERT model, training args, Trainer with early stopping.
    Returns configured Trainer and device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_labels
    ).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_w = f1_score(labels, preds, average='weighted')
        cm = confusion_matrix(labels, preds, labels=list(range(num_labels)))
        per_acc = cm.diagonal() / cm.sum(axis=1)
        m = {'accuracy': acc, 'f1_weighted': f1_w}
        for idx, cls in enumerate(class_names):
            m[f'accuracy_{cls}'] = float(per_acc[idx])
        return m

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=50,
        logging_dir=logging_dir,
        save_total_limit=3
    )
    early_stop = EarlyStoppingCallback(early_stopping_patience=early_stop_patience)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stop]
    )
    return trainer, device


def evaluate_and_save(
    trainer,
    test_dataset,
    class_names: list,
    label_encoder,
    tokenizer,
    save_dir: str = './hat_classifier_model'
):
    """
    Evaluate on test set, print manual classification report,
    and save model, tokenizer, and label encoder.
    """
    out = trainer.predict(test_dataset)
    y_true = out.label_ids
    y_pred = np.argmax(out.predictions, axis=-1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    preds = cm.sum(axis=0)

    precision = np.divide(tp, preds, out=np.zeros_like(tp, float), where=preds!=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, float), where=support!=0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros_like(tp, float), where=(precision+recall)!=0)

    total = support.sum()
    acc = tp.sum()/total
    macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()
    w_p = np.dot(precision, support)/total
    w_r = np.dot(recall, support)/total
    w_f1 = np.dot(f1, support)/total

    print("\n=== Classification Report ===")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for i, cls in enumerate(class_names):
        print(f"{cls:<12}{precision[i]:10.2f}{recall[i]:10.2f}{f1[i]:10.2f}{support[i]:10d}")
    print(f"{'accuracy':<12}{'':>10}{'':>10}{acc:10.2f}{total:10d}")
    print(f"{'macro avg':<12}{macro_p:10.2f}{macro_r:10.2f}{macro_f1:10.2f}{total:10d}")
    print(f"{'weighted avg':<12}{w_p:10.2f}{w_r:10.2f}{w_f1:10.2f}{total:10d}")

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    trainer.model.save_pretrained(save_dir)
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print('Artifacts saved to', save_dir)


if __name__ == '__main__':
    # Example use case
    raw_path = r"C:\Users\t00fe\Downloads\hand_labelled_dataset.json"
    data = prepare_hat_datasets(raw_path)
    trainer, device = prepare_trainer(
        data['train_dataset'],
        data['val_dataset'],
        num_labels=data['num_labels'],
        class_names=data['class_names']
    )
    trainer.train()
    evaluate_and_save(
        trainer,
        data['test_dataset'],
        data['class_names'],
        data['label_encoder'],
        data['tokenizer']
    )
