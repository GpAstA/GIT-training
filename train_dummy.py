# train_dummy.py
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from dummy_datasets import DummyDataset

def main():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # トークナイザーとモデルのロード
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # データセットとデータローダーの準備
    dataset = DummyDataset(num_samples=100)
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)
    
    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_dict({"text": [d[0] for d in dataset.data], "labels": [d[1] for d in dataset.data]})
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataloader = DataLoader(
        hf_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./output_dummy",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=10,
        logging_dir="./logs_dummy",
        save_strategy="no",
        report_to="none"
    )

    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset
    )

    # トレーニングの実行
    trainer.train()

if __name__ == "__main__":
    main()
