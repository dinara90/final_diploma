import argparse
import logging
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def compute_metrics(pred):
    """Вычисляет метрики качества для оценки модели."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
def train_classifier(data_path:str,
                     output_dir:str,
                     model_name:str="microsoft/deberta-v3-base",
                     test_size:float=0.15,
                     num_epochs:int=10,
                     batch_size:int=8,
                     gradient_accumulation_steps:int=2,
                     learning_rate:float=2e-5,
                     warmup_ratio:float=0.1,
                     weight_decay:float=0.01,
                     max_seq_length:int=512
):
    """
    Загружает данные, дообучает и сохраняет модель классификации секций.
    """
    logger.info(f"Запуск обучения модели: {model_name}")
    logger.info(f"Источник данных: {data_path}")
    logger.info(f"Директория вывода: {output_dir}")
    
    logger.info("Загрузка и подготовка данных...")
    try:
        df = pd.read_json(data_path)
        if "text" not in df.columns or 'label' not in df.columns:
            raise ValueError("JSON файл должен содержать колонки 'text' и 'label'")
        logger.info(f"Загружено {len(df)} записей.")
    except FileNotFoundError:
        logger.error(f"Ошибка: Файл данных не найден по пути {data_path}")
        return
    except Exception as e:
        logger.error(f"Ошибка при загрузке или чтении данных из {data_path}: {e}")
        return
    
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['text'].str.strip().astype(bool)]
    df.drop_duplicates(subset=['text', 'label'], inplace=True)
    logger.info(f"Осталось {len(df)} записей после очистки.")
    
    unique_labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(unique_labels)
    logger.info(f"Найденные метки ({num_labels}): {', '.join(unique_labels)}")
    
    df['label_id'] = df['label'].map(label2id)
    
    logger.info(f"Разделение данных на обучающую и валидационную (test_size={test_size})...")
    train_df, eval_df = train_test_split(df, test_size=test_size,
                                         random_state=42, stratify=df['label_id']
                                         )
    logger.info(f"Размер обучающей выборки: {len(train_df)}")
    logger.info(f"Размер валидационной выборки: {len(eval_df)}")
    
    logger.info("Вычисление весов классов для компенсации дисбаланса...")
    class_counts = train_df['label_id'].value_counts().sort_index()
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    
    class_weights = total_samples / (num_classes * class_counts)
    
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = class_weights_tensor.to(device)
    
    logger.info(f"Рассчитанные веса классов: {class_weights_tensor.cpu().numpy().tolist()}")
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label_id']])
    dataset_dict = DatasetDict({'train': train_dataset, 'eval': eval_dataset})
    
    logger.info(f"Загрузка токенизатора и модели {model_name}...")
    try:
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
            label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
        )
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели или токенизатора: {e}", exc_info=True)
        return
    
    logger.info("Токенизация данных...")
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_length)
    
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label_id', 'labels')
    tokenized_datasets.set_format("torch")
    
    logger.info("Настройка параметров обучения...")
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"Размер батча на устройство: {batch_size}")
    logger.info(f"Шагов накопления градиента: {gradient_accumulation_steps}")
    logger.info(f"Эффективный размер батча: {effective_batch_size}")
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    class WeightedLossTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Переопределенный метод для вычисления потерь с использованием весов классов.
            """
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
            
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='none',
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )
    
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['eval'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights_tensor
    )
    
    logger.info("Начало обучения...")
    try:
        train_result = trainer.train()
        logger.info("Обучение завершено.")
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}", exc_info=True)
        return
    
    logger.info("Оценка лучшей модели на валидационной выборке...")
    eval_metrics = trainer.evaluate()
    logger.info("Результаты оценки:")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"Сохранение лучшей модели и токенизатора в {output_dir}...")
    try:
        trainer.save_model(output_dir)
        label_map_path = os.path.join(output_dir, "label_map.json")
        with open(label_map_path, 'w') as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f)
        logger.info(f"Маппинг меток сохранен в {label_map_path}")
        logger.info("Модель, токенизатор и маппинг меток успешно сохранены.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении финальной модели: {e}", exc_info=True)
        
    logger.info("Скрипт завершен.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for DDR section classification.")
    parser.add_argument("--data_path", type=str, required=True, help="Путь к JSON файлу с размеченными данными (labeled_sections.json).")
    parser.add_argument("--output_dir", type=str, required=True, help="Директория для сохранения обученной модели и результатов.")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Имя модели для обучения.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Доля данных для валидационной выборки.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Количество эпох обучения.")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча на устройство.")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Количество шагов накопления градиента.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Скорость обучения.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Доля шагов для 'разогрева' learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Коэффициент L2 регуляризации.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Максимальная длина последовательности для токенизатора.")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    log_file_path = os.path.join(args.output_dir, "training_log.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    train_classifier(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        test_size=args.test_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length
    )