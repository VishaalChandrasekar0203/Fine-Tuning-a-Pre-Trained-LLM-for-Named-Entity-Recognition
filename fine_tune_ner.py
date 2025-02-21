# fine_tune_ner.py
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

# Load dataset
dataset = load_dataset("conll2003")

# Load pre-trained model and tokenizer
model_name = "bert-base-cased"
model = BertForTokenClassification.from_pretrained(model_name, num_labels=9)  # 9 classes in CoNLL-2003
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",  # Pad to max_length
        max_length=128,  # Set a fixed max_length
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token (e.g., [CLS], [SEP])
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Subword token
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Split dataset into train and test sets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Use a subset for quick training
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Load evaluation metric (Seqeval)
seqeval_metric = evaluate.load("seqeval")

# Define label names for CoNLL-2003
label_names = dataset["train"].features["ner_tags"].feature.names

# Define compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
model.save_pretrained("./fine_tuned_bert_ner")
tokenizer.save_pretrained("./fine_tuned_bert_ner")
