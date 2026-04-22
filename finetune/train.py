import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
 
 
# ─── Configuration ────────────────────────────────────────────────────────
BASE_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
OUTPUT_DIR = './finetune/finetuned-model'
NUM_LABELS = 3  # négatif (0), neutre (1), positif (2)
 
 
# ─── Charger le tokenizer et le modèle ───────────────────────────────────
print('Chargement du modèle de base...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True  # adapter la tête de classification
)
 
 
# ─── Tokenisation ────────────────────────────────────────────────────────
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128  # plus court = plus rapide pour l'entraînement
    )
 
# Charger le dataset préparé
print('Chargement du dataset...')
train_dataset = load_from_disk('finetune/data/train')
test_dataset  = load_from_disk('finetune/data/test')
 
# Tokeniser
train_tok = train_dataset.map(tokenize_function, batched=True)
test_tok  = test_dataset.map(tokenize_function, batched=True)
 
 
# ─── Métriques d'évaluation ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average='weighted')
    return {'accuracy': round(acc, 4), 'f1': round(f1, 4)}
 
 
# ─── Évaluation AVANT fine-tuning ────────────────────────────────────────
print('\n=== AVANT fine-tuning ===')
trainer_eval = Trainer(
    model=model,
    compute_metrics=compute_metrics
)
results_before = trainer_eval.evaluate(test_tok)
print(f'Accuracy avant : {results_before["eval_accuracy"]:.4f}')
print(f'F1 avant       : {results_before["eval_f1"]:.4f}')
 
 
# ─── Arguments d'entraînement ────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='no',
    logging_dir='./finetune/logs',
    logging_steps=10,
    report_to='none'  # désactiver WandB
)
 
 
# ─── Lancer le fine-tuning ────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    compute_metrics=compute_metrics
)
 
print('\nDébut du fine-tuning...')
trainer.train()
 
 
# ─── Évaluation APRÈS fine-tuning ────────────────────────────────────────
print('\n=== APRÈS fine-tuning ===')
results_after = trainer.evaluate(test_tok)
print(f'Accuracy après : {results_after["eval_accuracy"]:.4f}')
print(f'F1 après       : {results_after["eval_f1"]:.4f}')
 
improvement_acc = (results_after['eval_accuracy'] - results_before['eval_accuracy']) * 100
print(f'\nAmélioration accuracy : +{improvement_acc:.2f}%')
 
 
# ─── Sauvegarder le modèle fine-tuné ─────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'\nModèle sauvegardé dans {OUTPUT_DIR}')
print('Vous pouvez maintenant utiliser ce modèle dans sentiment.py !')
