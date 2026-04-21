import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
 
# Charger le CSV
df = pd.read_csv('finetune/data/dataset.csv')
print(f'Dataset chargé : {len(df)} exemples')
print(f'Distribution des labels :')
print(df['label'].value_counts())
 
# Nettoyer les données
df = df.dropna()
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)
 
# Vérification des labels
assert df['label'].isin([0, 1, 2]).all(), 'Labels invalides !'
 
# Diviser en train (80%) et test (20%)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
print(f'Entraînement : {len(train_df)} exemples')
print(f'Test         : {len(test_df)} exemples')
 
# Convertir en format HuggingFace
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))
 
# Sauvegarder
train_dataset.save_to_disk('finetune/data/train')
test_dataset.save_to_disk('finetune/data/test')
print('Dataset sauvegarde dans finetune/data/')
