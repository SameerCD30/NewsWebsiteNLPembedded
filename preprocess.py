import pandas as pd
from sklearn.model_selection import train_test_split

fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')
fake['label'] = 1
true['label'] = 0
df = pd.concat([fake, true], ignore_index=True)
df['text'] = df['title'] + ' ' + df['text']
df = df[['text', 'label']].dropna().drop_duplicates()
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")