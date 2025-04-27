import sqlite3
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ====================================================
# Step 1: Connect to the database and extract data
# ====================================================
# Connect to the SQLite database (ensure 'multicat.db' is in your working directory)
conn = sqlite3.connect('multicat.db')
cursor = conn.cursor()

# Define SQL query to extract required fields:
# - message_id: unique identifier for each message
# - participant: sender or participant info
# - utt: the utterance text (using corrected_text if available, otherwise asr_text)
# - sentiment and emotion: labels for the data
query = """
SELECT 
  original_uuid as message_id, 
  participant, 
  COALESCE(corrected_text, asr_text) as utt, 
  sentiment, 
  emotion
FROM utterance
WHERE sentiment IS NOT NULL
AND emotion IS NOT NULL
AND (corrected_text IS NOT NULL OR asr_text IS NOT NULL)
"""

# Execute the query and fetch all data
cursor.execute(query)
data = cursor.fetchall()
conn.close()

print(f"Total number of extracted records: {len(data)}")

# ====================================================
# Step 2: Process the data and map labels to indices
# ====================================================
# Define mapping dictionaries for sentiment and emotion labels
sent2idx = {'negative': 0, 'neutral': 1, 'positive': 2}
emo2idx = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}

# Process each record and convert string labels to numerical indices.
processed_data = []
for item in data:
    message_id, participant, utt, sentiment, emotion = item
    # Skip records with labels not found in the mapping dictionaries
    if sentiment not in sent2idx or emotion not in emo2idx:
        continue
    
    processed_data.append({
        'message_id': message_id,
        'participant': participant,
        'utt': utt,
        'sentiment': sent2idx[sentiment],
        'emotion': emo2idx[emotion]
    })

print(f"Number of processed records: {len(processed_data)}")

# ====================================================
# Step 3: Split the data into train, validation, and test sets (60:20:20)
# ====================================================
train_data, temp_data = train_test_split(processed_data, test_size=0.4, random_state=42)
dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
print(f"Training data: {len(train_data)} records, Validation data: {len(dev_data)} records, Test data: {len(test_data)} records")

# ====================================================
# Step 4: Create directories to save the processed data
# ====================================================
os.makedirs('data/text_data', exist_ok=True)
os.makedirs('data/ys_data', exist_ok=True)

# Define helper functions to extract text and label data for saving
def extract_text_data(dataset):
    """
    Extract text data for modeling.
    Each record includes:
      - audio_id: unique message identifier
      - x_utt: the actual text utterance
      - utt_length: the number of words in the utterance
    """
    return [{
        'audio_id': item['message_id'],
        'x_utt': item['utt'],
        'utt_length': len(item['utt'].split())
    } for item in dataset]

def extract_ys_data(dataset):
    """
    Extract label data for modeling.
    Each record includes:
      - audio_id: unique message identifier
      - ys: a list with sentiment and emotion labels (numerical indices)
    """
    return [{
        'audio_id': item['message_id'],
        'ys': [item['sentiment'], item['emotion']]
    } for item in dataset]

# ====================================================
# Step 5: Compute class weights for sentiment and emotion
# ====================================================
# Extract labels from the training data
train_sentiments = [item['sentiment'] for item in train_data]
train_emotions = [item['emotion'] for item in train_data]

# Print sentiment distribution
print("\nSentiment distribution:")
for sent, idx in sent2idx.items():
    count = train_sentiments.count(idx)
    print(f"{sent}: {count} ({count/len(train_sentiments)*100:.2f}%)")

# Print emotion distribution
print("\nEmotion distribution:")
for emo, idx in emo2idx.items():
    count = train_emotions.count(idx)
    print(f"{emo}: {count} ({count/len(train_emotions)*100:.2f}%)")

# ====================================================
# Step 6: Save the processed text and label data as pickle files
# ====================================================
with open('data/text_data/train.pickle', 'wb') as f:
    pickle.dump(extract_text_data(train_data), f)

with open('data/text_data/dev.pickle', 'wb') as f:
    pickle.dump(extract_text_data(dev_data), f)

with open('data/text_data/test.pickle', 'wb') as f:
    pickle.dump(extract_text_data(test_data), f)

with open('data/ys_data/train.pickle', 'wb') as f:
    pickle.dump(extract_ys_data(train_data), f)

with open('data/ys_data/dev.pickle', 'wb') as f:
    pickle.dump(extract_ys_data(dev_data), f)

with open('data/ys_data/test.pickle', 'wb') as f:
    pickle.dump(extract_ys_data(test_data), f)

# ====================================================
# Step 7: Calculate and save class weights to handle imbalanced data
# ====================================================
from sklearn.utils.class_weight import compute_class_weight
import torch

# Compute class weights for sentiment
sentiment_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_sentiments),
    y=train_sentiments
)

# Compute class weights for emotion
emotion_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_emotions),
    y=train_emotions
)

# Convert the computed weights to PyTorch tensors for use in model training
sentiment_weights = torch.tensor(sentiment_weights, dtype=torch.float)
emotion_weights = torch.tensor(emotion_weights, dtype=torch.float)

# Create directory for saving class weights
os.makedirs('data/class_weights', exist_ok=True)

# Save the class weights as a pickle file
with open('data/class_weights/asist_clsswts.pickle', 'wb') as f:
    pickle.dump([sentiment_weights, emotion_weights], f)

print("\nAll data has been successfully saved!")
print("Data saved at: data/")