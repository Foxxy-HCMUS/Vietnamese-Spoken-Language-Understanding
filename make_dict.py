import pandas as pd
import json
import config as cf

def read_file(filepath):
    lines = []
    with open(filepath) as f:
        for line in f.readlines():
            lines.append(json.loads(line))
    df = pd.DataFrame(lines)
    return df

train_data = read_file(cf.train_file)
train_data["sentence_len"] = train_data["words"].apply(len)
valid_data = read_file(cf.valid_file)

# Xây dựng vocab cho word và tag
words = list(train_data['words'].explode().unique())
slots = list(train_data['tags'].explode().unique())

# Tạo dict word to index, thêm 2 từ đặc biệt là Unknown và Padding
word2idx = {w : i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0

## Tạo dict slot to index, thêm 1 tag đặc biệt là Padding
# slot2idx = {t : i + 1 for i, t in enumerate(slots)}
# slot2idx["PAD"] = 0

slot2idx = {t : i for i, t in enumerate(slots)}

# Tạo 2 dict index to word và index to slot
idx2word = {i: w for w, i in word2idx.items()}
idx2slot = {i: w for w, i in slot2idx.items()}

# Tạo intent dict 
idx2intent = {i : val for i, val in enumerate(train_data["intent"].unique())}
intent2idx = {val : i for i, val in idx2intent.items()}

print('Number of training samples: ', len(train_data))
print('Number of test samples: ', len(valid_data))
print('Number of words: ', len(word2idx))
print('Number of intent labels: ', len(intent2idx))
print('Number of slot labels', len(slot2idx))
