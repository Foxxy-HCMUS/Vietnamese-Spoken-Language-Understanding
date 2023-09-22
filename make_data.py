from make_dict import train_data, valid_data, word2idx, slot2idx, intent2idx
from config import max_len
import numpy as np
from keras_preprocessing.sequence import pad_sequences

def make_idxdata(data):
    sentences = data["words"]
    slots = data["tags"]
    
    # Chuyển các câu về dạng vector of index
    sentence_idx = [[word2idx.get(w, word2idx['UNK']) for w in s] for s in sentences.values]

    # Padding các câu về max_len
    sentence_idx = pad_sequences(maxlen = max_len, sequences = sentence_idx, padding = "post", value = word2idx["PAD"]).tolist()

    # Chuyển các slot về dạng index
    slot_idx = [[slot2idx[w] for w in s] for s in slots.values]

    # Tiến hành padding về max_len
    slot_idx = pad_sequences(maxlen = max_len, sequences = slot_idx, padding = "post", value = slot2idx["O"]).tolist()

    # Chuyển intent về index
    intent_idx = [intent2idx[s] for s in data["intent"].values]
#     print(max_len)
    return list(zip(sentence_idx, sentences.apply(len).values, slot_idx, intent_idx))

train_data = make_idxdata(train_data)
valid_data = make_idxdata(valid_data)
# print(valid_data)
