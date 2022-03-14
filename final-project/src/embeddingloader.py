import numpy as np
import torch

vocab, embeddings = [], []

with open('D:\GloveEmbeddings\glove.6B.50d.txt', encoding='utf8') as file: 
    full_content = file.read().strip().split('\n')

for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)

vocab_np = np.array(vocab)
embedding_np = np.array(embeddings)

#print(vocab_npa) # A word in vocab_npa in index x will have its embedding in embs_npa in index x.
#print(embs_npa) 

embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_np).float()) # Load the embeddings into pytorch embeddings

# The embeddings can return the embedding of an individual word by supplying it with a LongTensor of the index of the word
input = torch.LongTensor([0])
print(embedding_layer(input))

# Giving it a LongTensor of index 0 will return the embedding for the word "the", because it is in index 0.
