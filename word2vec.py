import re
from collections import Counter

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset

def reset_weights(m):
    '''
    Функция для сброса весов модели.
    Применяется к каждому модулю модели.
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def prepare_data(text, window_size=2):
	text = re.sub(r'[^a-z@# ]', '', text.lower())
	tokens = text.split()
	vocab = set(tokens)
	word_to_index = {word: i for i, word in enumerate(vocab)}
	data = []
	for i in range(len(tokens)):
		for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
			if i != j:
				data.append((tokens[i], tokens[j]))    
	return data, word_to_index, len(vocab)


class SkipGramModelDataset(Dataset):

    def __init__(self, data, word_to_index):
        self.data = [(word_to_index[center], word_to_index[context]) for center, context in data]	
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return torch.tensor(self.data[index][0], dtype=torch.long), torch.tensor(self.data[index][1], dtype=torch.long)

class Word2VecSkipGramm(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGramm, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)
        self.activation_function = nn.LogSoftmax(dim=-1)

    def forward(self, center_word_index):
        hidden_layer = self.embeddings(center_word_index)
        out_layer = self.out_layer(hidden_layer)
        log_probs = self.activation_function(out_layer)
        return log_probs

def train_model(data, word_to_index, vocabulary_size, embedding_dim = 50, epochs = 10, batch_size = 1):
    dataset = SkipGramModelDataset(data, word_to_index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Word2VecSkipGramm(vocabulary_size, embedding_dim)

    loss_function = nn.NLLLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            model.zero_grad()
            log_probs = model(center)
            loss = loss_function(log_probs, context)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
    return model

def train(data):
    window_size = 2
    embedding_dim = 10
    epochs = 5
    batch_size = 1

    pair_data, word_to_index, vocabulary_size = prepare_data(data, window_size)

    model = train_model(pair_data, word_to_index, vocabulary_size, embedding_dim, epochs, batch_size)

    embeddings = model.embeddings.weight.data.numpy()

    index_to_word = {i:word for word, i in word_to_index.items()}
    w2v_dict = {index_to_word[index]: embeddings[index] for index in range(vocabulary_size)}
    model.apply(reset_weights)
    return w2v_dict

test_text = 'Captures Semantic Relationships: The skip-gram model effectively captures semantic relationships between words. It learns word embeddings that encode similar meanings and associations, allowing for tasks like word analogies and similarity calculations. Handles Rare Words: The skip-gram model performs well even with rare words or words with limited occurrences in the training data. It can generate meaningful representations for such words by leveraging the context in which they appear. Contextual Flexibility: The skip-gram model allows for flexible context definitions by using a window around each target word. This flexibility captures local and global word associations, resulting in richer semantic representations. Scalability: The skip-gram model can be trained efficiently on large-scale datasets due to its simplicity and parallelization potential. It can process vast amounts of text data to generate high-quality word embeddings.'

w2v_dict = train(test_text)


"""Wrod2Vec CBOW"""

def prepare_data_cbow(text, window_size=2):
    text = re.sub(r'[^a-z@#]', '', text.lower())
    tokens = text.split()
    vocab = set(tokens)
    word_to_ix = {word:i for i,word in enumerate(vocab)}
    data = []

    for i in range(window_size, len(tokens)):
         context = [tokens[i-j-1] for j in range(window_size)] + [tokens[i+j+1] for j in range(window_size)]
         target = tokens[i]
         data.append((context, target))
    return data, word_to_ix, len(vocab)


class Word2VecCBOWModel(nn.Module):
     def __init__(self, vocab_size, embedding_dim):
          super(Word2VecCBOWModel, self).__init__()
          self.embeddings = nn.Embedding(vocab_size, embedding_dim)
          self.out_layer = nn.Linear(embedding_dim, vocab_size)
          self.activation_function = nn.LogSoftmax(dim=1)

     def forward(self, center_word_idx):
          hidden_layer = torch.mean(self.embeddings(center_word_idx), dim=1)
          out_layer = self.out_layer(hidden_layer)
          activation = self.activation_function(out_layer)
          return activation


