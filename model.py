import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.distributions import Categorical
from tqdm.auto import tqdm, trange

import numpy as np
import os
import time
import functools

from utils import *

import music_handler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

example_1 = """X:88
T: Weige) zu Shi comin ch\`a gugu y?
N: C1517A
O: China, Jiangsu, Xixia
S: MM, 1, Neuese Sprastoehe | dcAzAz
d. Nanji diao]
M: Shinao
N: C0619
O: China, Shanxi Ming
S: Zhong he'to goihuan
N: C1646
O: China, Shanxi, Zuoquan
S: III, 566, 2)]
N: Zwish\`en* Suedem
N: +2Lied. Teedmen der Oktaviert.
M:cis Verzierungen.
R: Shange]
M: 3/4
L: 1/16
K: A
B2B2d4 | f2e2g2e2 | d8- | e4 | e8 |
e2B2A4
G2A2 | BdedB2B2 |
c.aao reng ta y\vi
N: Fie Prusikel, Essen is.
N: Die aufgerinsen, letzte Glissandi thaelt mit Ertiert. um
S: Sicao]
M: 2/4
L: 1/16
K: G
BGB4A2 | B2GFG2E2 | GA2Bd2B2 | EBAcE3G | E8 |
E2G2E2D2 | F2GGd2d2 | cd\ge8 |
e2d2e2d2 | AcAEG4 |
A3GA2A2 | B2d2A2G2 | EddA2 | GAGF | C2cc | c6- | d3fdA | G8"""

class ChineseMusicDataset(Dataset):

    def __init__(self, txt_file):
        with open(txt_file) as file:
            self.music_pieces = file.read().split("\n")

    def __len__(self):
        return len(self.music_pieces)


class LSTMMusicGenerator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(LSTMMusicGenerator, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, sequence):
        embeds = self.embeddings(sequence)
        lstm_output, (h, c) = self.lstm(embeds)
        output = self.fc(lstm_output)
        return output

def sample_indices(pred):
    return Categorical(logits=pred).sample()

def train(model,
          seq_length,
          batch_size,
          train_data,
          checkpoint_loc,
          lr=5e-3,
          criterion = nn.CrossEntropyLoss(),
          n_epoch = 2000
          ):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in trange(n_epoch):
        x_batch, y_batch = get_batch(train_data, seq_length, batch_size)

        x_batch = x_batch.to(device).long()
        y_batch = y_batch.to(device).long()

        pred= model(x_batch)
        pred = pred.permute(0, 2, 1)  # class logits should be in dim1

        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Epoch {epoch}, Loss: {loss:.4f}")
        if not (epoch % 100):
            tqdm.write(f"Loss: {loss:.4f}")
            save_path = os.path.join(checkpoint_loc, f"epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)

def generate_text(model, start_string, generation_length=1000):
    input_eval = torch.tensor([char2idx[s] for s in start_string]).long()
    input_eval = torch.unsqueeze(input_eval, 0)
    text_generated = []

    for i in range(generation_length):
        predictions = model(input_eval)
        # predictions = tf.squeeze(predictions, 0)

        predicted_id = sample_indices(predictions)
        input_eval = torch.tensor([[predicted_id]]).long()
        # input_eval = torch.unsqueeze([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

if __name__ == '__main__':
    songs = open("chinese.txt").read()
    vocab = sorted(set(songs))

    ### Hyperparameter setting and optimization ###

    # Optimization parameters:
    num_training_iterations = 2000  # Increase this to train longer
    batch_size = 4  # Experiment between 1 and 64
    seq_length = 100  # Experiment between 50 and 500
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

    # Model parameters:
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024  # Experiment between 1 and 2048

    # Checkpoint location:
    checkpoint_dir = 'training_checkpoints'

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    vectorized_songs = vectorize_string(songs, char2idx)
    model = LSTMMusicGenerator(vocab_size, embedding_dim, rnn_units)
    #
    # train(model, seq_length=seq_length, batch_size=batch_size, lr=learning_rate, checkpoint_loc=checkpoint_dir,
    #       train_data=vectorized_songs, n_epoch=num_training_iterations)
    model.load_state_dict(torch.load(os.path.join("training_checkpoints", "epoch1000.pt")))
    print(model)
    model.eval()

    song = generate_text(model, "X", generation_length=1000)

    print(song)

    # music_handler.play_generated_song(song)