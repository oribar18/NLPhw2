from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import f1_score
from gensim import downloader
import gensim
import numpy as np
import re
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence



def load_data(file_path, is_training=True):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    sentences_tags = []
    sentence_tags = []
    words = []
    word_to_label = {}

    for line in lines:
        line = re.sub(r'\ufeff', '', line).strip()
        if not line:
            if sentence and sentence_tags:
                # Check if any positive label exists in the sentence (only for training data)
                if is_training and any(tag == 1 for tag in sentence_tags):
                    sentences.append(sentence)
                    sentences_tags.append(sentence_tags)
                # For non-training data, add sentences without any filtering
                elif not is_training:
                    sentences.append(sentence)
                    sentences_tags.append(sentence_tags)
            sentence = []
            sentence_tags = []
        else:
            word, tag = line.split('\t')
            tag = 0 if tag.strip() == "O" else 1
            sentence.append(word)
            sentence_tags.append(tag)
            words.append(word)
            word_to_label[word] = tag

    if sentence and sentence_tags:  # Catch any trailing sentence not followed by a blank line
        if is_training and any(tag == 1 for tag in sentence_tags):
            sentences.append(sentence)
            sentences_tags.append(sentence_tags)
        elif not is_training:
            sentences.append(sentence)
            sentences_tags.append(sentence_tags)

    return sentences, sentences_tags, words, word_to_label


def build_set(sentences, model1, model2, sentences_tags):
    set_data = []
    set_tags = []
    all_tags = []
    representation_dict = {}

    # Check Gensim version and use appropriate attribute
    if gensim.__version__ < "4.0.0":
        vocab1 = model1.vocab
        vocab2 = model2.vocab
    else:
        vocab1 = model1.key_to_index
        vocab2 = model2.key_to_index

    for sentence, tags in zip(sentences, sentences_tags):
        all_tags += tags
        tokenized_sentence = []
        for word in sentence:
            if word not in representation_dict:
                # Handling OOV words for model1 (GloVe), using 'oov' for missing words
                if word in vocab1:
                    word_vec_1 = torch.as_tensor(model1[word].tolist())
                else:
                    # Attempt to use the lower case word if it exists, else 'oov' token if it exists, otherwise default to zeros
                    if word.lower() in vocab1:
                        word_vec_1 = torch.as_tensor(model1[word.lower()].tolist())
                    else:
                        word_vec_1 = torch.as_tensor(model1['oov'].tolist()) if 'oov' in vocab1 else torch.zeros(
                            model1.vector_size)

                # Handling OOV words for model2 (Word2Vec), using 'unk' for missing words
                if word in vocab2:
                    word_vec_2 = torch.as_tensor(model2[word].tolist())
                else:
                    # Attempt to use the lower case word if it exists, else 'unk' token if it exists, otherwise default to zeros
                    if word.lower() in vocab2:
                        word_vec_2 = torch.as_tensor(model2[word.lower()].tolist())
                    else:
                        word_vec_2 = torch.as_tensor(model2['unk'].tolist()) if 'unk' in vocab2 else torch.zeros(
                            model2.vector_size)

                final_vec = torch.cat((word_vec_1, word_vec_2))
                representation_dict[word] = final_vec
                tokenized_sentence.append(final_vec)
            else:
                tokenized_sentence.append(representation_dict[word])

        final_tokenized_sen = torch.stack(tokenized_sentence)
        set_data.append(final_tokenized_sen)
        set_tags.append(torch.tensor(tags, dtype=torch.long))

    return set_data, set_tags, all_tags


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-100)
    return xx_pad, yy_pad


class NerLSTM(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=256, num_layers=4):
        super(NerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-directional LSTM
        self.lstm = nn.LSTM(vec_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.5,
                            bidirectional=True)

        # First additional Linear and Tanh layers for more complexity
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim // 4)  # Intermediate layer
        self.Tanh1 = nn.Tanh()

        # Second additional Linear and Tanh layers
        self.linear2 = nn.Linear(hidden_dim // 4, hidden_dim // 8)  # Reduce dimensionality
        self.Tanh2 = nn.Tanh()

        # Final layer to predict classes
        self.linear3 = nn.Linear(hidden_dim // 8, num_classes)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        lstm_out, _ = self.lstm(input_ids)

        # Applying Tanh after the first Linear layer
        intermediate1 = self.Tanh1(self.linear1(lstm_out))

        # Applying second Linear and Tanh
        intermediate2 = self.Tanh2(self.linear2(intermediate1))

        # Final output layer
        outputs = self.linear3(intermediate2)

        if labels is not None:
            loss = self.loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            return outputs, loss
        return outputs, None


class NerDataset(Dataset):
    def __init__(self, file_path, model1, model2):
        if file_path == "data/train.tagged":
            sentences, original_tags, _, _ = load_data(file_path)
        else:
            sentences, original_tags, _, _ = load_data(file_path, False)

        self.data, self.tags, _ = build_set(sentences, model1, model2, original_tags)
        self.vector_dim = self.data[0].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.tags[idx]


def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data)
            predictions = torch.argmax(outputs, dim=2)
            all_predictions.append(predictions.view(-1).cpu().tolist())
            all_labels.append(labels.view(-1).cpu().tolist())
    valid_labels = [item for sublist in all_labels for item in sublist]
    valid_predictions = [item for sublist in all_predictions for item in sublist]
    f1 = f1_score(valid_labels, valid_predictions, average='binary')
    return f1


def train_and_evaluate(model, train_loader, dev_loader, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # Learning rate scheduler

    best_dev_f1 = -1
    no_improvement_count = 0
    patience = 5


    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            _, loss = model(data, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()  # Adjust the learning rate

        train_f1 = evaluate(model, train_loader, device)
        dev_f1 = evaluate(model, dev_loader, device)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            no_improvement_count = 0  # Reset counter
            print(
                f'Epoch {epoch + 1}, Loss: {np.mean(train_losses):.4f}, Train F1: {train_f1:.4f}, Dev F1: {dev_f1:.4f}')
            print(f'New best Dev F1: {best_dev_f1:.4f} at epoch {epoch + 1}')
        else:
            no_improvement_count += 1
            print(
                f'Epoch {epoch + 1}, Loss: {np.mean(train_losses):.4f}, Train F1: {train_f1:.4f}, Dev F1: {dev_f1:.4f}')
            print(f'No improvement in Dev F1. Count: {no_improvement_count}/{patience}. Best Dev F1: {best_dev_f1:.4f}')
            if no_improvement_count >= patience:
                print('Stopping training due to no improvement in Dev F1.')
                break  # Early stopping


def main():
    model1 = downloader.load('glove-twitter-200')
    model2 = downloader.load('word2vec-google-news-300')
    train_dataset = NerDataset("data/train.tagged", model1, model2)
    dev_dataset = NerDataset("data/dev.tagged", model1, model2)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=pad_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=pad_collate)
    combined_dataset = ConcatDataset([train_dataset, dev_dataset])

    # Create a single data loader for both datasets
    combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True, collate_fn=pad_collate)

    model = NerLSTM(vec_dim=train_dataset.vector_dim, num_classes=2)
    optimizer = Adam(model.parameters(), lr=0.0001)

    train_and_evaluate(model, combined_loader, dev_loader, optimizer, num_epochs=30)
    # if model is None:
    #     raise ValueError("The model returned from THIS.main() is None.")
    # return model


if __name__ == "__main__":
    main()