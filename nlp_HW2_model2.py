from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from gensim import downloader
import numpy as np
import re
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence


def load_data(file_path):
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
        sentences.append(sentence)
        sentences_tags.append(sentence_tags)

    return sentences, sentences_tags, words, word_to_label


def build_set(sentences, model1, model2, sentences_tags):
    set_data = []
    set_tags = []
    all_tags = []
    representation_dict = {}

    for sentence, tags in zip(sentences, sentences_tags):
        all_tags += tags
        tokenized_sentence = []
        for word in sentence:
            if word not in representation_dict:
                word_vec_1 = torch.as_tensor(model1[word].tolist()) if word in model1 else torch.zeros(model1.vector_size)
                word_vec_2 = torch.as_tensor(model2[word].tolist()) if word in model2 else torch.zeros(model2.vector_size)

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
    
    # Pad sequences to the maximum length of any sequence in the batch
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)  # Assumes 0 is an appropriate padding value
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-100)  # Assumes -100 is an appropriate padding value
    
    return xx_pad, yy_pad


class NerNN(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=128):
        super(NerNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
        self.third_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.activation(x)
        x = self.third_layer(x)

        if labels is not None:
            loss = self.loss(x.view(-1, 2), labels.view(-1))  # Adjust for CrossEntropyLoss expectations
            return x, loss
        return x, None


class NerDataset(Dataset):
    def __init__(self, file_path, model1, model2):
        sentences, original_tags, _, _ = load_data(file_path)
        self.data, self.tags, _ = build_set(sentences, model1, model2, original_tags)
        self.vector_dim = self.data[0].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.tags[idx]


def evaluate(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data)
            predictions = torch.argmax(outputs, dim=2)  # Get the index of the max log-probability
            
            # Flatten the predictions and labels to compute overall F1 score
            all_predictions.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    # Calculate F1 score, ignoring the padding label (-100)
    valid_indices = [i for i, label in enumerate(all_labels) if label != -100]
    valid_labels = [all_labels[i] for i in valid_indices]
    valid_predictions = [all_predictions[i] for i in valid_indices]
    
    f1 = f1_score(valid_labels, valid_predictions, average='binary')
    return f1


def train_and_evaluate(model, train_loader, dev_loader, optimizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
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
        
        train_f1 = evaluate(model, train_loader, device)
        dev_f1 = evaluate(model, dev_loader, device)
        
        print(f'Epoch {epoch+1}, Loss: {np.mean(train_losses):.4f}, Train F1: {train_f1:.4f}, Dev F1: {dev_f1:.4f}')


def main():
    model1 = downloader.load('glove-twitter-50')
    model2 = downloader.load('word2vec-google-news-300')
    train_dataset = NerDataset("data/train.tagged", model1, model2)
    dev_dataset = NerDataset("data/dev.tagged", model1, model2)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=pad_collate)


    model = NerNN(vec_dim=train_dataset.vector_dim, num_classes=2)
    optimizer = Adam(model.parameters(), lr=0.001)

    train_and_evaluate(model, train_loader, dev_loader, optimizer, num_epochs=10)


if __name__ == "__main__":
    main()
