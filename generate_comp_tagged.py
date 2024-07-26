from torch import nn
import torch
from torch.utils.data import DataLoader, ConcatDataset
from gensim import downloader
import re
import gensim
import nlp_HW2_comp as compModel


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    words = []

    for line in lines:
        line = re.sub(r'\ufeff', '', line)
        if line == '\t\n' or line == '\n':
            sentences.append(sentence)
            sentence = []
        else:
            word = line.split('\n')[0]
            sentence.append(word)
            words.append(word)

    return sentences, words


def build_set(sentences, model1, model2):
    set_data = []
    representation_dict = {}

    # Check Gensim version and use appropriate attribute
    if gensim.__version__ < "4.0.0":
        vocab1 = model1.vocab
        vocab2 = model2.vocab
    else:
        vocab1 = model1.key_to_index
        vocab2 = model2.key_to_index

    for sentence in sentences:
        tokenized_sentence = []
        for word in sentence:
            if word not in representation_dict:
                if word not in vocab1:
                    word_vec_1 = torch.as_tensor(model1['oov'].tolist())
                else:
                    word_vec_1 = torch.as_tensor(model1[word].tolist())

                if word not in vocab2:
                    word_vec_2 = torch.zeros(model2.vector_size)
                else:
                    word_vec_2 = torch.as_tensor(model2[word].tolist())

                final_vec = torch.cat((word_vec_1, word_vec_2))
                representation_dict[word] = torch.cat((word_vec_1, word_vec_2))
                tokenized_sentence.append(final_vec)
            else:
                tokenized_sentence.append(representation_dict[word])

        final_tokenized_sen = torch.stack(tokenized_sentence)
        set_data.append(final_tokenized_sen)

    return set_data


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, model1, model1_name, model2, model2_name):
        self.file_path = file_path
        self.sentences, words = load_data(file_path)
        self.vector_dim = int(re.findall(r'\d+', model1_name)[-1]) + int(re.findall(r'\d+', model2_name)[-1])
        self.tokenized_sen = build_set(self.sentences, model1, model2)


def predict(model, comp_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_predictions = []
    for sentence in comp_set.tokenized_sen:
        with torch.no_grad():
            outputs, _ = model(sentence)
        predictions = torch.argmax(outputs, dim=-1)
        all_predictions.append(predictions.view(-1).cpu().tolist())

    return all_predictions


def build_comp_tagged_file(sentences, predictions):
    with open('comp_316137371_314968595.tagged', 'w', encoding='utf-8-sig') as f:
        for sentence, tags in zip(sentences, predictions):
            tags = ["O" if tag == 0 else "I" for tag in tags]
            for word, tag in zip(sentence, tags):
                f.write(word + '\t' + str(tag) + '\n')
            f.write('\n')

def main():
    model1 = downloader.load('glove-twitter-200')
    model2 = downloader.load('word2vec-google-news-300')
    train_dataset = compModel.NerDataset("data/train.tagged", model1, model2)
    dev_dataset = compModel.NerDataset("data/dev.tagged", model1, model2)
    print('loaded models')
    train_loader = compModel.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=compModel.pad_collate)
    dev_loader = compModel.DataLoader(dev_dataset, batch_size=1, collate_fn=compModel.pad_collate)
    combined_dataset = ConcatDataset([train_dataset, dev_dataset])

    # Create a single data loader for both datasets
    combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True, collate_fn=compModel.pad_collate)

    model = compModel.NerLSTM(vec_dim=train_dataset.vector_dim, num_classes=2)
    optimizer = compModel.Adam(model.parameters(), lr=0.0001)

    compModel.train_and_evaluate(model, combined_loader, dev_loader, optimizer, num_epochs=30)

    model1_name = "glove-twitter-200"
    model2_name = "word2vec-google-news-300"
    comp_set = NerDataset("data/test.untagged", model1, model1_name, model2, model2_name)
    print('loaded dataset')
    print('loaded model')
    predictions = predict(model, comp_set)
    print('predicted')
    build_comp_tagged_file(comp_set.sentences, predictions)
    print('done')


if __name__ == '__main__':
    main()