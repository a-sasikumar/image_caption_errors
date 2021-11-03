import urllib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from transformers import DistilBertModel, DistilBertTokenizer


class ICErrorDataSet(Dataset):
    def __init__(self, X, y, text_tokenizer):
        sentences = [row[0] for row in X]
        self.encodings = text_tokenizer(sentences, truncation=True, padding=True)
        self.labels = np.array(y)

    def __getitem__(self, index):
        text_item = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        image_item = {}
        return {
            "text": text_item,
            "image": image_item
        }

    def __len__(self):
        return len(self.labels)


class CaptionErrorDetectorBase(nn.Module):
    def __init__(self):
        super(CaptionErrorDetectorBase, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.resnet_model = None
        self.linear1 = Linear(1000, 100)
        self.linear2 = Linear(100, 1)

    def forward(self, x):
        text_embeddings = self.bert_model(x["text"])
        image_embeddings = self.resnet_model(x["image"])
        return text_embeddings


def load_data():
    X = [
        ("person in read riding a motorcycle", "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg"),
        ("lady cutting cheese with reversed knife", "http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg"),
        ("girl touching a buffalo", "http://farm3.staticflickr.com/2169/2118578392_1193aa04a0_z.jpg")
    ]
    Y = ['true', 'true', 'false']

    sentences = [row[0] for row in X]

    # Embed text using BERT model.
    text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # {'input_ids': [[]], 'attention_mask': [[]]}
    inputs = text_tokenizer(sentences, return_tensors="pt", padding=True)
    # text_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])  # to get back the tokens.
    outputs = model(**inputs)

    train_dataset = ICErrorDataSet(X, Y, text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for data in train_dataloader:
        print(data)

    print('done')


def test_image_model():
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except Exception as e:
        print(f'Got exception {e}.')
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    print(input_batch.shape)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)


if __name__ == '__main__':
    # load_data()
    test_image_model()
