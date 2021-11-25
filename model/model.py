import os
import pickle
import sys
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
import wandb

device = torch.device('cpu')
# set computation device
if torch.cuda.is_available():
    # Get the GPU device name.
    device_name = torch.cuda.get_device_name()
    n_gpu = torch.cuda.device_count()
    print(f"Found device: {device_name}, n_gpu: {n_gpu}")
    device = torch.device("cuda")
print(device)


def load_invalid_urls() -> set:
    with open('../data/invalid_urls.pickle', 'rb') as f:
        return pickle.load(f)


invalid_urls = load_invalid_urls()


def persist_invalid_urls():
    with open('../data/invalid_urls.pickle', 'wb') as f:
        pickle.dump(invalid_urls, f, pickle.HIGHEST_PROTOCOL)


class ICErrorDataSet(Dataset):

    def __init__(self, X, y, text_tokenizer, image_preprocessor):
        self.data_root = "../data/images/"

        valid_indices1 = []
        # Save all images locally.
        for i, row in enumerate(X):
            if row[1] not in invalid_urls:
                if self._save_image_locally(row[1], row[2]):
                    valid_indices1.append(i)
                else:
                    invalid_urls.add(row[1])

        # Pre-process images and save as tensors.
        self.image_encodings = []
        valid_indices2 = []
        error_count = 0
        for i in valid_indices1:
            row = X[i]
            input_image = Image.open(self.data_root + row[2])
            try:
                input_tensor = image_preprocessor(input_image)
                self.image_encodings.append(input_tensor)
                valid_indices2.append(i)
            except Exception as e:
                error_count += 1
                invalid_urls.add(row[1])

        print('Errors: {}, locally saved images count: {}, final examples count: {}'.format(
            error_count, len(valid_indices1), len(valid_indices2)
        ))

        sentences = [X[i][0] for i in valid_indices2]
        self.encodings = text_tokenizer(sentences, truncation=True, padding=True)
        self.labels = torch.from_numpy(y[valid_indices2])

    def __getitem__(self, index):
        text_item = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        image_item = self.image_encodings[index]
        return {
                   "text": text_item,
                   "image": image_item
               }, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _save_image_locally(self, url, filename) -> bool:
        filepath = self.data_root + filename
        if not Path(filepath).is_file():
            try:
                print('downloading image {} in {}'.format(url, filepath))
                urllib.URLopener().retrieve(url, filepath)
            except Exception as e1:
                try:
                    print(f'Got exception {e1}.')
                    urllib.request.urlretrieve(url, filepath)
                except Exception as e2:
                    print(f'Got exception {e2}.')
                    return False
        return True


class CaptionErrorDetectorBase(nn.Module):
    def __init__(self):
        super(CaptionErrorDetectorBase, self).__init__()

        # Bert embedding for textual part.
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

        # ResNet embedding for image part. We use the last FCa layer's output.
        self.resnet_model = torchvision.models.resnet50(pretrained=True)

        # for param in self.resnet_model.parameters():
        #     param.requires_grad = False

        self.linear1 = Linear(768 + 1000, 100)
        self.linear2 = Linear(100, 2)

        # A variable to briefly tell how the architecture is.
        self.name = "DistilBert+ResNet+Linear2"

    def forward(self, x):
        text_embeddings = self.bert_model(**x["text"])  # (N, 8, 768)
        image_embeddings = self.resnet_model(x["image"])  # (N, 1000)
        text_cls_embeddings = text_embeddings.last_hidden_state[:, 0, :]  # (N, 768)
        text_image_append = torch.cat((text_cls_embeddings, image_embeddings), dim=1)  # (N, E1+E2)
        lin1out = self.linear1(text_image_append)  # (N, 100)
        output = self.linear2(torch.relu(lin1out))  # (N, 2)
        return output


def load_test_data(num_examples, batch_size=64) -> DataLoader:
    test_data = pd.read_csv(
        '../data/test_data/part-00000-9d7fcbd5-eed5-4753-a955-d0dd5f1d7bf1-c000.csv',
        escapechar='\\',
        nrows=num_examples
    )
    filename = test_data['file_name'].values.tolist()
    caption = test_data['caption'].values.tolist()
    urls = test_data['flickr_url'].values.tolist()
    target = test_data['foil'].values.reshape(-1).tolist()

    X_test = [(caption_, url_, filename_) for caption_, url_, filename_ in zip(caption, urls, filename)]
    Y_test = np.array(target) * 1

    text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', model_max_length=20)
    image_preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Loading test data.')
    test_dataset = ICErrorDataSet(X_test, Y_test, text_tokenizer, image_preprocessor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return test_dataloader


def load_train_val_data(num_examples=100, batch_size=20) -> (DataLoader, DataLoader):
    train_data = pd.read_csv(
        '../data/train_data/part-00000-079f7de7-8645-4478-a8dd-f3249585db1c-c000.csv',
        escapechar='\\',
        nrows=num_examples
    )
    filename = train_data['file_name'].values.tolist()
    caption = train_data['caption'].values.tolist()
    urls = train_data['flickr_url'].values.tolist()
    target = train_data['foil'].values.reshape(-1).tolist()

    X = [(caption_, url_, filename_) for caption_, url_, filename_ in zip(caption, urls, filename)]
    Y = np.array(target) * 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)

    text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', model_max_length=20)
    image_preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Loading training data.')
    train_dataset = ICErrorDataSet(X_train, y_train, text_tokenizer, image_preprocessor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Loading validation data.')
    val_dataset = ICErrorDataSet(X_test, y_test, text_tokenizer, image_preprocessor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def persist_model(my_model, path_root="../checkpoint/", append_name="") -> str:
    save_path = path_root + my_model.name + append_name
    torch.save(my_model.state_dict(), save_path)
    return save_path


def load_model_from_disk(save_path: str) -> nn.Module:
    my_model = CaptionErrorDetectorBase()
    my_model.load_state_dict(torch.load(save_path))
    my_model.eval()
    print('Model loaded from path {} successfully.'.format(save_path))
    return my_model


def train_model(num_examples=30, batch_size=10, max_epochs=10, print_every=1):
    print('Training started with num_example={}, batch_size={}, max_epochs={}'.format(
        num_examples, batch_size, max_epochs
    ))
    custom_name = "N={},b={},ep={}".format(num_examples, batch_size, max_epochs)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_loader, val_loader = load_train_val_data(num_examples=num_examples, batch_size=batch_size)
    print('Data loaded.')

    my_model = CaptionErrorDetectorBase()
    my_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(my_model.parameters(),
                            lr=1e-4,
                            eps=1e-5
                            )

    # Initialize wandb and add variables that you want associate with this run.
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    config = {
        "learning_rate": 1e-4,
        "train_size": len(train_loader.dataset),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "print_every": print_every,
        "architecture": my_model.name,
        "dataset": "FOIL-COCO"
    }
    wandb.init(project="cs682-image-captioning", entity="682f21team", config=config)

    best_val_acc = 0
    best_model_save_path = None

    for epoch in tqdm(range(max_epochs), total=max_epochs):
        print(f'Epoch: {epoch + 1}')

        train_loss = 0
        train_correct = 0

        # Training the model.
        my_model.train()
        for i, data in enumerate(train_loader):
            local_batch, local_labels = data[0], data[1]
            optimizer.zero_grad()

            text = local_batch['text']
            image = local_batch['image']
            input_ids = text['input_ids']
            attention_mask = text['attention_mask']
            device_input = {
                'image': image.to(device),
                'text': {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
            }
            device_local_labels = local_labels.to(device)
            text['input_ids'].requires_grad = False
            text['attention_mask'].requires_grad = False
            device_local_labels.requires_grad = False
            image.requires_grad = False

            outputs = my_model(device_input)
            loss = criterion(outputs, device_local_labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)

            train_batch_loss = loss.item()
            train_loss += train_batch_loss
            train_batch_correct = (preds == device_local_labels).sum().item()
            train_correct += train_batch_correct

            if i % print_every == 0:
                val_loss, val_acc = validate(val_loader, my_model, criterion, log_metrics=True)
                if val_acc > best_val_acc:
                    best_model_save_path = persist_model(my_model, append_name=custom_name)
                    best_val_acc = val_acc
                wandb.log({'train_batch_loss': train_batch_loss})

        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        print(f'\nEpoch: {epoch + 1}\nTrain Loss: {train_loss:.4f}, Train Acc: {train_accuracy}')
        wandb.log({
            'train_accuracy': train_accuracy,
            'train_loss': train_loss
        })
        val_loss, val_acc = validate(val_loader, my_model, criterion, log_metrics=True)
        if val_acc > best_val_acc:
            best_model_save_path = persist_model(my_model, append_name=custom_name)
            best_val_acc = val_acc
        print()

    print('training done.')
    return best_model_save_path


def validate(val_loader, my_model, loss_func, log_metrics=False):
    # Perform validation
    my_model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        # for i, data in tqdm(enumerate(val_loader), total=int(len(val_loader.dataset) / val_loader.batch_size)):
        for i, data in enumerate(val_loader):
            local_batch, local_labels = data[0], data[1]
            text = local_batch['text']
            image = local_batch['image']
            input_ids = text['input_ids']
            attention_mask = text['attention_mask']
            device_input = {
                'image': image.to(device),
                'text': {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
            }
            device_local_labels = local_labels.to(device)
            outputs = my_model(device_input)
            loss = loss_func(outputs, device_local_labels)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == device_local_labels).sum().item()

        val_loss = val_running_loss
        val_accuracy = 100. * val_running_correct / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
        if log_metrics:
            wandb.log({
                'val_accuracy': val_accuracy,
                'val_loss': val_loss
            })
        return val_loss, val_accuracy


def test_load_data():
    X_orig = [
        ("person in read riding a motorcycle", "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg", "motorcycle.jpg"),
        ("lady cutting cheese with reversed knife", "http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg", "cheese.jpg"),
        ("girl touching a buffalo", "http://farm3.staticflickr.com/2169/2118578392_1193aa04a0_z.jpg", "boy.jpg")
    ]
    Y_orig = ['true', 'true', 'false']

    X_panda = pd.DataFrame(X_orig)
    Y_panda = pd.DataFrame(Y_orig)

    X = X_panda.values.tolist()
    Y = (np.array(Y_panda.values.reshape(-1).tolist()) == 'true') * 1

    sentences = [row[0] for row in X]

    # Embed text using BERT model.
    text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # {'input_ids': [[]], 'attention_mask': [[]]}
    # inputs = text_tokenizer(sentences, return_tensors="pt", padding=True)
    # text_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])  # to get back the tokens.
    # outputs = model(**inputs)
    # return

    image_preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ICErrorDataSet(X, Y, text_tokenizer, image_preprocessor)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    my_model = CaptionErrorDetectorBase()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_model.parameters(),
            lr = 5e-5, # args.learning_rate - default is 5e-5
            eps = 1e-5
        )
    max_epochs = 2
    for epoch in range(max_epochs):
        # Training the model.
        my_model.train()
        for local_batch, local_labels in train_dataloader:
            # print(local_batch, local_labels)
            optimizer.zero_grad()

            outputs = my_model(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()

            print('loss={:.3g}'.format(loss))
    print('done')


def test_image_model():
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "../data/dog.jpg")
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

    # modify the model to get 800 output from linear layer.
    model.fc = nn.Linear(2048, 800)

    with torch.no_grad():
        output = model(input_batch)
        print(output)


def find_generalization_performance(best_model_save_path: str):
    print('Finding test time performance.')
    test_loader = load_test_data(num_examples=512)
    test_model = load_model_from_disk(best_model_save_path)
    test_model.to(device)
    validate(test_loader, test_model, nn.CrossEntropyLoss(), log_metrics=False)


if __name__ == '__main__':
    # test_image_model()

    # Save data locally so that training is faster.
    load_train_val_data(num_examples=10_000, batch_size=100)
    load_test_data(num_examples=1000, batch_size=64)
    persist_invalid_urls()

    # Perform actual training followed by test-time performance check.
    # max_epochs = 5
    # best_model_train_path = train_model(num_examples=10_000, batch_size=100, max_epochs=max_epochs, print_every=10)
    # find_generalization_performance(best_model_train_path)
    sys.exit(0)

"""
Some resources:
data generator: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
training loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
training/validation loop: https://debuggercafe.com/creating-efficient-image-data-loaders-in-pytorch-for-deep-learning/
"""
