import urllib
from pathlib import Path

import requests
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel, ViTModel, RobertaTokenizer

from models.model import load_invalid_urls, persist_model, load_model_from_disk

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    n_gpu = torch.cuda.device_count()
    print(f"Found device: {device_name}, n_gpu: {n_gpu}")
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print('device={}'.format(device))


invalid_urls = load_invalid_urls()


class ImageCaptioningDataset(Dataset):
    def __init__(self, X, y, tokenizer, feature_extractor):
        self.data_root = "../data/images/"
        valid_indices = self._find_valid_image_indices(X)
        sentences = []
        self.encoded_images = []
        for idx in valid_indices:
            row = X[idx]
            image = Image.open(self.data_root + row[2])
            encoded_image = feature_extractor(images=image, return_tensors="pt")
            self.encoded_images.append(encoded_image)
            sentences.append(X[idx][0])

        self.encoded_text = tokenizer(sentences, truncation=True, padding=True)

    def __getitem__(self, index):
        encodings = {
            'pixel_values': self.encoded_images[index].pixel_values,
            'labels': torch.tensor(self.encoded_text['input_ids'][index])
        }
        return encodings

    def __len__(self):
        return len(self.encoded_images)

    def _find_valid_image_indices(self, X) -> list:
        valid_indices1 = []
        # Save all images locally.
        for i, row in enumerate(X):
            if row[1] not in invalid_urls:
                if self._save_image_locally(row[1], row[2]):
                    valid_indices1.append(i)
                else:
                    invalid_urls.add(row[1])
        return valid_indices1

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


class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained('google/vit-base-patch16-224', 'roberta-base')
        self.model.config.decoder_start_token_id = tokenizer.bos_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.name = "ViT_encoder+RoBerta_decoder"

    def forward(self, x):
        # pixel_values has shape (N, 1, C, H, W). We remove the second dimension.
        outputs = self.model(pixel_values=x['pixel_values'].squeeze(dim=1), labels=x['labels'])
        return outputs


def test_dataset(num_examples=10, batch_size=16):
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

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_length=20)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    train_dataset = ImageCaptioningDataset(X_train, y_train, tokenizer, feature_extractor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    my_model = ImageCaptioningModel()
    my_model.to(device)
    optimizer = optim.AdamW(my_model.parameters(),
                            lr=1e-4,
                            eps=1e-5
                            )
    max_epochs = 5
    persist_path = None

    for epoch in tqdm(range(max_epochs)):
        # print(f'Epoch: {epoch + 1}')
        my_model.train()
        train_running_loss = 0
        for i, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            optimizer.zero_grad()
            outputs = my_model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            # print('train batch loss = {}'.format(loss.item()))
        print('\nep={}, train epoch loss = {}'.format(epoch + 1, train_running_loss))
        persist_path = persist_model(my_model, path_root="../checkpoint/ic/", append_name="_small_data")

    trained_model = load_model_from_disk(persist_path, ImageCaptioningModel())
    trained_model.to(device)
    for i, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            gen_ids = trained_model.model.generate(batch['pixel_values'].squeeze(dim=1))
            outputs = tokenizer.batch_decode(gen_ids)
            print('outputs={}'.format(outputs))
    print('done')


def process():
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    encoded_image = feature_extractor(images=image, return_tensors="pt")

    image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    image_output = image_model(**encoded_image)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    text = "Replace me by any text you'd like."
    encoded_text = tokenizer(text, return_tensors='pt')

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained('google/vit-base-patch16-224', 'roberta-base')
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # forward pass for model?
    output1 = model(decoder_input_ids=encoded_text['input_ids'], pixel_values=encoded_image['pixel_values'])
    output2 = model(labels=encoded_text['input_ids'], pixel_values=encoded_image['pixel_values'])

    # Inference from model
    generated_ids = model.generate(encoded_image.pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids)
    print(generated_text)
    print('done')


if __name__ == '__main__':
    # process()
    test_dataset()

'''
TODO:
Move tensors to GPU.
Train till loss is zero for small data.
Do caption prediction for train data and val data.
Add wandb tracking.
'''