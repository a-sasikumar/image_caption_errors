import os
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel, ViTModel, RobertaTokenizer, \
    BertTokenizer, GPT2Tokenizer

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

# Change the indices below to choose a specific encoder and decoder.
pretrained_encoder_name = ['google/vit-base-patch16-224-in21k', 'google/vit-base-patch16-224'][0]
pretrained_decoder_name = ['roberta-base', 'bert-base-uncased', 'gpt2'][2]


def load_pretrained_tokenizer(tokenizer_name: str, model_max_length: int):
    if tokenizer_name == 'roberta-base':
        tok = RobertaTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length)
    elif tokenizer_name == 'bert-base-uncased':
        tok = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=model_max_length)
    elif tokenizer_name == 'gpt2':
        tok = GPT2Tokenizer.from_pretrained(pretrained_decoder_name, model_max_length=model_max_length)
        tok.pad_token = tok.eos_token_id
    else:
        raise Exception('Tokenizer name {} isn\'t supported yet'.format(tokenizer_name))
    return tok


def load_encoder_decoder_model_and_tokenizer(
        encoder_pretrained_model_name,
        decoder_pretrained_model_name,
        model_max_length=20
):
    tokenizer = load_pretrained_tokenizer(decoder_pretrained_model_name, model_max_length)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name,
        decoder_pretrained_model_name
    )

    # Make some modifications to the model parameters based on decoder/tokenizer.
    if decoder_pretrained_model_name == 'gpt2':
        model.config.eos_token_id = model.config.decoder.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.config.decoder_start_token_id = tokenizer.eos_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.decoder.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


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
        print(f'original sentences = {sentences}')
        print('dataset initialized.')

    def __getitem__(self, index):
        encodings = {
            'pixel_values': self.encoded_images[index].pixel_values,
            'labels': torch.tensor(self.encoded_text['input_ids'][index]),
            'attention_mask': torch.tensor(self.encoded_text['attention_mask'][index])
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
        self.model, tokenizer = load_encoder_decoder_model_and_tokenizer(
            pretrained_encoder_name,
            pretrained_decoder_name
        )
        self.name = "Encoder_{}_Decoder_{}".format(pretrained_encoder_name, pretrained_decoder_name).replace('/', '-')

        # tokenizer = _load_pretrained_tokenizer(pretrained_decoder_name)
        # self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        #     'google/vit-base-patch16-224-in21k',
        #     # 'google/vit-base-patch16-224',
        #     pretrained_decoder_name
        # )
        #
        # if pretrained_decoder_name == 'gpt2':
        #     self.model.config.eos_token_id = self.model.config.decoder.eos_token_id
        #     self.model.config.pad_token_id = self.model.config.eos_token_id
        #     self.model.config.decoder_start_token_id = tokenizer.eos_token_id
        # else:
        #     self.model.config.pad_token_id = tokenizer.pad_token_id
        #     self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        # self.model.config.vocab_size = self.model.config.decoder.vocab_size
        # self.name = "ViT_encoder_21k+Gpt_decoder"
        #
        # self.model.decoder.resize_token_embeddings(len(tokenizer))  # from github training-scripts code.

    def forward(self, x):
        # pixel_values has shape (N, 1, C, H, W). We remove the second dimension.
        outputs = self.model(
            pixel_values=x['pixel_values'].squeeze(dim=1),
            labels=x['labels'],
            attention_mask=x['attention_mask']
        )
        return outputs


def start_run(
        num_examples=10, batch_size=16, print_every=1, max_epochs=5, wandb_mode="online",
        load_pretrained=False, model_checkpoint_path=None
):
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

    tokenizer = load_pretrained_tokenizer(pretrained_decoder_name, model_max_length=20)
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        pretrained_encoder_name
    )

    train_dataset = ImageCaptioningDataset(X_train, y_train, tokenizer, feature_extractor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ImageCaptioningDataset(X_test, y_test, tokenizer, feature_extractor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    if load_pretrained:
        my_model = load_model_from_disk(model_checkpoint_path, ImageCaptioningModel())
    else:
        my_model = ImageCaptioningModel()
    my_model.to(device)
    optimizer = optim.AdamW(my_model.parameters(),
                            lr=5e-4,
                            eps=1e-5
                            )
    persist_path = None

    # Initialize wandb and add variables that you want associate with this run.
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    config = {
        "learning_rate": 5e-4,
        "train_size": len(train_dataloader.dataset),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "print_every": 0,
        "architecture": my_model.name,
        "dataset": "FOIL-COCO"
    }
    wandb.init(project="cs682-image-captioning", entity="682f21team", config=config, mode=wandb_mode)

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
            if (i + 1) % print_every == 0:
                validate_ic(val_dataloader, my_model, tokenizer, log_metric=True)
        print('\nep={}, train epoch loss = {}'.format(epoch + 1, train_running_loss))
        validate_ic(val_dataloader, my_model, tokenizer, log_metric=True)
        wandb.log({
            'train_loss': train_running_loss
        })
        persist_path = persist_model(
            my_model,
            path_root="../checkpoint/ic/",
            append_name="_small_data_ep={}".format(max_epochs)
        )

    # Check predictions on training data.
    trained_model = load_model_from_disk(persist_path, ImageCaptioningModel())
    trained_model.to(device)
    train_running_loss = 0
    for i, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            outputs = trained_model(batch)
            train_running_loss += outputs.loss

            gen_ids = trained_model.model.generate(batch['pixel_values'].squeeze(dim=1))
            decoded_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            print('decoded_train_text={}'.format(decoded_text))
    print(f'train loss with trained model = {train_running_loss}')  # just for sanity check.
    print('done')


def validate_ic(dataloader, my_model, tokenizer, log_metric=False):
    my_model.eval()
    my_model.to(device)
    val_running_loss = 0
    for i, batch in enumerate(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            outputs = my_model(batch)
            val_running_loss += outputs.loss.item()

            gen_ids = my_model.model.generate(batch['pixel_values'].squeeze(dim=1))
            decoded_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            print('decoded_val_text={}'.format(decoded_text))
    # print(f'val loss = {val_running_loss}')
    if log_metric:
        wandb.log({
            'val_loss': val_running_loss
        })


if __name__ == '__main__':
    # wandb_mode can be online, offline or disabled.
    start_run(
        num_examples=20, batch_size=32, max_epochs=5, print_every=2, wandb_mode="disabled",
        # load_pretrained=True, model_checkpoint_path="../checkpoint/ic/ViT_encoder+Bert_decoder_small_data_ep=10"
    )

'''
TODO:
Train till loss is zero for small data.
'''