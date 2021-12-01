import os
import urllib
from enum import Enum
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
        tok = BertTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length)
    elif tokenizer_name == 'gpt2':
        tok = GPT2Tokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length)
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
        self.foil_word = []  # only useful for test data evaluation.
        for idx in valid_indices:
            row = X[idx]
            image = Image.open(self.data_root + row[2])
            encoded_image = feature_extractor(images=image, return_tensors="pt")
            self.encoded_images.append(encoded_image)
            sentences.append(X[idx][0])
            if len(row) >= 4:
                self.foil_word.append(row[3])
        self.encoded_text = tokenizer(sentences, truncation=True, padding=True)
        print(f'original sentences = {sentences}')
        print('dataset initialized.')

    def __getitem__(self, index):
        encodings = {
            'pixel_values': self.encoded_images[index].pixel_values,
            'labels': torch.tensor(self.encoded_text['input_ids'][index]),
            'attention_mask': torch.tensor(self.encoded_text['attention_mask'][index]),
        }
        if len(self.foil_word) > 0:
            encodings['foil_word'] = self.foil_word[index]
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

    def forward(self, x):
        # pixel_values has shape (N, 1, C, H, W). We remove the second dimension.
        outputs = self.model(
            pixel_values=x['pixel_values'].squeeze(dim=1),
            labels=x['labels'],
            attention_mask=x['attention_mask']
        )
        return outputs


def get_test_dataloader(num_examples=10, batch_size=16, foil_only=False):
    test_data = pd.read_csv(
        '../data/test_data/part-00000-9d7fcbd5-eed5-4753-a955-d0dd5f1d7bf1-c000.csv',
        escapechar='\\',
        nrows=num_examples * 2 if foil_only else num_examples
    )
    if foil_only:
        test_data = test_data[test_data['foil']]
    filename = test_data['file_name'].values.tolist()
    caption = test_data['caption'].values.tolist()
    urls = test_data['flickr_url'].values.tolist()
    target = test_data['foil'].values.reshape(-1).tolist()
    foil_word = test_data['foil_word'].values.tolist()

    X_test = [(caption_, url_, filename_, foil_word_) for caption_, url_, filename_, foil_word_ in zip(caption, urls, filename, foil_word)]
    Y_test = np.array(target) * 1

    tokenizer = load_pretrained_tokenizer(pretrained_decoder_name, model_max_length=20)
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        pretrained_encoder_name
    )

    test_dataset = ImageCaptioningDataset(X_test, Y_test, tokenizer, feature_extractor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_dataloader


def get_train_val_dataloaders(num_examples=10, batch_size=16, target_foil=None):
    train_data = pd.read_csv(
        '../data/train_data/part-00000-079f7de7-8645-4478-a8dd-f3249585db1c-c000.csv',
        escapechar='\\',
        nrows=num_examples * 2 if target_foil else num_examples
    )
    if target_foil:
        train_data = train_data[train_data['foil'] == target_foil]
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

    return train_dataloader, val_dataloader


def start_run(
        num_examples=10, batch_size=16, print_every=1, max_epochs=5, wandb_mode="online",
        load_pretrained=False, model_checkpoint_path=None, save_model=True
):
    train_dataloader, val_dataloader = get_train_val_dataloaders(num_examples, batch_size, target_foil=False)
    tokenizer = load_pretrained_tokenizer(pretrained_decoder_name, model_max_length=20)

    if load_pretrained:
        my_model = load_model_from_disk(model_checkpoint_path, ImageCaptioningModel())
    else:
        my_model = ImageCaptioningModel()
    my_model.to(device)
    lr = 1e-3
    optimizer = optim.AdamW(my_model.parameters(),
                            lr=lr,
                            eps=1e-5
                            )
    persist_path = None

    # Initialize wandb and add variables that you want associate with this run.
    os.environ.setdefault('WANDB_API_KEY', '713a778aae8db6219a582a6b794204a5af2cb75d')
    config = {
        "learning_rate": lr,
        "train_size": len(train_dataloader.dataset),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "print_every": print_every,
        "architecture": my_model.name,
        "dataset": "FOIL-COCO"
    }
    wandb.init(project="cs682-image-captioning", entity="682f21team", config=config, mode=wandb_mode)

    best_val_loss = np.inf
    for epoch in tqdm(range(max_epochs)):
        # print(f'Epoch: {epoch + 1}')
        my_model.train()
        train_running_loss = 0
        for i, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
                batch[k].requires_grad = False
            optimizer.zero_grad()
            outputs = my_model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_batch_loss = loss.item()
            train_running_loss += train_batch_loss
            # print('train batch loss = {}'.format(loss.item()))
            if (i + 1) % print_every == 0:
                _ = validate_ic(val_dataloader, my_model, tokenizer, log_metric=True)
                wandb.log({
                    'train_batch_loss': train_batch_loss
                })
        print('\nep={}, train epoch loss = {}'.format(epoch + 1, train_running_loss))
        val_loss = validate_ic(val_dataloader, my_model, tokenizer, log_metric=True, print_predictions=True)
        wandb.log({
            'train_loss': train_running_loss
        })
        if val_loss < best_val_loss or epoch == max_epochs - 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if save_model:
                append_name = "_lr={}_tr={}_bs={}_ep={}".format(
                    lr, len(train_dataloader.dataset), batch_size, max_epochs
                )
                if epoch == max_epochs - 1:
                    append_name += "_last_ep"
                persist_path = persist_model(
                    my_model,
                    path_root="../checkpoint/ic/",
                    append_name=append_name
                )

    # Check predictions on training data.
    check_predictions_on_train = False
    if persist_path and check_predictions_on_train:
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
        print(f'train loss with trained model = {train_running_loss}')
    print('done')


def access_performance(model_checkpoint_path, num_examples=100, batch_size=16):
    train_dataloader, val_dataloader = get_train_val_dataloaders(num_examples, batch_size, target_foil=False)
    tokenizer = load_pretrained_tokenizer(pretrained_decoder_name, model_max_length=20)
    my_model = load_model_from_disk(model_checkpoint_path, ImageCaptioningModel())
    validate_ic(val_dataloader, my_model, tokenizer, False, True)


def validate_ic(dataloader, my_model, tokenizer, log_metric=False, print_predictions=False):
    my_model.eval()
    my_model.to(device)
    val_running_loss = 0
    for i, batch in enumerate(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad():
            outputs = my_model(batch)
            val_running_loss += outputs.loss.item()

            if print_predictions:
                # gen_ids = my_model.model.generate(
                #     batch['pixel_values'].squeeze(dim=1)
                # )
                gen_ids = my_model.model.generate(
                    batch['pixel_values'].squeeze(dim=1), num_beams=4, max_length=20
                )
                decoded_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                print('decoded_val_text={}'.format(decoded_text))
    if log_metric:
        wandb.log({
            'val_loss': val_running_loss
        })
    return val_running_loss


def find_label_scores(model, my_labels, pixel_values, attention_mask, eos_token_id):
    """
    Gives scores to each possible caption representation so that we can find the most probable caption.
    :param model: pre-trained model to be used in eval mode.
    :param my_labels: torch tensor of shape (N, K) where N is the number of different captions and K is the length of
        every caption. my_label[i] has the label ids for ith caption.
    :param pixel_values: torch tensor of shape (1, 3, 224, 224) having pixel values for a single image corresponding to
        all the captions.
    :param attention_mask: torch tensor of shape (K,) having attention mask for all captions. All captions have the
        same attention mask.
    :param eos_token_id: EOS token id that the tokenizer uses.
    :return: torch tensor of shape (N,) where we have sum of log of prob of all captions according to the LM.
    """
    model.eval()
    N = my_labels.shape[0]
    batch = {
        'pixel_values': pixel_values.repeat(N, 1, 1, 1),
        'attention_mask': attention_mask.reshape(1, -1).repeat(N, 1),
        'labels': my_labels
    }
    for k, v in batch.items():
        batch[k] = v.to(device)
    outputs = model(batch)
    scores = []
    for i in range(N):
        prob = torch.softmax(outputs.logits[i], dim=1)  # (k, V) where k is the length of each caption.
        log_prob = torch.log(prob)  # (k, V)
        sum_log_prob = 0
        for ii in range(log_prob.shape[0]):
            most_prob_label_id = torch.argmax(log_prob[ii])
            if most_prob_label_id == eos_token_id:
                break
            sum_log_prob += log_prob[ii][most_prob_label_id]
        scores.append(sum_log_prob)
    return torch.tensor(scores, dtype=torch.float32)


def probability_distribution_of_caption(model_checkpoint_path: str, num_examples=5, batch_size=5):
    trained_model = load_model_from_disk(model_checkpoint_path, ImageCaptioningModel())
    trained_model.eval()
    trained_model.to(device)
    tokenizer = load_pretrained_tokenizer(pretrained_decoder_name, model_max_length=20)

    test_dataloader = get_test_dataloader(num_examples, batch_size, foil_only=True)
    for i, batch in enumerate(test_dataloader):
        for k, v in batch.items():
            if k != 'foil_word':
                batch[k] = v.to(device)
        with torch.no_grad():
            gen_ids = trained_model.model.generate(batch['pixel_values'].squeeze(dim=1))
            decoded_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            print('decoded_val_text={}'.format(decoded_text))

            outputs = trained_model(batch)

            candidate_labels = []
            # Generate multiple captions for each input caption by replacing a word.
            N = batch['labels'].shape[0]  # number of examples.
            for n in np.arange(N):
                attn_mask = batch['attention_mask'][n]
                num_tokens = attn_mask.shape[0]
                # number of tokens with attention mask 1.
                num_valid_tokens = num_tokens if attn_mask[-1] == 1 else (attn_mask == 0).nonzero(as_tuple=True)[0][0]
                my_labels = torch.zeros((num_valid_tokens, num_tokens), dtype=torch.int64)
                # replace every word with LM word.
                for j in np.arange(num_valid_tokens):
                    copy_labels = batch['labels'][n].clone().detach().to(torch.int64)
                    # replace jth word with best word according to LM word other than the same word.
                    _, top6 = torch.topk(outputs.logits[n][j], 6)
                    for idx in top6:
                        if idx != copy_labels[j] and idx not in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id]:
                            copy_labels[j] = idx
                            break
                    my_labels[j] = copy_labels
                candidate_labels.append(my_labels)

            # Find the text equivalent of every candidate for every example.
            for n in np.arange(N):
                my_labels = candidate_labels[n]
                decoded_captions_1 = tokenizer.batch_decode(my_labels)
                decoded_captions = tokenizer.batch_decode(my_labels, skip_special_tokens=True)
                # TODO: stop decoding when encountered EOS.
                print(f'all captions with candidates\n{decoded_captions}')

            # Find the best candidate for every example.
            for n in np.arange(N):
                my_labels = candidate_labels[n].clone().detach()
                label_scores = find_label_scores(
                    trained_model, my_labels, batch['pixel_values'][n], batch['attention_mask'][n],
                    tokenizer.eos_token_id
                )
                ci = torch.argmax(label_scores)
                foil_caption = tokenizer.decode(batch['labels'][n], skip_special_tokens=True)
                correct_caption = tokenizer.decode(my_labels[ci], skip_special_tokens=True)
                print(f'foil caption: {foil_caption}\ncorrected caption: {correct_caption}')
                detected_foil_word = tokenizer.convert_ids_to_tokens(batch['labels'][n])[ci]
                actual_foil_word = batch['foil_word'][n]
                print(f"detected foil word: {detected_foil_word}\nactual foil word: {actual_foil_word}")
    print('done')


if __name__ == '__main__':
    class WandbMode(Enum):
        ONLINE = "online"
        OFFLINE = "offline"
        DISABLED = "disabled"

    start_run(
        num_examples=1000, batch_size=16, max_epochs=10, print_every=2, wandb_mode=WandbMode.ONLINE.value,
        save_model=True,
        # load_pretrained=True, model_checkpoint_path="../checkpoint/ic/Encoder_google-vit-base-patch16-224_Decoder_gpt2_lr=0.001_tr=785_bs=16_ep=10_last_ep.pt"
    )

    # probability_distribution_of_caption(
    #     "../checkpoint/ic/Encoder_google-vit-base-patch16-224-in21k_Decoder_gpt2_small_data_ep=2",
    #     num_examples=5, batch_size=5
    # )

    # access_performance("../checkpoint/ic/Encoder_google-vit-base-patch16-224-in21k_Decoder_gpt2_lr=0.002_tr=785_bs=16_ep=10_last_ep.pt")
