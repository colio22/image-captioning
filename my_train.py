import argparse
import torch
from torch import nn
from torch.optim import Adam
import pickle
import os
from common.data.field import ImageDetectionsField, TextField, RawField
from common.data import COCODataset, DataLoader
from common.train import train
from common.utils.utils import create_dataset
from common.evaluation import PTBTokenizer, Cider
from model.transformer import GlobalEnhancedTransformer
# from models import build_encoder, build_decoder, Transformer

device = torch.device('cuda')


def train(model, loss_fn, optimizer, train_loader, text_field, epoch=0):
    model.train()  # Set model to training mode
    train_loss = []         # Empty list to add loss of each batch
    n = len(train_loader)   # Number of samples
    print_idx = int(n/10)   # Sample interval at which to print loss progress
    # Begin training
    for batch_idx, (img, target) in enumerate(train_loader):
        # Place data tensors on GPU
        img = img.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # Initialize gradients to 0
        output = model(img, target)    # Put input batch through model
        captions = target[:, 1:].contiguous()
        output = output[:, :-1].continuous()
        loss = loss_fn(output.view(-1, len(text_field.vocab)), captions.view(-1))   # Calculate loss
        loss.backward()        # Update weights
        optimizer.step()
        if batch_idx % print_idx == 0: # Log output 10 times per epoch
            print(f'Epoch {epoch}: [{batch_idx*len(img)}/{len(train_loader.dataset)}]') 
        # train_loss.append(loss.item()) # Add loss of batch to list
        
    return train_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Dual Transformer')
    parser.add_argument('--output', type=str, default='DualModel')
    parser.add_argument('--exp_name', type=str, default='dft')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--clip_path', type=str, default='coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
    parser.add_argument('--vinvl_path', type=str, default='coco/features/COCO2014_VinVL.hdf5')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print("Image Captioning Project")
    print(args.features_path)

    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
                           

    # Create the dataset
    dataset = COCODataset(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    # cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                       num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

    print(dataloader_train)

    model = GlobalEnhancedTransformer(len(text_field.vocab), 54, 64, 512, 64, 8, 3, 0.1)

    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    criterion = nn.NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])

    max_epoch = 3

    for epoch in range(1, max_epoch+1):
        loss = train(model, criterion, optim, dataloader_train, text_field, epoch)

    print(f'===Loss: {loss}')

if __name__ == "__main__":
    args = parse_args()
    main(args)
