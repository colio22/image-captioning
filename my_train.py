import argparse
import torch
import torchvision
from torch import nn
from torch.optim import Adam
import pickle
import os
from data.dataset import COCODataset
from transformers import AutoTokenizer
from model.transformer import GlobalEnhancedTransformer
from pycocoevalcap.cider.cider import Cider
import json
import torch.nn.functional as F


device = torch.device('cuda')
batch_size = 50


def train(model, loss_fn, optimizer, train_loader, tokenizer, epoch=0):
    """
    Trains a given model using a given optimizer and los function.
    """

    model.train()  # Set model to training mode

    train_loss = []         # Empty list to add loss of each batch
    n = len(train_loader)   # Number of samples
    print_idx = int(n/100)   # Sample interval at which to print loss progress

    # Begin training
    for batch_idx, (img, target, id) in enumerate(train_loader):
        print(f"Batch {batch_idx} of {int(len(train_loader)) + 1}")
        # Place data tensors on GPU
        img = img.to(device)

        # Tokenize input sequence
        tokens = tokenizer(target, padding=True, truncation=True, return_tensors='pt')
        token_ids = tokens['input_ids']
        token_ids = token_ids.to(device)
        # print(f"Token size: {token_ids.shape[0]}")
        expected_out = F.one_hot(token_ids[:,1:], num_classes=tokenizer.vocab_size)
        expected_out = expected_out.contiguous()

        optimizer.zero_grad()  # Initialize gradients to 0

        output = model(img, token_ids, batch_size)    # Put input batch through model
        # loss = loss_fn(output.view(-1, tokenizer.vocab_size), token_ids.view(-1))   # Calculate loss
        output = output[:,:-1].contiguous()
        loss = loss_fn(output.transpose(1, 2).float(), expected_out.transpose(1, 2).float())   # Calculate loss

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.sum().backward()        # Update weights
        optimizer.step()

        if batch_idx % print_idx == 0: # Log output 10 times per epoch
            print(f'Epoch {epoch}: [{batch_idx*len(img)}/{len(train_loader.dataset)}] Loss: {loss.sum().item():.3f}') 

        train_loss.append(loss.item()) # Add loss of batch to list
        
    return train_loss


def test(model: nn.Module,
        loss_fn: nn.modules.loss._Loss,
        test_loader: torch.utils.data.DataLoader,
        tokenizer,
        epoch: int=0):
    """
    Tests a given model against a test dataset.
    """

    model.eval() # Set model to test mode
    test_loss = 0         # Initialize total loss of test to 0
    test_predictions = {} # List of all generations

    # Begin testing
    with torch.no_grad():     # Turn gradients off for testing
        for images, targets, ids in test_loader:
            # Put input data on GPU
            images = images.to(device)
            # Tokenize input sequence
            tokens = tokenizer(targets, padding=True, truncation=True, return_tensors='pt')
            token_ids = tokens['input_ids']
            token_ids = token_ids.to(device)

            output = model(images, token_ids, batch_size)   # Put batch through model
            loss = loss_fn(output.view(-1, tokenizer.vocab_size), token_ids.view(-1))   # Calculate loss
            test_loss += loss.item()

    # Find average loss
    test_loss /= (len(test_loader.dataset) / test_loader.batch_size)
    test_stat = {'loss': test_loss}
    print(f"Test result on epoch {epoch}: Avg loss: {test_stat['loss']:.3f}")

    return test_stat


def generate_test_strings(model, data, tokenizer):
    """
    Creates a mapping of model-generated captions to the image
    they describe.
        model: The Transformer model to use for captioning
        data: The test dataset to be used
        tokenizer: Automatic tokenizer object used to brek apart input sequence
    Returns a dictionary of image ids and captions generated.
    """

    print("Generating Captions for each test image to prepare for metric evaluation")
    model.eval()        # Put model in evaluation mode
    caption_map = {}    # Dictionary to use to map images to generated captions

    # Turn gradients off to begin testing
    with torch.no_grad():
        for features, targets, ids in data:
            features = features.to(device)      # Put image on GPU

            # Iterate over every feature and id in batch
            for feature, id in zip(features, ids):
                enc_output, g_out = model.encoder(feature, batch_size=1)   # Put batch through model

                # Initialize input sequence with 'start of sentence' key
                predictions = [101*torch.ones([1,1], device=device).long()]

                # Allow for 50 words before terminating
                for i in range(50):
                    input_tokens = torch.cat(predictions, 1).to(device) # Add latest prediction to input list

                    # Feed current sequence through model
                    pred = model.decoder(input_tokens, enc_output, enc_output, g_out, batch_size=1)

                    # Use probability distribution to select which word comes next
                    dist = torch.distributions.Categorical(logits=pred[:, -1] / 0.5)
                    predicted_token = dist.sample().reshape(1, 1)
                    predictions.append(predicted_token)

                    # Stop generating caption when end of sentence key is produced
                    if predicted_token.item() == 102:
                        break

                # Decode tokens and add to dictionary
                pred_text = torch.cat(predictions, 1)
                pred_text_strings = tokenizer.decode(pred_text[0], skip_special_tokens=True)
                pred_text = "".join(pred_text_strings)
                caption_map[f'{id}'] = pred_text


def evaluate(generations, references):
    """
    Scores model based on various image captioning metrics
        generations: Generated captions
        references: Original captions
    """

    print("Evaluating CIDEr...")
    cider_eval = Cider()
    cider_score, _ = cider_eval.compute_score(references, generations)
    print(f"CIDEr Score: {cider_score}")


def parse_args():
    """
    Parses given command-line arguments
    """

    parser = argparse.ArgumentParser(description='Global Enhanced Transformer')
    parser.add_argument('--exp_name', type=str, default='GET')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print("Image Captioning Project")

    number_of_train_samples = 18000
    number_of_test_samples = int(number_of_train_samples * 0.2)


    # Create datasets from HDF file
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = COCODataset(args.features_path,  f'{args.annotation_folder}/captions_train2014.json', transform, limit=number_of_train_samples)
    test_dataset = COCODataset(args.features_path,  f'{args.annotation_folder}/captions_val2014.json', transform, limit=number_of_test_samples)

    # Create master list mapping captions to image ID
    reference_map = test_dataset.get_ref_dict()

    print(f'Training Dataset size: {len(train_dataset)}')
    print(f'Test Dataset size: {len(test_dataset)}')

    batch_size_train = args.batch_size
    batch_size_test = args.batch_size

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    print(train_loader)

    # Create tokenizer for input string
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # Clear memory from GPU because we will be using most of it
    torch.cuda.empty_cache()
 
    # Instantiate GET and put it on GPU
    model = GlobalEnhancedTransformer(vocab_size, 2048, 512, 64, 512, 8, 3, 0.1)
    model = model.to(device)

    # Select optimizer and loss function
    optim = Adam(model.parameters(), lr=0.1, betas=(0.9, 0.98))
    # criterion = nn.NLLLoss(ignore_index=tokenizer.vocab['[PAD]'])
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Train and test for desired number of epochs
    max_epoch = 1
    for epoch in range(1, max_epoch+1):
        loss = train(model, criterion, optim, train_loader, tokenizer, epoch)
        result = test(model, criterion, test_loader, tokenizer, epoch)

    # Save copy of model to drive
    print("Saving model...")
    torch.save(model.state_dict(), f'/content/drive/MyDrive/Colab Notebooks/ece570_project/{args.exp_name}.pth')

    # Generate captions for test dataset and map each to an image id
    generations = generate_test_strings(model, test_loader, tokenizer)

    # Save caption generations
    with open('/content/drive/MyDrive/Colab Notebooks/ece570_project/generations.json', 'w') as ref_file: 
        ref_file.write(json.dumps(generations))

    # Evaluate model perfromance with captioning metrics
    evaluate(generations, reference_map)


if __name__ == "__main__":
    args = parse_args()
    main(args)
