import argparse
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from data.dataset import COCODataset
from transformers import AutoTokenizer
from model.transformer import GlobalEnhancedTransformer
from pycocoevalcap.cider.cider import Cider
import json
import torch.nn.functional as F


# GPU Device instance to be used throughout project
device = torch.device('cuda')


def train(model, loss_fn, optimizer, train_loader, tokenizer, batch_size=50, epoch=0):
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

        # Generate 1-hot version of input showing which token of vocabulary to predict
        # Take from index 1 to end because token 0 of input should produce token 1
        # of expected_out
        expected_out = F.one_hot(token_ids[:,1:], num_classes=tokenizer.vocab_size)
        expected_out = expected_out.contiguous()    # Contiguous memory because of weird error

        optimizer.zero_grad()  # Initialize gradients to 0

        output = model(img, token_ids, batch_size)    # Put input batch through model

        # Discard last token of output sequence as garbage
        output = output[:,:-1].contiguous() # contiguous mem because of weird error
        loss = loss_fn(output.transpose(1, 2).float(), expected_out.transpose(1, 2).float())   # Calculate loss

        loss.sum().backward()        # Update weights
        optimizer.step()

        if batch_idx % print_idx == 0: # Log output 10 times per epoch
            print('\n---------------------------------')
            print(f'Epoch {epoch}: [{batch_idx*len(img)}/{len(train_loader.dataset)}] Loss: {loss.sum().item():.3f}') 
            print('---------------------------------\n')

        train_loss.append(loss.sum().item()) # Add loss of batch to list

        
    return train_loss


def generate_test_strings(model, features, ids, tokenizer, caption_map):
    """
    Creates a mapping of model-generated captions to the image
    they describe.
        model: The Transformer model to use for captioning
        data: The test dataset to be used
        tokenizer: Automatic tokenizer object used to brek apart input sequence
    Returns a dictionary of image ids and captions generated.

    Credit to Ditria for technique used to generate tokens
    (https://github.com/LukeDitria/pytorch_tutorials/blob/main/section14_transformers/solutions/Pytorch5_Transformer_Image_Captioning_Pytorch_Layers.ipynb)
    """

    model.eval()        # Put model in evaluation mode

    # Turn gradients off to begin testing
    with torch.no_grad():
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

    return caption_map


def test(model, loss_fn, test_loader, tokenizer, batch_size=50, epoch=0):
    """
    Tests a given model against a test dataset.
    """

    model.eval() # Set model to test mode
    test_loss = 0         # Initialize total loss of test to 0
    test_predictions = {} # List of all generations

    # Begin testing
    with torch.no_grad():     # Turn gradients off for testing
        for batch_idx, (images, targets, ids) in enumerate(test_loader):
            print(f"Batch {batch_idx} of {int(len(test_loader)) + 1}")
            # Put input data on GPU
            images = images.to(device)
            # Tokenize input sequence
            tokens = tokenizer(targets, padding=True, truncation=True, return_tensors='pt')
            token_ids = tokens['input_ids']
            token_ids = token_ids.to(device)

            # Use 1-hot encoding to transform expected token to place in vocabulary
            expected_out = F.one_hot(token_ids[:,1:], num_classes=tokenizer.vocab_size)
            expected_out = expected_out.contiguous() # Contiguous because of weird error

            output = model(images, token_ids, batch_size)   # Put batch through model

            # Throw out last token predicted as garbage
            output = output[:,:-1].contiguous() # Continous because of weird error
            loss = loss_fn(output.transpose(1, 2).float(), expected_out.transpose(1, 2).float())   # Calculate loss
            test_loss += loss.sum().item()  # Track total loss

            # Generate actual string captions
            test_predictions = generate_test_strings(model, images, ids, tokenizer, test_predictions)

    # Find average loss
    test_loss /= (len(test_loader.dataset) / test_loader.batch_size)
    test_stat = {'loss': test_loss, 'predictions': test_predictions}
    print(f"Test result on epoch {epoch}: Avg loss: {test_stat['loss']:.3f}")

    return test_stat



def evaluate(generations, references):
    """
    Scores model based on various image captioning metrics
        generations: Generated captions
        references: Original captions
    """

    print("Evaluating CIDEr...")

    # Ensure dictionaries have same keys
    possibly_missing_data = generations.keys()
    to_delete = []
    for key in references.keys():
      if key not in possibly_missing_data:
        to_delete.append(key)

    # Delete keys that do not match between reference and generated
    for key in to_delete:
        del references[key]

    # Cider expects captions as lists so we convert the strings to
    # lists of strings here
    cider_refs = {key: [val] for key, val in references.items()}
    cider_preds = {key: [val] for key, val in generations.items()}

    # Compute score
    cider_eval = Cider()
    cider_score, _ = cider_eval.compute_score(cider_refs, cider_preds)

    print("\n==============================")
    print(f" CIDEr Score: {cider_score}")
    print("==============================\n")

    # Print out 5 example captions and what the input was
    print("Example generations:")
    i = 0
    for gen, ref in zip(generations, references):
        print(f"\nExpected Caption: {ref}")
        print(f"Generated Caption: {gen}")
        i += 1

        if i == 5:
            break


def parse_args():
    """
    Parses given command-line arguments

    Credit to Cornia et al. and the M2 Transformer repository for this code.
    """

    parser = argparse.ArgumentParser(description='Global Enhanced Transformer')
    parser.add_argument('--exp_name', type=str, default='GET')
    parser.add_argument('--save_path', type=str, default="./")
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    """
    Main method
    """

    print("Image Captioning Project")

    # Limits on dataset for manageable train time
    number_of_train_samples = 18000
    number_of_test_samples = int(number_of_train_samples * 0.1)


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
    # Credit to Ditria for tokenizing method used
    # (https://github.com/LukeDitria/pytorch_tutorials/blob/main/section14_transformers/solutions/Pytorch5_Transformer_Image_Captioning_Pytorch_Layers.ipynb)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # Clear memory from GPU because we will be using most of it
    torch.cuda.empty_cache()
 
    # Instantiate GET and put it on GPU
    model = GlobalEnhancedTransformer(vocab_size, 2048, 512, 64, 512, 8, 3, 0.1)
    model = model.to(device)

    # If no model provided, proceed with training
    if args.load_model == None:
        # Select optimizer and loss function
        # Adam parameters based on the work of Cornia et al. in the
        # M2 Transformer
        optim = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98))
        criterion = nn.CrossEntropyLoss(reduction="none")

        # Train for desired number of epochs
        print("Starting training...\n")
        max_epoch = 1
        for epoch in range(1, max_epoch+1):
            loss = train(model, criterion, optim, train_loader, tokenizer, batch_size_train, epoch)

        # Save copy of model to drive
        print("Training complete. Saving model...\n")
        torch.save(model.state_dict(), f'{args.save_path}/{args.exp_name}.pth')
    else:   # If model provided, skip training and load
        print("Loading model...")
        model.load_state_dict(torch.load(args.load_model, weights_only=True))

    # Test for desired number of epochs
    print("Beginning tests...\n")
    for epoch in range(1, max_epoch+1):
        result = test(model, criterion, test_loader, tokenizer, batch_size_test, epoch)
        print("Testing complete.\n")

    # Evaluate model perfromance with CIDEr metric
    evaluate(result['predictions'], reference_map)

    # Save caption generations
    with open(f'{args.save_path}/generations_{args.exp_name}.json', 'w') as ref_file: 
        ref_file.write(json.dumps(result['predictions']))

    with open(f'{args.save_path}/references_{args.exp_name}.json', 'w') as ref_file: 
        ref_file.write(json.dumps(reference_map))


if __name__ == "__main__":
    args = parse_args()
    main(args)
