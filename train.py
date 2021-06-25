from tqdm import tqdm
from transformers.models.distilbert import DistilBertTokenizer
from model.SimpleCLIP import CLIPModel
from torch.optim import SGD, Adam, AdamW
from utils import get_loader, test_model

def train_model(images_path, annotation_path, batch_size, epoch_nums, run_on, lr, momentum, **kwargs):

    loader = get_loader(images_path, annotation_path, batch_size, **kwargs)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = CLIPModel(**kwargs)
    model = model.to(run_on)
    optimizer = SGD(model.parameters(), lr = lr, momentum=momentum)
    accuracies = list()

    for i in range(epoch_nums):
        average_loss = 0.
        for batchnum, (image, text) in tqdm(enumerate(loader)):

            optimizer.zero_grad()
            image, text = image.to(run_on)
            tokenized = tokenizer(text, max_length = 20, padding = 'max_length', return_tensors='pt')
            tokenized['input_ids'], tokenized['attention_mask'] = tokenized['input_ids'].cuda(), tokenized['attention_mask'].cuda()
            loss = model(images = image, **tokenized)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            if batchnum % 20 == 19:
                print(f'Batch {batchnum} average loss {average_loss / 20.}')
                average_loss = 0.

        curr_accuracy = test_model(model, tokenizer, run_on **kwargs)
        accuracies.append(curr_accuracy)

    return model, accuracies