from tqdm import tqdm
from transformers import CLIPModel, CLIPConfig, CLIPProcessor
from torch.optim import SGD, Adam, AdamW
from utils import get_loader, test_model

def train_model(images_path, annotation_path, batch_size, epoch_nums, run_on, lr, momentum, **kwargs):

    loader = get_loader(images_path, annotation_path, batch_size, **kwargs)

    model = CLIPModel(CLIPConfig())
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = model.to(run_on)
    optimizer = SGD(model.parameters(), lr = lr, momentum=momentum)
    accuracies = list()

    for i in range(epoch_nums):
        average_loss = 0.
        for batchnum, (image, text) in tqdm(enumerate(loader)):

            optimizer.zero_grad()
            image = image.to(run_on)
            inputs = processor.tokenizer(text, return_tensors='pt', padding=True)
            inputs['input_ids'], inputs['attention_mask'] = inputs['input_ids'].cuda(), inputs['attention_mask'].cuda()
            inputs['pixel_values'] = image
            outputs = model(**inputs, return_loss = True)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            if batchnum % 20 == 19:
                print(f'Batch {batchnum} average loss {average_loss / 20.}')
                average_loss = 0.

        curr_accuracy = test_model(model, processor, run_on, **kwargs)
        accuracies.append(curr_accuracy)

    return model, accuracies