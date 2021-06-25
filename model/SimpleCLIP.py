import torch
from torch import nn
import torch.nn.functional as F

from model.utils import get_resnets
from transformers.models.distilbert import DistilBertConfig, DistilBertModel

class ImageEncoder(nn.Module):

    def __init__(self, resnet_size = 34, pretrained = False, trainable = True, **kwargs):

        super(ImageEncoder, self).__init__()

        self.model = get_resnets(resnet_size, pretrained)
        del self.model.fc
        del self.model.avgpool
        self.last_conv = nn.Conv2d(512, 128, (5,5), 3)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.last_conv(x)

        return x


class TextEncoder(nn.Module):

    def __init__(self, pretrained = False, trainable = True, **kwargs):

        if pretrained:
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            self.model = DistilBertModel(DistilBertConfig)

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.5):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, temperature=1.0, image_embedding=3072, text_embedding=768, **kwargs):
        super().__init__()
        self.image_encoder = ImageEncoder(**kwargs)
        self.text_encoder = TextEncoder(**kwargs)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.image_embedding = image_embedding

    def forward(self, images, input_ids, attention_mask):
        # Getting Image and Text Features
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features.view(-1, self.image_embedding))
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def get_text_encoding(self, input_ids, attention_mask):

        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.text_projection(text_features)

        return text_embeddings

    def get_image_encoding(self, images):

        image_features = self.image_encoder(images)
        image_embeddings = self.image_projection(image_features.view(-1, self.image_embedding))
        return image_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()