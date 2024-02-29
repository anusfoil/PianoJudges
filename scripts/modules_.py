import numpy as np
import torch
from torch import nn
import torchmetrics
import torch.distributed as dist
from sklearn.metrics import classification_report, confusion_matrix
from einops import reduce, rearrange, repeat
import matplotlib.pyplot as plt
import itertools
import hook

import pytorch_lightning as pl
from .utils import init_encoder, load_embedding_from_hdf5, encoding_shape


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, kernel_size, dropout, padding, stride,
                 skip=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding, 
                              stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.skip = skip

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.skip:
            # Skip Connection - Ensuring the channel dimensions match
            if identity.shape != out.shape:
                if identity.shape[1] < out.shape[1]:
                    identity = repeat(identity, f"b c w h -> b (c repeat) w h", repeat=int(out.shape[1] / identity.shape[1]))
                else:
                    identity = identity[:, :out.shape[1], :, :]

            out += identity
        return out


class AudioCNNTransformer(nn.Module):
    def __init__(self, h, w, n_segs,
                 nhead, num_encoder_layers, dim_feedforward, dim_transformer, 
                 cnn_channels, kernel_size, dropout, padding, stride, out_classes=1):
        super(AudioCNNTransformer, self).__init__()
        # CNN blocks with skip connections
        self.conv_block1 = ConvBlock(1, cnn_channels, kernel_size, dropout, padding, stride)
        self.conv_block2 = ConvBlock(cnn_channels, 1, kernel_size, dropout, padding, stride)

        conv_out_h = self.conv_output_shape(h, kernel_size, stride, padding, 2)
        conv_out_w = self.conv_output_shape(w, kernel_size, stride, padding, 2)

        # fc layer that projects the dimension to transformer dimension
        self.fc = nn.Linear(conv_out_h * conv_out_w, dim_transformer)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_transformer,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Regression Layer
        self.regression = nn.Linear(dim_transformer * n_segs , out_classes)

    def forward(self, x):
        # x shape: (batch, n_segs, seg_timesteps * 2, embedding dimension)
        batch_size, n_segs, seq_len, emb_dim = x.shape

        # Apply CNN blocks
        x = x.view(batch_size * n_segs, 1, seq_len, emb_dim)  # Reshape for CNN
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Flatten and transpose for Transformer
        x = x.view(batch_size, n_segs, -1).transpose(0, 1)  # (n_segs, batch, flattened emb_dim)
        x = self.fc(x)

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(x)

        # Aggregate the output and reshape
        agg_output = transformer_output.transpose(0, 1).reshape(batch_size, -1)

        # Regression to get final output
        output = self.regression(agg_output).squeeze(-1)

        return output

    def conv_output_shape(self, input_dim, kernel_size, stride, padding, n=1):

        for _ in range(n):
            input_dim = ((input_dim - kernel_size + 2 * padding) / stride) + 1
        return int(input_dim)


class PredictionHead(pl.LightningModule):
    def __init__(self, cfg, embedding_dim, embedding_len):
        super(PredictionHead, self).__init__()
        self.save_hyperparameters()

        if cfg.task == 'rank': # ranking concatenate two embeddings
            embedding_len *= 2

        # Define your model architecture
        self.model = AudioCNNTransformer(
            h=embedding_dim, 
            w=embedding_len,
            n_segs=cfg.dataset.n_segs,
            out_classes=cfg.dataset.num_classes if cfg.objective == "classification" else 1,
            **cfg.model.args
        )
        self.learning_rate = cfg.learning_rate
        self.cfg = cfg
        self.encoder = cfg.encoder
        self.audio_inits = init_encoder(encoder=cfg.encoder, device=torch.device(f'cuda:{cfg.gpu[0]}'))   

        # Set criterion based on the task
        if cfg.objective == "classification":
            self.projection = nn.Linear(embedding_dim, cfg.dataset.num_classes)
            self.criterion = nn.CrossEntropyLoss()
            self.label_dtype = torch.long
        else:  # default to regression if not classification
            self.criterion = nn.MSELoss()
            self.label_dtype = torch.float32

        self.precision_metric = torchmetrics.Precision(num_classes=cfg.dataset.num_classes, average='macro', task='multiclass')
        self.recall = torchmetrics.Recall(num_classes=cfg.dataset.num_classes, average='macro', task='multiclass')
        self.f1 = torchmetrics.F1Score(num_classes=cfg.dataset.num_classes, average='macro', task='multiclass')
        self.accuracy = torchmetrics.Accuracy(num_classes=cfg.dataset.num_classes, task='multiclass')


    def forward(self, x):
        x = self.model(x)
        # If classification, add a softmax layer to project the logits to probabilities
        if self.cfg.objective == "classification":
            x = torch.softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        outputs, loss = self.train_valid_pass(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss = self.train_valid_pass(batch)
        self.log("val_loss", loss)

        labels, predicted_labels = self.outputs_conversion(batch, outputs)

        # Update metrics
        precision = self.precision_metric(predicted_labels, labels)
        recall = self.recall(predicted_labels, labels)
        f1_score = self.f1(predicted_labels, labels)
        accuracy = self.accuracy(predicted_labels, labels)

        print(predicted_labels, labels)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_f1_score', f1_score, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return {'true_labels': labels, 'predicted_labels': predicted_labels, 'predicted_outputs': outputs}


    def validation_epoch_end(self, outputs):
        all_true_labels = torch.cat([x['true_labels'] for x in outputs], dim=0).cpu().numpy()
        all_predicted_labels = torch.cat([x['predicted_labels'] for x in outputs], dim=0).cpu().numpy()

        # Calculate the confusion matrix
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        class_names = [str(i) for i in range(self.cfg.dataset.num_classes)]

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names, yticklabels=class_names,
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        # Check if wandb is used and log the plot
        if self.logger and isinstance(self.logger, pl.loggers.WandbLogger):
            import wandb
            wandb.log({"confusion_matrix": wandb.Image(fig)})

        plt.close(fig)


    def train_valid_pass(self, batch):

        emb = self.batch_to_embedding(batch).to(self.device)

        # Convert labels to tensor
        labels = torch.tensor(batch["label"], dtype=self.label_dtype, device=self.device)

        # Forward pass
        outputs = self(emb)
        loss = self.criterion(outputs, labels)

        return outputs, loss


    def test_step(self, batch, batch_idx):

        combined_emb = self.batch_to_embedding(batch)
        outputs = self(combined_emb)  # Regression output

        labels, predicted_labels = self.outputs_conversion(batch, outputs)


        # Store labels and predictions for later use in test_epoch_end
        return {'true_labels': labels, 'predicted_labels': predicted_labels, 'predicted_outputs': outputs}


    def test_epoch_end(self, outputs):
        # Concatenate all labels and predictions
        all_true_labels = torch.cat([x['true_labels'] for x in outputs], dim=0).cpu().numpy()
        all_predicted_labels = torch.cat([x['predicted_labels'] for x in outputs], dim=0).cpu().numpy()
        all_predicted_outputs = torch.cat([x['predicted_outputs'] for x in outputs], dim=0).cpu().numpy()

        # Print classification summary
        print(classification_report(all_true_labels, all_predicted_labels, labels=list(range(self.cfg.dataset.num_classes))))
        loss = self.criterion(torch.tensor(all_predicted_outputs), torch.tensor(all_true_labels))


    def outputs_conversion(self, batch, outputs):
        # Convert labels to tensor
        labels = torch.tensor(batch["label"], dtype=self.label_dtype, device=self.device)

        if self.cfg.objective == "classification":
            predicted_labels = outputs.max(dim=1)[1]    
        else:
            predicted_labels = self.output_to_label(outputs)        

        if self.cfg.task == 'diff': # diff: 1 - 9 -> 0 - 8
            labels = labels - 1
            predicted_labels = predicted_labels - 1

        return labels, predicted_labels

    def batch_to_embedding(self, batch):

        if 'audio_path_1' in batch:
            emb1 = load_embedding_from_hdf5(batch["audio_path_1"], self.encoder, self.device, self.cfg.max_segs).to(self.device)
            emb2 = load_embedding_from_hdf5(batch["audio_path_2"], self.encoder, self.device, self.cfg.max_segs).to(self.device)
            emb = torch.cat((emb1, emb2), dim=-2)  # (b, t1+t2, e)
        else:
            emb = load_embedding_from_hdf5(batch["audio_path"], self.encoder, self.device).to(self.device)

        return emb


    def output_to_label(self, output, threshold=1):
        '''thrshold: '''
        # Convert regression output to classification labels based on nearest point
        # res = torch.argmin(torch.abs(torch.stack([output + threshold, output, output - threshold])), dim=0) - 1

        label_classes = self.cfg.dataset.num_classes
        # convert the output to the nearest label class integer (clip to the range of label_classes)
        res = output.int().clip(0, label_classes - 1)

        return res


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



