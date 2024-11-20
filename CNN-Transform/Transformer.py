import torch.nn as nn
import torch.nn.functional as F


class onedCNN_Transformer(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_prob):
        super(onedCNN_Transformer, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.dropout1 = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.dropout2 = nn.Dropout(dropout_prob)
        self.bn2 = nn.BatchNorm1d(256)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.bn_transformer = nn.BatchNorm1d(hidden_size)
        self.flatten=nn.Flatten()
        self.classifier = nn.Linear(2816, output_size)

    def forward(self, x):
        x = self.cnn1(x.unsqueeze(1))  # Add channel dimension for CNN
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.cnn2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        # print(x.shape)[batch, 128, 40])
        x = x.transpose(1, 2)
        # print(x.shape)#([512, 40, 128])
        x = self.transformer_encoder(x)  # ( batch_size,sequence_length, embedding_dim)
        #print(x.shape)torch.Size([512, 40, 128])
        x=x.transpose(1,2)
        x = self.dropout3(x)
        x = self.bn_transformer(x)
        #print(x.shape)torch.Size([512, 128, 40])
        x = self.flatten(x)
        #print(x.shape)torch.Size([512, 5120])
        x = self.classifier(x)
        return x
