import torch.nn as nn
import torchvision.models as models
import torch
# Model Definition
class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=1, latent_dim=2048, lstm_layers=2, hidden_dim=1024, bidirectional=True, dropout_prob=0.3):
        super(DeepFakeDetector, self).__init__()
        
        resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-2])
        # self.feature_extractor = models.resnext50_32x4d(pretrained=True)
        # Freezing all layers initially
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freezing layers initially
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.lstm = nn.LSTM(
            latent_dim, hidden_dim, lstm_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout_prob
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        x = self.avgpool(x).view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return self.fc(x)
