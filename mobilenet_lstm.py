import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=2):
        super(MobileNetLSTM, self).__init__()

        # Load pretrained MobileNetV2 backbone
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.feature_extractor_out = mobilenet.last_channel

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_extractor_out,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 3, 224, 224)
        batch_size, seq_len, C, H, W = x.size()
        features = []

        for t in range(seq_len):
            f = self.feature_extractor(x[:, t])  # (batch, channels, h, w)
            f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1))  # (batch, channels, 1, 1)
            f = f.view(batch_size, -1)  # (batch, channels)
            features.append(f)

        features = torch.stack(features, dim=1)  # (batch, seq_len, channels)

        lstm_out, _ = self.lstm(features)       # (batch, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]           # (batch, hidden_size)

        logits = self.fc(last_out)              # (batch, num_classes)
        return logits