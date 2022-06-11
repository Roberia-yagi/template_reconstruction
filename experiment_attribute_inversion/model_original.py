from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, feature_size: int):
        super(Autoencoder, self).__init__()
        self.dense_enc1 = nn.Linear(feature_size, feature_size / 2)
        self.bn1 = nn.BatchNorm1d(feature_size / 2)
        self.dense_enc2 = nn.Linear(feature_size / 2, feature_size /4)
        self.bn2 = nn.BatchNorm1d(feature_size /4)
        self.dense_enc3 = nn.Linear(feature_size /4, feature_size /8)
    
        self.dense_dec1 = nn.Linear(feature_size /8, feature_size /4)
        self.bn4 = nn.BatchNorm1d(feature_size / 4)
        self.dense_dec2 = nn.Linear(feature_size / 4, feature_size / 2)
        self.bn5 = nn.BatchNorm1d(feature_size / 2)
        self.drop1 = nn.Dropout(p=0.2)
        self.dense_dec3 = nn.Linear(feature_size / 2, feature_size)

    def encoder(self, x):
        x = nn.ReLU(self.dense_enc1(x))
        x = self.bn1(x)
        x = nn.ReLU(self.dense_enc2(x))
        x = self.bn2(x)
        x = self.dense_enc3(x)
        return x

    def decoder(self, x):
        x = nn.ReLU(self.dense_dec1(x))
        x = self.bn4(x)
        x = nn.ReLU(self.dense_dec2(x))
        x = self.bn5(x)
        x = self.drop1(x)
        x = self.dense_dec3(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z