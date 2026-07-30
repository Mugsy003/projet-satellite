import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        # On ne renvoie que les états cachés pour initialiser le décodeur
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden, cell):
        # x: (batch_size, seq_len, input_dim)
        # hidden, cell viennent de l'encodeur
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        # Projection de l'état caché de chaque pas de temps vers la prédiction (ET)
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = Encoder(encoder_input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(decoder_input_dim, hidden_dim, output_dim, num_layers, dropout)
        
    def forward(self, source, target_features):
        """
        source: (batch_size, lookback_len, encoder_input_dim) -> Le passé (NDVI, SAVI, ET_PT_SINRH, Ta, Rn, RH)
        target_features: (batch_size, forecast_len, decoder_input_dim) -> Le futur (Ta, Rn, RH, DOY_sin, DOY_cos)
        """
        # 1. L'encodeur traite la séquence passée
        hidden, cell = self.encoder(source)
        
        # 2. Le décodeur traite les features futures (météo prévisionnelle)
        # Il est initialisé avec l'état (hidden, cell) de l'encodeur qui agit comme vecteur de contexte (mémoire du passé)
        predictions, _, _ = self.decoder(target_features, hidden, cell)
        
        # predictions: (batch_size, forecast_len, output_dim)
        return predictions
