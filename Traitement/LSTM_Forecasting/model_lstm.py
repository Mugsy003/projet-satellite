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
        # input_dim prend désormais les features + la prédiction de la veille
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x_step, hidden, cell):
        # x_step: (batch_size, 1, input_dim)
        outputs, (hidden, cell) = self.lstm(x_step, (hidden, cell))
        prediction = self.fc(outputs)
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = Encoder(encoder_input_dim, hidden_dim, num_layers, dropout)
        # Le décodeur prend decoder_input_dim (2) + 1 (la prédiction ET précédente)
        self.decoder = Decoder(decoder_input_dim + 1, hidden_dim, output_dim, num_layers, dropout)
        
    def forward(self, source, target_features):
        """
        source: (batch_size, lookback_len, 7) -> Le passé
        target_features: (batch_size, forecast_len, 3) -> Le futur (Ta, RH, Rs)
        """
        batch_size = source.shape[0]
        forecast_len = target_features.shape[1]
        
        # --- INPUT FEATURE DROPOUT sur la Radiation ---
        # On force le modèle à s'appuyer sur Ta et RH 10% du temps (Simulation de nuages intenses)
        if self.training and torch.rand(1).item() < 0.1:
            source = source.clone()
            target_features = target_features.clone()
            # Rn est à l'index 6 dans source
            source[:, :, 6] = 0.0
            # Rs est à l'index 2 dans target_features
            target_features[:, :, 2] = 0.0
            
        # 1. Encodeur traite tout le passé
        hidden, cell = self.encoder(source)
        
        # 2. Décodeur Autorégressif
        predictions = []
        
        # Le premier point d'ancrage est le vrai ET du tout dernier jour du passé
        # Dans source, PT_SINRH_ET est à l'index 3
        prev_pred = source[:, -1:, 3:4] # shape: (batch_size, 1, 1)
        
        for t in range(forecast_len):
            # Météo prévue pour le jour t
            curr_target = target_features[:, t:t+1, :] # shape: (batch_size, 1, 2)
            
            # On colle la météo avec l'ET de la veille
            decoder_input = torch.cat([curr_target, prev_pred], dim=2) # shape: (batch_size, 1, 3)
            
            # Prédiction pour le jour t
            pred_t, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions.append(pred_t)
            
            # L'ET de la veille devient la prédiction que l'on vient de faire ! (Teacher Forcing = 0%)
            prev_pred = pred_t
            
        predictions = torch.cat(predictions, dim=1) # shape: (batch_size, forecast_len, 1)
        return predictions
