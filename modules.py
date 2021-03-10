import torch
import torch.nn as nn


class MLBExplainerModel(nn.Module):
    """Embedding model for learning feature importance and impact of MLB game features
    """
    def __init__(self, input_levels, target_sizes, feat_size=64):
        super().__init__()

        self.embedders = nn.ModuleDict({
            k: nn.Embedding(v.n_levels, v.emb_dim)
            for k, v
            in input_levels.items()
        })
        for layer in self.embedders.values():
            layer.weight.data = nn.init.normal_(layer.weight.data, mean=0, std=0.01)
        
        env_in_features = sum([
            v.emb_dim
            for k, v
            in input_levels.items()
            if k not in {'teams', 'pitchers', 'managers'}
            
        ])
        team_in_features = sum([
            v.emb_dim
            for k, v
            in input_levels.items()
            if k in {'teams', 'pitchers', 'managers'}
        ])
        
        env_hidden_features = int(env_in_features//2)
        team_hidden_features = int(team_in_features//2)
        out_features = feat_size
        self.env_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(env_in_features, env_hidden_features),
            nn.BatchNorm1d(env_hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(env_hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
        self.team_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(team_in_features, team_hidden_features),
            nn.BatchNorm1d(team_hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(team_hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
        
        in_features = out_features*3  # env output, team, opponent
        hidden_features = in_features//2
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, feat_size),
        )

        self.target_classifiers = nn.ModuleDict({
            f'{k}_{side}': nn.Linear(feat_size, n_targets)
            for k, n_targets in target_sizes.items()
            for side in ['team', 'opp']
        })
        self.target_classifiers['Win'] = nn.Linear(feat_size, 1)
        
    def forward(self, x, device=None):
        
        # Encodings
        environment = torch.cat([
            self.embedders['years'](torch.tensor(x[:, 0], dtype=torch.long, device=device)),
            self.embedders['months'](torch.tensor(x[:, 1], dtype=torch.long, device=device)),
            self.embedders['dow'](torch.tensor(x[:, 2], dtype=torch.long, device=device)),
            self.embedders['time_of_day'](torch.tensor(x[:, 3], dtype=torch.long, device=device)),
            self.embedders['parks'](torch.tensor(x[:, 4], dtype=torch.long, device=device)),
            self.embedders['leagues'](torch.tensor(x[:, 5], dtype=torch.long, device=device)),
            self.embedders['umps'](torch.tensor(x[:, 6], dtype=torch.long, device=device)),
            self.embedders['home_away'](torch.tensor(x[:, -1], dtype=torch.long, device=device)),
        ], dim=1)
        environment = self.env_encoder(environment)
        
        team = torch.cat([
            self.embedders['teams'](torch.tensor(x[:, 10], dtype=torch.long, device=device)),
            self.embedders['pitchers'](torch.tensor(x[:, 11], dtype=torch.long, device=device)),
            self.embedders['managers'](torch.tensor(x[:, 12], dtype=torch.long, device=device)),
        ], dim=1)
        team = self.team_encoder(team)

        opponent = torch.cat([
            self.embedders['teams'](torch.tensor(x[:, 7], dtype=torch.long, device=device)),
            self.embedders['pitchers'](torch.tensor(x[:, 8], dtype=torch.long, device=device)),
            self.embedders['managers'](torch.tensor(x[:, 9], dtype=torch.long, device=device)),
        ], dim=1)
        opponent = self.team_encoder(opponent)
        
        # Feed forward to feature vector
        features = torch.cat((environment, team, opponent), dim=1)
        features = self.ffn(features)
        
        # Classifiers
        yh = dict()
        for k, classifier in self.target_classifiers.items():
            if k != 'Win':
                # Use cumsum to enforce monotonic increase in prob
                yh[k] = torch.sigmoid(torch.cumsum(classifier(features), dim=-1))  
            else:
                yh[k] = torch.sigmoid(classifier(features).squeeze(-1))

        return yh