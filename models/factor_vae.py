import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .factor_encoder import FactorEncoder
from .factor_predictor import FactorPredictor
from .factor_decoder import FactorDecoder
from utils import reparameterize

class FactorVAE(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_portfolios=20, num_factors=8):
        super(FactorVAE, self).__init__()
        self.feature_extractor = FeatureExtractor(num_features, hidden_size)
        self.factor_encoder = FactorEncoder(hidden_size, num_portfolios, num_factors)
        self.factor_predictor = FactorPredictor(hidden_size, num_factors)
        self.factor_decoder = FactorDecoder(hidden_size, num_factors)

    def get_contrastive_representation(self, x):
        """
        NEW METHOD: Extracts the deterministic mean for contrastive learning, 
        avoiding the reparameterization noise of the VAE.
        """
        e = self.feature_extractor(x)
        mu_prior, _ = self.factor_predictor(e) # Ignore sigma
        return mu_prior

    def forward(self, x, y=None):
        # 1. Extract latent features
        e = self.feature_extractor(x) # (N, H)
        
        # 2. Predict Prior Factors (Always computed)
        mu_prior, sigma_prior = self.factor_predictor(e)
        
        if self.training and y is not None:
            # --- TRAINING PHASE ---
            # 3. Extract Optimal Posterior Factors (Oracle)
            mu_post, sigma_post = self.factor_encoder(y, e)
            
            # 4. Reconstruct returns using posterior factors
            mu_y, sigma_y = self.factor_decoder(e, mu_post, sigma_post)
            
            return mu_y, sigma_y, mu_post, sigma_post, mu_prior, sigma_prior
        
        else:
            # --- PREDICTION PHASE ---
            mu_y, sigma_y = self.factor_decoder(e, mu_prior, sigma_prior)
            return mu_y, sigma_y