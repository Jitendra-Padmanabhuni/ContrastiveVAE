import torch
import torch.nn.functional as F

def reparameterize(mu, sigma):
    """Samples from a Gaussian distribution using the reparameterization trick."""
    eps = torch.randn_like(sigma)
    return mu + eps * sigma

def kl_divergence(mu_post, sigma_post, mu_prior, sigma_prior):
    var_post = (sigma_post ** 2) + 1e-8
    var_prior = (sigma_prior ** 2) + 1e-8
    kl = 0.5 * (torch.log(var_prior / var_post) + (var_post + (mu_post - mu_prior)**2) / var_prior - 1)
    return kl.sum(dim=-1).mean()

def negative_log_likelihood(y_true, mu_y, sigma_y):
    var_y = (sigma_y ** 2) + 1e-8
    nll = 0.5 * torch.log(2 * torch.pi * var_y) + ((y_true - mu_y)**2) / (2 * var_y)
    return nll.mean()

def symmetric_masked_info_nce_loss(mu1, mu2, market_returns, temperature=0.1, return_threshold=0.005):
    """
    Symmetric Contrastive Loss applied to deterministic prior means.
    mu1: (Batch_Days, K) - View 1 (Clean Subsamples)
    mu2: (Batch_Days, K) - View 2 (Masked Subsamples)
    """
    batch_size = mu1.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=mu1.device, requires_grad=True)
        
    mu1_norm = F.normalize(mu1, dim=1)
    mu2_norm = F.normalize(mu2, dim=1)
    
    # Calculate the regime mask based on the median returns
    returns_matrix = market_returns.unsqueeze(1) - market_returns.unsqueeze(0)
    similar_regime_mask = (torch.abs(returns_matrix) < return_threshold).float()
    
    # Keep the diagonal unmasked (these are our same-day positive pairs)
    eye = torch.eye(batch_size, device=mu1.device)
    similar_regime_mask = similar_regime_mask * (1.0 - eye)
    
    # Penalty to remove false negatives
    penalty = similar_regime_mask * 1e9
    
    # View 1 -> View 2 Logits
    logits_12 = torch.matmul(mu1_norm, mu2_norm.T) / temperature
    logits_12 = logits_12 - penalty
    
    # View 2 -> View 1 Logits
    logits_21 = torch.matmul(mu2_norm, mu1_norm.T) / temperature
    logits_21 = logits_21 - penalty
    
    labels = torch.arange(batch_size, device=mu1.device)
    
    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)
    
    return (loss_12 + loss_21) / 2.0