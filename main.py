import os
import argparse
import torch
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from models.factor_vae import FactorVAE
from data.qlib_dataset import initialize_qlib, get_qlib_dataloader
# Add to your imports
from utils import kl_divergence, negative_log_likelihood, info_nce_loss

# --- Hyperparameters ---
SEQ_LEN = 20
NUM_FEATURES = 158  
HIDDEN_SIZE = 64
NUM_PORTFOLIOS = 20
NUM_FACTORS = 8
GAMMA = 1.0       
LR = 1e-4
EPOCHS = 1
RISK_AVERSION = 0.5
LAMBDA_CON = 0.5       # Weight of the Contrastive Loss
CONTRASTIVE_DAYS = 16  # Accumulate 16 days before stepping optimizer
SUBSAMPLE_RATIO = 0.8  # Keep 80% of stocks for each contrastive view 
MODEL_PATH = "factor_vae_weights.pth" 

# --- Portfolio Backtest Hyperparameters (from Paper) ---
TOP_K = 50  # Number of stocks to hold in the portfolio
TOP_N = 5   # Maximum number of dropped/swapped stocks per day (Turnover constraint)


def calculate_rank_ic(y_true, y_pred):
    """Calculates Rank IC using Spearman correlation."""
    if len(y_true) < 2: return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return ic


def calculate_portfolio_metrics(daily_returns):
    """Computes Annualized Return, Sharpe Ratio, and Max Drawdown."""
    returns = np.array(daily_returns)
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
        
    # Annualized Return (assuming 252 trading days)
    ar = np.mean(returns) * 252
    
    # Sharpe Ratio
    std = np.std(returns)
    sr = (np.mean(returns) / std * np.sqrt(252)) if std > 1e-8 else 0.0
    
    # Maximum Drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / np.maximum(running_max, 1e-8)
    mdd = np.abs(np.min(drawdowns))
    
    return ar, sr, mdd


def get_topk_drop_portfolio(scores, prev_portfolio, k=50, n=5):
    """
    Implements the TopK-Drop strategy. 
    Selects k stocks, ensuring intersection with previous portfolio >= k - n.
    """
    num_stocks = len(scores)
    
    # If not enough stocks available for the day, just buy everything we can
    if num_stocks <= k:
        return set(range(num_stocks))

    # Sort stock indices by risk-adjusted score (highest to lowest)
    sorted_indices = np.argsort(scores)[::-1]
    
    if not prev_portfolio:
        return set(sorted_indices[:k])
        
    top_k = set(sorted_indices[:k])
    intersection = top_k.intersection(prev_portfolio)
    
    # Check if turnover constraint is met natively
    if len(intersection) >= (k - n):
        return top_k
        
    # If constraint failed, we must force-keep highest scoring stocks from the old portfolio
    shortfall = (k - n) - len(intersection)
    
    # Old stocks not currently in the top K, sorted by best scores
    prev_not_in_top = [idx for idx in sorted_indices if idx in prev_portfolio and idx not in top_k]
    
    # New top K stocks not in old portfolio, sorted by worst scores (to drop them)
    top_not_in_prev = [idx for idx in sorted_indices[::-1] if idx in top_k and idx not in prev_portfolio]
    
    # Swap out the worst new stocks for the best old stocks to meet constraint
    for i in range(shortfall):
        if i < len(prev_not_in_top) and i < len(top_not_in_prev):
            top_k.remove(top_not_in_prev[i])
            top_k.add(prev_not_in_top[i])
            
    return top_k


def get_cached_dataloader(start_date, end_date, cache_filename):
    """Loads dataset from disk if it exists, otherwise fetches from Qlib and saves it."""
    if os.path.exists(cache_filename):
        print(f"Loading cached dataset from {cache_filename}...")
        return torch.load(cache_filename)
    
    print(f"Generating dataset from {start_date} to {end_date} (This will take time)...")
    loader = get_qlib_dataloader(start_date, end_date, seq_len=SEQ_LEN)
    
    data_list = []
    for date, x_batch, y_batch in loader:
        data_list.append((date, x_batch, y_batch))
        
    torch.save(data_list, cache_filename)
    print(f"Dataset successfully cached to {cache_filename}!")
    return data_list


def evaluate(model, device, val_loader):
    """Runs evaluation and backtests the TDrisk portfolio strategy."""
    model.eval()
    ic_list = []
    daily_portfolio_returns = []
    prev_portfolio = set()
    
    with torch.no_grad():
        for date, x_batch, y_batch in val_loader:
            # Must have at least K stocks to construct the portfolio
            if x_batch.shape[0] < TOP_K: 
                continue

            x_batch = x_batch.to(device)
            
            # Inference Mode
            mu_y, sigma_y = model(x_batch)
            
            # Extract flattened arrays for indexing and sorting
            mu_np = mu_y.cpu().numpy().flatten()
            sigma_np = sigma_y.cpu().numpy().flatten()
            y_true = y_batch.numpy().flatten()
            
            # 1. Standard Rank IC (based on Expected Return)
            ic = calculate_rank_ic(y_true, mu_np)
            ic_list.append(ic)
            
            # 2. TDrisk Score Calculation (Expected Return - Risk Penalty)
            tdrisk_scores = mu_np - 0.5*(RISK_AVERSION * sigma_np) #0.5 risk penalty
            
            # 3. Apply TopK-Drop Portfolio Selection
            current_portfolio = get_topk_drop_portfolio(tdrisk_scores, prev_portfolio, k=TOP_K, n=TOP_N)
            
            # 4. Calculate Portfolio Daily Return (Equal Weight)
            if current_portfolio:
                port_return = np.mean(y_true[list(current_portfolio)])
                daily_portfolio_returns.append(port_return)
                
            prev_portfolio = current_portfolio
            
    # Calculate Averages
    avg_ic = sum(ic_list) / len(ic_list) if ic_list else 0.0
    ar, sr, mdd = calculate_portfolio_metrics(daily_portfolio_returns)
    
    print(f"Validation Rank IC: {avg_ic:.4f} | AR: {ar*100:.2f}% | SR: {sr:.4f} | MDD: {mdd*100:.2f}%")
    return avg_ic


def train():
    initialize_qlib()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device: {device} ---")
    
    model = FactorVAE(NUM_FEATURES, HIDDEN_SIZE, NUM_PORTFOLIOS, NUM_FACTORS)
    model.to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Contrastive Accumulators
        optimizer.zero_grad()
        z1_batch, z2_batch = [], []
        vae_loss_accum = 0.0
        valid_days = 0

        train_loader = get_qlib_dataloader("2010-01-01", "2017-12-31", seq_len=SEQ_LEN)
        
        for date, x_batch, y_batch in train_loader:
            if x_batch.shape[0] < NUM_PORTFOLIOS:
                continue 
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # --- 1. Standard VAE Pass (using ALL stocks) ---
            mu_y, sigma_y, mu_post, sigma_post, mu_prior, sigma_prior = model(x_batch, y_batch)
            
            nll = negative_log_likelihood(y_batch, mu_y, sigma_y)
            kl = kl_divergence(mu_post, sigma_post, mu_prior, sigma_prior)
            vae_loss_accum += (nll + GAMMA * kl)
            
            # --- 2. Contrastive Pass (using Subsampled Views) ---
            N_total = x_batch.shape[0]
            subset_size = int(N_total * SUBSAMPLE_RATIO)
            
            # Generate two random permutations of stock indices
            idx1 = torch.randperm(N_total, device=device)[:subset_size]
            idx2 = torch.randperm(N_total, device=device)[:subset_size]
            
            view1, view2 = x_batch[idx1], x_batch[idx2]
            
            # Extract prior factors for both views
            z1 = model.get_prior_factors(view1) # Shape: (K,)
            z2 = model.get_prior_factors(view2) # Shape: (K,)
            
            z1_batch.append(z1)
            z2_batch.append(z2)
            valid_days += 1
            
            # --- 3. Compute Multi-Task Loss and Step (Every 16 Days) ---
            if valid_days == CONTRASTIVE_DAYS:
                Z1 = torch.stack(z1_batch) # Shape: (16, K)
                Z2 = torch.stack(z2_batch) # Shape: (16, K)
                
                # Calculate InfoNCE across the 16 days
                con_loss = info_nce_loss(Z1, Z2)
                
                # Average VAE loss over the 16 days
                avg_vae_loss = vae_loss_accum / CONTRASTIVE_DAYS
                
                # Joint Objective
                total_loss = avg_vae_loss + (LAMBDA_CON * con_loss)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                # Reset accumulators for next batch of days
                optimizer.zero_grad()
                z1_batch, z2_batch = [], []
                vae_loss_accum = 0.0
                valid_days = 0

        # Catch any remaining days at the end of the epoch
        if valid_days > 1:
            Z1 = torch.stack(z1_batch)
            Z2 = torch.stack(z2_batch)
            con_loss = info_nce_loss(Z1, Z2)
            avg_vae_loss = vae_loss_accum / valid_days
            total_loss = avg_vae_loss + (LAMBDA_CON * con_loss)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
        
        # Evaluate out-of-sample
        val_loader = get_qlib_dataloader("2018-01-01", "2019-12-31", seq_len=SEQ_LEN)
        evaluate(model, device, val_loader)

    print("Training Complete. Saving model weights...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Weights successfully saved to {MODEL_PATH}")


def eval_only():
    initialize_qlib()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating on device: {device} ---")
    
    model = FactorVAE(NUM_FEATURES, HIDDEN_SIZE, NUM_PORTFOLIOS, NUM_FACTORS)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded existing weights from {MODEL_PATH}")
    else:
        print(f"Error: {MODEL_PATH} not found. You must train the model first!")
        return
        
    model.to(device)
    
    val_loader = get_qlib_dataloader("2018-01-01", "2019-12-31", seq_len=SEQ_LEN)
    evaluate(model, device, val_loader)
    evaluate(model, device, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FactorVAE Runner")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], 
                        help="Choose whether to train the model or evaluate existing weights.")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        eval_only()