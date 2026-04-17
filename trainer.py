"""
Training orchestration for NTS‑NOTEARS (Global and Adaptive Window).
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from nts_notears_model import NTS_NOTEARS
from lbfgsb_scipy import LBFGSBScipy
from utils import squared_loss, reshape_for_model_forward
from change_point_detector import universe_adaptive_start_date
from selector import select_top_etf
from push_results import push_daily_result


def train_nts_notears(X_train, X_val, n_lags, lambda1, lambda2, w_threshold, max_iter, h_tol, rho_max, device):
    """Train NTS‑NOTEARS using dual ascent."""
    d = X_train.shape[1]
    dims = [d] + config.HIDDEN_DIMS + [1]
    model = NTS_NOTEARS(dims, n_lags=n_lags).to(device)

    X_train_t = reshape_for_model_forward(X_train, model, device=device)
    X_val_t   = reshape_for_model_forward(X_val, model, device=device)

    rho, alpha, h = 1.0, 0.0, np.inf
    for iteration in range(max_iter):
        optimizer = LBFGSBScipy(model.parameters())
        optimizer.assign_bounds(model)
        while rho < rho_max:
            def closure():
                optimizer.zero_grad()
                X_hat = model(X_train_t)
                loss = squared_loss(X_hat, torch.tensor(X_train[model.simultaneous_idx:], dtype=torch.float32, device=device))
                h_val = model.h_func()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * model.l2_reg()
                l1_reg = 0
                if isinstance(lambda1, list):
                    for kidx in range(model.kernel_size):
                        l1_reg += lambda1[kidx] * model.fc1_l1_reg(kidx)
                else:
                    l1_reg = lambda1 * model.fc1_l1_reg()
                primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj
            optimizer.step(closure)
            with torch.no_grad():
                h_new = model.h_func().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        if h_new <= h_tol:
            break
    return model


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}
    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    hit_rate = (ret_series > 0).mean()
    cum_return = (1 + ret_series).prod() - 1
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": max_dd, "hit_rate": hit_rate, "cum_return": cum_return,
        "n_days": len(ret_series)
    }


def train_global(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Global Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    total_days = len(returns)
    train_end = int(total_days * config.TRAIN_RATIO)
    val_end   = train_end + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end]
    val_ret   = returns.iloc[train_end:val_end]
    test_ret  = returns.iloc[val_end:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_ret.values)
    X_val   = scaler.transform(val_ret.values)
    X_test  = scaler.transform(test_ret.values)

    model = train_nts_notears(
        X_train, X_val, n_lags=config.N_LAGS,
        lambda1=config.LAMBDA1, lambda2=config.LAMBDA2,
        w_threshold=config.W_THRESHOLD, max_iter=config.MAX_ITER,
        h_tol=config.H_TOL, rho_max=config.RHO_MAX, device=config.DEVICE
    )

    # Predict next‑day returns on test set
    model.eval()
    with torch.no_grad():
        X_test_t = reshape_for_model_forward(X_test, model, device=config.DEVICE)
        pred = model(X_test_t).cpu().numpy()
    last_pred = pred[-1]  # prediction for next day
    top_idx = np.argmax(last_pred)
    top_etf = tickers[top_idx]
    pred_return = float(last_pred[top_idx])

    metrics = evaluate_etf(top_etf, test_ret)
    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_adaptive(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    if returns.empty:
        return {"ticker": None, "pred_return": None, "metrics": {}, "change_point_date": None, "lookback_days": 0}

    cp_date = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
    if end_date <= cp_date:
        end_date = returns.index[-1] - pd.Timedelta(days=10)
    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret = returns.loc[train_mask]
    test_ret  = returns.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print(f"  Insufficient training days ({len(train_ret)}). Falling back to global.")
        return train_global(universe, returns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_ret.values)
    X_test  = scaler.transform(test_ret.values) if len(test_ret) > 0 else X_train

    model = train_nts_notears(
        X_train, X_train[-len(train_ret)//5:] or X_train[-10:],
        n_lags=config.N_LAGS, lambda1=config.LAMBDA1, lambda2=config.LAMBDA2,
        w_threshold=config.W_THRESHOLD, max_iter=config.MAX_ITER,
        h_tol=config.H_TOL, rho_max=config.RHO_MAX, device=config.DEVICE
    )

    model.eval()
    with torch.no_grad():
        X_test_t = reshape_for_model_forward(X_test, model, device=config.DEVICE)
        pred = model(X_test_t).cpu().numpy()
    last_pred = pred[-1] if len(pred) > 0 else np.zeros(len(tickers))
    top_idx = np.argmax(last_pred)
    top_etf = tickers[top_idx]
    pred_return = float(last_pred[top_idx])

    metrics = evaluate_etf(top_etf, test_ret) if len(test_ret) > 0 else {}
    lookback = (returns.index[-1] - cp_date).days
    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "metrics": metrics,
        "change_point_date": cp_date.strftime("%Y-%m-%d"),
        "lookback_days": lookback,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"\n{'='*50}\nProcessing {universe.upper()}\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        global_res = train_global(universe, returns)
        adaptive_res = train_adaptive(universe, returns)
        all_results[universe] = {"global": global_res, "adaptive": adaptive_res}
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
