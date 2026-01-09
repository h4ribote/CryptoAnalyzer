import csv
import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# --- 1. データ集約ロジック ---

def resample_ohlc(data, factor):
    n = len(data['close'])
    num_bins = n // factor
    if num_bins == 0: return None
    limit = num_bins * factor
    return {
        'open': data['open'][:limit].reshape(-1, factor)[:, 0],
        'high': np.max(data['high'][:limit].reshape(-1, factor), axis=1),
        'low': np.min(data['low'][:limit].reshape(-1, factor), axis=1),
        'close': data['close'][:limit].reshape(-1, factor)[:, -1],
        'volume': np.sum(data['volume'][:limit].reshape(-1, factor), axis=1)
    }

# --- 2. 指標計算ロジック ---

def calc_sma(data, window):
    ret = np.full_like(data, np.nan)
    if len(data) < window: return ret
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ret[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
    return ret

def calc_ema(data, window):
    """NaNを考慮したEMA計算"""
    ret = np.full_like(data, np.nan)
    # 最初にNaNでないインデックスを探す
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < window: return ret
    
    start_idx = valid_indices[0]
    alpha = 2 / (window + 1)
    
    # 最初の有効なウィンドウの平均で初期化
    first_window_end = start_idx + window
    if first_window_end > len(data): return ret
    
    ret[first_window_end-1] = np.mean(data[start_idx:first_window_end])
    for i in range(first_window_end, len(data)):
        if np.isnan(data[i]):
            ret[i] = ret[i-1]
        else:
            ret[i] = (data[i] - ret[i-1]) * alpha + ret[i-1]
    return ret

def calc_rsi(prices, window=14):
    deltas = np.insert(np.diff(prices), 0, 0)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    avg_gain = calc_sma(gains, window)
    avg_loss = calc_sma(losses, window)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    return 100 - (100 / (1 + rs))

def calc_roc(data, window=12):
    ret = np.full_like(data, np.nan)
    if len(data) < window: return ret
    ret[window:] = ((data[window:] - data[:-window]) / (data[:-window] + 1e-9)) * 100
    return ret

def calc_bb_width(data, window=20):
    sma = calc_sma(data, window)
    std = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        std[i] = np.std(data[i - window + 1 : i + 1])
    return (4 * std) / (sma + 1e-9)

def calc_obv(close, volume):
    obv = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    obv_change = np.zeros_like(obv)
    obv_change[1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1e-9)
    return obv_change

def calc_macd_hist(data):
    ema12 = calc_ema(data, 12)
    ema26 = calc_ema(data, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    return macd_line - signal_line

def get_indicators(data):
    close = data['close']
    vol = data['volume']
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)
    
    return {
        'sma_7': calc_sma(close, 7),
        'sma_30': calc_sma(close, 30),
        'rsi': calc_rsi(close, 14),
        'roc': calc_roc(close, 12),
        'bb_width': calc_bb_width(close, 20),
        'obv_change': calc_obv(close, vol),
        'macd_hist': calc_macd_hist(close),
        'returns': returns,
        'lag1': np.roll(returns, 1)
    }

def create_dataset(data, target_shift=1):
    inds = get_indicators(data)
    close = data['close']
    X, y = [], []
    # 指標が安定するまで少し長めに飛ばす (MACD 26+9=35 なので 50あれば安全)
    feature_keys = ['sma_7', 'sma_30', 'rsi', 'roc', 'bb_width', 'obv_change', 'macd_hist', 'returns', 'lag1']
    
    for i in range(50, len(close) - target_shift):
        feats = [inds[k][i] for k in feature_keys]
        if np.any(np.isnan(feats)): continue
        X.append(feats)
        y.append(1 if close[i + target_shift] > close[i] else 0)
    return np.array(X), np.array(y)

# --- 3. 学習メインロジック ---

def train_for_symbol(symbol, all_5m_data):
    periods = {
        'short': {'factor': 1, 'shift': 3, 'name': '短期'},
        'mid':   {'factor': 12, 'shift': 4, 'name': '中期'},
        'long':  {'factor': 288, 'shift': 1, 'name': '長期'}
    }
    
    if not os.path.exists('model'): os.makedirs('model')
    
    for key, cfg in periods.items():
        resampled = resample_ohlc(all_5m_data, cfg['factor'])
        if resampled is None: continue
        X, y = create_dataset(resampled, target_shift=cfg['shift'])
        
        if len(X) < 30: # データの閾値を少し緩和
            print(f"  [!] {symbol} {cfg['name']}: 有効な学習データが不足しています (サンプル数: {len(X)})")
            continue
            
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        model_filename = f'model/model_{symbol}_{key}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] {cfg['name']}モデル保存: {model_filename} (サンプル数: {len(X)})")

def main():
    history_dir = 'history'
    if not os.path.exists(history_dir):
        print(f"エラー: '{history_dir}' ディレクトリが見つかりません。")
        return

    symbol_files = {}
    for filename in os.listdir(history_dir):
        if filename.endswith('.csv'):
            symbol = filename.split('-')[0]
            if symbol not in symbol_files: symbol_files[symbol] = []
            symbol_files[symbol].append(os.path.join(history_dir, filename))

    for symbol, files in symbol_files.items():
        print(f"\n>>> シンボル: {symbol} の学習を開始します ({len(files)} ファイル)")
        
        combined_data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'open_time': []}
        for f_path in files:
            with open(f_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        for k in combined_data.keys(): combined_data[k].append(float(row[k]))
                    except: continue
        
        if not combined_data['close']: continue
        indices = np.argsort(combined_data['open_time'])
        data_dict = {k: np.array(combined_data[k])[indices] for k in ['open', 'high', 'low', 'close', 'volume']}
        
        train_for_symbol(symbol, data_dict)

if __name__ == "__main__":
    main()
