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
        'close': data['close'][:limit].reshape(-1, factor)[:, -1]
    }

# --- 2. 指標計算ロジック ---

def calc_sma(data, window):
    ret = np.full_like(data, np.nan)
    if len(data) < window: return ret
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ret[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
    return ret

def calc_rsi(prices, window=14):
    deltas = np.insert(np.diff(prices), 0, 0)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    avg_gain = calc_sma(gains, window)
    avg_loss = calc_sma(losses, window)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    return 100 - (100 / (1 + rs))

def get_indicators(data):
    close = data['close']
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    return {
        'sma_7': calc_sma(close, 7), 'sma_30': calc_sma(close, 30),
        'rsi': calc_rsi(close, 14), 'returns': returns, 'lag1': np.roll(returns, 1)
    }

def create_dataset(data, target_shift=1):
    inds = get_indicators(data)
    close = data['close']
    X, y = [], []
    for i in range(30, len(close) - target_shift):
        feats = [inds[k][i] for k in ['sma_7', 'sma_30', 'rsi', 'returns', 'lag1']]
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
    
    for key, cfg in periods.items():
        resampled = resample_ohlc(all_5m_data, cfg['factor'])
        if resampled is None: continue
        X, y = create_dataset(resampled, target_shift=cfg['shift'])
        if len(X) < 50: 
            print(f"  [!] {symbol} {cfg['name']}: データ不足のためスキップ")
            continue
            
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # モデル保存名をシンボル込みにする
        model_filename = f'model/model_{symbol}_{key}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] {cfg['name']}モデル保存: {model_filename}")

def main():
    history_dir = 'history'
    if not os.path.exists(history_dir):
        print(f"エラー: '{history_dir}' ディレクトリが見つかりません。")
        return

    # history内のファイルをシンボルごとに分類
    symbol_files = {}
    for filename in os.listdir(history_dir):
        if filename.endswith('.csv'):
            # ファイル名が 'SOL_USDT-Min5...' なら 'SOL_USDT' を抽出
            symbol = filename.split('-')[0]
            if symbol not in symbol_files: symbol_files[symbol] = []
            symbol_files[symbol].append(os.path.join(history_dir, filename))

    if not symbol_files:
        print("CSVファイルが見つかりませんでした。")
        return

    for symbol, files in symbol_files.items():
        print(f"\n>>> シンボル: {symbol} の学習を開始します ({len(files)} ファイル)")
        
        combined_data = {'open': [], 'high': [], 'low': [], 'close': [], 'open_time': []}
        for f_path in files:
            with open(f_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        for k in combined_data.keys(): combined_data[k].append(float(row[k]))
                    except: continue
        
        # 時間順にソートしてnumpy配列化
        indices = np.argsort(combined_data['open_time'])
        data_dict = {k: np.array(combined_data[k])[indices] for k in ['open', 'high', 'low', 'close']}
        
        train_for_symbol(symbol, data_dict)

if __name__ == "__main__":
    main()
