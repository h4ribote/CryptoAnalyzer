import json
import urllib.request
import urllib.error
import pickle
import numpy as np
import os

# --- 指標計算ロジック (Train.pyと共通) ---

def calc_sma(data, window):
    ret = np.full_like(data, np.nan)
    if len(data) < window: return ret
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ret[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
    return ret

def calc_ema(data, window):
    ret = np.full_like(data, np.nan)
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < window: return ret
    start_idx = valid_indices[0]
    alpha = 2 / (window + 1)
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
        if close[i] > close[i-1]: obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]: obv[i] = obv[i-1] - volume[i]
        else: obv[i] = obv[i-1]
    obv_change = np.zeros_like(obv)
    obv_change[1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1e-9)
    return obv_change

def calc_macd_hist(data):
    ema12 = calc_ema(data, 12)
    ema26 = calc_ema(data, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    return macd_line - signal_line

def get_latest_feature(data):
    close = data['close']
    vol = data['volume']
    
    inds = {
        'sma_7': calc_sma(close, 7),
        'sma_30': calc_sma(close, 30),
        'rsi': calc_rsi(close, 14),
        'roc': calc_roc(close, 12),
        'bb_width': calc_bb_width(close, 20),
        'obv_change': calc_obv(close, vol),
        'macd_hist': calc_macd_hist(close)
    }
    
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)
    
    # Train.pyのfeature_keysと同じ順序
    feats = [
        inds['sma_7'][-1], inds['sma_30'][-1], inds['rsi'][-1], 
        inds['roc'][-1], inds['bb_width'][-1], inds['obv_change'][-1], 
        inds['macd_hist'][-1], returns[-1], returns[-2]
    ]
    return np.array([feats]), inds['rsi'][-1]

# --- API取得 ---

def fetch_mexc_klines(symbol, interval):
    mexc_symbol = symbol.replace("_", "")
    url = f"https://api.mexc.com/api/v3/klines?symbol={mexc_symbol}&interval={interval}&limit=100"
    
    try:
        with urllib.request.urlopen(url) as response:
            klines = json.loads(response.read().decode())
            return {
                'close': np.array([float(k[4]) for k in klines]),
                'volume': np.array([float(k[5]) for k in klines])
            }
    except Exception as e:
        print(f" [!] API Error ({interval}): {e}")
        return None

# --- 予測メイン ---

def run_analysis(symbol):
    print(f"\n" + "="*50)
    print(f" 市場分析レポート: {symbol} (拡張版)")
    print("="*50)
    
    periods = [
        ('short', '5m', '短期'), 
        ('mid', '60m', '中期'), 
        ('long', '1d', '長期')
    ]
    
    results = []
    for key, interval, label in periods:
        model_file = f'model/model_{symbol}_{key}.pkl'
        if not os.path.exists(model_file):
            print(f" [!] {label}モデルが見つかりません: {model_file}")
            continue
            
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
        data = fetch_mexc_klines(symbol, interval)
        if data is None: continue
            
        X, rsi = get_latest_feature(data)
        if np.any(np.isnan(X)):
            print(f" [!] {label}計算エラー: 指標に欠損値が含まれています。")
            continue
            
        prob = model.predict_proba(X)[0][1] * 100
        results.append((label, prob, rsi))

    if not results: return

    print(f"{'期間':<10} | {'上昇確率':<10} | {'RSI':<8} | {'判定'}")
    print("-" * 50)
    for label, prob, rsi in results:
        status = "強気" if prob > 60 else "弱気" if prob < 40 else "中立"
        print(f"{label:<10} | {prob:>7.2f} % | {rsi:>6.1f} | {status}")
    
    avg_p = sum(r[1] for r in results) / len(results)
    print("-" * 50)
    print(f" 総合センチメント指数: {avg_p:.2f} %")

def main():
    if not os.path.exists('model'):
        print("モデルディレクトリが見つかりません。")
        return

    available_symbols = set()
    for f in os.listdir('model'):
        if f.startswith('model_') and f.endswith('.pkl'):
            name_body = f[6:-4]
            for key in ['short', 'mid', 'long']:
                if name_body.endswith('_' + key):
                    symbol = name_body[:-(len(key)+1)]
                    available_symbols.add(symbol)
                    break

    sorted_symbols = sorted(list(available_symbols))
    if not sorted_symbols:
        print("有効なモデルが見つかりません。")
        return

    print(f"予測可能なペア: {', '.join(sorted_symbols)}")
    target_symbol = input("予測するペア名を入力してください: ").strip()

    if target_symbol not in available_symbols:
        print(f"エラー: ペア '{target_symbol}' は存在しません。")
        return

    run_analysis(target_symbol)

if __name__ == "__main__":
    main()
