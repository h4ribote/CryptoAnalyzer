import json
import urllib.request
import urllib.error
import pickle
import numpy as np
import os

# --- 指標計算ロジック (Train.pyと同一) ---

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

def get_latest_feature(data):
    close = data['close']
    inds = {
        'sma_7': calc_sma(close, 7), 'sma_30': calc_sma(close, 30),
        'rsi': calc_rsi(close, 14)
    }
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    
    feats = [inds['sma_7'][-1], inds['sma_30'][-1], inds['rsi'][-1], returns[-1], returns[-2]]
    return np.array([feats]), inds['rsi'][-1]

# --- API取得 ---

def fetch_mexc_klines(symbol, interval):
    # API用シンボル (例: SOLUSDT)
    mexc_symbol = symbol.replace("_", "")
    # MEXC v3では 1h ではなく 60m を使用するのが確実
    url = f"https://api.mexc.com/api/v3/klines?symbol={mexc_symbol}&interval={interval}&limit=100"
    
    try:
        with urllib.request.urlopen(url) as response:
            klines = json.loads(response.read().decode())
            return {'close': np.array([float(k[4]) for k in klines])}
    except urllib.error.HTTPError as e:
        # HTTPエラー（400など）の場合、パラメータ不正の可能性があるため表示
        print(f" [!] API HTTP Error ({interval}): {e.code} {e.reason}")
        return None
    except Exception as e:
        print(f" [!] API Error ({interval}): {e}")
        return None

# --- 予測メイン ---

def run_analysis(symbol):
    print(f"\n" + "="*50)
    print(f" 市場分析レポート: {symbol}")
    print("="*50)
    
    # 修正点: '1h' を '60m' に変更
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
        if data is None:
            print(f" [!] {label}データの取得に失敗しました。")
            continue
            
        X, rsi = get_latest_feature(data)
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

if __name__ == "__main__":
    # 分析したいシンボルを指定
    symbols_to_analyze = ["SOL_USDT",] 
    
    for s in symbols_to_analyze:
        run_analysis(s)
