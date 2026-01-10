import json
import urllib.request
import urllib.error
import pickle
import numpy as np
import os
import indicators as ind

# --- 指標計算ロジック (Train.pyと共通) ---

def get_latest_feature(data):
    # 指標を一括計算
    inds = ind.get_indicators(data)

    # Train.pyのfeature_keysと同じ順序
    # ['sma_7', 'sma_30', 'rsi', 'roc', 'bb_width', 'obv_change', 'macd_hist', 'returns', 'lag1', 'lag2', 'lag3', 'atr', 'stoch_k', 'stoch_d', 'cci', 'adx', 'bb_pct_b', 'sma_dev_7', 'hour_sin', 'hour_cos']

    feats = [
        inds['sma_7'][-1],
        inds['sma_30'][-1],
        inds['rsi'][-1],
        inds['roc'][-1],
        inds['bb_width'][-1],
        inds['obv_change'][-1],
        inds['macd_hist'][-1],
        inds['returns'][-1],
        inds['lag1'][-1],
        inds['lag2'][-1],
        inds['lag3'][-1],
        inds['atr'][-1],
        inds['stoch_k'][-1],
        inds['stoch_d'][-1],
        inds['cci'][-1],
        inds['adx'][-1],
        inds['bb_pct_b'][-1],
        inds['sma_dev_7'][-1],
        inds['hour_sin'][-1],
        inds['hour_cos'][-1]
    ]
    return np.array([feats]), inds['rsi'][-1]

# --- API取得 ---

def fetch_mexc_klines(symbol, interval):
    mexc_symbol = symbol.replace("_", "")
    url = f"https://api.mexc.com/api/v3/klines?symbol={mexc_symbol}&interval={interval}&limit=100"

    try:
        with urllib.request.urlopen(url) as response:
            klines = json.loads(response.read().decode())
            # API returns: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            # Using indices: 0: Time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume
            return {
                'open_time': np.array([float(k[0]) for k in klines]),
                'open': np.array([float(k[1]) for k in klines]),
                'high': np.array([float(k[2]) for k in klines]),
                'low': np.array([float(k[3]) for k in klines]),
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
