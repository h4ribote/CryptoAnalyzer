import csv
import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import indicators as ind

# --- 2. 学習データ作成ロジック ---

def create_dataset(data, target_shift=1):
    inds = ind.get_indicators(data)
    close = data['close']
    X, y = [], []
    # 指標が安定するまで少し長めに飛ばす (MACD 26+9=35 なので 50あれば安全)
    feature_keys = [
        'sma_7', 'sma_30', 'rsi', 'roc', 'bb_width', 'obv_change', 'macd_hist',
        'returns', 'lag1', 'lag2', 'lag3',
        'atr', 'stoch_k', 'stoch_d', 'cci', 'adx', 'bb_pct_b', 'sma_dev_7',
        'hour_sin', 'hour_cos'
    ]

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
        resampled = ind.resample_ohlc(all_5m_data, cfg['factor'])
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

    available_symbols = sorted(list(symbol_files.keys()))
    if not available_symbols:
        print("学習可能なデータが見つかりません。")
        return

    print(f"学習可能なペア: {', '.join(available_symbols)}")

    selected_symbol = input("学習するペア名を入力してください: ").strip()

    if selected_symbol not in symbol_files:
        print(f"エラー: ペア '{selected_symbol}' は存在しません。")
        return

    files = symbol_files[selected_symbol]
    print(f"\n>>> シンボル: {selected_symbol} の学習を開始します ({len(files)} ファイル)")

    combined_data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'open_time': []}
    for f_path in files:
        with open(f_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    for k in combined_data.keys(): combined_data[k].append(float(row[k]))
                except: continue

    if not combined_data['close']:
        print("有効なデータがありません。")
        return

    indices = np.argsort(combined_data['open_time'])
    data_dict = {k: np.array(combined_data[k])[indices] for k in ['open', 'high', 'low', 'close', 'volume', 'open_time']}

    train_for_symbol(selected_symbol, data_dict)

if __name__ == "__main__":
    main()
