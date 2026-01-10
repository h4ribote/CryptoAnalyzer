import os
import csv
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. データ集約・指標計算ロジック (Train.py から複製) ---

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

# --- 2. 評価用データセット作成 ---

def create_eval_dataset(data, target_shift=1):
    """
    Train.pyのcreate_datasetをベースに、
    バックテスト用の価格データ(entry, exit)も返すように拡張
    """
    inds = get_indicators(data)
    close = data['close']
    X, y, trade_prices = [], [], []
    feature_keys = ['sma_7', 'sma_30', 'rsi', 'roc', 'bb_width', 'obv_change', 'macd_hist', 'returns', 'lag1']

    for i in range(50, len(close) - target_shift):
        feats = [inds[k][i] for k in feature_keys]
        if np.any(np.isnan(feats)): continue
        X.append(feats)
        # ラベル: target_shift 後に価格が上がっていれば 1、そうでなければ 0
        y.append(1 if close[i + target_shift] > close[i] else 0)
        # バックテスト用: (エントリー価格, エグジット価格)
        trade_prices.append((close[i], close[i + target_shift]))

    return np.array(X), np.array(y), trade_prices

# --- 3. 評価・バックテストロジック ---

def evaluate_performance(symbol, all_5m_data):
    periods = {
        'short': {'factor': 1, 'shift': 3, 'name': '短期 (15分後予測)'},
        'mid':   {'factor': 12, 'shift': 4, 'name': '中期 (4時間後予測)'},
        'long':  {'factor': 288, 'shift': 1, 'name': '長期 (1日後予測)'}
    }

    print(f"\n{'='*60}")
    print(f" モデル性能評価レポート: {symbol}")
    print(f"{'='*60}")

    for key, cfg in periods.items():
        print(f"\n>>> {cfg['name']} モデル評価中...")

        # 1. モデル読み込み
        model_path = f"model/model_{symbol}_{key}.pkl"
        if not os.path.exists(model_path):
            print(f"  [!] モデルファイルが見つかりません: {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 2. データ準備
        resampled = resample_ohlc(all_5m_data, cfg['factor'])
        if resampled is None:
            print("  [!] データ不足のためリサンプリングできませんでした。")
            continue

        X, y_true, trade_prices = create_eval_dataset(resampled, target_shift=cfg['shift'])
        if len(X) == 0:
            print("  [!] テスト可能なデータがありません。")
            continue

        # 3. 予測
        y_pred = model.predict(X)

        # 4. 機械学習指標の計算
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # 5. バックテスト (簡易シミュレーション)
        # ルール: 予測が1なら買い、target_shift後に売り。予測が0なら何もしない。
        # トランザクションコストは考慮しない。
        total_return = 0.0
        wins = 0
        losses = 0
        trade_count = 0

        for i, prediction in enumerate(y_pred):
            if prediction == 1:
                entry_price, exit_price = trade_prices[i]
                trade_ret = (exit_price - entry_price) / entry_price
                total_return += trade_ret
                trade_count += 1
                if trade_ret > 0:
                    wins += 1
                else:
                    losses += 1

        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0.0

        # 6. 結果出力
        print(f"  [ML Metrics]")
        print(f"    - Accuracy  (正解率): {acc:.2%}")
        print(f"    - Precision (適合率): {prec:.2%}")
        print(f"    - Recall    (再現率): {rec:.2%}")
        print(f"    - F1 Score  (F値)   : {f1:.2f}")
        print(f"    - Confusion Matrix:\n{cm}")

        print(f"  [Backtest Simulation]")
        print(f"    - Total Trades (取引回数): {trade_count}")
        print(f"    - Win Rate     (勝率)    : {win_rate:.2f}%")
        print(f"    - Total Return (累積損益): {total_return:.2%}")

# --- 4. メイン処理 ---

def main():
    history_dir = 'history_for_test'
    if not os.path.exists(history_dir):
        print(f"エラー: '{history_dir}' ディレクトリが見つかりません。")
        return

    # 利用可能なシンボルのリストアップ
    symbol_files = {}
    for filename in os.listdir(history_dir):
        if filename.endswith('.csv'):
            symbol = filename.split('-')[0]
            if symbol not in symbol_files: symbol_files[symbol] = []
            symbol_files[symbol].append(os.path.join(history_dir, filename))

    available_symbols = sorted(list(symbol_files.keys()))
    if not available_symbols:
        print("データが見つかりません。")
        return

    print(f"テスト可能なペア: {', '.join(available_symbols)}")
    selected_symbol = input("テストするペア名を入力してください: ").strip()

    if selected_symbol not in symbol_files:
        print(f"エラー: ペア '{selected_symbol}' は存在しません。")
        return

    # データ読み込み・結合
    files = symbol_files[selected_symbol]
    print(f"\n>>> データを読み込んでいます... ({len(files)} ファイル)")

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

    # 時系列ソート
    indices = np.argsort(combined_data['open_time'])
    data_dict = {k: np.array(combined_data[k])[indices] for k in ['open', 'high', 'low', 'close', 'volume']}

    # 評価実行
    evaluate_performance(selected_symbol, data_dict)

if __name__ == "__main__":
    main()
