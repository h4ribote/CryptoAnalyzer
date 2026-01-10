import os
import csv
import json
import pickle
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. データ集約・指標計算ロジック (Train.py から複製) ---

def resample_ohlc(data, factor):
    n = len(data['close'])
    num_bins = n // factor
    if num_bins == 0: return None
    limit = num_bins * factor
    ret = {
        'open': data['open'][:limit].reshape(-1, factor)[:, 0],
        'high': np.max(data['high'][:limit].reshape(-1, factor), axis=1),
        'low': np.min(data['low'][:limit].reshape(-1, factor), axis=1),
        'close': data['close'][:limit].reshape(-1, factor)[:, -1],
        'volume': np.sum(data['volume'][:limit].reshape(-1, factor), axis=1)
    }
    if 'open_time' in data:
        ret['open_time'] = data['open_time'][:limit].reshape(-1, factor)[:, 0]
    return ret

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
    バックテスト用の価格データ(OHLC)も返すように拡張
    """
    inds = get_indicators(data)
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']

    X, y, ohlc_data = [], [], []
    feature_keys = ['sma_7', 'sma_30', 'rsi', 'roc', 'bb_width', 'obv_change', 'macd_hist', 'returns', 'lag1']

    # 予測用データは i 時点の指標を使う
    # OHLCデータは、シミュレーション用に全期間保持する必要があるため、少し扱いを変える
    # ここでは、X[k] に対応するデータポイントの「次の足」からの値動きを見るために、
    # インデックスを合わせる。

    # i は予測を行う時点。i+1 以降の価格でトレードを行う。
    start_idx = 50
    end_idx = len(close) - target_shift # ML評価用のラベル生成のため

    open_time = data.get('open_time')

    for i in range(start_idx, end_idx):
        feats = [inds[k][i] for k in feature_keys]
        if np.any(np.isnan(feats)): continue
        X.append(feats)
        # ML評価用ラベル (直近予測精度確認のため残す)
        y.append(1 if close[i + target_shift] > close[i] else 0)

        # バックテスト用: i時点の価格と、その後の価格データへの参照が必要
        # ここでは単純にインデックスiを返すか、必要なデータを返す
        # シミュレーションループで i+1 からのデータを参照する
        item = {
            'index': i,
            'open': open_[i],
            'high': high[i],
            'low': low[i],
            'close': close[i]
        }
        if open_time is not None:
            item['open_time'] = open_time[i]
        ohlc_data.append(item)

    return np.array(X), np.array(y), ohlc_data, data # data全体も返す

# --- 3. 評価・バックテストロジック ---

def load_config():
    default_config = {
        "enable_short": False,
        "threshold_long": 0.5,
        "threshold_short": 0.5,
        "take_profit": 0.02,
        "stop_loss": 0.01,
        "initial_capital": 10000.0,
        "trading_fee": 0.0,
        "max_positions": 1,
        "ml_metrics": True
    }
    config_path = 'config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                return {**default_config, **user_config}
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")
            return default_config
    else:
        print("設定ファイル(config.json)が見つかりません。デフォルト値を使用します。")
        return default_config

def evaluate_performance(symbol, all_5m_data):
    periods = {
        'short': {'factor': 1, 'shift': 3, 'name': '短期 (15分後予測)'},
        'mid':   {'factor': 12, 'shift': 4, 'name': '中期 (4時間後予測)'},
        'long':  {'factor': 288, 'shift': 1, 'name': '長期 (1日後予測)'}
    }

    config = load_config()
    print(f"現在の設定: {json.dumps(config, indent=2, ensure_ascii=False)}")

    print(f"\n{'='*60}")
    print(f" モデル性能評価レポート: {symbol}")
    print(f"{'='*60}")

    for key, cfg in periods.items():
        print(f"\n>>> {cfg['name']} モデル評価中...")

        model_path = f"model/model_{symbol}_{key}.pkl"
        if not os.path.exists(model_path):
            print(f"  [!] モデルファイルが見つかりません: {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        resampled = resample_ohlc(all_5m_data, cfg['factor'])
        if resampled is None:
            print("  [!] データ不足のためリサンプリングできませんでした。")
            continue

        X, y_true, eval_ohlc, full_data = create_eval_dataset(resampled, target_shift=cfg['shift'])
        if len(X) == 0:
            print("  [!] テスト可能なデータがありません。")
            continue

        # 予測
        y_pred = model.predict(X)
        try:
            y_pred_proba = model.predict_proba(X)
        except AttributeError:
            y_pred_proba = np.vstack([1-y_pred, y_pred]).T

        # ML指標
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)

        # --- 新バックテストロジック ---
        # パラメータ取得
        th_long = config.get('threshold_long', 0.5)
        th_short = config.get('threshold_short', 0.5)
        enable_short = config.get('enable_short', False)
        tp_pct = config.get('take_profit', 0.02)
        sl_pct = config.get('stop_loss', 0.01)
        fee = config.get('trading_fee', 0.0)
        max_pos = config.get('max_positions', 1)

        active_positions = [] # list of dict: {type, entry_price, tp_price, sl_price, entry_step}
        closed_trades = [] # list of return

        # 全データ配列へのアクセスショートカット
        high_arr = full_data['high']
        low_arr = full_data['low']
        close_arr = full_data['close']

        # eval_ohlc は X (予測時点) に対応している。
        # X[i] で予測した結果は、次の足 (index+1) のOpenでエントリーすると仮定するのが一般的だが、
        # 簡易化のため「予測時点のClose」でエントリーし、その後の価格変動を見るとする。
        # ただし、リアルタイム性を考慮すると「iのCloseで判断 -> i+1のOpenで約定」が正しいが、
        # ここでは元のロジックに合わせ「iのCloseで約定」とする。(create_eval_datasetのtrade_prices準拠)

        # しかし、保有期間無制限のため、i+1, i+2... とループを進める必要がある。
        # create_eval_datasetの戻り値は「予測が発生した時点」のリスト。
        # これは時系列順に並んでいるので、これをメインループとして回す。

        for i in range(len(X)):
            current_idx = eval_ohlc[i]['index']
            current_close = close_arr[current_idx]

            # 1. 既存ポジションの決済判定 (現在の足のHigh/Lowで判定)
            # エントリーした「次の足」からHigh/Lowチェックを行うべきだが、
            # ループは「予測地点」単位で進んでいる。予測地点が連続しているならこれで良い。
            # しかし、create_eval_datasetでスキップがある場合（指標計算不可など）、飛ぶ可能性がある。
            # 正確には full_data を1ステップずつ回すべきだが、今回は X に対応するステップのみで簡易シミュレーションする。
            # (より厳密にするなら、Xのindex間の隙間も埋めてチェックする必要があるが、一旦簡略化)

            # 注: エントリー直後の同一足での決済は考慮しない（エントリーはCloseと仮定するため、変動は次の足から）
            # よって、ポジションリストにあるのは「以前にエントリーしたもの」のみとする。

            # しかし、ループが「予測発生時」にしか回らないと、予測が発生しない期間の価格変動で決済できない。
            # そこで、シミュレーションループは `start_idx` から `end_idx` まで 1ステップずつ回す形に変更する。
            pass

        # --- 再設計: ステップ実行型シミュレーション ---

        # 範囲: 最初の予測可能地点(50) から データ末尾まで
        sim_start = 50
        sim_end = len(close_arr)

        # X, y_pred_proba は圧縮されている（NaN除去等）ため、元の時系列インデックスとのマッピングが必要
        # map_idx_to_pred: full_data_index -> (prob_up, prob_down) or None
        map_idx_to_pred = {}
        for k, item in enumerate(eval_ohlc):
            idx = item['index']
            map_idx_to_pred[idx] = y_pred_proba[k]

        active_positions = []
        closed_trades = []

        open_time_arr = full_data.get('open_time')

        for t in range(sim_start, sim_end):
            # 現在の足の価格
            h = full_data['high'][t]
            l = full_data['low'][t]
            c = full_data['close'][t]
            curr_time = open_time_arr[t] if open_time_arr is not None else None

            # 1. 既存ポジションの決済チェック (High/Low判定)
            # 優先度: SL > TP (保守的)

            remaining_positions = []
            for pos in active_positions:
                p_type = pos['type']
                entry = pos['entry_price']
                tp = pos['tp_price']
                sl = pos['sl_price']

                # エントリーした足(pos['idx'] == t)では決済しないルールにする（Closeエントリーのため）
                if pos['idx'] == t:
                    remaining_positions.append(pos)
                    continue

                executed = False
                close_rate = 0.0
                reason = ''

                if p_type == 'long':
                    # Long: Low <= SL ?
                    if l <= sl:
                        close_rate = sl
                        executed = True
                        reason = 'sl'
                    # Long: High >= TP ?
                    elif h >= tp:
                        close_rate = tp
                        executed = True
                        reason = 'tp'

                    if executed:
                        raw_ret = (close_rate - entry) / entry

                elif p_type == 'short':
                    # Short: High >= SL ?
                    if h >= sl:
                        close_rate = sl
                        executed = True
                        reason = 'sl'
                    # Short: Low <= TP ?
                    elif l <= tp:
                        close_rate = tp
                        executed = True
                        reason = 'tp'

                    if executed:
                        raw_ret = (entry - close_rate) / entry

                if executed:
                    # 手数料
                    final_ret = raw_ret - fee

                    closed_trades.append({
                        'symbol': symbol,
                        'entry_index': pos['idx'],
                        'entry_price': entry,
                        'entry_time': pos.get('entry_time'),
                        'exit_index': t,
                        'exit_price': close_rate,
                        'exit_time': curr_time,
                        'type': p_type,
                        'return': final_ret,
                        'reason': reason
                    })
                else:
                    remaining_positions.append(pos)

            active_positions = remaining_positions

            # 2. 新規エントリー判定 (Close時点)
            if len(active_positions) < max_pos:
                if t in map_idx_to_pred:
                    prob = map_idx_to_pred[t] # [prob_down, prob_up]
                    prob_up = prob[1]

                    # エントリー判定
                    entry_type = None
                    if prob_up >= th_long:
                        entry_type = 'long'
                    elif enable_short and prob_up <= th_short:
                        entry_type = 'short'

                    if entry_type:
                        entry_price = c # Closeでエントリー

                        if entry_type == 'long':
                            tp_price = entry_price * (1 + tp_pct)
                            sl_price = entry_price * (1 - sl_pct)
                        else:
                            tp_price = entry_price * (1 - tp_pct)
                            sl_price = entry_price * (1 + sl_pct)

                        active_positions.append({
                            'type': entry_type,
                            'entry_price': entry_price,
                            'entry_time': curr_time,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'idx': t
                        })

        # ループ終了

        # 集計
        total_trades = len(closed_trades)
        win_trades = [tr for tr in closed_trades if tr['return'] > 0]
        loss_trades = [tr for tr in closed_trades if tr['return'] <= 0]

        wins = len(win_trades)
        losses = len(loss_trades)
        long_count = len([tr for tr in closed_trades if tr['type'] == 'long'])
        short_count = len([tr for tr in closed_trades if tr['type'] == 'short'])

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        total_return = sum(tr['return'] for tr in closed_trades)

        # 追加指標
        gross_profit = sum(tr['return'] for tr in win_trades)
        gross_loss = sum(abs(tr['return']) for tr in loss_trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean([tr['return'] for tr in win_trades]) if wins > 0 else 0.0
        avg_loss = np.mean([tr['return'] for tr in loss_trades]) if losses > 0 else 0.0
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

        # Max Drawdown (資産曲線ベース)
        equity_curve = [config['initial_capital']]
        for tr in closed_trades:
            equity_curve.append(equity_curve[-1] * (1 + tr['return']))
        equity_curve = np.array(equity_curve)

        peak = np.maximum.accumulate(equity_curve)
        if len(peak) > 0:
            drawdowns = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdowns)
        else:
            max_drawdown = 0.0

        # 結果出力
        if config.get('ml_metrics', True):
            print(f"  [ML Metrics]")
            print(f"    - Accuracy  (正解率): {acc:.2%}")
            print(f"    - Precision (適合率): {prec:.2%}")
            print(f"    - Recall    (再現率): {rec:.2%}")
            print(f"    - F1 Score  (F値)   : {f1:.2f}")
            print(f"    - Confusion Matrix:")
            print(f"      TN {cm[0][0]}  FP {cm[0][1]}  FN {cm[1][0]}  TP {cm[1][1]}")

        print(f"  [Backtest Simulation]")
        print(f"    - Total Closed Trades : {total_trades} (L: {long_count}, S: {short_count})")
        print(f"    - Win Rate            : {win_rate:.2f}%")
        print(f"    - Total Return        : {total_return:.2%}")
        print(f"    - Profit Factor       : {profit_factor:.2f}")
        print(f"    - Max Drawdown        : {max_drawdown:.2%}")
        print(f"    - Pending Positions   : {len(active_positions)} (Excluded)")
        print(f"    - Parameters          : TP={tp_pct}, SL={sl_pct}, MaxPos={max_pos}")

        # JSON出力
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        results = {
            'config': config,
            'period': key,
            'metrics': {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_return': total_return,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'payoff_ratio': payoff_ratio
            },
            'ml_metrics': {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            },
            'trades': closed_trades
        }

        output_dir = 'backtest_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ts = int(time.time())
        filename = f"{output_dir}/backtest_{symbol}_{key}_{ts}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [Output] 結果を保存しました: {filename}")
        except Exception as e:
            print(f"  [!] 結果の保存に失敗しました: {e}")

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
    data_dict = {k: np.array(combined_data[k])[indices] for k in ['open', 'high', 'low', 'close', 'volume', 'open_time']}

    # 評価実行
    evaluate_performance(selected_symbol, data_dict)

if __name__ == "__main__":
    main()
