import os
import csv
import json
import pickle
import time
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import indicators as ind

# --- 2. 評価用データセット作成 ---

def create_eval_dataset(data, target_shift=1):
    """
    MLモデル予測用の特徴量を作成する
    """
    inds = ind.get_indicators(data)
    close = data['close']

    X, y, ohlc_data = [], [], []
    feature_keys = [
        'sma_7', 'sma_30', 'rsi', 'roc', 'bb_width', 'obv_change', 'macd_hist',
        'returns', 'lag1', 'lag2', 'lag3',
        'atr', 'stoch_k', 'stoch_d', 'cci', 'adx', 'bb_pct_b', 'sma_dev_7',
        'hour_sin', 'hour_cos'
    ]

    start_idx = 50
    end_idx = len(close) - target_shift

    open_time = data.get('open_time')

    for i in range(start_idx, end_idx):
        feats = [inds[k][i] for k in feature_keys]
        if np.any(np.isnan(feats)): continue
        X.append(feats)
        # 評価用ラベル (念のため)
        y.append(1 if close[i + target_shift] > close[i] else 0)

        item = {
            'index': i,
            'open_time': open_time[i] if open_time is not None else None
        }
        ohlc_data.append(item)

    return np.array(X), np.array(y), ohlc_data

# --- 3. シミュレーター ---

class BacktestSimulator:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.fee = config.get('trading_fee', 0.0)
        self.max_pos = config.get('max_positions', 1)
        self.time_limit = config.get('time_limit', 0)

        self.active_positions = []
        self.closed_trades = []
        # equity_curve は毎トレード終了時に更新
        self.equity = self.initial_capital

    def run(self, full_data, signals):
        """
        full_data: 5分足データの辞書 (open_time, open, high, low, close)
        signals: { timestamp_ms: prob_up }  (エントリー可能な時刻 -> 上昇確率)
        """
        # データ配列
        times = full_data['open_time']
        opens = full_data['open']
        highs = full_data['high']
        lows = full_data['low']
        # closes = full_data['close'] # Closeはエントリー価格決定には使わないが、念のため

        # パラメータ
        tp_pct = self.config.get('take_profit', 0.02)
        sl_pct = self.config.get('stop_loss', 0.01)
        th_long = self.config.get('threshold_long', 0.5)
        th_short = self.config.get('threshold_short', 0.5)
        enable_short = self.config.get('enable_short', False)

        # 全期間ループ
        for i in range(len(times)):
            t_curr = times[i]
            op = opens[i]
            hi = highs[i]
            lo = lows[i]

            # --- 1. 既存ポジションの決済判定 ---
            # 5分足の中でのHigh/Lowで判定する
            # ※ 厳密にはHighとLowどちらが先に起きたか不明だが、
            #    ここでは保守的に「SLが先にヒットする」と仮定するか、
            #    あるいは同足で両方ヒットしたらSL優先とするのが通例。

            next_positions = []
            for pos in self.active_positions:
                p_type = pos['type']
                entry_price = pos['entry_price']
                tp_price = pos['tp_price']
                sl_price = pos['sl_price']

                is_closed = False
                close_price = 0.0
                reason = ""

                # --- 0. 時間切れ判定 (優先) ---
                if self.time_limit and self.time_limit > 0:
                    # open_time is in ms, time_limit in minutes
                    elapsed_ms = t_curr - pos['entry_time']
                    if elapsed_ms >= self.time_limit * 60 * 1000:
                        is_closed = True
                        close_price = op
                        reason = 'time_limit'
                        # 時間切れの場合はOpenで決済し、リターン計算へ

                if is_closed:
                    if p_type == 'long':
                        raw_ret = (close_price - entry_price) / entry_price
                    else: # short
                        raw_ret = (entry_price - close_price) / entry_price

                else:
                    if p_type == 'long':
                        # SL判定 (LowがSL以下なら)
                        if lo <= sl_price:
                            is_closed = True
                            close_price = sl_price
                            reason = 'sl'
                        # TP判定 (HighがTP以上なら)
                        elif hi >= tp_price:
                            is_closed = True
                            close_price = tp_price
                            reason = 'tp'

                        if is_closed:
                            # 両方ヒットした場合の優先順位: SL優先 (保守的)
                            if (lo <= sl_price) and (hi >= tp_price):
                                close_price = sl_price
                                reason = 'sl'

                            # リターン計算
                            raw_ret = (close_price - entry_price) / entry_price

                    elif p_type == 'short':
                        # SL判定 (HighがSL以上なら)
                        if hi >= sl_price:
                            is_closed = True
                            close_price = sl_price
                            reason = 'sl'
                        # TP判定 (LowがTP以下なら)
                        elif lo <= tp_price:
                            is_closed = True
                            close_price = tp_price
                            reason = 'tp'

                        if is_closed:
                            if (hi >= sl_price) and (lo <= tp_price):
                                close_price = sl_price
                                reason = 'sl'

                            raw_ret = (entry_price - close_price) / entry_price

                if is_closed:
                    final_ret = raw_ret - self.fee
                    self.equity *= (1 + final_ret)

                    self.closed_trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': t_curr,
                        'type': p_type,
                        'entry_price': entry_price,
                        'exit_price': close_price,
                        'return': final_ret,
                        'reason': reason
                    })
                else:
                    next_positions.append(pos)

            self.active_positions = next_positions

            # --- 2. 新規エントリー判定 ---
            # シグナルがあるか？
            if len(self.active_positions) < self.max_pos:
                if t_curr in signals:
                    prob_up = signals[t_curr]

                    # 判定
                    entry_type = None
                    if prob_up >= th_long:
                        entry_type = 'long'
                    elif enable_short and prob_up <= th_short:
                        entry_type = 'short'

                    if entry_type:
                        # Openでエントリー
                        entry_price = op

                        if entry_type == 'long':
                            tp = entry_price * (1 + tp_pct)
                            sl = entry_price * (1 - sl_pct)
                        else:
                            tp = entry_price * (1 - tp_pct)
                            sl = entry_price * (1 + sl_pct)

                        self.active_positions.append({
                            'type': entry_type,
                            'entry_time': t_curr,
                            'entry_price': entry_price,
                            'tp_price': tp,
                            'sl_price': sl
                        })

    def get_metrics(self):
        trades = self.closed_trades
        total_trades = len(trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }

        wins = [t for t in trades if t['return'] > 0]
        losses = [t for t in trades if t['return'] <= 0]

        win_rate = len(wins) / total_trades
        total_return = self.equity / self.initial_capital - 1.0

        gross_profit = sum(t['return'] for t in wins)
        gross_loss = sum(abs(t['return']) for t in losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # DD計算用
        equity_curve = [self.initial_capital]
        curr = self.initial_capital
        max_dd = 0.0
        peak = curr

        for t in trades:
            curr *= (1 + t['return'])
            if curr > peak: peak = curr
            dd = (peak - curr) / peak
            if dd > max_dd: max_dd = dd
            equity_curve.append(curr)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd
        }

# --- 4. 最適化・実行ロジック ---

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
        "ml_metrics": True,
        "time_limit_short": 15,
        "time_limit_mid": 240,
        "time_limit_long": 1440
    }
    config_path = 'config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                return {**default_config, **user_config}
        except Exception:
            return default_config
    else:
        return default_config

def evaluate_performance(symbol, all_5m_data, optimize=False):
    config = load_config()
    periods = {
        'short': {'factor': 1, 'shift': 3, 'name': '短期 (15分後予測)'},
        'mid':   {'factor': 12, 'shift': 4, 'name': '中期 (4時間後予測)'},
        'long':  {'factor': 288, 'shift': 1, 'name': '長期 (1日後予測)'}
    }

    print(f"\n{'='*60}")
    print(f" バックテスト実行: {symbol}")
    if optimize:
        print(" ※ パラメータ最適化モード")
    else:
        print(f" ※ 現在の設定: TP={config['take_profit']}, SL={config['stop_loss']}, TH_L={config['threshold_long']}")
    print(f"{'='*60}")

    for key, cfg in periods.items():
        print(f"\n>>> {cfg['name']} モデル...")

        model_path = f"model/model_{symbol}_{key}.pkl"
        if not os.path.exists(model_path):
            print(f"  [Skip] モデルファイルなし: {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # リサンプリング & 予測
        resampled = ind.resample_ohlc(all_5m_data, cfg['factor'])
        if resampled is None: continue

        X, _, eval_ohlc = create_eval_dataset(resampled, target_shift=cfg['shift'])
        if len(X) == 0: continue

        # 予測実行
        try:
            y_pred_proba = model.predict_proba(X)
        except AttributeError:
            print("  [Error] モデルが確率を出力できません。")
            continue

        # Signal Map作成: timestamp -> prob_up
        # 予測に使った足の期間 = factor * 5分
        # open_time は足の開始時刻。予測結果が使えるのは「足の終了時刻」＝「次の足の開始時刻」
        duration_ms = cfg['factor'] * 5 * 60 * 1000

        signals = {}
        for i in range(len(X)):
            # この予測が出た足の開始時刻
            start_time = eval_ohlc[i]['open_time']
            if start_time is None: continue

            # エントリー可能時刻
            entry_time = start_time + duration_ms

            # 上昇確率 (クラス1)
            prob_up = y_pred_proba[i][1]
            signals[entry_time] = prob_up

        # --- 最適化モード ---
        if optimize:
            # グリッド定義
            tp_range = [0.01, 0.02, 0.03, 0.04, 0.05]
            sl_range = [0.005, 0.01, 0.02, 0.03]
            th_long_range = [0.5, 0.55, 0.6, 0.65, 0.7]
            # short閾値は "prob_up <= th" なので小さい方が確信度高い
            th_short_range = [0.5, 0.45, 0.4, 0.35, 0.3]
            time_limit_range = [0, 15, 30, 60, 120, 240, 480, 1440]

            best_metric = -999.0
            best_params = {}
            best_result = {}

            combinations = list(itertools.product(tp_range, sl_range, th_long_range, th_short_range, time_limit_range))
            print(f"  [Info] {len(combinations)} 通りの組み合わせを検証します...")

            for tp, sl, th_l, th_s, t_lim in combinations:
                # 設定コピー & 上書き
                test_cfg = config.copy()
                test_cfg['take_profit'] = tp
                test_cfg['stop_loss'] = sl
                test_cfg['threshold_long'] = th_l
                test_cfg['threshold_short'] = th_s
                test_cfg['time_limit'] = t_lim

                sim = BacktestSimulator(test_cfg)
                sim.run(all_5m_data, signals)
                metrics = sim.get_metrics()

                # 評価基準: ここでは Profit Factor 重視 (あるいは Total Return)
                # トレード回数が少なすぎる(例えば5回未満)場合は除外するなど
                score = metrics['total_return'] # とりあえずリターン最大化

                if metrics['total_trades'] >= 10 and score > best_metric:
                    best_metric = score
                    best_params = {'TP': tp, 'SL': sl, 'TH_L': th_l, 'TH_S': th_s, 'TimeLimit': t_lim}
                    best_result = metrics

            print(f"  [Best Result]")
            print(f"    Params : {best_params}")
            print(f"    Return : {best_result.get('total_return', 0):.2%}")
            print(f"    WinRate: {best_result.get('win_rate', 0):.2%}")
            print(f"    PF     : {best_result.get('profit_factor', 0):.2f}")
            print(f"    Trades : {best_result.get('total_trades', 0)}")
            print(f"    DD     : {best_result.get('max_drawdown', 0):.2%}")

        # --- 通常モード ---
        else:
            run_cfg = config.copy()
            run_cfg['time_limit'] = config.get(f'time_limit_{key}', 0)

            sim = BacktestSimulator(run_cfg)
            sim.run(all_5m_data, signals)
            metrics = sim.get_metrics()

            print(f"  [Result]")
            print(f"    Total Return : {metrics['total_return']:.2%}")
            print(f"    Win Rate     : {metrics['win_rate']:.2%}")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Total Trades : {metrics['total_trades']}")
            print(f"    Max Drawdown : {metrics['max_drawdown']:.2%}")

            # 結果保存
            output_dir = 'backtest_results'
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            ts = int(time.time())
            fname = f"{output_dir}/{symbol}_{key}_{ts}.json"

            # datetimeなどは含まれていないのでそのままdump可能
            save_data = {
                'config': config,
                'metrics': metrics,
                'trades': sim.closed_trades
            }
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"  [Save] {fname}")

def main():
    history_dir = 'history_for_test'
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
        print("データが見つかりません。")
        return

    print(f"テスト可能なペア: {', '.join(available_symbols)}")
    selected_symbol = input("テストするペア名を入力してください: ").strip()

    if selected_symbol not in symbol_files:
        print(f"エラー: ペア '{selected_symbol}' は存在しません。")
        return

    # モード選択
    print("\n実行モードを選択:")
    print("1. 現在の設定でバックテスト (詳細ログ出力)")
    print("2. パラメータ最適化 (グリッドサーチ)")
    mode_input = input("選択 (1/2): ").strip()
    is_optimize = (mode_input == '2')

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

    indices = np.argsort(combined_data['open_time'])
    data_dict = {k: np.array(combined_data[k])[indices] for k in combined_data.keys()}

    evaluate_performance(selected_symbol, data_dict, optimize=is_optimize)

if __name__ == "__main__":
    main()
