import os
import csv
import json
import time
import pickle
import numpy as np
import urllib.request
import urllib.error
import itertools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. Indicators Logic
# ==========================================

def resample_ohlc(data, factor):
    """
    OHLCデータをリサンプリングする。
    data: {'open': np.array, 'high': np.array, ...}
    factor: 圧縮率 (例: 1分足データを5分足にするなら5)
    """
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

    # open_timeの処理 (存在する場合のみ)
    if 'open_time' in data and data['open_time'] is not None:
        if isinstance(data['open_time'], (np.ndarray, list)) and len(data['open_time']) >= limit:
             ret['open_time'] = np.array(data['open_time'])[:limit].reshape(-1, factor)[:, 0]

    return ret

# --- Basic Indicators ---

def calc_sma(data, window):
    ret = np.full_like(data, np.nan)
    if len(data) < window: return ret
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ret[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
    return ret

def calc_ema(data, window):
    """NaNを考慮したEMA計算"""
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

def calc_rma(data, window):
    """Wilder's Smoothing (Running Moving Average)"""
    ret = np.full_like(data, np.nan)
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < window: return ret

    start_idx = valid_indices[0]
    alpha = 1 / window

    first_window_end = start_idx + window
    if first_window_end > len(data): return ret

    # Initialize with SMA
    ret[first_window_end-1] = np.mean(data[start_idx:first_window_end])
    for i in range(first_window_end, len(data)):
        if np.isnan(data[i]):
            ret[i] = ret[i-1]
        else:
            ret[i] = alpha * data[i] + (1 - alpha) * ret[i-1]
    return ret

# --- Advanced Indicators ---

def calc_rsi(prices, window=14):
    deltas = np.insert(np.diff(prices), 0, 0)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    avg_gain = calc_rma(gains, window)
    avg_loss = calc_rma(losses, window)

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Fix: if avg_loss is 0 (pure gain), RSI should be 100
    rsi[avg_loss == 0] = 100
    return rsi

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

def calc_bb_pct_b(data, window=20):
    sma = calc_sma(data, window)
    std = np.full_like(data, np.nan)
    for i in range(window - 1, len(data)):
        std[i] = np.std(data[i - window + 1 : i + 1])
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (data - lower) / (upper - lower + 1e-9)

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

def calc_atr(high, low, close, window=14):
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        h_l = high[i] - low[i]
        h_cp = abs(high[i] - close[i-1])
        l_cp = abs(low[i] - close[i-1])
        tr[i] = max(h_l, h_cp, l_cp)

    return calc_rma(tr, window)

def calc_sma_nan_safe(data, window):
    ret = np.full_like(data, np.nan)
    if len(data) < window: return ret

    kernel = np.ones(window) / window
    c = np.convolve(data, kernel, mode='valid')
    ret[window-1:] = c
    return ret

def calc_stoch(high, low, close, k_window=14, d_window=3):
    n = len(close)
    if n < k_window:
        return np.full(n, np.nan), np.full(n, np.nan)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        low_windows = sliding_window_view(low, window_shape=k_window)
        high_windows = sliding_window_view(high, window_shape=k_window)

        mins = np.min(low_windows, axis=1)
        maxs = np.max(high_windows, axis=1)

        lowest_low = np.concatenate([np.full(k_window - 1, np.nan), mins])
        highest_high = np.concatenate([np.full(k_window - 1, np.nan), maxs])

    except (ImportError, AttributeError):
        lowest_low = np.full_like(low, np.nan)
        highest_high = np.full_like(high, np.nan)
        for i in range(k_window - 1, n):
            lowest_low[i] = np.min(low[i - k_window + 1 : i + 1])
            highest_high[i] = np.max(high[i - k_window + 1 : i + 1])

    denom = highest_high - lowest_low

    fast_k = np.full_like(close, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        fast_k = 100 * (close - lowest_low) / denom

    fast_k = np.clip(fast_k, 0, 100)

    slow_k = calc_sma_nan_safe(fast_k, d_window)
    slow_d = calc_sma_nan_safe(slow_k, d_window)

    return slow_k, slow_d

def calc_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma_tp = calc_sma(tp, window)

    mean_dev = np.full_like(tp, np.nan)
    for i in range(window - 1, len(tp)):
        mean_dev[i] = np.mean(np.abs(tp[i - window + 1 : i + 1] - sma_tp[i]))

    cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-9)
    return cci

def calc_adx(high, low, close, window=14):
    up_move = np.zeros_like(high)
    down_move = np.zeros_like(low)

    up_move[1:] = high[1:] - high[:-1]
    down_move[1:] = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    tr_smooth = calc_rma(tr, window)
    plus_dm_smooth = calc_rma(plus_dm, window)
    minus_dm_smooth = calc_rma(minus_dm, window)

    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-9)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-9)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = calc_rma(dx, window)

    return adx

# --- Main Feature Generation ---

def get_indicators(data):
    close = data['close']
    high = data.get('high', close)
    low = data.get('low', close)
    vol = data['volume']

    inds = {
        'sma_7': calc_sma(close, 7),
        'sma_30': calc_sma(close, 30),
        'rsi': calc_rsi(close, 14),
        'roc': calc_roc(close, 12),
        'bb_width': calc_bb_width(close, 20),
        'obv_change': calc_obv(close, vol),
        'macd_hist': calc_macd_hist(close),
    }

    inds['atr'] = calc_atr(high, low, close, 14)
    inds['stoch_k'], inds['stoch_d'] = calc_stoch(high, low, close)
    inds['cci'] = calc_cci(high, low, close)
    inds['adx'] = calc_adx(high, low, close)
    inds['bb_pct_b'] = calc_bb_pct_b(close, 20)
    inds['sma_dev_7'] = (close - inds['sma_7']) / (inds['sma_7'] + 1e-9)

    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)
    inds['returns'] = returns

    inds['lag1'] = np.roll(returns, 1)
    inds['lag2'] = np.roll(returns, 2)
    inds['lag3'] = np.roll(returns, 3)

    if 'open_time' in data and data['open_time'] is not None:
        t0 = data['open_time'][0]
        if t0 > 1e11: # likely ms
            scale = 3600000
        else: # likely seconds
            scale = 3600

        hours = (data['open_time'] / scale) % 24
        inds['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        inds['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    else:
        inds['hour_sin'] = np.zeros_like(close)
        inds['hour_cos'] = np.zeros_like(close)

    return inds


# ==========================================
# 2. Shared Data Processing & Configuration
# ==========================================

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
        except Exception:
            return default_config
    else:
        return default_config

def get_symbol_files(directory):
    if not os.path.exists(directory):
        return {}

    symbol_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            symbol = filename.split('-')[0]
            if symbol not in symbol_files: symbol_files[symbol] = []
            symbol_files[symbol].append(os.path.join(directory, filename))
    return symbol_files

def load_and_merge_data(files):
    """
    複数のCSVファイルを読み込んで結合する。
    注意: 連続していないデータを結合する場合、指標計算前に結合すると不連続点でおかしくなる。
    ここでは「一つのペアの連続したデータ」として扱うための結合を行う。
    複数ペアを混ぜる場合は、指標計算後に特徴量を結合すべき。
    """
    combined_data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'open_time': []}

    # ファイル名でソートして読み込む（時系列順にするため）
    files = sorted(files)

    for f_path in files:
        with open(f_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    for k in combined_data.keys(): combined_data[k].append(float(row[k]))
                except: continue

    if not combined_data['close']:
        return None

    # 時刻順にソート（念のため）
    indices = np.argsort(combined_data['open_time'])
    data_dict = {k: np.array(combined_data[k])[indices] for k in combined_data.keys()}
    return data_dict

# ==========================================
# 3. Training Module
# ==========================================

def create_training_dataset(data, target_shift=1):
    inds = get_indicators(data)
    close = data['close']
    X, y = [], []

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

def train_general_model(symbol_files):
    """
    全ての指定されたペアのデータを使って単一の汎用モデルを学習する
    """
    periods = {
        'short': {'factor': 1, 'shift': 3, 'name': '短期'},
        'mid':   {'factor': 12, 'shift': 4, 'name': '中期'},
        'long':  {'factor': 288, 'shift': 1, 'name': '長期'}
    }

    if not os.path.exists('model'): os.makedirs('model')

    for key, cfg in periods.items():
        print(f"\n>>> {cfg['name']} モデルのデータ準備中...")

        all_X = []
        all_y = []

        for symbol, files in symbol_files.items():
            # 各シンボルごとにデータを読み込み、指標計算を行う
            # これにより、異なるシンボル間の不連続性を回避する
            data_dict = load_and_merge_data(files)
            if data_dict is None: continue

            resampled = resample_ohlc(data_dict, cfg['factor'])
            if resampled is None: continue

            X, y = create_training_dataset(resampled, target_shift=cfg['shift'])

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            print(f"  [!] {cfg['name']}: 学習データがありません。")
            continue

        # 全ペアのデータを結合
        X_combined = np.concatenate(all_X)
        y_combined = np.concatenate(all_y)

        print(f"  学習開始 (総サンプル数: {len(X_combined)})...")
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_combined, y_combined)

        model_filename = f'model/model_ALL_{key}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] {cfg['name']}モデル保存: {model_filename}")

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
        X, y = create_training_dataset(resampled, target_shift=cfg['shift'])

        if len(X) < 30:
            print(f"  [!] {symbol} {cfg['name']}: 有効な学習データが不足しています (サンプル数: {len(X)})")
            continue

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        model_filename = f'model/model_{symbol}_{key}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] {cfg['name']}モデル保存: {model_filename} (サンプル数: {len(X)})")

def run_training_mode():
    history_dir = 'history'
    if not os.path.exists(history_dir):
        print(f"エラー: '{history_dir}' ディレクトリが見つかりません。")
        return

    symbol_files = get_symbol_files(history_dir)
    available_symbols = sorted(list(symbol_files.keys()))
    if not available_symbols:
        print("学習可能なデータが見つかりません。")
        return

    print(f"学習可能なペア: {', '.join(available_symbols)}")
    print("すべてのペアを学習して汎用モデルを作る場合は 'ALL' と入力してください。")

    selected_symbol = input("学習するペア名 (または ALL): ").strip()

    if selected_symbol.upper() == 'ALL':
        print(f"\n>>> 全ペア ({len(available_symbols)}種) を使用して汎用モデルを学習します...")
        train_general_model(symbol_files)
    elif selected_symbol in symbol_files:
        files = symbol_files[selected_symbol]
        print(f"\n>>> シンボル: {selected_symbol} の学習を開始します ({len(files)} ファイル)")
        data_dict = load_and_merge_data(files)
        if data_dict is None:
            print("有効なデータがありません。")
            return
        train_for_symbol(selected_symbol, data_dict)
    else:
        print(f"エラー: ペア '{selected_symbol}' は存在しません。")

# ==========================================
# 4. Backtest Module
# ==========================================

def create_eval_dataset(data, target_shift=1):
    """
    MLモデル予測用の特徴量を作成する
    """
    inds = get_indicators(data)
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
        y.append(1 if close[i + target_shift] > close[i] else 0)

        item = {
            'index': i,
            'open_time': open_time[i] if open_time is not None else None
        }
        ohlc_data.append(item)

    return np.array(X), np.array(y), ohlc_data

class BacktestSimulator:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.fee = config.get('trading_fee', 0.0)
        self.max_pos = config.get('max_positions', 1)

        self.active_positions = []
        self.closed_trades = []
        self.equity = self.initial_capital

    def run(self, full_data, signals):
        times = full_data['open_time']
        opens = full_data['open']
        highs = full_data['high']
        lows = full_data['low']

        tp_pct = self.config.get('take_profit', 0.02)
        sl_pct = self.config.get('stop_loss', 0.01)
        th_long = self.config.get('threshold_long', 0.5)
        th_short = self.config.get('threshold_short', 0.5)
        enable_short = self.config.get('enable_short', False)

        for i in range(len(times)):
            t_curr = times[i]
            op = opens[i]
            hi = highs[i]
            lo = lows[i]

            next_positions = []
            for pos in self.active_positions:
                p_type = pos['type']
                entry_price = pos['entry_price']
                tp_price = pos['tp_price']
                sl_price = pos['sl_price']

                is_closed = False
                close_price = 0.0
                reason = ""

                if p_type == 'long':
                    if lo <= sl_price:
                        is_closed = True
                        close_price = sl_price
                        reason = 'sl'
                    elif hi >= tp_price:
                        is_closed = True
                        close_price = tp_price
                        reason = 'tp'

                    if is_closed:
                        if (lo <= sl_price) and (hi >= tp_price):
                            close_price = sl_price
                            reason = 'sl'
                        raw_ret = (close_price - entry_price) / entry_price

                elif p_type == 'short':
                    if hi >= sl_price:
                        is_closed = True
                        close_price = sl_price
                        reason = 'sl'
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

            if len(self.active_positions) < self.max_pos:
                if t_curr in signals:
                    prob_up = signals[t_curr]
                    entry_type = None
                    if prob_up >= th_long:
                        entry_type = 'long'
                    elif enable_short and prob_up <= th_short:
                        entry_type = 'short'

                    if entry_type:
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

        curr = self.initial_capital
        max_dd = 0.0
        peak = curr
        for t in trades:
            curr *= (1 + t['return'])
            if curr > peak: peak = curr
            dd = (peak - curr) / peak
            if dd > max_dd: max_dd = dd

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd
        }

def evaluate_performance(symbol, all_5m_data, optimize=False, use_model_symbol=None):
    """
    use_model_symbol: どのシンボルのモデルを使うか (Noneならsymbolと同じ、'ALL'なら汎用モデル)
    """
    config = load_config()
    periods = {
        'short': {'factor': 1, 'shift': 3, 'name': '短期 (15分後予測)'},
        'mid':   {'factor': 12, 'shift': 4, 'name': '中期 (4時間後予測)'},
        'long':  {'factor': 288, 'shift': 1, 'name': '長期 (1日後予測)'}
    }

    # モデルの識別子
    model_key_name = use_model_symbol if use_model_symbol else symbol

    print(f"\n{'='*60}")
    print(f" バックテスト実行: {symbol}")
    print(f" 使用モデル: {model_key_name}")
    if optimize:
        print(" ※ パラメータ最適化モード")
    else:
        print(f" ※ 現在の設定: TP={config['take_profit']}, SL={config['stop_loss']}, TH_L={config['threshold_long']}")
    print(f"{'='*60}")

    for key, cfg in periods.items():
        print(f"\n>>> {cfg['name']} モデル...")

        model_path = f"model/model_{model_key_name}_{key}.pkl"
        if not os.path.exists(model_path):
            print(f"  [Skip] モデルファイルなし: {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        resampled = resample_ohlc(all_5m_data, cfg['factor'])
        if resampled is None: continue

        X, _, eval_ohlc = create_eval_dataset(resampled, target_shift=cfg['shift'])
        if len(X) == 0: continue

        try:
            y_pred_proba = model.predict_proba(X)
        except AttributeError:
            print("  [Error] モデルが確率を出力できません。")
            continue

        duration_ms = cfg['factor'] * 5 * 60 * 1000
        signals = {}
        for i in range(len(X)):
            start_time = eval_ohlc[i]['open_time']
            if start_time is None: continue
            entry_time = start_time + duration_ms
            prob_up = y_pred_proba[i][1]
            signals[entry_time] = prob_up

        if optimize:
            tp_range = [0.01, 0.02, 0.03, 0.04, 0.05]
            sl_range = [0.005, 0.01, 0.02, 0.03]
            th_long_range = [0.5, 0.55, 0.6, 0.65, 0.7]
            th_short_range = [0.5, 0.45, 0.4, 0.35, 0.3]

            best_metric = -999.0
            best_params = {}
            best_result = {}

            combinations = list(itertools.product(tp_range, sl_range, th_long_range, th_short_range))
            print(f"  [Info] {len(combinations)} 通りの組み合わせを検証します...")

            for tp, sl, th_l, th_s in combinations:
                test_cfg = config.copy()
                test_cfg['take_profit'] = tp
                test_cfg['stop_loss'] = sl
                test_cfg['threshold_long'] = th_l
                test_cfg['threshold_short'] = th_s

                sim = BacktestSimulator(test_cfg)
                sim.run(all_5m_data, signals)
                metrics = sim.get_metrics()
                score = metrics['total_return']

                if metrics['total_trades'] >= 10 and score > best_metric:
                    best_metric = score
                    best_params = {'TP': tp, 'SL': sl, 'TH_L': th_l, 'TH_S': th_s}
                    best_result = metrics

            print(f"  [Best Result]")
            print(f"    Params : {best_params}")
            print(f"    Return : {best_result.get('total_return', 0):.2%}")
            print(f"    WinRate: {best_result.get('win_rate', 0):.2%}")
            print(f"    PF     : {best_result.get('profit_factor', 0):.2f}")
            print(f"    Trades : {best_result.get('total_trades', 0)}")
            print(f"    DD     : {best_result.get('max_drawdown', 0):.2%}")

        else:
            sim = BacktestSimulator(config)
            sim.run(all_5m_data, signals)
            metrics = sim.get_metrics()

            print(f"  [Result]")
            print(f"    Total Return : {metrics['total_return']:.2%}")
            print(f"    Win Rate     : {metrics['win_rate']:.2%}")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Total Trades : {metrics['total_trades']}")
            print(f"    Max Drawdown : {metrics['max_drawdown']:.2%}")

            output_dir = 'backtest_results'
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            ts = int(time.time())
            fname = f"{output_dir}/{symbol}_{model_key_name}_{key}_{ts}.json"
            save_data = {'config': config, 'metrics': metrics, 'trades': sim.closed_trades}
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"  [Save] {fname}")

def run_backtest_mode():
    history_dir = 'history_for_test'
    if not os.path.exists(history_dir):
        print(f"エラー: '{history_dir}' ディレクトリが見つかりません。")
        return

    symbol_files = get_symbol_files(history_dir)
    available_symbols = sorted(list(symbol_files.keys()))
    if not available_symbols:
        print("データが見つかりません。")
        return

    print(f"テスト可能なペア: {', '.join(available_symbols)}")
    selected_symbol = input("テストするペア名を入力してください: ").strip()

    if selected_symbol not in symbol_files:
        print(f"エラー: ペア '{selected_symbol}' は存在しません。")
        return

    # モデル選択
    print("\n使用するモデルを選択:")
    print(f"1. {selected_symbol} 専用モデル")
    print("2. 汎用モデル (ALL)")
    model_choice = input("選択 (1/2): ").strip()
    use_model = 'ALL' if model_choice == '2' else selected_symbol

    # モード選択
    print("\n実行モードを選択:")
    print("1. 現在の設定でバックテスト (詳細ログ出力)")
    print("2. パラメータ最適化 (グリッドサーチ)")
    mode_input = input("選択 (1/2): ").strip()
    is_optimize = (mode_input == '2')

    files = symbol_files[selected_symbol]
    print(f"\n>>> データを読み込んでいます... ({len(files)} ファイル)")

    data_dict = load_and_merge_data(files)
    if data_dict is None:
        print("有効なデータがありません。")
        return

    evaluate_performance(selected_symbol, data_dict, optimize=is_optimize, use_model_symbol=use_model)

# ==========================================
# 5. Prediction Module
# ==========================================

def get_latest_feature(data):
    inds = get_indicators(data)

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

def fetch_mexc_klines(symbol, interval):
    mexc_symbol = symbol.replace("_", "")
    url = f"https://api.mexc.com/api/v3/klines?symbol={mexc_symbol}&interval={interval}&limit=100"

    try:
        with urllib.request.urlopen(url) as response:
            klines = json.loads(response.read().decode())
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

def run_analysis(symbol, use_model_symbol=None):
    model_key = use_model_symbol if use_model_symbol else symbol

    print(f"\n" + "="*50)
    print(f" 市場分析レポート: {symbol} (拡張版)")
    print(f" 使用モデル: {model_key}")
    print("="*50)

    periods = [
        ('short', '5m', '短期'),
        ('mid', '60m', '中期'),
        ('long', '1d', '長期')
    ]

    results = []
    for key, interval, label in periods:
        model_file = f'model/model_{model_key}_{key}.pkl'
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

def run_prediction_mode():
    if not os.path.exists('model'):
        print("モデルディレクトリが見つかりません。")
        return

    available_symbols = set()
    has_general_model = False

    for f in os.listdir('model'):
        if f.startswith('model_') and f.endswith('.pkl'):
            name_body = f[6:-4]
            # Check for general model
            if name_body.startswith('ALL_'):
                has_general_model = True

            for key in ['short', 'mid', 'long']:
                if name_body.endswith('_' + key):
                    symbol = name_body[:-(len(key)+1)]
                    available_symbols.add(symbol)
                    break

    sorted_symbols = sorted(list(available_symbols))
    if not sorted_symbols and not has_general_model:
        print("有効なモデルが見つかりません。")
        return

    print(f"予測可能なペア (モデルあり): {', '.join(sorted_symbols)}")
    if has_general_model:
        print("※ 汎用モデル (ALL) が利用可能です。")

    target_symbol = input("予測するペア名を入力してください: ").strip()

    # モデル選択
    use_model = target_symbol
    if has_general_model:
        if target_symbol not in sorted_symbols:
            print(f"ペア '{target_symbol}' の専用モデルはありませんが、汎用モデルで予測します。")
            use_model = 'ALL'
        else:
            print("\n使用するモデルを選択:")
            print(f"1. {target_symbol} 専用モデル")
            print("2. 汎用モデル (ALL)")
            choice = input("選択 (1/2): ").strip()
            if choice == '2':
                use_model = 'ALL'

    run_analysis(target_symbol, use_model_symbol=use_model)

# ==========================================
# 6. Main Entry Point
# ==========================================

def main():
    print("========================================")
    print("       Unified Crypto Model Tool        ")
    print("========================================")
    print("1. Train Model (学習)")
    print("2. Backtest / Optimize (検証・最適化)")
    print("3. Predict (予測 - MEXC API)")
    print("0. Exit")

    choice = input("\nSelect mode (0-3): ").strip()

    if choice == '1':
        run_training_mode()
    elif choice == '2':
        run_backtest_mode()
    elif choice == '3':
        run_prediction_mode()
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
