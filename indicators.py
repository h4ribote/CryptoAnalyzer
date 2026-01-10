import numpy as np

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
    # Typically RSI uses RMA (Wilder's), but SMA is used in original code.
    # We'll stick to SMA to match original logic or switch to RMA for standard RSI?
    # Original used SMA. Let's keep SMA for consistency with previous version unless requested.
    # But standard RSI uses RMA. Let's upgrade to RMA for better quality.
    avg_gain = calc_rma(gains, window)
    avg_loss = calc_rma(losses, window)

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
    # TR = max(H-L, |H-Cp|, |L-Cp|)
    # index 0 is H-L
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
    # np.convolve handles NaNs correctly (output is NaN only for windows containing NaN)
    # unlike cumsum which propagates NaN forever.
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
    high = data.get('high', close) # Fallback if not available
    low = data.get('low', close)
    vol = data['volume']

    # 既存の指標
    # RSI: SMA -> RMAに変更したため値が少し変わる可能性あり
    inds = {
        'sma_7': calc_sma(close, 7),
        'sma_30': calc_sma(close, 30),
        'rsi': calc_rsi(close, 14),
        'roc': calc_roc(close, 12),
        'bb_width': calc_bb_width(close, 20),
        'obv_change': calc_obv(close, vol),
        'macd_hist': calc_macd_hist(close),
    }

    # 新規指標
    inds['atr'] = calc_atr(high, low, close, 14)
    inds['stoch_k'], inds['stoch_d'] = calc_stoch(high, low, close)
    inds['cci'] = calc_cci(high, low, close)
    inds['adx'] = calc_adx(high, low, close)
    inds['bb_pct_b'] = calc_bb_pct_b(close, 20)
    inds['sma_dev_7'] = (close - inds['sma_7']) / (inds['sma_7'] + 1e-9)

    # Returns & Lags
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)
    inds['returns'] = returns

    inds['lag1'] = np.roll(returns, 1)
    inds['lag2'] = np.roll(returns, 2)
    inds['lag3'] = np.roll(returns, 3)

    # Time Features
    if 'open_time' in data and data['open_time'] is not None:
        # Assuming open_time is in milliseconds (standard for crypto APIs)
        # However, checking history file format might be good.
        # If it's unix timestamp (seconds), /3600 gives hours.
        # If ms, /3600000.
        # Let's infer from magnitude.

        # Example: 1685577600 (2023-06-01) -> 10 digits -> seconds
        # 1685577600000 -> 13 digits -> ms

        # Check first element
        t0 = data['open_time'][0]
        if t0 > 1e11: # likely ms
            scale = 3600000
        else: # likely seconds
            scale = 3600

        # Hour of day (UTC)
        # (t / 3600) % 24
        hours = (data['open_time'] / scale) % 24
        inds['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        inds['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    else:
        # Fill with 0 if no time data
        inds['hour_sin'] = np.zeros_like(close)
        inds['hour_cos'] = np.zeros_like(close)

    return inds
