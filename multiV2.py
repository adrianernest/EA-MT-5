import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime, timezone

# ==========================================
# KONFIGURASI UTAMA
# ==========================================
SYMBOL            = "XAUUSD"
TIMEFRAME         = mt5.TIMEFRAME_M1
MAGIC             = 303030

# === MULTI ORDER ===
JUMLAH_ORDER      = 3
LOT_PER_ORDER     = 5.00

# === FORCE CLOSE TARGET ===
TARGET_PROFIT_USD = 0.01
TARGET_PIPS       = 1

# === PROTEKSI ===
MAX_LOSS_USD      = 5.0
MAX_HOLD_DETIK    = 120

# === THRESHOLD AI ===
THRESHOLD         = 0.53

# === FILTER PASAR ===
ADX_MIN           = 20        # Hanya entry jika pasar trending (ADX > 20)
BB_SQUEEZE_RATIO  = 0.8       # Deteksi squeeze Bollinger Bands

# Jam trading: 24 jam penuh
ALLOWED_HOURS     = list(range(0, 24))

# ==========================================
# INDIKATOR LENGKAP
# ==========================================
def compute_indicators(df):

    # ── RSI 7 ────────────────────────────────────────
    diff = df['close'].diff()
    up   = diff.clip(lower=0)
    dn   = -diff.clip(upper=0)
    df['RSI_7'] = 100 - (100 / (1 + up.ewm(span=7, adjust=False).mean()
                                   / dn.ewm(span=7, adjust=False).mean()))

    # ── EMA Trend ────────────────────────────────────
    df['EMA_9']   = df['close'].ewm(span=9,  adjust=False).mean()
    df['EMA_21']  = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_50']  = df['close'].ewm(span=50, adjust=False).mean()   # BARU: trend menengah
    df['EMA_cross']   = df['EMA_9']  - df['EMA_21']
    df['EMA_vs_50']   = df['close']  - df['EMA_50']                 # posisi harga vs EMA50

    # ── Stochastic ───────────────────────────────────
    lo = df['low'].rolling(14).min()
    hi = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - lo) / (hi - lo)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ── ATR ──────────────────────────────────────────
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR_14']      = tr.rolling(14).mean()
    df['ATR_ratio']   = df['ATR_14'] / df['ATR_14'].rolling(50).mean()  # BARU: volatilitas relatif

    # ── ADX (Average Directional Index) — BARU ───────
    # +DM dan -DM
    df['up_move']   = df['high'].diff()
    df['dn_move']   = -df['low'].diff()
    df['+DM'] = np.where((df['up_move'] > df['dn_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['dn_move'] > df['up_move']) & (df['dn_move'] > 0), df['dn_move'], 0)
    atr14 = df['ATR_14'].replace(0, np.nan)
    df['+DI'] = 100 * df['+DM'].ewm(span=14, adjust=False).mean() / atr14
    df['-DI'] = 100 * df['-DM'].ewm(span=14, adjust=False).mean() / atr14
    dx = 100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI']).replace(0, np.nan)
    df['ADX']  = dx.ewm(span=14, adjust=False).mean()
    df['DI_diff'] = df['+DI'] - df['-DI']   # positif = bullish, negatif = bearish

    # ── MACD (12, 26, 9) — BARU ──────────────────────
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']   # histogram positif = momentum naik

    # ── Bollinger Bands — BARU ───────────────────────
    bb_mid            = df['close'].rolling(20).mean()
    bb_std            = df['close'].rolling(20).std()
    df['BB_upper']    = bb_mid + 2 * bb_std
    df['BB_lower']    = bb_mid - 2 * bb_std
    df['BB_width']    = (df['BB_upper'] - df['BB_lower']) / bb_mid   # lebar bands
    df['BB_pct']      = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])  # posisi di dalam band
    df['BB_squeeze']  = df['BB_width'] / df['BB_width'].rolling(50).mean()  # squeeze detector

    # ── CCI (14) — BARU ──────────────────────────────
    tp_cci            = (df['high'] + df['low'] + df['close']) / 3
    ma_cci            = tp_cci.rolling(14).mean()
    md_cci            = tp_cci.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI']         = (tp_cci - ma_cci) / (0.015 * md_cci)

    # ── Williams %R — BARU ───────────────────────────
    lo14_w = df['low'].rolling(14).min()
    hi14_w = df['high'].rolling(14).max()
    df['WillR'] = -100 * (hi14_w - df['close']) / (hi14_w - lo14_w)

    # ── OBV (On Balance Volume) — BARU ───────────────
    if 'tick_volume' in df.columns:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['tick_volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['tick_volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV']       = obv
        df['OBV_trend'] = pd.Series(obv).ewm(span=10, adjust=False).mean().values
        df['OBV_diff']  = df['OBV'] - df['OBV_trend']   # OBV di atas/bawah trendnya
        df['vol_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
    else:
        df['OBV_diff']  = 0.0
        df['vol_ratio'] = 1.0

    # ── Volume Spike — BARU ──────────────────────────
    df['vol_spike'] = df['vol_ratio'].apply(lambda x: 1 if x > 1.5 else 0)

    # ── VWAP proxy ───────────────────────────────────
    df['vwap']    = ((df['high'] + df['low'] + df['close']) / 3).rolling(20).mean()
    df['vs_vwap'] = df['close'] - df['vwap']

    # ── Higher High / Lower Low (Struktur Pasar) — BARU
    df['HH'] = df['high'].rolling(5).max()
    df['LL']  = df['low'].rolling(5).min()
    df['structure_bull'] = ((df['high'] > df['HH'].shift(1)) & (df['low'] > df['LL'].shift(1))).astype(int)
    df['structure_bear'] = ((df['high'] < df['HH'].shift(1)) & (df['low'] < df['LL'].shift(1))).astype(int)

    # ── Pin Bar Detector — BARU ──────────────────────
    body      = (df['close'] - df['open']).abs()
    up_wick   = df['high'] - df[['close','open']].max(axis=1)
    lo_wick   = df[['close','open']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    df['pin_bull'] = ((lo_wick > body * 2) & (lo_wick > up_wick * 2)).astype(int)  # ekor bawah panjang
    df['pin_bear'] = ((up_wick > body * 2) & (up_wick > lo_wick * 2)).astype(int)  # ekor atas panjang

    # ── Momentum & Candle Shape ───────────────────────
    df['mom_3']   = df['close'] - df['close'].shift(3)
    df['mom_7']   = df['close'] - df['close'].shift(7)   # BARU: momentum lebih panjang
    df['body']    = df['close'] - df['open']
    df['up_wick'] = up_wick
    df['lo_wick'] = lo_wick

    # Bersihkan kolom helper
    df.drop(columns=['up_move','dn_move','+DM','-DM','+DI','-DI','HH','LL'], inplace=True, errors='ignore')

    return df

# ==========================================
# FEATURES UNTUK AI
# ==========================================
FEATURES = [
    # Trend
    'EMA_cross', 'EMA_vs_50',
    # Momentum
    'RSI_7', 'MACD_hist', 'CCI', 'WillR', 'mom_3', 'mom_7',
    # Oscillator
    'stoch_k', 'stoch_d',
    # Kekuatan Trend
    'ADX', 'DI_diff',
    # Volatilitas
    'ATR_14', 'ATR_ratio', 'BB_width', 'BB_pct', 'BB_squeeze',
    # Volume
    'vol_ratio', 'OBV_diff',
    # VWAP
    'vs_vwap',
    # Price Action
    'body', 'up_wick', 'lo_wick', 'structure_bull', 'structure_bear',
    'pin_bull', 'pin_bear', 'vol_spike'
]

# ==========================================
# DATA PIPELINE
# ==========================================
def get_data(n=3000):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = compute_indicators(df)
    move = df['close'].shift(-1) - df['close']
    df['Target'] = np.where(move >  0.20,  1,
                   np.where(move < -0.20,  0, -1))
    df = df[df['Target'] != -1].copy()
    df.dropna(inplace=True)
    return df

# ==========================================
# TRAIN MODEL
# ==========================================
def train_model(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, df['Target'])
    return model, scaler

# ==========================================
# KONFIRMASI MULTI-LAYER (DIPERKUAT)
# ==========================================
def confirm_buy(r):
    """
    Scoring sistem — minimal 5 dari 8 kondisi harus terpenuhi.
    Semakin tinggi score, semakin kuat sinyal BUY.
    """
    conditions = [
        r['EMA_cross']    >  0,          # EMA9 di atas EMA21
        r['EMA_vs_50']    >  0,          # Harga di atas EMA50
        r['MACD_hist']    >  0,          # MACD histogram positif
        r['stoch_k']      < 35,          # Stochastic oversold
        r['DI_diff']      >  0,          # +DI > -DI (bullish ADX)
        r['vs_vwap']      >  0,          # Harga di atas VWAP
        r['mom_3']        >  0,          # Momentum 3 bar positif
        r['structure_bull'] == 1,        # Struktur Higher High/Low
        r['CCI']          < -80,         # CCI oversold (reversal)
        r['pin_bull']     == 1,          # Ada pin bar bullish
    ]
    score = sum(conditions)
    return score >= 4   # minimal 4 dari 10 kondisi

def confirm_sell(r):
    conditions = [
        r['EMA_cross']    <  0,
        r['EMA_vs_50']    <  0,
        r['MACD_hist']    <  0,
        r['stoch_k']      > 65,
        r['DI_diff']      <  0,
        r['vs_vwap']      <  0,
        r['mom_3']        <  0,
        r['structure_bear'] == 1,
        r['CCI']          > 80,
        r['pin_bear']     == 1,
    ]
    score = sum(conditions)
    return score >= 4

def is_market_valid(r):
    """
    Filter kualitas pasar sebelum entry.
    Hindari entry di pasar sideways atau volatilitas ekstrem.
    """
    adx_ok      = r['ADX'] > ADX_MIN              # Pasar harus trending
    not_extreme = r['ATR_ratio'] < 2.5            # Hindari volatilitas ekstrem
    not_squeeze = r['BB_squeeze'] > 0.5           # Hindari saat BB terlalu sempit
    return adx_ok and not_extreme and not_squeeze

# ==========================================
# EKSEKUSI MULTI ORDER
# ==========================================
def open_multi_orders(direction):
    sym  = mt5.symbol_info(SYMBOL)
    if not sym: return []
    tick    = mt5.symbol_info_tick(SYMBOL)
    pt      = sym.point
    tickets = []
    sl_pts  = 500

    for i in range(JUMLAH_ORDER):
        if direction == 'BUY':
            price = tick.ask
            sl    = price - sl_pts * pt
            tp    = price + sl_pts * pt
            otype = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl    = price + sl_pts * pt
            tp    = price - sl_pts * pt
            otype = mt5.ORDER_TYPE_SELL

        res = mt5.order_send({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       LOT_PER_ORDER,
            "type":         otype,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "magic":        MAGIC,
            "comment":      f"ScalpV3#{i+1}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            tickets.append(res.order)
            print(f"  ✅ Order #{i+1} DIBUKA | Ticket:{res.order} | Price:{price:.2f}")
        else:
            code = res.retcode if res else "None"
            print(f"  ❌ Order #{i+1} GAGAL | Kode:{code}")
            if code == 10027:
                print("  ⚠️  Aktifkan 'Algo Trading' di toolbar MT5!")
        time.sleep(0.1)

    return tickets

# ==========================================
# FORCE CLOSE
# ==========================================
def force_close_order(position):
    tick = mt5.symbol_info_tick(SYMBOL)
    if position.type == mt5.ORDER_TYPE_BUY:
        close_price = tick.bid
        close_type  = mt5.ORDER_TYPE_SELL
    else:
        close_price = tick.ask
        close_type  = mt5.ORDER_TYPE_BUY
    return mt5.order_send({
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       position.volume,
        "type":         close_type,
        "position":     position.ticket,
        "price":        close_price,
        "magic":        MAGIC,
        "comment":      "ForceClose",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })

def force_close_all(reason=""):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return
    closed = 0
    total  = 0
    for pos in positions:
        if pos.magic != MAGIC: continue
        res = force_close_order(pos)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            total  += pos.profit
            closed += 1
    if closed > 0:
        print(f"  🔒 FORCE CLOSE {closed} order | Profit: ${total:.2f} | {reason}")

# ==========================================
# MONITOR POSISI
# ==========================================
def monitor_and_close(open_time):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return False

    my_pos = [p for p in positions if p.magic == MAGIC]
    if not my_pos: return False

    sym          = mt5.symbol_info(SYMBOL)
    pt           = sym.point if sym else 0.01
    total_profit = sum(p.profit for p in my_pos)
    elapsed      = (datetime.now(timezone.utc) - open_time).total_seconds()

    if total_profit >= TARGET_PROFIT_USD:
        force_close_all(f"Target profit ${TARGET_PROFIT_USD} ✅")
        return False
    if total_profit <= -MAX_LOSS_USD:
        force_close_all(f"Batas loss -${MAX_LOSS_USD} ⛔")
        return False
    if elapsed >= MAX_HOLD_DETIK:
        force_close_all(f"Timeout {MAX_HOLD_DETIK}s ⏱️")
        return False

    tick = mt5.symbol_info_tick(SYMBOL)
    for pos in my_pos:
        pip_profit = ((tick.bid - pos.price_open) if pos.type == mt5.ORDER_TYPE_BUY
                      else (pos.price_open - tick.ask)) / pt
        if pip_profit >= TARGET_PIPS:
            res = force_close_order(pos)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  💰 Ticket {pos.ticket} | +{pip_profit:.1f} pip | ${pos.profit:.2f}")

    print(f"  📊 {len(my_pos)} posisi | P/L: ${total_profit:.2f} | {elapsed:.0f}s")
    return True

# ==========================================
# SINYAL ENTRY
# ==========================================
def get_signal(model, scaler, n_bars=300):
    live = get_data(n_bars)
    if live is None or len(live) < 5:
        return None, None, None
    row   = live.iloc[-1]
    X     = scaler.transform(live[FEATURES].iloc[[-1]])
    probs = model.predict_proba(X)[0]
    cls   = list(model.classes_)
    p_up  = probs[cls.index(1)] if 1 in cls else 0
    p_dn  = probs[cls.index(0)] if 0 in cls else 0
    return p_up, p_dn, row

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ Gagal inisialisasi MT5!"); quit()

    print("=" * 55)
    print(f"  MULTI-ORDER SCALPER v3 | {SYMBOL} M1")
    print(f"  Indikator  : RSI, EMA(9/21/50), Stoch, MACD,")
    print(f"               ADX, BB, CCI, WillR, OBV,")
    print(f"               Pin Bar, HH/LL Structure, Vol Spike")
    print(f"  Order      : {JUMLAH_ORDER} x {LOT_PER_ORDER} lot")
    print(f"  Target     : ${TARGET_PROFIT_USD} profit / {TARGET_PIPS} pip per order")
    print(f"  Max Loss   : ${MAX_LOSS_USD}")
    print(f"  Timeout    : {MAX_HOLD_DETIK} detik")
    print(f"  ADX Filter : > {ADX_MIN} (trending only)")
    print("=" * 55)

    model    = None
    scaler   = None
    it       = 0
    in_trade = False
    open_time = None

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Retrain setiap 60 iterasi (~10 menit)
            if it % 60 == 0:
                print(f"\n[{now:%H:%M:%S}] 🔄 Training model ({len(FEATURES)} fitur)...")
                data = get_data(3000)
                if data is not None and len(data) > 200:
                    model, scaler = train_model(data)
                    print(f"  → Selesai | {len(data)} sample | {len(FEATURES)} fitur")
                else:
                    print("  → Data tidak cukup, skip.")

            if model is None:
                time.sleep(10); it += 1; continue

            # Monitor posisi aktif
            if in_trade:
                still_open = monitor_and_close(open_time)
                if not still_open:
                    in_trade = False
                    print(f"[{now:%H:%M:%S}] ✅ Semua posisi ditutup.")
                time.sleep(2); it += 1; continue

            # Cari sinyal
            p_up, p_dn, row = get_signal(model, scaler)
            if row is None:
                time.sleep(10); it += 1; continue

            # Filter pasar dulu
            if not is_market_valid(row):
                print(f"[{now:%H:%M:%S}] ⏸️  Pasar tidak ideal | ADX:{row['ADX']:.1f} BB_sq:{row['BB_squeeze']:.2f}")
                time.sleep(10); it += 1; continue

            # Entry BUY
            if p_up > THRESHOLD and confirm_buy(row):
                print(f"\n[{now:%H:%M:%S}] 🚀 SINYAL BUY | AI:{p_up*100:.1f}% | ADX:{row['ADX']:.1f}")
                print(f"  RSI:{row['RSI_7']:.1f} | MACD:{row['MACD_hist']:.3f} | "
                      f"Stoch:{row['stoch_k']:.1f} | CCI:{row['CCI']:.1f}")
                tickets = open_multi_orders('BUY')
                if tickets:
                    in_trade  = True
                    open_time = datetime.now(timezone.utc)

            # Entry SELL
            elif p_dn > THRESHOLD and confirm_sell(row):
                print(f"\n[{now:%H:%M:%S}] 📉 SINYAL SELL | AI:{p_dn*100:.1f}% | ADX:{row['ADX']:.1f}")
                print(f"  RSI:{row['RSI_7']:.1f} | MACD:{row['MACD_hist']:.3f} | "
                      f"Stoch:{row['stoch_k']:.1f} | CCI:{row['CCI']:.1f}")
                tickets = open_multi_orders('SELL')
                if tickets:
                    in_trade  = True
                    open_time = datetime.now(timezone.utc)

            else:
                print(f"[{now:%H:%M:%S}] ⏳ Tunggu... UP:{p_up*100:.1f}% DN:{p_dn*100:.1f}% "
                      f"| ADX:{row['ADX']:.1f} | BB_sq:{row['BB_squeeze']:.2f}")

            it += 1
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n⚠️  Bot dihentikan.")
        force_close_all("Manual stop")
    finally:
        mt5.shutdown()
        print("MT5 disconnected.")