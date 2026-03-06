import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import time
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════╗
# ║         POWER SCALPER PRO — XAUUSD M1               ║
# ║   Ensemble AI + Tick Monitor + Smart Risk Mgmt      ║
# ╚══════════════════════════════════════════════════════╝

# ==========================================
# KONFIGURASI
# ==========================================
SYMBOL          = "XAUUSD"
TIMEFRAME       = mt5.TIMEFRAME_M1
MAGIC           = 404040

# === MULTI ORDER ===
JUMLAH_ORDER    = 3
LOT_PER_ORDER   = 5.00

# === PROFIT TARGET ===
TARGET_PROFIT_USD = 0.01     # Force close semua jika total profit >= $0.01
TARGET_PIPS       = 1        # Force close per order jika >= 1 pip profit

# === PROTEKSI ===
MAX_LOSS_USD      = 5.0      # Cut loss jika rugi >= $5
MAX_HOLD_DETIK    = 60       # Timeout 60 detik (lebih agresif)
MAX_SPREAD_POINTS = 30       # Skip entry jika spread > 30 points (terlalu lebar)

# === FILTER AI ===
THRESHOLD_ENTRY   = 0.53    
ADX_MIN           = 18       # Pasar harus trending
SCORE_MIN         = 3        # Minimal 3 kondisi konfirmasi terpenuhi

# === TRAILING STOP ===
TRAILING_AKTIF    = True     # Aktifkan trailing stop
TRAILING_PIPS     = 2        # Geser SL setiap 2 pip profit tambahan

# ==========================================
# INDIKATOR LENGKAP
# ==========================================
def compute_indicators(df):

    # RSI multi-periode
    for span in [5, 7, 14]:
        diff = df['close'].diff()
        up   = diff.clip(lower=0)
        dn   = -diff.clip(upper=0)
        df[f'RSI_{span}'] = 100 - (100 / (1 + up.ewm(span=span, adjust=False).mean()
                                             / dn.ewm(span=span, adjust=False).mean()))

    # EMA stack
    for span in [5, 9, 21, 50, 100]:
        df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    df['EMA_cross_fast'] = df['EMA_5']  - df['EMA_9']
    df['EMA_cross_mid']  = df['EMA_9']  - df['EMA_21']
    df['EMA_cross_slow'] = df['EMA_21'] - df['EMA_50']
    df['vs_EMA100']      = df['close']  - df['EMA_100']

    # Stochastic
    lo = df['low'].rolling(14).min()
    hi = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - lo) / (hi - lo)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_cross'] = df['stoch_k'] - df['stoch_d']

    # ATR multi-periode
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR_7']     = tr.rolling(7).mean()
    df['ATR_14']    = tr.rolling(14).mean()
    df['ATR_ratio'] = df['ATR_7'] / df['ATR_14']   # volatilitas relatif jangka pendek

    # ADX
    up_move = df['high'].diff()
    dn_move = -df['low'].diff()
    pdm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    ndm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr14 = df['ATR_14'].replace(0, np.nan)
    pdi   = 100 * pd.Series(pdm, index=df.index).ewm(span=14, adjust=False).mean() / atr14
    ndi   = 100 * pd.Series(ndm, index=df.index).ewm(span=14, adjust=False).mean() / atr14
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df['ADX']     = dx.ewm(span=14, adjust=False).mean()
    df['DI_diff'] = pdi - ndi

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    df['MACD_cross']  = np.sign(df['MACD_hist']) - np.sign(df['MACD_hist'].shift(1))

    # Bollinger Bands
    bb_mid           = df['close'].rolling(20).mean()
    bb_std           = df['close'].rolling(20).std()
    df['BB_upper']   = bb_mid + 2 * bb_std
    df['BB_lower']   = bb_mid - 2 * bb_std
    df['BB_width']   = (df['BB_upper'] - df['BB_lower']) / bb_mid
    df['BB_pct']     = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_squeeze'] = df['BB_width'] / df['BB_width'].rolling(50).mean()

    # CCI
    tp     = (df['high'] + df['low'] + df['close']) / 3
    ma_tp  = tp.rolling(14).mean()
    md_tp  = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (tp - ma_tp) / (0.015 * md_tp)

    # Williams %R
    df['WillR'] = -100 * (df['high'].rolling(14).max() - df['close']) / \
                         (df['high'].rolling(14).max() - df['low'].rolling(14).min())

    # OBV
    if 'tick_volume' in df.columns:
        obv = [0]
        for i in range(1, len(df)):
            v = df['tick_volume'].iloc[i]
            obv.append(obv[-1] + v if df['close'].iloc[i] > df['close'].iloc[i-1]
                       else obv[-1] - v if df['close'].iloc[i] < df['close'].iloc[i-1]
                       else obv[-1])
        obv_s          = pd.Series(obv, index=df.index)
        df['OBV_diff'] = obv_s - obv_s.ewm(span=10, adjust=False).mean()
        df['vol_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
    else:
        df['OBV_diff']  = 0.0
        df['vol_ratio'] = 1.0

    df['vol_spike'] = (df['vol_ratio'] > 1.8).astype(int)

    # VWAP proxy
    df['vwap']    = ((df['high'] + df['low'] + df['close']) / 3).rolling(20).mean()
    df['vs_vwap'] = df['close'] - df['vwap']

    # Market Structure (HH/LL)
    df['HH'] = df['high'].rolling(5).max()
    df['LL']  = df['low'].rolling(5).min()
    df['structure_bull'] = ((df['high'] > df['HH'].shift(1)) & (df['low'] > df['LL'].shift(1))).astype(int)
    df['structure_bear'] = ((df['high'] < df['HH'].shift(1)) & (df['low'] < df['LL'].shift(1))).astype(int)
    df.drop(columns=['HH','LL'], inplace=True, errors='ignore')

    # Pin Bar
    body    = (df['close'] - df['open']).abs()
    up_wick = df['high'] - df[['close','open']].max(axis=1)
    lo_wick = df[['close','open']].min(axis=1) - df['low']
    df['pin_bull']  = ((lo_wick > body * 2) & (lo_wick > up_wick * 2)).astype(int)
    df['pin_bear']  = ((up_wick > body * 2) & (up_wick > lo_wick * 2)).astype(int)

    # Candle properties
    df['body']      = df['close'] - df['open']
    df['up_wick']   = up_wick
    df['lo_wick']   = lo_wick
    df['body_ratio'] = body / (df['high'] - df['low']).replace(0, np.nan)

    # Momentum multi-periode
    for p in [2, 3, 5, 8]:
        df[f'mom_{p}'] = df['close'] - df['close'].shift(p)

    # Rate of Change
    df['ROC_5']  = df['close'].pct_change(5) * 100
    df['ROC_10'] = df['close'].pct_change(10) * 100

    return df

# ==========================================
# FEATURES (36 fitur)
# ==========================================
FEATURES = [
    # RSI multi
    'RSI_5', 'RSI_7', 'RSI_14',
    # EMA
    'EMA_cross_fast', 'EMA_cross_mid', 'EMA_cross_slow', 'vs_EMA100',
    # Stochastic
    'stoch_k', 'stoch_d', 'stoch_cross',
    # Volatilitas
    'ATR_7', 'ATR_14', 'ATR_ratio',
    # Trend strength
    'ADX', 'DI_diff',
    # MACD
    'MACD_hist', 'MACD_cross',
    # Bollinger
    'BB_width', 'BB_pct', 'BB_squeeze',
    # Oscillator
    'CCI', 'WillR',
    # Volume
    'vol_ratio', 'OBV_diff', 'vol_spike',
    # VWAP
    'vs_vwap',
    # Price Action
    'body', 'up_wick', 'lo_wick', 'body_ratio',
    'structure_bull', 'structure_bear', 'pin_bull', 'pin_bear',
    # Momentum
    'mom_2', 'mom_3', 'mom_5', 'mom_8',
    'ROC_5', 'ROC_10'
]

# ==========================================
# DATA PIPELINE
# ==========================================
def get_data(n=4000):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = compute_indicators(df)
    move = df['close'].shift(-1) - df['close']
    df['Target'] = np.where(move >  0.15,  1,
                   np.where(move < -0.15,  0, -1))
    df = df[df['Target'] != -1].copy()
    df.dropna(inplace=True)
    return df

# ==========================================
# ENSEMBLE MODEL (3 model digabung)
# ==========================================
def train_model(df):
    """
    Gabungkan 3 model berbeda:
    1. Random Forest     — bagus untuk pola non-linear
    2. Gradient Boosting — bagus untuk akurasi
    3. Logistic Reg      — bagus untuk probabilitas kalibasi
    Voting majority → sinyal lebih robust
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    y = df['Target']

    rf  = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, class_weight='balanced',
            random_state=42, n_jobs=-1)

    gb  = GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.05, random_state=42)

    lr  = CalibratedClassifierCV(
            LogisticRegression(max_iter=500, class_weight='balanced', random_state=42))

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft',    # rata-rata probabilitas
        weights=[3, 2, 1] # RF paling dipercaya
    )
    ensemble.fit(X, y)
    print(f"  → Ensemble: RF + GradBoost + LogReg | {len(FEATURES)} fitur | {len(df)} sample")
    return ensemble, scaler

# ==========================================
# FILTER SPREAD
# ==========================================
def get_spread():
    tick = mt5.symbol_info_tick(SYMBOL)
    sym  = mt5.symbol_info(SYMBOL)
    if not tick or not sym: return 999
    return int((tick.ask - tick.bid) / sym.point)

# ==========================================
# FILTER KUALITAS PASAR
# ==========================================
def is_market_valid(r):
    spread = get_spread()
    if spread > MAX_SPREAD_POINTS:
        return False, f"Spread terlalu lebar: {spread} pts"
    if r['ADX'] < ADX_MIN:
        return False, f"ADX lemah: {r['ADX']:.1f}"
    if r['ATR_ratio'] > 2.8:
        return False, f"Volatilitas ekstrem: {r['ATR_ratio']:.2f}"
    return True, "OK"

# ==========================================
# SCORING KONFIRMASI
# ==========================================
def score_buy(r):
    return sum([
        r['EMA_cross_mid']   >  0,
        r['EMA_cross_slow']  >  0,
        r['MACD_hist']       >  0,
        r['DI_diff']         >  0,
        r['stoch_k']         < 40,
        r['stoch_cross']     >  0,
        r['vs_vwap']         >  0,
        r['RSI_7']           < 60,
        r['CCI']             < -50,
        r['BB_pct']          < 0.4,
        r['mom_3']           >  0,
        r['structure_bull']  == 1,
        r['pin_bull']        == 1,
        r['vol_spike']       == 1,
    ])

def score_sell(r):
    return sum([
        r['EMA_cross_mid']   <  0,
        r['EMA_cross_slow']  <  0,
        r['MACD_hist']       <  0,
        r['DI_diff']         <  0,
        r['stoch_k']         > 60,
        r['stoch_cross']     <  0,
        r['vs_vwap']         <  0,
        r['RSI_7']           > 40,
        r['CCI']             > 50,
        r['BB_pct']          > 0.6,
        r['mom_3']           <  0,
        r['structure_bear']  == 1,
        r['pin_bear']        == 1,
        r['vol_spike']       == 1,
    ])

# ==========================================
# EKSEKUSI MULTI ORDER
# ==========================================
def open_multi_orders(direction, atr_val):
    sym  = mt5.symbol_info(SYMBOL)
    if not sym: return []
    tick    = mt5.symbol_info_tick(SYMBOL)
    pt      = sym.point
    tickets = []

    # SL dinamis berdasarkan ATR
    sl_pts = max(200, int(atr_val * 2 / pt))

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
            "comment":      f"PowerScalp#{i+1}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            tickets.append(res.order)
            print(f"  ✅ Order #{i+1} | Ticket:{res.order} | {direction} @ {price:.2f} | SL:{sl_pts}pts")
        else:
            code = res.retcode if res else "None"
            print(f"  ❌ Order #{i+1} GAGAL | Kode:{code}")
            if code == 10027:
                print("  ⚠️  Aktifkan 'Algo Trading' di toolbar MT5!")
        time.sleep(0.05)

    return tickets

# ==========================================
# TRAILING STOP
# ==========================================
def apply_trailing_stop(positions):
    if not TRAILING_AKTIF: return
    sym  = mt5.symbol_info(SYMBOL)
    if not sym: return
    pt   = sym.point
    tick = mt5.symbol_info_tick(SYMBOL)

    for pos in positions:
        if pos.magic != MAGIC: continue
        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - TRAILING_PIPS * pt * 10
            if new_sl > pos.sl + pt:   # SL hanya boleh naik
                mt5.order_send({
                    "action":   mt5.TRADE_ACTION_SLTP,
                    "symbol":   SYMBOL,
                    "position": pos.ticket,
                    "sl":       new_sl,
                    "tp":       pos.tp,
                })
        else:
            new_sl = tick.ask + TRAILING_PIPS * pt * 10
            if new_sl < pos.sl - pt:   # SL hanya boleh turun
                mt5.order_send({
                    "action":   mt5.TRADE_ACTION_SLTP,
                    "symbol":   SYMBOL,
                    "position": pos.ticket,
                    "sl":       new_sl,
                    "tp":       pos.tp,
                })

# ==========================================
# FORCE CLOSE
# ==========================================
def force_close_order(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    return mt5.order_send({
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     pos.ticket,
        "price":        close_price,
        "magic":        MAGIC,
        "comment":      "FC",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })

def force_close_all(reason=""):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return 0
    total = 0
    closed = 0
    for pos in positions:
        if pos.magic != MAGIC: continue
        res = force_close_order(pos)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            total  += pos.profit
            closed += 1
    if closed:
        emoji = "✅" if total >= 0 else "⛔"
        print(f"  🔒 CLOSE {closed} order | ${total:.2f} {emoji} | {reason}")
    return total

# ==========================================
# MONITOR POSISI (TICK-LEVEL)
# ==========================================
def monitor_and_close(open_time):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return False
    my_pos = [p for p in positions if p.magic == MAGIC]
    if not my_pos: return False

    sym          = mt5.symbol_info(SYMBOL)
    pt           = sym.point if sym else 0.01
    tick         = mt5.symbol_info_tick(SYMBOL)
    total_profit = sum(p.profit for p in my_pos)
    elapsed      = (datetime.now(timezone.utc) - open_time).total_seconds()

    # Kondisi close semua
    if total_profit >= TARGET_PROFIT_USD:
        force_close_all(f"Target profit ${TARGET_PROFIT_USD} ✅")
        return False
    if total_profit <= -MAX_LOSS_USD:
        force_close_all(f"Max loss -${MAX_LOSS_USD} ⛔")
        return False
    if elapsed >= MAX_HOLD_DETIK:
        force_close_all(f"Timeout {MAX_HOLD_DETIK}s ⏱️")
        return False

    # Trailing stop
    apply_trailing_stop(my_pos)

    # Close per order jika sudah profit
    for pos in my_pos:
        pip_profit = ((tick.bid - pos.price_open) if pos.type == mt5.ORDER_TYPE_BUY
                      else (pos.price_open - tick.ask)) / pt
        if pip_profit >= TARGET_PIPS:
            res = force_close_order(pos)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  💰 #{pos.ticket} +{pip_profit:.1f}pip | ${pos.profit:.2f}")

    print(f"  📊 {len(my_pos)} posisi | P/L:${total_profit:.2f} | {elapsed:.0f}s "
          f"| Spread:{get_spread()}pts")
    return True

# ==========================================
# SINYAL ENTRY
# ==========================================
def get_signal(model, scaler, n=300):
    live = get_data(n)
    if live is None or len(live) < 10: return None, None, None
    row   = live.iloc[-1]
    X     = scaler.transform(live[FEATURES].iloc[[-1]])
    probs = model.predict_proba(X)[0]
    cls   = list(model.classes_)
    p_up  = probs[cls.index(1)] if 1 in cls else 0
    p_dn  = probs[cls.index(0)] if 0 in cls else 0
    return p_up, p_dn, row

# ==========================================
# STATISTIK SESI
# ==========================================
class SessionStats:
    def __init__(self):
        self.wins = 0; self.losses = 0; self.total_pnl = 0.0
        self.trades = 0

    def record(self, pnl):
        self.trades += 1
        self.total_pnl += pnl
        if pnl > 0: self.wins += 1
        else: self.losses += 1

    def print(self):
        wr = (self.wins / self.trades * 100) if self.trades > 0 else 0
        print(f"\n  📈 STATISTIK SESI")
        print(f"  Trade  : {self.trades} | W:{self.wins} L:{self.losses}")
        print(f"  Winrate: {wr:.1f}%")
        print(f"  Total P/L: ${self.total_pnl:.2f}")

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ Gagal inisialisasi MT5!"); quit()

    print("╔" + "═"*52 + "╗")
    print("║       POWER SCALPER PRO — XAUUSD M1           ║")
    print("║  Ensemble AI: RF + GradBoost + LogReg          ║")
    print(f"║  Order  : {JUMLAH_ORDER} x {LOT_PER_ORDER} lot                         ║")
    print(f"║  Target : ${TARGET_PROFIT_USD} profit / {TARGET_PIPS} pip per order         ║")
    print(f"║  Max Loss: ${MAX_LOSS_USD} | Timeout: {MAX_HOLD_DETIK}s                ║")
    print(f"║  Spread Filter: >{MAX_SPREAD_POINTS} pts skip                  ║")
    print(f"║  Trailing Stop: {'ON' if TRAILING_AKTIF else 'OFF'} ({TRAILING_PIPS} pip)                    ║")
    print("╚" + "═"*52 + "╝\n")

    model     = None
    scaler    = None
    it        = 0
    in_trade  = False
    open_time = None
    stats     = SessionStats()

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Retrain setiap 90 iterasi (~15 menit)
            if it % 90 == 0:
                print(f"\n[{now:%H:%M:%S}] 🔄 Training Ensemble Model...")
                data = get_data(4000)
                if data is not None and len(data) > 300:
                    model, scaler = train_model(data)
                else:
                    print("  → Data tidak cukup.")

            if model is None:
                time.sleep(5); it += 1; continue

            # Monitor posisi aktif — cek setiap 1 detik
            if in_trade:
                still_open = monitor_and_close(open_time)
                if not still_open:
                    in_trade = False
                    stats.print()
                time.sleep(1); it += 1; continue

            # Cari sinyal entry
            p_up, p_dn, row = get_signal(model, scaler)
            if row is None:
                time.sleep(5); it += 1; continue

            # Validasi pasar
            valid, reason = is_market_valid(row)
            if not valid:
                print(f"[{now:%H:%M:%S}] ⏸️  Skip — {reason}")
                time.sleep(5); it += 1; continue

            sc_buy  = score_buy(row)
            sc_sell = score_sell(row)

            # Entry BUY
            if p_up > THRESHOLD_ENTRY and sc_buy >= SCORE_MIN:
                print(f"\n[{now:%H:%M:%S}] 🚀 BUY | AI:{p_up*100:.1f}% | Score:{sc_buy}/14 | ADX:{row['ADX']:.1f}")
                print(f"  RSI:{row['RSI_7']:.1f} | MACD:{row['MACD_hist']:.3f} | "
                      f"Stoch:{row['stoch_k']:.1f} | CCI:{row['CCI']:.1f} | Spread:{get_spread()}pts")
                tickets = open_multi_orders('BUY', row['ATR_14'])
                if tickets:
                    in_trade  = True
                    open_time = datetime.now(timezone.utc)

            # Entry SELL
            elif p_dn > THRESHOLD_ENTRY and sc_sell >= SCORE_MIN:
                print(f"\n[{now:%H:%M:%S}] 📉 SELL | AI:{p_dn*100:.1f}% | Score:{sc_sell}/14 | ADX:{row['ADX']:.1f}")
                print(f"  RSI:{row['RSI_7']:.1f} | MACD:{row['MACD_hist']:.3f} | "
                      f"Stoch:{row['stoch_k']:.1f} | CCI:{row['CCI']:.1f} | Spread:{get_spread()}pts")
                tickets = open_multi_orders('SELL', row['ATR_14'])
                if tickets:
                    in_trade  = True
                    open_time = datetime.now(timezone.utc)

            else:
                print(f"[{now:%H:%M:%S}] ⏳ UP:{p_up*100:.1f}% DN:{p_dn*100:.1f}% "
                      f"| BuyScore:{sc_buy} SellScore:{sc_sell} "
                      f"| ADX:{row['ADX']:.1f} | Spread:{get_spread()}pts")

            it += 1
            time.sleep(5)   # lebih cepat dari sebelumnya (5 detik)

    except KeyboardInterrupt:
        print("\n⚠️  Bot dihentikan.")
        force_close_all("Manual stop")
        stats.print()
    finally:
        mt5.shutdown()
        print("MT5 disconnected.")