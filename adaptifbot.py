import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import threading
import time
import json
import os
from datetime import datetime, timezone
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║      ADAPTIVE LEARNING SCALPER — XAUUSD M1                  ║
# ║  Belajar dari setiap trade, semakin pintar setiap sesi       ║
# ║  Memory → Feedback Loop → Adaptive Threshold → Self-Improve ║
# ╚══════════════════════════════════════════════════════════════╝

# ==========================================
# KONFIGURASI
# ==========================================
SYMBOL        = "XAUUSD"
TIMEFRAME     = mt5.TIMEFRAME_M1
MAGIC         = 707070

JUMLAH_ORDER  = 3
LOT_PER_ORDER = 5.00

TARGET_PROFIT_USD = 0.01
TARGET_PIPS       = 1
MAX_LOSS_USD      = 5.0
MAX_HOLD_DETIK    = 45
MAX_SPREAD        = 25

# Threshold awal — akan berubah otomatis
THRESHOLD_BASE    = 0.65
THRESHOLD_MIN     = 0.58     # batas bawah threshold
THRESHOLD_MAX     = 0.80     # batas atas threshold
MIN_ACCURACY      = 0.53

MONITOR_INTERVAL  = 0.2
SEARCH_INTERVAL   = 3

# Memory file — simpan riwayat trade permanen
MEMORY_FILE       = "trade_memory.json"

# Jumlah trade terakhir untuk evaluasi adaptif
WINDOW_EVAL       = 20       # evaluasi 20 trade terakhir

# ==========================================
# MEMORI TRADE — INTI PEMBELAJARAN
# ==========================================
class TradeMemory:
    """
    Menyimpan setiap trade beserta:
    - Kondisi pasar saat entry (fitur indikator)
    - Arah yang dipilih (BUY/SELL)
    - Hasil (profit/loss)
    - Timestamp

    Data ini digunakan untuk:
    1. Retrain model dengan contoh nyata dari trading
    2. Hitung win rate adaptif
    3. Sesuaikan threshold secara otomatis
    """
    def __init__(self, file=MEMORY_FILE):
        self.file    = file
        self.records = deque(maxlen=5000)   # simpan max 5000 trade terakhir
        self.load()

    def save_entry(self, features_dict, direction, entry_price, timestamp):
        """Simpan kondisi saat entry."""
        record = {
            "timestamp":   timestamp.isoformat(),
            "direction":   direction,
            "entry_price": entry_price,
            "features":    features_dict,
            "result":      None,    # diisi setelah close
            "pnl":         None,
        }
        self.records.append(record)
        return len(self.records) - 1   # return index

    def save_result(self, pnl):
        """Update hasil trade terakhir setelah close."""
        if not self.records: return
        last = self.records[-1]
        last["result"] = "WIN" if pnl > 0 else "LOSS"
        last["pnl"]    = pnl
        self._persist()

    def get_recent_trades(self, n=WINDOW_EVAL):
        """Ambil n trade terakhir yang sudah ada hasilnya."""
        done = [r for r in self.records if r["result"] is not None]
        return done[-n:]

    def win_rate(self, n=WINDOW_EVAL):
        trades = self.get_recent_trades(n)
        if not trades: return 0.5
        wins = sum(1 for t in trades if t["result"] == "WIN")
        return wins / len(trades)

    def get_loss_patterns(self):
        """
        Ambil kondisi saat LOSS — ini yang harus dihindari model.
        Return sebagai DataFrame untuk retraining.
        """
        losses = [r for r in self.records
                  if r["result"] == "LOSS" and r["features"] is not None]
        if not losses: return None
        rows = []
        for r in losses:
            row = r["features"].copy()
            row["Target"] = 0 if r["direction"] == "BUY" else 1
            rows.append(row)
        return pd.DataFrame(rows)

    def get_win_patterns(self):
        """Ambil kondisi saat WIN untuk diperkuat."""
        wins = [r for r in self.records
                if r["result"] == "WIN" and r["features"] is not None]
        if not wins: return None
        rows = []
        for r in wins:
            row = r["features"].copy()
            row["Target"] = 1 if r["direction"] == "BUY" else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def total_pnl(self):
        done = [r for r in self.records if r["pnl"] is not None]
        return sum(r["pnl"] for r in done)

    def stats(self):
        done  = [r for r in self.records if r["result"] is not None]
        total = len(done)
        if total == 0: return "Belum ada trade"
        wins  = sum(1 for r in done if r["result"] == "WIN")
        pnl   = sum(r["pnl"] for r in done)
        wr    = wins / total * 100
        return (f"Total:{total} | W:{wins} L:{total-wins} | "
                f"WR:{wr:.1f}% | P/L:${pnl:.2f}")

    def _persist(self):
        """Simpan ke file JSON agar tidak hilang saat restart."""
        try:
            data = list(self.records)
            with open(self.file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Gagal simpan memory: {e}")

    def load(self):
        """Load riwayat trade dari file jika ada."""
        if not os.path.exists(self.file): return
        try:
            with open(self.file, 'r') as f:
                data = json.load(f)
            self.records = deque(data, maxlen=5000)
            done = sum(1 for r in self.records if r["result"])
            print(f"  📂 Memory loaded: {len(self.records)} trade "
                  f"({done} selesai) dari {self.file}")
        except Exception as e:
            print(f"  ⚠️  Gagal load memory: {e}")

# ==========================================
# ADAPTIVE THRESHOLD
# ==========================================
class AdaptiveThreshold:
    """
    Threshold naik otomatis saat win rate buruk,
    turun saat win rate bagus.
    Tujuan: selektif saat performa jelek,
            lebih aktif saat performa bagus.
    """
    def __init__(self, base=THRESHOLD_BASE):
        self.current = base
        self.history = []

    def update(self, win_rate, n_trades):
        if n_trades < 5:
            return   # butuh minimal 5 trade dulu

        old = self.current

        if win_rate < 0.35:
            # Win rate sangat buruk → naikkan threshold drastis
            self.current = min(self.current + 0.04, THRESHOLD_MAX)
            reason = f"WR sangat buruk {win_rate*100:.1f}%"
        elif win_rate < 0.45:
            # Win rate buruk → naikkan threshold
            self.current = min(self.current + 0.02, THRESHOLD_MAX)
            reason = f"WR buruk {win_rate*100:.1f}%"
        elif win_rate > 0.65:
            # Win rate bagus → turunkan sedikit (lebih aktif)
            self.current = max(self.current - 0.01, THRESHOLD_MIN)
            reason = f"WR bagus {win_rate*100:.1f}%"
        elif win_rate > 0.55:
            # Win rate cukup baik → pertahankan
            reason = f"WR stabil {win_rate*100:.1f}%"
        else:
            reason = f"WR normal {win_rate*100:.1f}%"

        if self.current != old:
            print(f"  🎯 Threshold: {old:.2f} → {self.current:.2f} | {reason}")

        self.history.append({
            "threshold": self.current,
            "win_rate":  win_rate,
            "time":      datetime.now(timezone.utc).isoformat()
        })

    def get(self):
        return self.current

# ==========================================
# INDIKATOR + FITUR DELTA
# ==========================================
def compute_features(df):
    # RSI delta
    diff = df['close'].diff()
    up, dn = diff.clip(lower=0), -diff.clip(upper=0)
    rsi = 100 - (100 / (1 + up.ewm(span=7, adjust=False).mean()
                           / dn.ewm(span=7, adjust=False).mean()))
    df['RSI']        = rsi
    df['RSI_delta']  = rsi - rsi.shift(1)
    df['RSI_delta2'] = rsi - rsi.shift(3)

    # EMA slope
    for s in [5, 9, 21, 50]:
        df[f'EMA_{s}'] = df['close'].ewm(span=s, adjust=False).mean()
    df['EMA_fast']      = df['EMA_5']  - df['EMA_9']
    df['EMA_mid']       = df['EMA_9']  - df['EMA_21']
    df['EMA9_slope']    = df['EMA_9']  - df['EMA_9'].shift(2)
    df['EMA21_slope']   = df['EMA_21'] - df['EMA_21'].shift(2)
    df['EMA_gap']       = df['EMA_9']  - df['EMA_21']
    df['EMA_gap_delta'] = df['EMA_gap'] - df['EMA_gap'].shift(1)
    df['vs_EMA50']      = df['close']  - df['EMA_50']

    # MACD delta
    ema8  = df['close'].ewm(span=8,  adjust=False).mean()
    ema17 = df['close'].ewm(span=17, adjust=False).mean()
    macd  = ema8 - ema17
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    df['MACD_hist']       = hist
    df['MACD_hist_delta'] = hist - hist.shift(1)
    df['MACD_cross']      = np.sign(hist) - np.sign(hist.shift(1))

    # Stochastic slope
    lo = df['low'].rolling(8).min()
    hi = df['high'].rolling(8).max()
    stoch = 100 * (df['close'] - lo) / (hi - lo)
    df['stoch_k']     = stoch
    df['stoch_slope'] = stoch - stoch.shift(2)

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR']       = tr.rolling(7).mean()
    df['ATR_ratio'] = tr / df['ATR']

    # ADX
    up_m = df['high'].diff()
    dn_m = -df['low'].diff()
    pdm  = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    ndm  = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr  = df['ATR'].replace(0, np.nan)
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(span=14, adjust=False).mean() / atr
    ndi  = 100 * pd.Series(ndm, index=df.index).ewm(span=14, adjust=False).mean() / atr
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df['ADX']     = dx.ewm(span=14, adjust=False).mean()
    df['DI_diff'] = pdi - ndi

    # Candle pattern
    df['candle_dir']  = np.sign(df['close'] - df['open'])
    body    = (df['close'] - df['open']).abs()
    up_wick = df['high'] - df[['close','open']].max(axis=1)
    lo_wick = df[['close','open']].min(axis=1) - df['low']
    rng     = (df['high'] - df['low']).replace(0, np.nan)
    df['body_ratio']    = body    / rng
    df['up_wick_ratio'] = up_wick / rng
    df['lo_wick_ratio'] = lo_wick / rng
    df['dir_consistency'] = (df['candle_dir'] +
                              df['candle_dir'].shift(1) +
                              df['candle_dir'].shift(2))

    # Volume
    if 'tick_volume' in df.columns:
        df['vol_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(10).mean()
        df['vol_delta'] = df['tick_volume'] - df['tick_volume'].shift(1)
    else:
        df['vol_ratio'] = 1.0
        df['vol_delta'] = 0.0

    # Posisi harga
    df['pct_from_high5'] = (df['close'] - df['high'].rolling(5).max()) / df['close']
    df['pct_from_low5']  = (df['close'] - df['low'].rolling(5).min())  / df['close']

    # Momentum
    df['mom_2'] = df['close'] - df['close'].shift(2)
    df['mom_5'] = df['close'] - df['close'].shift(5)

    return df

FEATURES = [
    'RSI', 'RSI_delta', 'RSI_delta2',
    'EMA_fast', 'EMA_mid', 'EMA9_slope', 'EMA21_slope',
    'EMA_gap', 'EMA_gap_delta', 'vs_EMA50',
    'MACD_hist', 'MACD_hist_delta', 'MACD_cross',
    'stoch_k', 'stoch_slope',
    'ATR', 'ATR_ratio',
    'ADX', 'DI_diff',
    'candle_dir', 'body_ratio', 'up_wick_ratio', 'lo_wick_ratio',
    'dir_consistency',
    'vol_ratio', 'vol_delta',
    'pct_from_high5', 'pct_from_low5',
    'mom_2', 'mom_5',
]

# ==========================================
# DATA PIPELINE
# ==========================================
def get_data(n=3000):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = compute_features(df)
    move = df['close'].shift(-1) - df['close']
    df['Target'] = np.where(move >  0.10,  1,
                   np.where(move < -0.10,  0, -1))
    df = df[df['Target'] != -1].copy()
    df.dropna(inplace=True)
    return df

# ==========================================
# TRAIN MODEL + FEEDBACK DARI MEMORY
# ==========================================
def train_model(base_df, memory: TradeMemory):
    """
    Training dengan 3 lapis data:
    1. Data historis MT5 (ribuan bar)
    2. Pola WIN dari trade nyata (diperkuat 3x)
    3. Pola LOSS dari trade nyata (dibalik & diperkuat 2x)
    """
    frames = [base_df[FEATURES + ['Target']].copy()]

    # Tambahkan pola WIN — diperkuat 3x
    win_df = memory.get_win_patterns()
    if win_df is not None and len(win_df) >= 3:
        # Filter hanya kolom yang ada di FEATURES
        cols = [c for c in FEATURES if c in win_df.columns]
        win_df = win_df[cols + ['Target']].dropna()
        # Duplikasi 3x untuk memberi bobot lebih
        for _ in range(3):
            frames.append(win_df)
        print(f"  📗 WIN patterns: {len(win_df)} sample (bobot 3x)")

    # Tambahkan pola LOSS — dengan label dibalik (ajarkan apa yang TIDAK boleh)
    loss_df = memory.get_loss_patterns()
    if loss_df is not None and len(loss_df) >= 3:
        cols = [c for c in FEATURES if c in loss_df.columns]
        loss_df = loss_df[cols + ['Target']].dropna()
        for _ in range(2):
            frames.append(loss_df)
        print(f"  📕 LOSS patterns: {len(loss_df)} sample (bobot 2x, label dibalik)")

    # Gabung semua data
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(combined[FEATURES])
    y = combined['Target'].values

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X, y)

    # Cross validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = scores.mean()

    # Feature importance top 5
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    top5 = importance.nlargest(5).index.tolist()
    print(f"  → Akurasi: {accuracy*100:.1f}% | "
          f"Data: {len(combined)} sample | "
          f"Top: {', '.join(top5)}")

    return model, scaler, accuracy

# ==========================================
# CEK SPREAD
# ==========================================
def get_spread():
    tick = mt5.symbol_info_tick(SYMBOL)
    sym  = mt5.symbol_info(SYMBOL)
    if not tick or not sym: return 999
    return round((tick.ask - tick.bid) / sym.point)

# ==========================================
# FORCE CLOSE
# ==========================================
def force_close_one(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    return mt5.order_send({
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       pos.volume,
        "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "position":     pos.ticket,
        "price":        tick.bid if pos.type == 0 else tick.ask,
        "magic":        MAGIC,
        "comment":      "FC",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })

def force_close_all(reason=""):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return 0.0
    total = 0.0
    for pos in positions:
        if pos.magic != MAGIC: continue
        res = force_close_one(pos)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            total += pos.profit
    sign = "✅" if total >= 0 else "⛔"
    print(f"  🔒 CLOSE | ${total:.4f} {sign} | {reason}")
    return total

# ==========================================
# THREAD MONITOR POSISI
# ==========================================
class PositionMonitor(threading.Thread):
    def __init__(self, memory: TradeMemory):
        super().__init__(daemon=True)
        self.memory    = memory
        self.active    = False
        self.open_time = None
        self._lock     = threading.Lock()
        self.last_pnl  = 0.0

    def start_trade(self):
        with self._lock:
            self.active    = True
            self.open_time = datetime.now(timezone.utc)
            self.last_pnl  = 0.0

    def end_trade(self, pnl):
        with self._lock:
            self.active   = False
            self.last_pnl = pnl
        # Simpan hasil ke memory — inilah feedback loop
        self.memory.save_result(pnl)
        result = "✅ WIN" if pnl > 0 else "⛔ LOSS"
        print(f"  📝 Trade selesai: {result} ${pnl:.4f} | {self.memory.stats()}")

    def is_active(self):
        with self._lock:
            return self.active

    def run(self):
        sym = mt5.symbol_info(SYMBOL)
        pt  = sym.point if sym else 0.01

        while True:
            if not self.active:
                time.sleep(0.1)
                continue

            try:
                positions = mt5.positions_get(symbol=SYMBOL)
                if not positions:
                    self.end_trade(0.0)
                    continue

                my_pos = [p for p in positions if p.magic == MAGIC]
                if not my_pos:
                    self.end_trade(0.0)
                    continue

                tick         = mt5.symbol_info_tick(SYMBOL)
                total_profit = sum(p.profit for p in my_pos)
                elapsed      = (datetime.now(timezone.utc) - self.open_time).total_seconds()

                # Close semua — profit tercapai
                if total_profit >= TARGET_PROFIT_USD:
                    pnl = force_close_all(f"PROFIT ${total_profit:.4f}")
                    self.end_trade(total_profit)
                    continue

                # Cut loss
                if total_profit <= -MAX_LOSS_USD:
                    force_close_all(f"CUT LOSS ${total_profit:.2f}")
                    self.end_trade(total_profit)
                    continue

                # Timeout
                if elapsed >= MAX_HOLD_DETIK:
                    force_close_all(f"TIMEOUT {elapsed:.0f}s")
                    self.end_trade(total_profit)
                    continue

                # Close per order yang sudah profit
                for pos in my_pos:
                    pip = ((tick.bid - pos.price_open) if pos.type == 0
                           else (pos.price_open - tick.ask)) / pt
                    if pip >= TARGET_PIPS:
                        res = force_close_one(pos)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"  💰 #{pos.ticket} +{pip:.1f}pip ${pos.profit:.4f}")

            except Exception as e:
                print(f"  ⚠️ Monitor error: {e}")

            time.sleep(MONITOR_INTERVAL)

# ==========================================
# BUKA ORDER
# ==========================================
def open_orders(direction, atr):
    sym  = mt5.symbol_info(SYMBOL)
    if not sym: return []
    tick = mt5.symbol_info_tick(SYMBOL)
    pt   = sym.point
    sl_pts = max(150, int(atr * 2 / pt))
    tickets = []

    for i in range(JUMLAH_ORDER):
        price = tick.ask if direction == 'BUY' else tick.bid
        sl = (price - sl_pts * pt) if direction == 'BUY' else (price + sl_pts * pt)
        tp = (price + sl_pts * pt) if direction == 'BUY' else (price - sl_pts * pt)

        res = mt5.order_send({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       LOT_PER_ORDER,
            "type":         mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "magic":        MAGIC,
            "comment":      f"ALS#{i+1}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            tickets.append(res.order)
            print(f"  ✅ #{i+1} {direction} @ {price:.2f}")
        else:
            code = res.retcode if res else "?"
            print(f"  ❌ #{i+1} GAGAL kode:{code}")
            if code == 10027:
                print("  ⚠️  Aktifkan Algo Trading di MT5!")
        time.sleep(0.05)
    return tickets

# ==========================================
# SINYAL + KONFIRMASI
# ==========================================
def get_signal(model, scaler):
    live = get_data(300)
    if live is None or len(live) < 5:
        return None, None, None
    row   = live.iloc[-1]
    X     = scaler.transform(live[FEATURES].iloc[[-1]])
    probs = model.predict_proba(X)[0]
    cls   = list(model.classes_)
    p_up  = probs[cls.index(1)] if 1 in cls else 0
    p_dn  = probs[cls.index(0)] if 0 in cls else 0
    return p_up, p_dn, row

def row_to_dict(row):
    """Konversi row pandas ke dict untuk disimpan di memory."""
    return {f: float(row[f]) for f in FEATURES if f in row.index}

def confirm_buy(r):
    return sum([
        r['RSI_delta']        >  0,
        r['RSI_delta2']       >  0,
        r['MACD_hist_delta']  >  0,
        r['EMA_gap_delta']    >  0,
        r['stoch_slope']      >  0,
        r['DI_diff']          >  0,
        r['dir_consistency']  >  0,
        r['lo_wick_ratio']    > 0.3,
        r['vol_delta']        >  0,
        r['pct_from_low5']    < 0.005,
    ]) >= 5

def confirm_sell(r):
    return sum([
        r['RSI_delta']        <  0,
        r['RSI_delta2']       <  0,
        r['MACD_hist_delta']  <  0,
        r['EMA_gap_delta']    <  0,
        r['stoch_slope']      <  0,
        r['DI_diff']          <  0,
        r['dir_consistency']  <  0,
        r['up_wick_ratio']    > 0.3,
        r['vol_delta']        >  0,
        r['pct_from_high5']   > -0.005,
    ]) >= 5

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ MT5 gagal!"); quit()

    print("╔" + "═"*56 + "╗")
    print("║     ADAPTIVE LEARNING SCALPER — XAUUSD M1        ║")
    print("║  Belajar dari setiap trade, semakin pintar        ║")
    print(f"║  Order   : {JUMLAH_ORDER} x {LOT_PER_ORDER} lot                          ║")
    print(f"║  Target  : ${TARGET_PROFIT_USD} / {TARGET_PIPS} pip → langsung close         ║")
    print(f"║  Threshold awal: {THRESHOLD_BASE} (adaptif {THRESHOLD_MIN}~{THRESHOLD_MAX})       ║")
    print(f"║  Memory  : {MEMORY_FILE}                    ║")
    print("╚" + "═"*56 + "╝\n")

    # Inisialisasi komponen
    memory    = TradeMemory()
    adaptive  = AdaptiveThreshold()
    monitor   = PositionMonitor(memory)
    monitor.start()

    model    = None
    scaler   = None
    accuracy = 0.0
    it       = 0

    # Tampilkan statistik dari sesi sebelumnya
    stats_str = memory.stats()
    if stats_str != "Belum ada trade":
        print(f"  📊 Riwayat sebelumnya: {stats_str}\n")

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Retrain setiap 100 iterasi ATAU setelah setiap 5 trade selesai
            recent = memory.get_recent_trades(5)
            need_retrain = (it % 100 == 0) or \
                           (len(recent) > 0 and
                            recent[-1]["result"] is not None and
                            it > 0)

            if need_retrain:
                print(f"\n[{now:%H:%M:%S}] 🔄 Retrain dengan feedback memory...")
                base_data = get_data(3000)
                if base_data is not None and len(base_data) > 300:
                    model, scaler, accuracy = train_model(base_data, memory)

                    # Update threshold berdasarkan win rate
                    wr = memory.win_rate(WINDOW_EVAL)
                    n  = len(memory.get_recent_trades(WINDOW_EVAL))
                    adaptive.update(wr, n)
                else:
                    print("  → Data tidak cukup.")

            if model is None:
                time.sleep(3); it += 1; continue

            # Tunggu jika ada posisi aktif
            if monitor.is_active():
                time.sleep(0.2); it += 1; continue

            # Cek spread
            spread = get_spread()
            if spread > MAX_SPREAD:
                print(f"[{now:%H:%M:%S}] ⏸️  Spread {spread}pts")
                time.sleep(3); it += 1; continue

            # Cek akurasi
            if accuracy < MIN_ACCURACY:
                print(f"[{now:%H:%M:%S}] ⚠️  Akurasi {accuracy*100:.1f}% rendah, skip.")
                time.sleep(5); it += 1; continue

            # Ambil sinyal
            p_up, p_dn, row = get_signal(model, scaler)
            if row is None:
                time.sleep(3); it += 1; continue

            # ADX filter
            if row['ADX'] < 20:
                print(f"[{now:%H:%M:%S}] ⏸️  ADX:{row['ADX']:.1f} | Spread:{spread}")
                time.sleep(3); it += 1; continue

            threshold = adaptive.get()
            wr_now    = memory.win_rate(WINDOW_EVAL)

            # ── Entry BUY ──
            if p_up > threshold and confirm_buy(row):
                print(f"\n[{now:%H:%M:%S}] 🚀 BUY")
                print(f"  AI:{p_up*100:.1f}% > {threshold*100:.0f}% | "
                      f"ADX:{row['ADX']:.1f} | WR:{wr_now*100:.1f}%")
                tickets = open_orders('BUY', row['ATR'])
                if tickets:
                    # Simpan kondisi entry ke memory
                    memory.save_entry(
                        row_to_dict(row), 'BUY',
                        mt5.symbol_info_tick(SYMBOL).ask,
                        datetime.now(timezone.utc)
                    )
                    monitor.start_trade()

            # ── Entry SELL ──
            elif p_dn > threshold and confirm_sell(row):
                print(f"\n[{now:%H:%M:%S}] 📉 SELL")
                print(f"  AI:{p_dn*100:.1f}% > {threshold*100:.0f}% | "
                      f"ADX:{row['ADX']:.1f} | WR:{wr_now*100:.1f}%")
                tickets = open_orders('SELL', row['ATR'])
                if tickets:
                    memory.save_entry(
                        row_to_dict(row), 'SELL',
                        mt5.symbol_info_tick(SYMBOL).bid,
                        datetime.now(timezone.utc)
                    )
                    monitor.start_trade()

            else:
                print(f"[{now:%H:%M:%S}] ⏳ UP:{p_up*100:.1f}% DN:{p_dn*100:.1f}% "
                      f"| Thr:{threshold:.2f} | WR:{wr_now*100:.1f}% "
                      f"| ADX:{row['ADX']:.1f}")

            it += 1
            time.sleep(SEARCH_INTERVAL)

    except KeyboardInterrupt:
        print("\n⚠️  Bot dihentikan.")
        force_close_all("Manual stop")
        print(f"\n📊 STATISTIK AKHIR: {memory.stats()}")
    finally:
        mt5.shutdown()
        print("Selesai.")