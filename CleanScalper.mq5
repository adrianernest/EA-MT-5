//+------------------------------------------------------------------+
//|                                                                  |
//|         ██████╗██╗     ███████╗ █████╗ ███╗   ██╗               |
//|        ██╔════╝██║     ██╔════╝██╔══██╗████╗  ██║               |
//|        ██║     ██║     █████╗  ███████║██╔██╗ ██║               |
//|        ██║     ██║     ██╔══╝  ██╔══██║██║╚██╗██║               |
//|        ╚██████╗███████╗███████╗██║  ██║██║ ╚████║               |
//|         ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝               |
//|                                                                  |
//|              CLEAN SCALPER — XAUUSD                             |
//|                                                                  |
//|  FILOSOFI: Fewer rules, better rules                            |
//|                                                                  |
//|  HANYA 3 INDIKATOR INTI:                                        |
//|  1. EMA 50 H1  → Arah trend besar (filter utama)               |
//|  2. ADX 14     → Pastikan pasar trending, bukan sideways        |
//|  3. ATR 14     → SL/TP dinamis sesuai volatilitas               |
//|                                                                  |
//|  LOGIKA ENTRY (sangat sederhana):                               |
//|  BUY  = Harga di atas EMA50 H1 + ADX > 20 + Candle pullback    |
//|  SELL = Harga di bawah EMA50 H1 + ADX > 20 + Candle pullback   |
//|                                                                  |
//|  KENAPA INI LEBIH BAIK:                                         |
//|  → EMA 50 H1 adalah indikator paling diikuti institusi          |
//|  → ADX memastikan ada energy di balik pergerakan                |
//|  → ATR membuat SL/TP realistis sesuai kondisi pasar             |
//|  → Candle pullback = entry di harga bagus, bukan kejar harga    |
//|  → Sedikit filter = lebih banyak sinyal berkualitas             |
//|                                                                  |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>
CTrade trade;

//+------------------------------------------------------------------+
//| INPUT                                                            |
//+------------------------------------------------------------------+
input group "━━━ IDENTITAS ━━━"
input ulong  Magic          = 777777;
input string Tag            = "CLEAN";

input group "━━━ LOT & ORDER ━━━"
input double Lot            = 0.01;     // Lot per order
input int    MaxOrders      = 3;        // Max order serentak

input group "━━━ EXIT ━━━"
input double ATR_TP_Multi   = 1.5;      // TP = ATR × nilai ini
input double ATR_SL_Multi   = 1.0;      // SL = ATR × nilai ini
input double MaxLossUSD     = 5.0;      // Hard cut loss ($)
input int    TimeoutMenit   = 60;       // Timeout posisi (menit)
input bool   UseTrailing    = true;     // Trailing stop aktif

input group "━━━ FILTER MASUK ━━━"
input double ADX_Min        = 22.0;     // ADX minimal
input int    MaxSpread      = 20;       // Max spread (points)
input int    CooldownMenit  = 5;        // Jeda antar entry (menit)

input group "━━━ FILTER JAM ━━━"
input bool   UseHourFilter  = true;
input int    StartHour      = 7;        // Mulai jam 7 (London open)
input int    EndHour        = 20;       // Selesai jam 20

input group "━━━ PARAMETER ━━━"
input int    EMA_Period     = 50;       // EMA trend utama
input int    ADX_Period     = 14;
input int    ATR_Period     = 14;

//+------------------------------------------------------------------+
//| HANDLE                                                           |
//+------------------------------------------------------------------+

// H1 — penentu trend besar
int hH1_ema;
int hH1_adx;

// M15 — konfirmasi trend menengah
int hM15_ema;
int hM15_adx;

// M1 — timing entry + volatilitas
int hM1_atr;
int hM1_adx;

//+------------------------------------------------------------------+
//| STATE                                                            |
//+------------------------------------------------------------------+
datetime g_lastBar    = 0;
datetime g_openTime   = 0;
datetime g_lastEntry  = 0;
double   g_highWater  = -999999;
double   g_initBal    = 0;
int      g_wins       = 0;
int      g_losses     = 0;
double   g_pnl        = 0;

//+------------------------------------------------------------------+
//| INIT                                                             |
//+------------------------------------------------------------------+
int OnInit() {
    trade.SetExpertMagicNumber(Magic);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    trade.SetAsyncMode(false);

    g_initBal = AccountInfoDouble(ACCOUNT_BALANCE);

    // Hanya 6 handle total — ringan dan cepat
    hH1_ema  = iMA(Symbol(),  PERIOD_H1, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
    hH1_adx  = iADX(Symbol(), PERIOD_H1, ADX_Period);
    hM15_ema = iMA(Symbol(),  PERIOD_M15, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
    hM15_adx = iADX(Symbol(), PERIOD_M15, ADX_Period);
    hM1_atr  = iATR(Symbol(), PERIOD_M1, ATR_Period);
    hM1_adx  = iADX(Symbol(), PERIOD_M1, ADX_Period);

    if(hH1_ema==INVALID_HANDLE || hH1_adx==INVALID_HANDLE ||
       hM15_ema==INVALID_HANDLE|| hM1_atr==INVALID_HANDLE) {
        Alert("CLEAN SCALPER: Gagal init!");
        return INIT_FAILED;
    }

    Print("╔════════════════════════════════════════╗");
    Print("║      CLEAN SCALPER — AKTIF             ║");
    Print("╠════════════════════════════════════════╣");
    Print("║  Pair    : ", Symbol());
    Print("║  Lot     : ", MaxOrders, " x ", Lot);
    Print("║  Filter  : EMA", EMA_Period, "H1 + ADX", ADX_Min, "+ ATR");
    Print("║  Jam     : ", StartHour, ":00 - ", EndHour, ":00");
    Print("║  Spread  : max ", MaxSpread, "pts");
    Print("╚════════════════════════════════════════╝");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| DEINIT                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    IndicatorRelease(hH1_ema);  IndicatorRelease(hH1_adx);
    IndicatorRelease(hM15_ema); IndicatorRelease(hM15_adx);
    IndicatorRelease(hM1_atr);  IndicatorRelease(hM1_adx);

    int total = g_wins + g_losses;
    double wr = total > 0 ? (double)g_wins/total*100 : 0;
    Print("╔════════════════════════════════════════╗");
    Print("║          STATISTIK SESI                ║");
    Print("╠════════════════════════════════════════╣");
    Print("║  Trade : ", total, " (W:", g_wins, " L:", g_losses, ")");
    Print("║  WR    : ", DoubleToString(wr, 1), "%");
    Print("║  P/L   : $", DoubleToString(g_pnl, 2));
    Print("╚════════════════════════════════════════╝");
}

//+------------------------------------------------------------------+
//| HELPER                                                           |
//+------------------------------------------------------------------+
double V(int h, int buf, int shift=0) {
    double a[]; ArraySetAsSeries(a, true);
    if(CopyBuffer(h, buf, 0, shift+2, a) < shift+1) return 0;
    return a[shift];
}

//+------------------------------------------------------------------+
//| HITUNG POSISI & PROFIT                                          |
//+------------------------------------------------------------------+
int NPos() {
    int n=0;
    for(int i=0; i<PositionsTotal(); i++)
        if(PositionGetSymbol(i)==Symbol() &&
           PositionGetInteger(POSITION_MAGIC)==(long)Magic) n++;
    return n;
}

double NPnL() {
    double s=0;
    for(int i=0; i<PositionsTotal(); i++)
        if(PositionGetSymbol(i)==Symbol() &&
           PositionGetInteger(POSITION_MAGIC)==(long)Magic)
            s += PositionGetDouble(POSITION_PROFIT);
    return s;
}

//+------------------------------------------------------------------+
//| CLOSE SEMUA                                                     |
//+------------------------------------------------------------------+
void CloseAll(string why) {
    double pnl = NPnL();
    int n = 0;
    trade.SetExpertMagicNumber(Magic);
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong t = PositionGetTicket(i);
        if(PositionSelectByTicket(t))
            if(PositionGetString(POSITION_SYMBOL)==Symbol() &&
               (ulong)PositionGetInteger(POSITION_MAGIC)==Magic)
                if(trade.PositionClose(t)) n++;
    }
    if(n > 0) {
        g_pnl += pnl;
        if(pnl >= 0) g_wins++; else g_losses++;
        Print("🔒 ", n, " | $", DoubleToString(pnl,4),
              pnl>=0?" ✅":" ⛔", " | ", why,
              " W:", g_wins, " L:", g_losses,
              " $", DoubleToString(g_pnl,2));
    }
    g_openTime = 0; g_highWater = -999999;
}

//+------------------------------------------------------------------+
//| TRAILING STOP + CLOSE PROFIT                                    |
//+------------------------------------------------------------------+
void ManagePositions(double atr) {
    double pt  = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    double tpPts = ATR_TP_Multi * atr / pt;

    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong t = PositionGetTicket(i);
        if(!PositionSelectByTicket(t)) continue;
        if(PositionGetString(POSITION_SYMBOL) != Symbol()) continue;
        if((ulong)PositionGetInteger(POSITION_MAGIC) != Magic) continue;

        double op   = PositionGetDouble(POSITION_PRICE_OPEN);
        double cSL  = PositionGetDouble(POSITION_SL);
        double cTP  = PositionGetDouble(POSITION_TP);
        int    typ  = (int)PositionGetInteger(POSITION_TYPE);
        double cur  = (typ==0) ? bid : ask;
        double pip  = (typ==0) ? (cur-op)/pt : (op-cur)/pt;

        // Trailing: geser SL setiap kali profit bertambah 0.5 ATR
        if(UseTrailing && pip >= tpPts * 0.5) {
            double trailDist = ATR_SL_Multi * atr * 0.6;
            double newSL = (typ==0) ? cur - trailDist : cur + trailDist;
            // Minimal breakeven
            double beSL  = (typ==0) ? op + 2*pt : op - 2*pt;
            newSL = (typ==0) ? MathMax(newSL, beSL) : MathMin(newSL, beSL);
            bool better = (typ==0) ? newSL > cSL : newSL < cSL;
            if(better) trade.PositionModify(t, newSL, cTP);
        }

        // Close saat TP tercapai
        if(pip >= tpPts) {
            double pf = PositionGetDouble(POSITION_PROFIT);
            if(trade.PositionClose(t))
                Print("💰 #",t," +",DoubleToString(pip,1),
                      "pip $",DoubleToString(pf,4));
        }
    }
}

//+------------------------------------------------------------------+
//| OPEN ORDERS                                                     |
//+------------------------------------------------------------------+
void OpenOrders(bool buy, double atr) {
    double pt   = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double slPts = ATR_SL_Multi * atr / pt;
    double tpPts = ATR_TP_Multi * atr / pt;

    // Pastikan SL minimal 100 points untuk XAUUSD
    slPts = MathMax(slPts, 100);
    tpPts = MathMax(tpPts, slPts * ATR_TP_Multi);

    for(int i=0; i<MaxOrders; i++) {
        double price, sl, tp;
        if(buy) {
            price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
            sl    = price - slPts * pt;
            tp    = price + tpPts * pt;
            if(!trade.Buy(Lot, Symbol(), price, sl, tp,
                          Tag+"#"+IntegerToString(i+1)))
                Print("  ❌ BUY gagal: ", trade.ResultRetcode(),
                      trade.ResultRetcode()==10027?" ← AKTIFKAN ALGO TRADING!":"");
            else
                Print("  ✅ BUY #",i+1," @",DoubleToString(price,2),
                      " SL:",DoubleToString(sl,2),
                      " TP:",DoubleToString(tp,2));
        } else {
            price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
            sl    = price + slPts * pt;
            tp    = price - tpPts * pt;
            if(!trade.Sell(Lot, Symbol(), price, sl, tp,
                           Tag+"#"+IntegerToString(i+1)))
                Print("  ❌ SELL gagal: ", trade.ResultRetcode(),
                      trade.ResultRetcode()==10027?" ← ALGO TRADING!":"");
            else
                Print("  ✅ SELL #",i+1," @",DoubleToString(price,2),
                      " SL:",DoubleToString(sl,2),
                      " TP:",DoubleToString(tp,2));
        }
        Sleep(30);
    }
    g_openTime  = TimeCurrent();
    g_lastEntry = TimeCurrent();
    g_highWater = 0;
}

//+------------------------------------------------------------------+
//| ════ LOGIKA UTAMA — 3 LANGKAH SAJA ════                        |
//|                                                                  |
//| LANGKAH 1: TREND FILTER (H1 EMA50)                             |
//|   "Kemana arah trend besar hari ini?"                           |
//|   Harga di atas EMA50 H1 = uptrend → hanya BUY                 |
//|   Harga di bawah EMA50 H1 = downtrend → hanya SELL             |
//|                                                                  |
//| LANGKAH 2: MOMENTUM FILTER (ADX)                               |
//|   "Apakah trend ini punya tenaga?"                              |
//|   ADX M1 > 22 = ada tenaga → lanjut                            |
//|   ADX M1 < 22 = lemah/sideways → skip                          |
//|                                                                  |
//| LANGKAH 3: ENTRY TIMING (Pullback ke EMA M15)                  |
//|   "Masuk di harga bagus, bukan kejar harga"                    |
//|   Uptrend: harga pullback ke EMA M15 lalu bounce naik → BUY   |
//|   Downtrend: harga pullback ke EMA M15 lalu bounce turun→ SELL |
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckSignal(bool &isBuy, double &atr) {
    double price   = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    atr            = V(hM1_atr, 0);

    //--- LANGKAH 1: TREND FILTER H1
    double ema_h1  = V(hH1_ema, 0);
    double adx_h1  = V(hH1_adx, 0);

    // H1 harus trending (bukan sideways)
    if(adx_h1 < 18) return false;

    bool uptrend   = price > ema_h1;
    bool downtrend = price < ema_h1;

    // Terlalu dekat EMA H1 (dalam 0.1% range) = zona abu-abu, skip
    double dist = MathAbs(price - ema_h1) / ema_h1;
    if(dist < 0.001) return false;

    //--- LANGKAH 2: MOMENTUM FILTER M1
    double adx_m1  = V(hM1_adx, 0);
    double pdi_m1  = V(hM1_adx, 1);
    double ndi_m1  = V(hM1_adx, 2);

    if(adx_m1 < ADX_Min) return false;

    // DI harus searah dengan trend H1
    if(uptrend   && pdi_m1 < ndi_m1) return false;
    if(downtrend && ndi_m1 < pdi_m1) return false;

    //--- LANGKAH 3: ENTRY TIMING — PULLBACK KE EMA M15
    double ema_m15 = V(hM15_ema, 0);
    double adx_m15 = V(hM15_adx, 0);

    // Ambil data candle M1 (2 bar terakhir yang sudah selesai)
    double c1 = iClose(Symbol(), PERIOD_M1, 1);
    double o1 = iOpen(Symbol(),  PERIOD_M1, 1);
    double h1 = iHigh(Symbol(),  PERIOD_M1, 1);
    double l1 = iLow(Symbol(),   PERIOD_M1, 1);
    double c2 = iClose(Symbol(), PERIOD_M1, 2);

    double body   = MathAbs(c1 - o1);
    double range  = h1 - l1;
    if(range < atr * 0.1) return false;  // Candle terlalu kecil

    // BUY SETUP
    // Kondisi: uptrend H1 + harga di atas EMA M15 + candle bullish
    // atau harga baru bounce dari EMA M15
    if(uptrend) {
        bool aboveEmaM15   = price > ema_m15;
        bool bullCandle    = c1 > o1 && body > range * 0.4;
        bool bouncedEma    = l1 <= ema_m15 * 1.001 && c1 > ema_m15;
        bool consistentUp  = c1 > c2;

        bool entryOK = aboveEmaM15 && (bullCandle || bouncedEma) && consistentUp;
        if(entryOK) { isBuy = true; return true; }
    }

    // SELL SETUP
    // Kondisi: downtrend H1 + harga di bawah EMA M15 + candle bearish
    if(downtrend) {
        bool belowEmaM15   = price < ema_m15;
        bool bearCandle    = c1 < o1 && body > range * 0.4;
        bool bouncedEma    = h1 >= ema_m15 * 0.999 && c1 < ema_m15;
        bool consistentDn  = c1 < c2;

        bool entryOK = belowEmaM15 && (bearCandle || bouncedEma) && consistentDn;
        if(entryOK) { isBuy = false; return true; }
    }

    return false;
}

//+------------------------------------------------------------------+
//| ONTICK                                                          |
//+------------------------------------------------------------------+
void OnTick() {

    double atr = V(hM1_atr, 0);

    //════════════════════════════════════════
    // MONITOR POSISI — SETIAP TICK
    //════════════════════════════════════════
    if(NPos() > 0) {
        double pnl     = NPnL();
        int    elapsed = (int)(TimeCurrent() - g_openTime) / 60;

        if(pnl > g_highWater) g_highWater = pnl;

        // Cut loss
        if(pnl <= -MaxLossUSD) {
            CloseAll("CUT LOSS $"+DoubleToString(pnl,2));
            return;
        }
        // Timeout
        if(elapsed >= TimeoutMenit) {
            CloseAll("TIMEOUT "+IntegerToString(elapsed)+"min");
            return;
        }
        // Trailing + close profit per order
        ManagePositions(atr);

        if(NPos() == 0) {
            Print("✅ Selesai | W:", g_wins, " L:", g_losses,
                  " $", DoubleToString(g_pnl,2));
        }
        return;
    }

    //════════════════════════════════════════
    // CARI SINYAL — HANYA DI BAR M1 BARU
    //════════════════════════════════════════
    datetime barNow = iTime(Symbol(), PERIOD_M1, 0);
    if(barNow == g_lastBar) return;
    g_lastBar = barNow;

    // Filter jam
    if(UseHourFilter) {
        MqlDateTime tm; TimeToStruct(TimeCurrent(), tm);
        if(tm.hour < StartHour || tm.hour >= EndHour) return;
    }

    // Filter spread
    int spread = (int)SymbolInfoInteger(Symbol(), SYMBOL_SPREAD);
    if(spread > MaxSpread) {
        Print("⏸ Spread:", spread, "pts — skip");
        return;
    }

    // Cooldown
    if(g_lastEntry > 0 &&
       (int)(TimeCurrent()-g_lastEntry)/60 < CooldownMenit) return;

    //════════════════════════════════════════
    // 3 LANGKAH ANALISA
    //════════════════════════════════════════
    bool  isBuy = false;
    double atrSignal = 0;
    bool  signal = CheckSignal(isBuy, atrSignal);

    if(!signal) {
        // Log ringkas setiap bar — tidak spam
        double ema_h1 = V(hH1_ema, 0);
        double price  = SymbolInfoDouble(Symbol(), SYMBOL_BID);
        string trendStr = price > ema_h1 ? "⬆UP" : "⬇DN";
        Print("⏳ ",trendStr,
              " ADX_H1:", DoubleToString(V(hH1_adx,0),1),
              " ADX_M1:", DoubleToString(V(hM1_adx,0),1),
              " Sprd:", spread);
        return;
    }

    //════════════════════════════════════════
    // ENTRY
    //════════════════════════════════════════
    double ema_h1 = V(hH1_ema, 0);
    double price  = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double tpPts  = ATR_TP_Multi * atrSignal /
                    SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double slPts  = ATR_SL_Multi * atrSignal /
                    SymbolInfoDouble(Symbol(), SYMBOL_POINT);

    Print("");
    Print("╔════════════════════════════════════════╗");
    Print("║  🎯 ENTRY ", isBuy?"BUY ⬆":"SELL ⬇");
    Print("╠════════════════════════════════════════╣");
    Print("║  Trend  : Harga ", isBuy?"DI ATAS":"DI BAWAH",
          " EMA", EMA_Period, " H1");
    Print("║  EMA H1 : ", DoubleToString(ema_h1,2),
          " | Harga: ", DoubleToString(price,2));
    Print("║  ADX H1 : ", DoubleToString(V(hH1_adx,0),1),
          " | ADX M1: ", DoubleToString(V(hM1_adx,0),1));
    Print("║  ATR    : ", DoubleToString(atrSignal,2),
          " | TP:", DoubleToString(tpPts,1), "pts",
          " SL:", DoubleToString(slPts,1), "pts");
    Print("║  Spread : ", spread, " pts");
    Print("╚════════════════════════════════════════╝");

    OpenOrders(isBuy, atrSignal);
}
//+------------------------------------------------------------------+
