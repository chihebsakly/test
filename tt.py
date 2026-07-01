"""
"""
????????????????????????????????????????????????????????????????????
?        BOT EXPERT TRADING BTC/USD - VERSION ULTIME              ?
?        40+ Scénarios d'opportunité - Architecture Modulaire     ?
????????????????????????????????????????????????????????????????????
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import logging
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime

# ??????????????????????????????????????????????????????????????
# CONFIGURATION GLOBALE
# ??????????????????????????????????????????????????????????????
SCRIPT_DIR = os.path.join(os.path.expanduser("~"), "BotTrading")
os.makedirs(SCRIPT_DIR, exist_ok=True)

SYMBOL = "BTCUSD"
LOT = 0.05              # Lot fixe unique
MAGIC_NUMBER = 123456
TIMEFRAME = mt5.TIMEFRAME_M5
HIST_BOUGIES = 2016  # 7 jours
LOOP_INTERVAL = 2    # secondes
LOG_FILE = os.path.join(SCRIPT_DIR, "bot_expert.log")
SCORE_MIN_ENTRY = 55  # Seuil minimum pour entrer
SCORE_AGGRESSIVE = 80  # Seuil mode agressif
MAX_POSITIONS = 1     # 1 seul trade à la fois
SOLDE = 27            # Solde du compte en $
STOP_LOSS_MAX = 8     # Perte max en $ (protection solde)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("BotExpert")


# ??????????????????????????????????????????????????????????????
# STRUCTURES DE DONNÉES
# ??????????????????????????????????????????????????????????????
@dataclass
class MarketSnapshot:
    """Capture complète de l'état du marché à un instant T"""
    # Prix
    prix: float
    open_price: float
    high_price: float
    low_price: float

    # RSI
    rsi_7: float
    rsi_14: float
    rsi_21: float
    rsi_prev_7: float       # RSI 7 bougie précédente
    rsi_prev_14: float      # RSI 14 bougie précédente

    # Moyennes mobiles
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float

    # Volatilité
    atr: float
    atr_percent: float      # ATR en % du prix
    bollinger_upper: float
    bollinger_lower: float
    bollinger_mid: float
    bollinger_width: float  # Largeur relative des bandes

    # Momentum
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    stoch_k: float
    stoch_d: float

    # Volume
    volume_ratio: float     # Volume actuel vs moyenne
    obv_slope: float        # Pente OBV (flux d'argent)

    # Mouvements de prix
    price_change_1b: float  # % changement 1 bougie
    price_change_3b: float  # % changement 3 bougies
    price_change_5b: float  # % changement 5 bougies
    price_change_12b: float # % changement 1h (12×5min)
    price_change_24b: float # % changement 2h

    # Mouvements relatifs à ATR (CRUCIAL pour BTC)
    move_1b_in_atr: float   # Mouvement 1 bougie en multiples d'ATR
    move_3b_in_atr: float   # Mouvement 3 bougies en multiples d'ATR
    move_5b_in_atr: float   # Mouvement 5 bougies en multiples d'ATR

    # Breakout / Structure
    is_new_high_2h: bool
    is_new_low_2h: bool
    is_new_high_4h: bool
    is_new_low_4h: bool
    distance_from_high_2h: float
    distance_from_low_2h: float
    high_2h: float
    low_2h: float
    high_4h: float
    low_4h: float

    # Pentes / Divergences
    rsi_slope_5: float      # Pente RSI sur 5 bougies
    rsi_slope_10: float     # Pente RSI sur 10 bougies
    price_slope_5: float    # Pente prix sur 5 bougies
    price_slope_10: float   # Pente prix sur 10 bougies

    # Bougies japonaises
    is_hammer: bool
    is_shooting_star: bool
    is_engulfing_bull: bool
    is_engulfing_bear: bool
    is_doji: bool
    candle_body_ratio: float  # Corps vs mèches

    # Seuils adaptatifs
    quantile_high_rsi: float
    quantile_low_rsi: float

    # Contexte temporel
    consecutive_green: int   # Bougies vertes consécutives
    consecutive_red: int     # Bougies rouges consécutives


@dataclass
class Signal:
    """Signal de trading avec score et justification"""
    direction: str          # "BUY", "SELL", "NONE"
    score: int              # 0-100+
    reasons: List[str]
    is_aggressive: bool
    scenario_count: int     # Nombre de scénarios déclenchés


# ??????????????????????????????????????????????????????????????
# DATA ENGINE - Calcul de tous les indicateurs
# ??????????????????????????????????????????????????????????????
class DataEngine:
    def __init__(self, symbol: str, timeframe: int, history_size: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.history_size = history_size

    def get_snapshot(self) -> Optional[MarketSnapshot]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.history_size)
        if rates is None or len(rates) < 250:
            logger.warning("Données insuffisantes")
            return None

        df = pd.DataFrame(rates)

        # --- RSI ---
        df['rsi_7'] = RSIIndicator(df['close'], 7).rsi()
        df['rsi_14'] = RSIIndicator(df['close'], 14).rsi()
        df['rsi_21'] = RSIIndicator(df['close'], 21).rsi()

        # --- EMAs ---
        df['ema_9'] = EMAIndicator(df['close'], 9).ema_indicator()
        df['ema_21'] = EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50'] = EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = EMAIndicator(df['close'], 200).ema_indicator()

        # --- ATR ---
        atr_ind = AverageTrueRange(df['high'], df['low'], df['close'], 14)
        df['atr'] = atr_ind.average_true_range()

        # --- Bollinger ---
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()

        # --- MACD ---
        macd_ind = MACD(df['close'])
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_hist'] = macd_ind.macd_diff()

        # --- Stochastic ---
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], 14, 3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # --- OBV ---
        obv = OnBalanceVolumeIndicator(df['close'], df['tick_volume'])
        df['obv'] = obv.on_balance_volume()

        # --- Volume moyenne ---
        df['vol_ma'] = df['tick_volume'].rolling(50).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # --- Changements de prix ---
        price_change_1b = (last['close'] - prev['close']) / prev['close'] * 100
        price_change_3b = (last['close'] - df.iloc[-4]['close']) / df.iloc[-4]['close'] * 100
        price_change_5b = (last['close'] - df.iloc[-6]['close']) / df.iloc[-6]['close'] * 100
        price_change_12b = (last['close'] - df.iloc[-13]['close']) / df.iloc[-13]['close'] * 100
        price_change_24b = (last['close'] - df.iloc[-25]['close']) / df.iloc[-25]['close'] * 100

        # --- Mouvements en multiples d'ATR (adaptatif à la volatilité) ---
        current_atr = last['atr'] if last['atr'] > 0 else 1
        move_1b_in_atr = abs(last['close'] - prev['close']) / current_atr
        move_3b_in_atr = abs(last['close'] - df.iloc[-4]['close']) / current_atr
        move_5b_in_atr = abs(last['close'] - df.iloc[-6]['close']) / current_atr

        # --- Breakout 2h (24 bougies) ---
        window_2h = df.tail(24)
        high_2h = window_2h['high'].max()
        low_2h = window_2h['low'].min()
        is_new_high_2h = (last['high'] >= high_2h and prev['high'] < high_2h)
        is_new_low_2h = (last['low'] <= low_2h and prev['low'] > low_2h)

        # --- Breakout 4h (48 bougies) ---
        window_4h = df.tail(48)
        high_4h = window_4h['high'].max()
        low_4h = window_4h['low'].min()
        is_new_high_4h = (last['high'] >= high_4h and prev['high'] < high_4h)
        is_new_low_4h = (last['low'] <= low_4h and prev['low'] > low_4h)

        dist_high_2h = (high_2h - last['close']) / last['close'] * 100
        dist_low_2h = (last['close'] - low_2h) / last['close'] * 100

        # --- Pentes ---
        rsi_slope_5 = df['rsi_14'].iloc[-5:].diff().mean()
        rsi_slope_10 = df['rsi_14'].iloc[-10:].diff().mean()
        price_slope_5 = df['close'].iloc[-5:].pct_change().mean() * 100
        price_slope_10 = df['close'].iloc[-10:].pct_change().mean() * 100

        # --- Volume ratio ---
        vol_ratio = last['tick_volume'] / last['vol_ma'] if last['vol_ma'] > 0 else 1.0

        # --- OBV slope ---
        obv_slope = df['obv'].iloc[-5:].diff().mean()

        # --- Bougies japonaises ---
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['close'], last['open'])
        lower_wick = min(last['close'], last['open']) - last['low']
        total_range = last['high'] - last['low'] if last['high'] != last['low'] else 0.001
        candle_body_ratio = body / total_range

        is_hammer = (lower_wick > body * 2 and upper_wick < body * 0.5 and last['close'] > last['open'])
        is_shooting_star = (upper_wick > body * 2 and lower_wick < body * 0.5 and last['close'] < last['open'])
        is_doji = (candle_body_ratio < 0.1)
        is_engulfing_bull = (last['close'] > last['open'] and prev['close'] < prev['open'] and
                             last['close'] > prev['open'] and last['open'] < prev['close'])
        is_engulfing_bear = (last['close'] < last['open'] and prev['close'] > prev['open'] and
                             last['close'] < prev['open'] and last['open'] > prev['close'])

        # --- Bougies consécutives ---
        consecutive_green = 0
        consecutive_red = 0
        for i in range(len(df) - 1, max(len(df) - 20, 0), -1):
            if df.iloc[i]['close'] > df.iloc[i]['open']:
                if consecutive_red == 0:
                    consecutive_green += 1
                else:
                    break
            else:
                if consecutive_green == 0:
                    consecutive_red += 1
                else:
                    break

        # --- Quantiles adaptatifs ---
        recent_rsi = df['rsi_14'].tail(500)
        q_high = recent_rsi.quantile(0.90)
        q_low = recent_rsi.quantile(0.10)

        # --- Bollinger width ---
        bb_width = (last['bb_upper'] - last['bb_lower']) / last['bb_mid'] * 100 if last['bb_mid'] > 0 else 0

        return MarketSnapshot(
            prix=last['close'],
            open_price=last['open'],
            high_price=last['high'],
            low_price=last['low'],
            rsi_7=last['rsi_7'],
            rsi_14=last['rsi_14'],
            rsi_21=last['rsi_21'],
            rsi_prev_7=prev['rsi_7'],
            rsi_prev_14=prev['rsi_14'],
            ema_9=last['ema_9'],
            ema_21=last['ema_21'],
            ema_50=last['ema_50'],
            ema_200=last['ema_200'],
            atr=last['atr'],
            atr_percent=(last['atr'] / last['close'] * 100),
            bollinger_upper=last['bb_upper'],
            bollinger_lower=last['bb_lower'],
            bollinger_mid=last['bb_mid'],
            bollinger_width=bb_width,
            macd_line=last['macd'],
            macd_signal=last['macd_signal'],
            macd_hist=last['macd_hist'],
            macd_hist_prev=prev['macd_hist'],
            stoch_k=last['stoch_k'],
            stoch_d=last['stoch_d'],
            volume_ratio=vol_ratio,
            obv_slope=obv_slope,
            price_change_1b=price_change_1b,
            price_change_3b=price_change_3b,
            price_change_5b=price_change_5b,
            price_change_12b=price_change_12b,
            price_change_24b=price_change_24b,
            move_1b_in_atr=move_1b_in_atr,
            move_3b_in_atr=move_3b_in_atr,
            move_5b_in_atr=move_5b_in_atr,
            is_new_high_2h=is_new_high_2h,
            is_new_low_2h=is_new_low_2h,
            is_new_high_4h=is_new_high_4h,
            is_new_low_4h=is_new_low_4h,
            distance_from_high_2h=dist_high_2h,
            distance_from_low_2h=dist_low_2h,
            high_2h=high_2h,
            low_2h=low_2h,
            high_4h=high_4h,
            low_4h=low_4h,
            rsi_slope_5=rsi_slope_5,
            rsi_slope_10=rsi_slope_10,
            price_slope_5=price_slope_5,
            price_slope_10=price_slope_10,
            is_hammer=is_hammer,
            is_shooting_star=is_shooting_star,
            is_engulfing_bull=is_engulfing_bull,
            is_engulfing_bear=is_engulfing_bear,
            is_doji=is_doji,
            candle_body_ratio=candle_body_ratio,
            quantile_high_rsi=q_high,
            quantile_low_rsi=q_low,
            consecutive_green=consecutive_green,
            consecutive_red=consecutive_red
        )


# ??????????????????????????????????????????????????????????????
# SIGNAL GENERATOR - 40+ SCÉNARIOS D'OPPORTUNITÉ
# ??????????????????????????????????????????????????????????????
class SignalGenerator:
    """
    Chaque scénario ajoute un score. Le total détermine l'action.
    Score >= 55 : Entrée standard
    Score >= 80 : Entrée agressive (lot augmenté)
    """

    def evaluate(self, snap: MarketSnapshot) -> Signal:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 1 : RSI - Scénarios 1 à 8               ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 1 : RSI7 survendu adaptatif ---
        if snap.rsi_7 <= snap.quantile_low_rsi:
            buy_score += 20
            buy_reasons.append(f"[S1] RSI7={snap.rsi_7:.1f} <= seuil adaptatif {snap.quantile_low_rsi:.1f}")

        # --- Scénario 2 : RSI7 suracheté adaptatif ---
        if snap.rsi_7 >= snap.quantile_high_rsi:
            sell_score += 20
            sell_reasons.append(f"[S2] RSI7={snap.rsi_7:.1f} >= seuil adaptatif {snap.quantile_high_rsi:.1f}")

        # --- Scénario 3 : RSI14 zone forte achat ---
        if snap.rsi_14 <= 35:
            buy_score += 15
            buy_reasons.append(f"[S3] RSI14={snap.rsi_14:.1f} survendu")

        # --- Scénario 4 : RSI14 zone forte vente ---
        if snap.rsi_14 >= 65:
            sell_score += 15
            sell_reasons.append(f"[S4] RSI14={snap.rsi_14:.1f} suracheté")

        # --- Scénario 5 : RSI7 extrême bas (opportunité pure) ---
        if snap.rsi_7 <= 25:
            buy_score += 25
            buy_reasons.append(f"[S5] ? RSI7={snap.rsi_7:.1f} EXTRÊME BAS ? opportunité")

        # --- Scénario 6 : RSI7 extrême haut (opportunité pure) ---
        if snap.rsi_7 >= 75:
            sell_score += 25
            sell_reasons.append(f"[S6] ? RSI7={snap.rsi_7:.1f} EXTRÊME HAUT ? opportunité")

        # --- Scénario 7 : RSI rebond rapide depuis l'extrême ---
        if snap.rsi_prev_7 <= 20 and snap.rsi_7 > 25:
            buy_score += 15
            buy_reasons.append(f"[S7] RSI7 rebondit de {snap.rsi_prev_7:.1f} à {snap.rsi_7:.1f}")

        # --- Scénario 8 : RSI chute rapide depuis l'extrême ---
        if snap.rsi_prev_7 >= 80 and snap.rsi_7 < 75:
            sell_score += 15
            sell_reasons.append(f"[S8] RSI7 chute de {snap.rsi_prev_7:.1f} à {snap.rsi_7:.1f}")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 2 : MOUVEMENT BRUSQUE - Scénarios 9 à 14?
        # ?  Basé sur ATR (adaptatif) + % pour les gros moves   ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 9 : Mouvement rapide 1 bougie > 1.5×ATR (MICRO-BURST) ---
        if snap.move_1b_in_atr >= 1.5:
            if snap.price_change_1b > 0:
                # Hausse brusque ? potentiel continuation ou retour
                if snap.rsi_7 < 65:
                    buy_score += 15
                    buy_reasons.append(f"[S9] ?? Micro-burst HAUT {snap.move_1b_in_atr:.1f}×ATR, RSI frais ? continuation")
                else:
                    sell_score += 15
                    sell_reasons.append(f"[S9] ?? Micro-burst HAUT {snap.move_1b_in_atr:.1f}×ATR, RSI chaud ? retour")
            else:
                if snap.rsi_7 > 35:
                    sell_score += 15
                    sell_reasons.append(f"[S9] ?? Micro-burst BAS {snap.move_1b_in_atr:.1f}×ATR, RSI frais ? continuation")
                else:
                    buy_score += 15
                    buy_reasons.append(f"[S9] ?? Micro-burst BAS {snap.move_1b_in_atr:.1f}×ATR, RSI épuisé ? rebond")

        # --- Scénario 10 : Mouvement 3 bougies > 2×ATR ---
        if snap.move_3b_in_atr >= 2.0:
            if snap.price_change_3b > 0:
                if snap.rsi_7 < 70:
                    buy_score += 20
                    buy_reasons.append(f"[S10] ?? Rush HAUT 3b = {snap.move_3b_in_atr:.1f}×ATR ? momentum")
                else:
                    sell_score += 20
                    sell_reasons.append(f"[S10] ?? Rush HAUT 3b = {snap.move_3b_in_atr:.1f}×ATR + RSI haut ? retour")
            else:
                if snap.rsi_7 > 30:
                    sell_score += 20
                    sell_reasons.append(f"[S10] ?? Chute 3b = {snap.move_3b_in_atr:.1f}×ATR ? momentum baissier")
                else:
                    buy_score += 20
                    buy_reasons.append(f"[S10] ?? Chute 3b = {snap.move_3b_in_atr:.1f}×ATR + RSI bas ? rebond")

        # --- Scénario 11 : Mouvement 3 bougies > 3×ATR (VIOLENT) ---
        if snap.move_3b_in_atr >= 3.0:
            if snap.price_change_3b < 0:
                buy_score += 25
                buy_reasons.append(f"[S11] ?? Crash violent {snap.move_3b_in_atr:.1f}×ATR ? rebond fort probable")
            else:
                sell_score += 25
                sell_reasons.append(f"[S11] ?? Pump violent {snap.move_3b_in_atr:.1f}×ATR ? correction forte")

        # --- Scénario 12 : Mouvement % 1h (garde le % pour les gros moves) ---
        if snap.price_change_12b <= -1.5:
            buy_score += 20
            buy_reasons.append(f"[S12] ?? Crash 1h: {snap.price_change_12b:.2f}% ? rebond")
        if snap.price_change_12b >= 1.5:
            sell_score += 20
            sell_reasons.append(f"[S12] ?? Pump 1h: +{snap.price_change_12b:.2f}% ? correction")

        # --- Scénario 13 : Chute 2h progressive ---
        if snap.price_change_24b <= -2.5:
            buy_score += 20
            buy_reasons.append(f"[S13] ?? Chute 2h: {snap.price_change_24b:.2f}% ? zone d'achat")

        # --- Scénario 14 : Hausse 2h progressive ---
        if snap.price_change_24b >= 2.5:
            sell_score += 20
            sell_reasons.append(f"[S14] ?? Hausse 2h: +{snap.price_change_24b:.2f}% ? zone de vente")

        # --- Scénario 14b : Mouvement 5 bougies > 2.5×ATR (nouveau) ---
        if snap.move_5b_in_atr >= 2.5:
            if snap.price_change_5b > 0 and snap.rsi_7 >= 65:
                sell_score += 15
                sell_reasons.append(f"[S14b] Accélération 5b {snap.move_5b_in_atr:.1f}×ATR + RSI haut ? vente")
            elif snap.price_change_5b < 0 and snap.rsi_7 <= 35:
                buy_score += 15
                buy_reasons.append(f"[S14b] Accélération 5b {snap.move_5b_in_atr:.1f}×ATR + RSI bas ? achat")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 3 : BREAKOUT / PIC - Scénarios 15 à 20  ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 15 : Nouveau pic haut 2h + RSI frais ? continuation ---
        if snap.is_new_high_2h and snap.rsi_7 < 70:
            buy_score += 20
            buy_reasons.append(f"[S15] ?? Breakout HAUT 2h + RSI frais ? continuation")

        # --- Scénario 16 : Nouveau pic haut 2h + RSI épuisé ? retournement ---
        if snap.is_new_high_2h and snap.rsi_7 >= 70:
            sell_score += 15
            sell_reasons.append(f"[S16] ??? Pic 2h + RSI épuisé ? retournement")

        # --- Scénario 17 : Nouveau creux 2h + RSI frais ? continuation baisse ---
        if snap.is_new_low_2h and snap.rsi_7 > 30:
            sell_score += 20
            sell_reasons.append(f"[S17] ?? Breakdown BAS 2h + RSI frais ? continuation")

        # --- Scénario 18 : Nouveau creux 2h + RSI épuisé ? rebond ---
        if snap.is_new_low_2h and snap.rsi_7 <= 30:
            buy_score += 15
            buy_reasons.append(f"[S18] ??? Creux 2h + RSI épuisé ? rebond")

        # --- Scénario 19 : Breakout 4h haut (signal fort) ---
        if snap.is_new_high_4h:
            if snap.rsi_7 < 75:
                buy_score += 25
                buy_reasons.append(f"[S19] ???? Breakout HAUT 4h ? forte continuation")
            else:
                sell_score += 10
                sell_reasons.append(f"[S19] Pic 4h + RSI saturé")

        # --- Scénario 20 : Breakout 4h bas (signal fort) ---
        if snap.is_new_low_4h:
            if snap.rsi_7 > 25:
                sell_score += 25
                sell_reasons.append(f"[S20] ???? Breakdown BAS 4h ? forte continuation")
            else:
                buy_score += 10
                buy_reasons.append(f"[S20] Creux 4h + RSI épuisé ? rebond")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 4 : DIVERGENCES - Scénarios 21 à 24     ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 21 : Divergence haussière courte (5 bougies) ---
        if snap.rsi_slope_5 > 0.5 and snap.price_slope_5 < -0.05:
            buy_score += 20
            buy_reasons.append(f"[S21] ?? Divergence haussière rapide RSI? Prix?")

        # --- Scénario 22 : Divergence baissière courte (5 bougies) ---
        if snap.rsi_slope_5 < -0.5 and snap.price_slope_5 > 0.05:
            sell_score += 20
            sell_reasons.append(f"[S22] ?? Divergence baissière rapide RSI? Prix?")

        # --- Scénario 23 : Divergence haussière longue (10 bougies) ---
        if snap.rsi_slope_10 > 0.3 and snap.price_slope_10 < -0.03:
            buy_score += 15
            buy_reasons.append(f"[S23] ?? Divergence haussière prolongée")

        # --- Scénario 24 : Divergence baissière longue (10 bougies) ---
        if snap.rsi_slope_10 < -0.3 and snap.price_slope_10 > 0.03:
            sell_score += 15
            sell_reasons.append(f"[S24] ?? Divergence baissière prolongée")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 5 : BOLLINGER BANDS - Scénarios 25 à 28 ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 25 : Prix touche bande inférieure ---
        if snap.prix <= snap.bollinger_lower:
            buy_score += 15
            buy_reasons.append(f"[S25] Prix touche Bollinger BAS ? rebond")

        # --- Scénario 26 : Prix touche bande supérieure ---
        if snap.prix >= snap.bollinger_upper:
            sell_score += 15
            sell_reasons.append(f"[S26] Prix touche Bollinger HAUT ? rejet")

        # --- Scénario 27 : Squeeze Bollinger + mouvement haussier ---
        if snap.bollinger_width < 1.5 and snap.price_change_1b > 0.3:
            buy_score += 15
            buy_reasons.append(f"[S27] ?? Squeeze BB + expansion haussière")

        # --- Scénario 28 : Squeeze Bollinger + mouvement baissier ---
        if snap.bollinger_width < 1.5 and snap.price_change_1b < -0.3:
            sell_score += 15
            sell_reasons.append(f"[S28] ?? Squeeze BB + expansion baissière")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 6 : MACD - Scénarios 29 à 32            ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 29 : Croisement MACD haussier ---
        if snap.macd_hist > 0 and snap.macd_hist_prev <= 0:
            buy_score += 15
            buy_reasons.append(f"[S29] ? Croisement MACD haussier")

        # --- Scénario 30 : Croisement MACD baissier ---
        if snap.macd_hist < 0 and snap.macd_hist_prev >= 0:
            sell_score += 15
            sell_reasons.append(f"[S30] ? Croisement MACD baissier")

        # --- Scénario 31 : MACD histogramme croissant fort ---
        if snap.macd_hist > 0 and snap.macd_hist > snap.macd_hist_prev * 1.5:
            buy_score += 10
            buy_reasons.append(f"[S31] MACD momentum haussier accélère")

        # --- Scénario 32 : MACD histogramme décroissant fort ---
        if snap.macd_hist < 0 and snap.macd_hist < snap.macd_hist_prev * 1.5:
            sell_score += 10
            sell_reasons.append(f"[S32] MACD momentum baissier accélère")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 7 : STOCHASTIQUE - Scénarios 33 à 35    ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 33 : Stochastique survendu + croisement ---
        if snap.stoch_k < 20 and snap.stoch_k > snap.stoch_d:
            buy_score += 15
            buy_reasons.append(f"[S33] Stoch survendu + croisement haussier K={snap.stoch_k:.0f}")

        # --- Scénario 34 : Stochastique suracheté + croisement ---
        if snap.stoch_k > 80 and snap.stoch_k < snap.stoch_d:
            sell_score += 15
            sell_reasons.append(f"[S34] Stoch suracheté + croisement baissier K={snap.stoch_k:.0f}")

        # --- Scénario 35 : Double confirmation Stoch + RSI ---
        if snap.stoch_k < 25 and snap.rsi_7 < 30:
            buy_score += 20
            buy_reasons.append(f"[S35] ?? Double survendu: Stoch+RSI")
        if snap.stoch_k > 75 and snap.rsi_7 > 70:
            sell_score += 20
            sell_reasons.append(f"[S35] ?? Double suracheté: Stoch+RSI")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 8 : VOLUME - Scénarios 36 à 38          ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 36 : Volume spike (confirmation) ---
        if snap.volume_ratio > 1.5:
            buy_score += 8
            sell_score += 8
            buy_reasons.append(f"[S36] ?? Volume x{snap.volume_ratio:.1f}")
            sell_reasons.append(f"[S36] ?? Volume x{snap.volume_ratio:.1f}")

        # --- Scénario 37 : Volume explosion + direction ---
        if snap.volume_ratio > 2.5 and snap.price_change_1b > 0.2:
            buy_score += 15
            buy_reasons.append(f"[S37] ?? Volume explosion HAUSSIER x{snap.volume_ratio:.1f}")
        if snap.volume_ratio > 2.5 and snap.price_change_1b < -0.2:
            sell_score += 15
            sell_reasons.append(f"[S37] ?? Volume explosion BAISSIER x{snap.volume_ratio:.1f}")

        # --- Scénario 38 : OBV confirme la direction ---
        if snap.obv_slope > 0 and snap.price_slope_5 > 0:
            buy_score += 8
            buy_reasons.append(f"[S38] OBV confirme flux acheteur")
        if snap.obv_slope < 0 and snap.price_slope_5 < 0:
            sell_score += 8
            sell_reasons.append(f"[S38] OBV confirme flux vendeur")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 9 : BOUGIES JAPONAISES - Scénarios 39-42?
        # ????????????????????????????????????????????????????????

        # --- Scénario 39 : Marteau (hammer) en zone basse ---
        if snap.is_hammer and snap.rsi_7 < 40:
            buy_score += 15
            buy_reasons.append(f"[S39] ?? Marteau + RSI bas ? retournement haussier")

        # --- Scénario 40 : Étoile filante en zone haute ---
        if snap.is_shooting_star and snap.rsi_7 > 60:
            sell_score += 15
            sell_reasons.append(f"[S40] ? Étoile filante + RSI haut ? retournement baissier")

        # --- Scénario 41 : Engulfing haussier ---
        if snap.is_engulfing_bull:
            buy_score += 15
            buy_reasons.append(f"[S41] ?? Engulfing haussier ? renversement")

        # --- Scénario 42 : Engulfing baissier ---
        if snap.is_engulfing_bear:
            sell_score += 15
            sell_reasons.append(f"[S42] ?? Engulfing baissier ? renversement")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 10 : EMA (BONUS) - Scénarios 43 à 46    ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 43 : Prix au-dessus EMA200 (bonus tendance) ---
        if snap.prix > snap.ema_200:
            buy_score += 5
            buy_reasons.append(f"[S43] Prix > EMA200 (tendance)")

        # --- Scénario 44 : Prix en-dessous EMA200 (bonus tendance) ---
        if snap.prix < snap.ema_200:
            sell_score += 5
            sell_reasons.append(f"[S44] Prix < EMA200 (tendance)")

        # --- Scénario 45 : Golden cross EMA9/21 ---
        if snap.ema_9 > snap.ema_21 and snap.prix > snap.ema_9:
            buy_score += 8
            buy_reasons.append(f"[S45] EMA9 > EMA21 + Prix au-dessus")

        # --- Scénario 46 : Death cross EMA9/21 ---
        if snap.ema_9 < snap.ema_21 and snap.prix < snap.ema_9:
            sell_score += 8
            sell_reasons.append(f"[S46] EMA9 < EMA21 + Prix en-dessous")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 11 : CONTEXTE / MOMENTUM - Scén. 47-52  ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 47 : Contre-tendance RSI fort (EMA ne bloque pas) ---
        if snap.rsi_7 >= 70 and snap.prix > snap.ema_200:
            sell_score += 8
            sell_reasons.append(f"[S47] ?? Suracheté malgré hausse ? retournement")
        if snap.rsi_7 <= 30 and snap.prix < snap.ema_200:
            buy_score += 8
            buy_reasons.append(f"[S47] ?? Survendu malgré baisse ? rebond")

        # --- Scénario 48 : Séquence bougies rouges consécutives (épuisement vendeurs) ---
        if snap.consecutive_red >= 5:
            buy_score += 15
            buy_reasons.append(f"[S48] {snap.consecutive_red} bougies rouges ? épuisement vendeurs")

        # --- Scénario 49 : Séquence bougies vertes consécutives (épuisement acheteurs) ---
        if snap.consecutive_green >= 5:
            sell_score += 15
            sell_reasons.append(f"[S49] {snap.consecutive_green} bougies vertes ? épuisement acheteurs")

        # --- Scénario 50 : Prix proche du plus bas 2h (support) ---
        if snap.distance_from_low_2h < 0.15 and not snap.is_new_low_2h:
            buy_score += 10
            buy_reasons.append(f"[S50] Prix proche support 2h ({snap.distance_from_low_2h:.2f}%)")

        # --- Scénario 51 : Prix proche du plus haut 2h (résistance) ---
        if snap.distance_from_high_2h < 0.15 and not snap.is_new_high_2h:
            sell_score += 10
            sell_reasons.append(f"[S51] Prix proche résistance 2h ({snap.distance_from_high_2h:.2f}%)")

        # --- Scénario 52 : Accélération soudaine (1 bougie > 2×ATR) ---
        if snap.move_1b_in_atr >= 2.0:
            if snap.price_change_1b > 0:
                if snap.rsi_7 < 65:
                    buy_score += 15
                    buy_reasons.append(f"[S52] ? Bougie géante haussière {snap.move_1b_in_atr:.1f}×ATR")
                else:
                    sell_score += 10
                    sell_reasons.append(f"[S52] Bougie géante haussière + RSI haut ? épuisement")
            else:
                if snap.rsi_7 > 35:
                    sell_score += 15
                    sell_reasons.append(f"[S52] ? Bougie géante baissière {snap.move_1b_in_atr:.1f}×ATR")
                else:
                    buy_score += 10
                    buy_reasons.append(f"[S52] Bougie géante baissière + RSI bas ? rebond")

        # ????????????????????????????????????????????????????????
        # ?  CATÉGORIE 12 : COMBOS RARES - Scénarios 53 à 58   ?
        # ?  (Configurations à forte probabilité)                ?
        # ????????????????????????????????????????????????????????

        # --- Scénario 53 : COMBO ULTIME BUY ---
        # RSI extrême + Mouvement ATR fort + Volume
        if snap.rsi_7 <= 25 and snap.move_3b_in_atr >= 2.0 and snap.volume_ratio > 1.3:
            buy_score += 30
            buy_reasons.append(f"[S53] ?? COMBO ULTIME: RSI extrême + Crash {snap.move_3b_in_atr:.1f}×ATR + Volume")

        # --- Scénario 54 : COMBO ULTIME SELL ---
        if snap.rsi_7 >= 75 and snap.move_3b_in_atr >= 2.0 and snap.volume_ratio > 1.3:
            sell_score += 30
            sell_reasons.append(f"[S54] ?? COMBO ULTIME: RSI extrême + Pump {snap.move_3b_in_atr:.1f}×ATR + Volume")

        # --- Scénario 55 : Triple convergence BUY ---
        # Stoch survendu + RSI survendu + Bollinger basse
        if snap.stoch_k < 20 and snap.rsi_7 < 30 and snap.prix <= snap.bollinger_lower * 1.005:
            buy_score += 25
            buy_reasons.append(f"[S55] ???? Triple convergence BUY: Stoch+RSI+BB")

        # --- Scénario 56 : Triple convergence SELL ---
        if snap.stoch_k > 80 and snap.rsi_7 > 70 and snap.prix >= snap.bollinger_upper * 0.995:
            sell_score += 25
            sell_reasons.append(f"[S56] ???? Triple convergence SELL: Stoch+RSI+BB")

        # --- Scénario 57 : Rebond sur EMA200 + confirmation ---
        if abs(snap.prix - snap.ema_200) / snap.ema_200 < 0.002:  # Prix très proche EMA200
            if snap.rsi_7 < 50 and snap.prix > snap.ema_200:
                buy_score += 15
                buy_reasons.append(f"[S57] ?? Rebond sur EMA200 confirmé")
            elif snap.rsi_7 > 50 and snap.prix < snap.ema_200:
                sell_score += 15
                sell_reasons.append(f"[S57] ?? Rejet EMA200 confirmé")

        # --- Scénario 58 : Doji après tendance forte (indécision ? retournement) ---
        if snap.is_doji:
            if snap.consecutive_red >= 3:
                buy_score += 12
                buy_reasons.append(f"[S58] Doji après {snap.consecutive_red} rouges ? retournement?")
            if snap.consecutive_green >= 3:
                sell_score += 12
                sell_reasons.append(f"[S58] Doji après {snap.consecutive_green} vertes ? retournement?")

        # ????????????????????????????????????????????????????????
        # ?  DÉCISION FINALE                                     ?
        # ????????????????????????????????????????????????????????

        buy_scenarios = len(buy_reasons)
        sell_scenarios = len(sell_reasons)

        if buy_score >= SCORE_MIN_ENTRY and buy_score > sell_score:
            return Signal("BUY", buy_score, buy_reasons, buy_score >= SCORE_AGGRESSIVE, buy_scenarios)
        elif sell_score >= SCORE_MIN_ENTRY and sell_score > buy_score:
            return Signal("SELL", sell_score, sell_reasons, sell_score >= SCORE_AGGRESSIVE, sell_scenarios)
        else:
            best = max(buy_score, sell_score)
            reasons = buy_reasons if buy_score >= sell_score else sell_reasons
            count = buy_scenarios if buy_score >= sell_score else sell_scenarios
            return Signal("NONE", best, reasons, False, count)


# ══════════════════════════════════════════════════════════════
# RISK MANAGER - Gestion dynamique du risque
# ══════════════════════════════════════════════════════════════
class RiskManager:
    def __init__(self):
        self.entry_score = 0       # Score au moment de l'entrée
        self.max_profit_seen = 0   # Plus haut profit vu (trailing)

    def on_new_trade(self, signal: Signal):
        """Appelé quand un nouveau trade est ouvert"""
        self.entry_score = signal.score
        self.max_profit_seen = 0

    def compute_lot(self, signal: Signal, atr: float, prix: float) -> float:
        """Lot fixe 0.05 - Solde 27$, on ne risque pas plus"""
        return LOT

    def should_close(self, pos, snap: MarketSnapshot) -> Tuple[bool, str]:
        """
        Fermeture intelligente selon la force du signal d'entrée :
        - Signal faible (55-70)  : TP rapide, sécuriser vite (1-5$)
        - Signal fort (70-85)    : Laisser courir vers 10-15$
        - Signal agressif (85+)  : Viser 25-30$+ avec trailing
        """
        profit = pos.profit

        # Mise à jour du profit max vu (pour trailing)
        if profit > self.max_profit_seen:
            self.max_profit_seen = profit

        # ═══ STOP LOSS (toujours actif) ═══
        if profit <= -STOP_LOSS_MAX:
            return True, f"⛔ Stop protection solde ({profit:.2f}$ / max -{STOP_LOSS_MAX}$)"

        atr_stop = snap.atr * 1.5
        estimated_loss = min(atr_stop * LOT * 10, STOP_LOSS_MAX)
        if profit <= -estimated_loss:
            return True, f"Stop ATR ({profit:.2f}$ <= -{estimated_loss:.2f}$)"

        # ═══ TAKE PROFIT selon force du signal ═══
        if self.entry_score >= SCORE_AGGRESSIVE:
            return self._tp_aggressive(pos, snap, profit)
        elif self.entry_score >= 70:
            return self._tp_strong(pos, snap, profit)
        else:
            return self._tp_safe(pos, snap, profit)

    def _tp_safe(self, pos, snap, profit) -> Tuple[bool, str]:
        """Signal faible (55-70) → TP rapide 1-5$"""
        if pos.type == 0:  # BUY
            if snap.rsi_7 >= 55 and profit >= 1.5:
                return True, f"TP safe BUY: RSI={snap.rsi_7:.1f}, +{profit:.2f}$"
            if snap.rsi_7 >= 70 and profit >= 0.5:
                return True, f"TP safe rapide: RSI={snap.rsi_7:.1f}"
        elif pos.type == 1:  # SELL
            if snap.rsi_7 <= 45 and profit >= 1.5:
                return True, f"TP safe SELL: RSI={snap.rsi_7:.1f}, +{profit:.2f}$"
            if snap.rsi_7 <= 30 and profit >= 0.5:
                return True, f"TP safe rapide: RSI={snap.rsi_7:.1f}"

        # MACD retourne
        if pos.type == 0 and snap.macd_hist < 0 and snap.macd_hist_prev >= 0 and profit >= 1:
            return True, f"TP safe MACD: +{profit:.2f}$"
        if pos.type == 1 and snap.macd_hist > 0 and snap.macd_hist_prev <= 0 and profit >= 1:
            return True, f"TP safe MACD: +{profit:.2f}$"

        # Trailing à 3$
        if profit >= 3 and 40 <= snap.rsi_7 <= 60:
            return True, f"Trailing safe: +{profit:.2f}$ + RSI neutre"

        # Max safe = 5$
        if profit >= 5:
            return True, f"TP safe MAX: +{profit:.2f}$"

        return False, ""

    def _tp_strong(self, pos, snap, profit) -> Tuple[bool, str]:
        """Signal fort (70-85) → Viser 10-15$"""
        if pos.type == 0:  # BUY
            if snap.rsi_7 >= 75 and profit >= 5:
                return True, f"TP fort BUY: RSI={snap.rsi_7:.1f}, +{profit:.2f}$"
            if snap.rsi_7 >= 80 and profit >= 2:
                return True, f"TP fort saturé: RSI={snap.rsi_7:.1f}"
        elif pos.type == 1:  # SELL
            if snap.rsi_7 <= 25 and profit >= 5:
                return True, f"TP fort SELL: RSI={snap.rsi_7:.1f}, +{profit:.2f}$"
            if snap.rsi_7 <= 20 and profit >= 2:
                return True, f"TP fort saturé: RSI={snap.rsi_7:.1f}"

        # MACD retourne (seulement si déjà en profit)
        if pos.type == 0 and snap.macd_hist < 0 and snap.macd_hist_prev >= 0 and profit >= 3:
            return True, f"TP fort MACD: +{profit:.2f}$"
        if pos.type == 1 and snap.macd_hist > 0 and snap.macd_hist_prev <= 0 and profit >= 3:
            return True, f"TP fort MACD: +{profit:.2f}$"

        # Trailing : profit recule de 40% depuis le max
        if self.max_profit_seen >= 5 and profit <= self.max_profit_seen * 0.6:
            return True, f"Trailing fort: max={self.max_profit_seen:.2f}$, actuel=+{profit:.2f}$"

        # Ne jamais laisser +8$ devenir 0
        if self.max_profit_seen >= 8 and profit <= 2:
            return True, f"Protection: max {self.max_profit_seen:.2f}$ → actuel +{profit:.2f}$"

        # Max fort = 15$
        if profit >= 15:
            return True, f"TP fort MAX: +{profit:.2f}$ 🎉"

        return False, ""

    def _tp_aggressive(self, pos, snap, profit) -> Tuple[bool, str]:
        """Signal agressif (85+) → Viser 25-30$+"""
        if pos.type == 0:  # BUY
            if snap.rsi_7 >= 85 and profit >= 10:
                return True, f"TP agressif BUY: RSI={snap.rsi_7:.1f}, +{profit:.2f}$"
        elif pos.type == 1:  # SELL
            if snap.rsi_7 <= 15 and profit >= 10:
                return True, f"TP agressif SELL: RSI={snap.rsi_7:.1f}, +{profit:.2f}$"

        # MACD retourne (seulement gros profit)
        if pos.type == 0 and snap.macd_hist < 0 and snap.macd_hist_prev >= 0 and profit >= 8:
            return True, f"TP agressif MACD: +{profit:.2f}$"
        if pos.type == 1 and snap.macd_hist > 0 and snap.macd_hist_prev <= 0 and profit >= 8:
            return True, f"TP agressif MACD: +{profit:.2f}$"

        # Trailing serré : profit recule de 30% depuis le max
        if self.max_profit_seen >= 10 and profit <= self.max_profit_seen * 0.7:
            return True, f"Trailing agressif: max={self.max_profit_seen:.2f}$, actuel=+{profit:.2f}$"

        # Ne jamais laisser +15$ devenir 0
        if self.max_profit_seen >= 15 and profit <= 5:
            return True, f"Protection agressif: max {self.max_profit_seen:.2f}$"

        # Max agressif = 30$
        if profit >= 30:
            return True, f"TP agressif MAX: +{profit:.2f}$ 🏆🏆"

        return False, ""


# ??????????????????????????????????????????????????????????????
# ORDER EXECUTOR
# ??????????????????????????????????????????????????????????????
class OrderExecutor:
    def open_order(self, direction: str, lot: float) -> bool:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error("Tick indisponible")
            return False

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "BotExpert_v2"
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"? OUVERT {direction} | Lot={lot} | Prix={price:.2f}")
            return True
        else:
            code = result.retcode if result else "None"
            logger.error(f"? ÉCHEC {direction} | Code={code}")
            return False

    def close_position(self, pos, reason: str) -> bool:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return False

        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Close_v2"
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"?? FERMÉ | Raison: {reason} | Profit: {pos.profit:.2f}$")
            return True
        else:
            logger.error(f"? Échec fermeture | Raison: {reason}")
            return False


# ??????????????????????????????????????????????????????????????
# AFFICHAGE DASHBOARD
# ??????????????????????????????????????????????????????????????
class Dashboard:
    @staticmethod
    def display(snap: MarketSnapshot, signal: Signal, positions):
        os.system('cls' if os.name == 'nt' else 'clear')

        print("????????????????????????????????????????????????????????????")
        print("?     ?? BOT EXPERT TRADING BTC - 40+ SCÉNARIOS          ?")
        print("????????????????????????????????????????????????????????????")
        print(f"?  ? {datetime.now().strftime('%H:%M:%S')}                                          ?")
        print("????????????????????????????????????????????????????????????")
        print(f"?  ?? Prix    : {snap.prix:>10.2f} $                          ?")
        print(f"?  ?? RSI(7)  : {snap.rsi_7:>6.1f}  | RSI(14): {snap.rsi_14:>6.1f}           ?")
        print(f"?  ?? Stoch K : {snap.stoch_k:>6.1f}  | Stoch D: {snap.stoch_d:>6.1f}           ?")
        print(f"?  ?? EMA9/21 : {snap.ema_9:>10.2f} / {snap.ema_21:>10.2f}       ?")
        print(f"?  ?? EMA50   : {snap.ema_50:>10.2f} | EMA200: {snap.ema_200:>10.2f}  ?")
        print(f"?  ?? ATR     : {snap.atr:>8.2f}  ({snap.atr_percent:.2f}%)                ?")
        print(f"?  ?? BB      : {snap.bollinger_lower:.0f} | {snap.bollinger_mid:.0f} | {snap.bollinger_upper:.0f}    ?")
        print(f"?  ?? MACD H  : {snap.macd_hist:>+8.2f}                            ?")
        print(f"?  ?? Volume  : x{snap.volume_ratio:.1f}                                  ?")
        print("????????????????????????????????????????????????????????????")
        print(f"?  ? Prix 1b: {snap.price_change_1b:>+6.2f}% | 3b: {snap.price_change_3b:>+6.2f}%           ?")
        print(f"?  ? Prix 1h: {snap.price_change_12b:>+6.2f}% | 2h: {snap.price_change_24b:>+6.2f}%           ?")
        print(f"?  Move ATR  : 1b={snap.move_1b_in_atr:.1f}x | 3b={snap.move_3b_in_atr:.1f}x | 5b={snap.move_5b_in_atr:.1f}x  ?")
        print(f"?  Bougies   : ??×{snap.consecutive_green} | ??×{snap.consecutive_red}                       ?")

        if snap.is_new_high_2h:
            print("?  ?? NOUVEAU PIC HAUT 2H !                              ?")
        if snap.is_new_low_2h:
            print("?  ?? NOUVEAU CREUX BAS 2H !                             ?")
        if snap.is_new_high_4h:
            print("?  ???? NOUVEAU PIC HAUT 4H !                            ?")
        if snap.is_new_low_4h:
            print("?  ???? NOUVEAU CREUX BAS 4H !                           ?")

        print("????????????????????????????????????????????????????????????")

        # Signal
        dir_icon = "?? BUY" if signal.direction == "BUY" else "?? SELL" if signal.direction == "SELL" else "? ATTENTE"
        print(f"?  SIGNAL: {dir_icon} | Score: {signal.score}/100 | Scénarios: {signal.scenario_count}  ?")

        if signal.is_aggressive:
            print("?  ??? MODE AGRESSIF ACTIVÉ ???                       ?")

        print("????????????????????????????????????????????????????????????")
        print("?  Raisons:                                              ?")
        for r in signal.reasons[:10]:  # Max 10 lignes affichées
            print(f"?    ? {r[:52]:<52} ?")

        print("????????????????????????????????????????????????????????????")

        # Position
        if positions:
            pos = positions[0]
            pos_type = "BUY" if pos.type == 0 else "SELL"
            profit_icon = "??" if pos.profit >= 0 else "??"
            print(f"?  {profit_icon} POSITION: {pos_type} | Profit: {pos.profit:>+8.2f}$          ?")
        else:
            print("?  ?? Aucune position ouverte                           ?")

        print("????????????????????????????????????????????????????????????")


# ??????????????????????????????????????????????????????????????
# BOUCLE PRINCIPALE
# ??????????????????????????????????????????????????????????????
def main():
    if not mt5.initialize():
        print("? Impossible de se connecter à MetaTrader 5")
        return

    if not mt5.symbol_select(SYMBOL, True):
        print(f"? Symbole {SYMBOL} non disponible")
        mt5.shutdown()
        return

    data_engine = DataEngine(SYMBOL, TIMEFRAME, HIST_BOUGIES)
    signal_gen = SignalGenerator()
    risk_mgr = RiskManager()
    executor = OrderExecutor()
    dashboard = Dashboard()

    logger.info("??? BOT EXPERT V2 DÉMARRÉ - 40+ Scénarios ???")
    print("?? Démarrage du Bot Expert Trading...")
    time.sleep(1)

    try:
        while True:
            snap = data_engine.get_snapshot()

            if snap is None:
                print("? En attente de données marché...")
                time.sleep(LOOP_INTERVAL)
                continue

            signal = signal_gen.evaluate(snap)
            positions = mt5.positions_get(symbol=SYMBOL)

            # Affichage
            dashboard.display(snap, signal, positions)

            # --- Gestion position existante ---
            if positions:
                pos = positions[0]
                should_close, reason = risk_mgr.should_close(pos, snap)
                if should_close:
                    executor.close_position(pos, reason)
                    logger.info(f"FERMETURE: {reason}")

            # --- Ouverture nouvelle position (1 seul trade max) ---
            elif signal.direction != "NONE":
                lot = risk_mgr.compute_lot(signal, snap.atr, snap.prix)
                success = executor.open_order(signal.direction, lot)
                if success:
                    risk_mgr.on_new_trade(signal)
                    logger.info(
                        f"OUVERTURE {signal.direction} | Score={signal.score} | "
                        f"Scénarios={signal.scenario_count} | Lot={lot} | "
                        f"Raisons: {'; '.join(signal.reasons[:5])}"
                    )
            else:
                print(f"\n  ?? Score actuel: {signal.score}/{SCORE_MIN_ENTRY} requis")

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        logger.info("??? BOT ARRÊTÉ PAR L'UTILISATEUR ???")
        mt5.shutdown()
        print("\n?? Bot arrêté proprement.")
    except Exception as e:
        logger.error(f"ERREUR FATALE: {e}")
        mt5.shutdown()
        raise


if __name__ == "__main__":
    main()
