# -*- coding: utf-8 -*-
"""
BOT EXPERT TRADING BTC/USD - VERSION ULTIME
40+ Scenarios d'opportunite - Architecture Modulaire
Lot fixe 0.05 - 1 seul trade
"""

import MetaTrader5 as mt5
import pandas as pd
import time
import os
import logging
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

# ==============================================================
# CONFIGURATION GLOBALE
# ==============================================================
SCRIPT_DIR = os.path.join(os.path.expanduser("~"), "BotTrading")
os.makedirs(SCRIPT_DIR, exist_ok=True)

SYMBOL = "BTCUSD"
LOT = 0.05              # Lot fixe unique
MAGIC_NUMBER = 123456
TIMEFRAME = mt5.TIMEFRAME_M5
HIST_BOUGIES = 2016     # 7 jours
LOOP_INTERVAL = 2       # secondes
LOG_FILE = os.path.join(SCRIPT_DIR, "bot_expert.log")
SCORE_MIN_ENTRY = 150   # Score minimum 1er trade (150/100 = tres selectif)
SCORE_DCA_MIN = 200     # Score minimum 2eme trade (DCA = tres sur)
SCORE_AGGRESSIVE = 180  # Seuil mode agressif (combo extreme)
WARMUP_LOOPS = 15       # Attendre 15 boucles (~30s) observer le marche avant de trader
MAX_POSITIONS = 2       # 1 seul trade normal + 1 DCA urgence seulement (score 200+)
STOP_LOSS_MAX = 15      # Perte max en $ PAR TRADE (stop fixe -15$)
STOP_LOSS_TOTAL = 20    # Perte max TOTALE pour les 2 trades combines
COOLDOWN_SECONDS = 60   # Attente apres fermeture avant re-entrer
MAX_SPREAD_USD = 60     # Spread max acceptable (XM = ~50$, marge de securite)
MIN_SCORE_DIFF = 15     # Difference min entre BUY et SELL score
TRADE_HISTORY_FILE = os.path.join(SCRIPT_DIR, "trades_history.csv")
MAX_CONSECUTIVE_LOSSES = 3   # Apres 3 pertes d'affilee, augmenter le seuil
DAILY_LOSS_LIMIT = 12        # Arret si perte totale session >= 12$
DAILY_PROFIT_TARGET = 20     # Objectif journalier (info seulement)
# RSI seuils d'entree (plus precis que le score seul)
RSI_BUY_1 = 18              # 1er trade BUY si RSI7 <= cette valeur
RSI_BUY_DCA = 12            # DCA BUY si RSI7 <= cette valeur
RSI_SELL_1 = 79             # 1er trade SELL si RSI7 >= cette valeur
RSI_SELL_DCA = 87           # DCA SELL si RSI7 >= cette valeur
RSI_NEUTRAL_LOW = 40        # RSI zone neutre basse (apres SL, attendre que RSI passe ici)
RSI_NEUTRAL_HIGH = 60       # RSI zone neutre haute
# Stop Loss reel en points (envoye a MT5, visible sur le graphique)
SL_POINTS = 300             # Stop Loss en points (= ~15$ pour 0.05 lot)
TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M15  # Timeframe de confirmation
TIMEFRAME_FOND = mt5.TIMEFRAME_M30     # Timeframe de fond (tendance generale)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("BotExpert")


# ==============================================================
# STRUCTURES DE DONNEES
# ==============================================================
@dataclass
class MarketSnapshot:
    # Prix
    prix: float
    open_price: float
    high_price: float
    low_price: float

    # RSI
    rsi_7: float
    rsi_14: float
    rsi_21: float
    rsi_prev_7: float
    rsi_prev_14: float

    # Moyennes mobiles
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float

    # Volatilite
    atr: float
    atr_percent: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_mid: float
    bollinger_width: float

    # Momentum
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    stoch_k: float
    stoch_d: float

    # Volume
    volume_ratio: float
    obv_slope: float

    # Mouvements de prix (%)
    price_change_1b: float
    price_change_3b: float
    price_change_5b: float
    price_change_12b: float
    price_change_24b: float

    # Mouvements relatifs a ATR (CRUCIAL pour BTC)
    move_1b_in_atr: float
    move_3b_in_atr: float
    move_5b_in_atr: float

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
    rsi_slope_5: float
    rsi_slope_10: float
    price_slope_5: float
    price_slope_10: float

    # Bougies japonaises
    is_hammer: bool
    is_shooting_star: bool
    is_engulfing_bull: bool
    is_engulfing_bear: bool
    is_doji: bool
    candle_body_ratio: float

    # Seuils adaptatifs
    quantile_high_rsi: float
    quantile_low_rsi: float

    # Contexte temporel
    consecutive_green: int
    consecutive_red: int

    # Multi-timeframe M15 (confirmation)
    rsi_14_m15: float
    ema_50_m15: float
    ema_200_m15: float
    macd_hist_m15: float
    trend_m15: str          # "UP", "DOWN", "NEUTRAL"

    # Multi-timeframe M30 (tendance de fond)
    rsi_14_m30: float
    trend_m30: str          # "UP", "DOWN", "NEUTRAL"
    macd_hist_m30: float


@dataclass
class Signal:
    direction: str          # "BUY", "SELL", "NONE"
    score: int              # 0-100+
    reasons: List[str]
    is_aggressive: bool
    scenario_count: int


# ==============================================================
# DATA ENGINE - Calcul de tous les indicateurs
# ==============================================================
class DataEngine:
    def __init__(self, symbol: str, timeframe: int, history_size: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.history_size = history_size

    def get_snapshot(self) -> Optional[MarketSnapshot]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.history_size)
        if rates is None or len(rates) < 250:
            logger.warning("Donnees insuffisantes")
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

        # --- Mouvements en multiples d'ATR ---
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

        # --- Bougies consecutives ---
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

        # --- Multi-timeframe M15 ---
        rates_m15 = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME_CONFIRM, 0, 200)
        rsi_14_m15 = 50.0
        ema_50_m15 = last['close']
        ema_200_m15 = last['close']
        macd_hist_m15 = 0.0
        trend_m15 = "NEUTRAL"
        if rates_m15 is not None and len(rates_m15) >= 200:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15['rsi_14'] = RSIIndicator(df_m15['close'], 14).rsi()
            df_m15['ema_50'] = EMAIndicator(df_m15['close'], 50).ema_indicator()
            df_m15['ema_200'] = EMAIndicator(df_m15['close'], 200).ema_indicator()
            macd_m15 = MACD(df_m15['close'])
            df_m15['macd_hist'] = macd_m15.macd_diff()
            last_m15 = df_m15.iloc[-1]
            rsi_14_m15 = last_m15['rsi_14']
            ema_50_m15 = last_m15['ema_50']
            ema_200_m15 = last_m15['ema_200']
            macd_hist_m15 = last_m15['macd_hist']
            if last_m15['ema_50'] > last_m15['ema_200'] and last_m15['close'] > last_m15['ema_50']:
                trend_m15 = "UP"
            elif last_m15['ema_50'] < last_m15['ema_200'] and last_m15['close'] < last_m15['ema_50']:
                trend_m15 = "DOWN"

        # --- Multi-timeframe M30 (tendance de fond) ---
        rates_m30 = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME_FOND, 0, 200)
        rsi_14_m30 = 50.0
        macd_hist_m30 = 0.0
        trend_m30 = "NEUTRAL"
        if rates_m30 is not None and len(rates_m30) >= 200:
            df_m30 = pd.DataFrame(rates_m30)
            df_m30['rsi_14'] = RSIIndicator(df_m30['close'], 14).rsi()
            df_m30['ema_50'] = EMAIndicator(df_m30['close'], 50).ema_indicator()
            df_m30['ema_200'] = EMAIndicator(df_m30['close'], 200).ema_indicator()
            macd_m30 = MACD(df_m30['close'])
            df_m30['macd_hist'] = macd_m30.macd_diff()
            last_m30 = df_m30.iloc[-1]
            rsi_14_m30 = last_m30['rsi_14']
            macd_hist_m30 = last_m30['macd_hist']
            if last_m30['ema_50'] > last_m30['ema_200'] and last_m30['close'] > last_m30['ema_50']:
                trend_m30 = "UP"
            elif last_m30['ema_50'] < last_m30['ema_200'] and last_m30['close'] < last_m30['ema_50']:
                trend_m30 = "DOWN"

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
            consecutive_red=consecutive_red,
            rsi_14_m15=rsi_14_m15,
            ema_50_m15=ema_50_m15,
            ema_200_m15=ema_200_m15,
            macd_hist_m15=macd_hist_m15,
            trend_m15=trend_m15,
            rsi_14_m30=rsi_14_m30,
            trend_m30=trend_m30,
            macd_hist_m30=macd_hist_m30
        )


# ==============================================================
# DATA VALIDATOR - Verification integrite des donnees
# ==============================================================
class DataValidator:
    """Verifie que les donnees historiques sont fiables avant de trader"""

    def __init__(self):
        self.last_valid_price = 0
        self.last_data_time = 0
        self.data_warnings = []

    def validate_snapshot(self, snap: MarketSnapshot) -> Tuple[bool, List[str]]:
        """Verifie la coherence du snapshot. Retourne (ok, warnings)"""
        warnings = []

        # 1. Prix aberrant (0 ou negatif)
        if snap.prix <= 0:
            warnings.append("CRITIQUE: Prix <= 0")
            return False, warnings

        # 2. Prix a change de plus de 10% depuis derniere lecture (anomalie)
        if self.last_valid_price > 0:
            change_pct = abs(snap.prix - self.last_valid_price) / self.last_valid_price * 100
            if change_pct > 10:
                warnings.append(f"CRITIQUE: Prix change de {change_pct:.1f}% en 2s (anomalie)")
                return False, warnings

        # 3. RSI hors limites (doit etre 0-100)
        if not (0 <= snap.rsi_7 <= 100) or not (0 <= snap.rsi_14 <= 100):
            warnings.append(f"CRITIQUE: RSI hors limites (RSI7={snap.rsi_7:.1f}, RSI14={snap.rsi_14:.1f})")
            return False, warnings

        # 4. ATR negatif ou zero (impossible)
        if snap.atr <= 0:
            warnings.append("CRITIQUE: ATR <= 0")
            return False, warnings

        # 5. Donnees gelees (meme prix que la derniere fois = marche ferme?)
        if self.last_valid_price > 0 and snap.prix == self.last_valid_price:
            elapsed = time.time() - self.last_data_time
            if elapsed > 60:  # Meme prix depuis > 60s
                warnings.append(f"ATTENTION: Prix inchange depuis {elapsed:.0f}s (marche gele?)")
                # Pas critique, on continue mais avec avertissement

        # 6. Volume zero (pas de marche)
        if snap.volume_ratio <= 0:
            warnings.append("ATTENTION: Volume = 0")

        # 7. Bollinger inversees (anomalie indicateur)
        if snap.bollinger_upper < snap.bollinger_lower:
            warnings.append("CRITIQUE: Bollinger inversees")
            return False, warnings

        # 8. EMA200 aberrante (trop loin du prix = donnees insuffisantes)
        if snap.ema_200 > 0:
            ema_dist = abs(snap.prix - snap.ema_200) / snap.prix * 100
            if ema_dist > 20:
                warnings.append(f"ATTENTION: EMA200 a {ema_dist:.1f}% du prix (donnees recentes?)")

        # 9. Stochastique hors limites
        if not (0 <= snap.stoch_k <= 100) or not (0 <= snap.stoch_d <= 100):
            warnings.append(f"ATTENTION: Stoch hors limites K={snap.stoch_k:.1f} D={snap.stoch_d:.1f}")

        # Tout est OK, mettre a jour
        self.last_valid_price = snap.prix
        self.last_data_time = time.time()
        self.data_warnings = warnings

        return True, warnings

    def validate_trade_history(self) -> Tuple[bool, str]:
        """Verifie que le fichier trades_history.csv est valide"""
        if not os.path.exists(TRADE_HISTORY_FILE):
            return True, "Pas de fichier historique (premier lancement)"

        try:
            size = os.path.getsize(TRADE_HISTORY_FILE)
            if size == 0:
                return True, "Fichier historique vide"

            # Lire et verifier le format
            df = pd.read_csv(TRADE_HISTORY_FILE)

            # Verifier colonnes attendues
            expected_cols = ['date', 'profit', 'cumul', 'trades', 'wins']
            if not all(col in df.columns for col in expected_cols):
                # Fichier corrompu, le recreer
                os.rename(TRADE_HISTORY_FILE, TRADE_HISTORY_FILE + ".bak")
                return False, "Fichier historique corrompu (backup cree, nouveau fichier)"

            # Verifier coherence: wins <= trades
            if len(df) > 0:
                last_row = df.iloc[-1]
                if last_row['wins'] > last_row['trades']:
                    return False, "Incoherence: wins > trades dans historique"

            # Verifier taille (limiter a 1000 lignes max)
            if len(df) > 1000:
                # Garder les 500 derniers
                df.tail(500).to_csv(TRADE_HISTORY_FILE, index=False)
                return True, f"Historique tronque: {len(df)} -> 500 lignes"

            return True, f"Historique OK: {len(df)} trades, cumul={last_row['cumul']:.2f}$"

        except Exception as e:
            return False, f"Erreur lecture historique: {e}"

    def check_mt5_connection(self) -> Tuple[bool, str]:
        """Verifie que MT5 est toujours connecte et que le symbole est actif"""
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            return False, "Symbole non disponible"

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return False, "Tick non disponible"

        # Verifier que le tick n'est pas trop vieux (> 30s)
        tick_time = tick.time
        now = int(time.time())
        if now - tick_time > 30:
            return False, f"Tick ancien: {now - tick_time}s (marche ferme?)"

        return True, "MT5 OK"

    def full_check(self, snap: Optional[MarketSnapshot] = None) -> Tuple[bool, List[str]]:
        """Check complet : MT5 + donnees + historique"""
        all_warnings = []

        # Check MT5
        mt5_ok, mt5_msg = self.check_mt5_connection()
        if not mt5_ok:
            all_warnings.append(f"MT5: {mt5_msg}")
            return False, all_warnings

        # Check snapshot
        if snap is not None:
            snap_ok, snap_warnings = self.validate_snapshot(snap)
            all_warnings.extend(snap_warnings)
            if not snap_ok:
                return False, all_warnings

        # Check historique (toutes les 50 iterations = ~100s)
        # On ne le fait pas a chaque boucle pour la perf
        return True, all_warnings


# ==============================================================
# SIGNAL GENERATOR - 64 SCENARIOS D'OPPORTUNITE
# ==============================================================
class SignalGenerator:

    def evaluate(self, snap: MarketSnapshot) -> Signal:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # ==========================================================
        # CATEGORIE 1 : RSI - Scenarios 1 a 8
        # ==========================================================

        # S1 : RSI7 survendu adaptatif
        if snap.rsi_7 <= snap.quantile_low_rsi:
            buy_score += 20
            buy_reasons.append(f"[S1] RSI7={snap.rsi_7:.1f} <= seuil {snap.quantile_low_rsi:.1f}")

        # S2 : RSI7 surachete adaptatif
        if snap.rsi_7 >= snap.quantile_high_rsi:
            sell_score += 20
            sell_reasons.append(f"[S2] RSI7={snap.rsi_7:.1f} >= seuil {snap.quantile_high_rsi:.1f}")

        # S3 : RSI14 zone forte achat
        if snap.rsi_14 <= 35:
            buy_score += 15
            buy_reasons.append(f"[S3] RSI14={snap.rsi_14:.1f} survendu")

        # S4 : RSI14 zone forte vente
        if snap.rsi_14 >= 65:
            sell_score += 15
            sell_reasons.append(f"[S4] RSI14={snap.rsi_14:.1f} surachete")

        # S5 : RSI7 extreme bas
        if snap.rsi_7 <= 25:
            buy_score += 25
            buy_reasons.append(f"[S5] RSI7={snap.rsi_7:.1f} EXTREME BAS")

        # S6 : RSI7 extreme haut
        if snap.rsi_7 >= 75:
            sell_score += 25
            sell_reasons.append(f"[S6] RSI7={snap.rsi_7:.1f} EXTREME HAUT")

        # S7 : RSI rebond rapide depuis extreme
        if snap.rsi_prev_7 <= 20 and snap.rsi_7 > 25:
            buy_score += 15
            buy_reasons.append(f"[S7] RSI7 rebondit {snap.rsi_prev_7:.1f} -> {snap.rsi_7:.1f}")

        # S8 : RSI chute rapide depuis extreme
        if snap.rsi_prev_7 >= 80 and snap.rsi_7 < 75:
            sell_score += 15
            sell_reasons.append(f"[S8] RSI7 chute {snap.rsi_prev_7:.1f} -> {snap.rsi_7:.1f}")

        # ==========================================================
        # CATEGORIE 2 : MOUVEMENT BRUSQUE (ATR) - Scenarios 9 a 14
        # ==========================================================

        # S9 : Micro-burst 1 bougie > 1.5x ATR
        if snap.move_1b_in_atr >= 1.5:
            if snap.price_change_1b > 0:
                if snap.rsi_7 < 65:
                    buy_score += 15
                    buy_reasons.append(f"[S9] Micro-burst HAUT {snap.move_1b_in_atr:.1f}xATR RSI frais")
                else:
                    sell_score += 15
                    sell_reasons.append(f"[S9] Micro-burst HAUT {snap.move_1b_in_atr:.1f}xATR RSI chaud")
            else:
                if snap.rsi_7 > 35:
                    sell_score += 15
                    sell_reasons.append(f"[S9] Micro-burst BAS {snap.move_1b_in_atr:.1f}xATR RSI frais")
                else:
                    buy_score += 15
                    buy_reasons.append(f"[S9] Micro-burst BAS {snap.move_1b_in_atr:.1f}xATR RSI epuise")

        # S10 : Rush 3 bougies > 2x ATR
        if snap.move_3b_in_atr >= 2.0:
            if snap.price_change_3b > 0:
                if snap.rsi_7 < 70:
                    buy_score += 20
                    buy_reasons.append(f"[S10] Rush HAUT 3b={snap.move_3b_in_atr:.1f}xATR momentum")
                else:
                    sell_score += 20
                    sell_reasons.append(f"[S10] Rush HAUT 3b={snap.move_3b_in_atr:.1f}xATR RSI haut retour")
            else:
                if snap.rsi_7 > 30:
                    sell_score += 20
                    sell_reasons.append(f"[S10] Chute 3b={snap.move_3b_in_atr:.1f}xATR momentum baissier")
                else:
                    buy_score += 20
                    buy_reasons.append(f"[S10] Chute 3b={snap.move_3b_in_atr:.1f}xATR RSI bas rebond")

        # S11 : Mouvement violent 3b > 3x ATR
        if snap.move_3b_in_atr >= 3.0:
            if snap.price_change_3b < 0:
                buy_score += 25
                buy_reasons.append(f"[S11] CRASH violent {snap.move_3b_in_atr:.1f}xATR rebond fort")
            else:
                sell_score += 25
                sell_reasons.append(f"[S11] PUMP violent {snap.move_3b_in_atr:.1f}xATR correction")

        # S12 : Mouvement 1h en %
        if snap.price_change_12b <= -1.5:
            buy_score += 20
            buy_reasons.append(f"[S12] Crash 1h: {snap.price_change_12b:.2f}% rebond")
        if snap.price_change_12b >= 1.5:
            sell_score += 20
            sell_reasons.append(f"[S12] Pump 1h: +{snap.price_change_12b:.2f}% correction")

        # S13 : Chute 2h progressive
        if snap.price_change_24b <= -2.5:
            buy_score += 20
            buy_reasons.append(f"[S13] Chute 2h: {snap.price_change_24b:.2f}% zone achat")

        # S14 : Hausse 2h progressive
        if snap.price_change_24b >= 2.5:
            sell_score += 20
            sell_reasons.append(f"[S14] Hausse 2h: +{snap.price_change_24b:.2f}% zone vente")

        # S14b : Acceleration 5b > 2.5x ATR
        if snap.move_5b_in_atr >= 2.5:
            if snap.price_change_5b > 0 and snap.rsi_7 >= 65:
                sell_score += 15
                sell_reasons.append(f"[S14b] Accel 5b {snap.move_5b_in_atr:.1f}xATR+RSI haut vente")
            elif snap.price_change_5b < 0 and snap.rsi_7 <= 35:
                buy_score += 15
                buy_reasons.append(f"[S14b] Accel 5b {snap.move_5b_in_atr:.1f}xATR+RSI bas achat")

        # ==========================================================
        # CATEGORIE 3 : BREAKOUT / PIC - Scenarios 15 a 20
        # ==========================================================

        # S15 : Nouveau pic haut 2h + RSI frais
        if snap.is_new_high_2h and snap.rsi_7 < 70:
            buy_score += 20
            buy_reasons.append("[S15] BREAKOUT HAUT 2h + RSI frais continuation")

        # S16 : Nouveau pic haut 2h + RSI epuise
        if snap.is_new_high_2h and snap.rsi_7 >= 70:
            sell_score += 15
            sell_reasons.append("[S16] Pic 2h + RSI epuise retournement")

        # S17 : Nouveau creux 2h + RSI frais
        if snap.is_new_low_2h and snap.rsi_7 > 30:
            sell_score += 20
            sell_reasons.append("[S17] BREAKDOWN BAS 2h + RSI frais continuation")

        # S18 : Nouveau creux 2h + RSI epuise
        if snap.is_new_low_2h and snap.rsi_7 <= 30:
            buy_score += 15
            buy_reasons.append("[S18] Creux 2h + RSI epuise rebond")

        # S19 : Breakout 4h haut
        if snap.is_new_high_4h:
            if snap.rsi_7 < 75:
                buy_score += 25
                buy_reasons.append("[S19] BREAKOUT HAUT 4h forte continuation")
            else:
                sell_score += 10
                sell_reasons.append("[S19] Pic 4h + RSI sature")

        # S20 : Breakdown 4h bas
        if snap.is_new_low_4h:
            if snap.rsi_7 > 25:
                sell_score += 25
                sell_reasons.append("[S20] BREAKDOWN BAS 4h forte continuation")
            else:
                buy_score += 10
                buy_reasons.append("[S20] Creux 4h + RSI epuise rebond")

        # ==========================================================
        # CATEGORIE 4 : DIVERGENCES - Scenarios 21 a 24
        # ==========================================================

        # S21 : Divergence haussiere rapide
        if snap.rsi_slope_5 > 0.5 and snap.price_slope_5 < -0.05:
            buy_score += 20
            buy_reasons.append("[S21] Divergence haussiere rapide RSI+ Prix-")

        # S22 : Divergence baissiere rapide
        if snap.rsi_slope_5 < -0.5 and snap.price_slope_5 > 0.05:
            sell_score += 20
            sell_reasons.append("[S22] Divergence baissiere rapide RSI- Prix+")

        # S23 : Divergence haussiere longue
        if snap.rsi_slope_10 > 0.3 and snap.price_slope_10 < -0.03:
            buy_score += 15
            buy_reasons.append("[S23] Divergence haussiere prolongee")

        # S24 : Divergence baissiere longue
        if snap.rsi_slope_10 < -0.3 and snap.price_slope_10 > 0.03:
            sell_score += 15
            sell_reasons.append("[S24] Divergence baissiere prolongee")

        # ==========================================================
        # CATEGORIE 5 : BOLLINGER BANDS - Scenarios 25 a 28
        # ==========================================================

        # S25 : Prix touche bande inferieure
        if snap.prix <= snap.bollinger_lower:
            buy_score += 15
            buy_reasons.append("[S25] Prix touche Bollinger BAS rebond")

        # S26 : Prix touche bande superieure
        if snap.prix >= snap.bollinger_upper:
            sell_score += 15
            sell_reasons.append("[S26] Prix touche Bollinger HAUT rejet")

        # S27 : Squeeze BB + mouvement haussier
        if snap.bollinger_width < 1.5 and snap.price_change_1b > 0.3:
            buy_score += 15
            buy_reasons.append("[S27] Squeeze BB + expansion haussiere")

        # S28 : Squeeze BB + mouvement baissier
        if snap.bollinger_width < 1.5 and snap.price_change_1b < -0.3:
            sell_score += 15
            sell_reasons.append("[S28] Squeeze BB + expansion baissiere")

        # ==========================================================
        # CATEGORIE 6 : MACD - Scenarios 29 a 32
        # ==========================================================

        # S29 : Croisement MACD haussier
        if snap.macd_hist > 0 and snap.macd_hist_prev <= 0:
            buy_score += 15
            buy_reasons.append("[S29] Croisement MACD haussier")

        # S30 : Croisement MACD baissier
        if snap.macd_hist < 0 and snap.macd_hist_prev >= 0:
            sell_score += 15
            sell_reasons.append("[S30] Croisement MACD baissier")

        # S31 : MACD histogramme accelere haussier
        if snap.macd_hist > 0 and snap.macd_hist_prev > 0 and snap.macd_hist > snap.macd_hist_prev * 1.5:
            buy_score += 10
            buy_reasons.append("[S31] MACD momentum haussier accelere")

        # S32 : MACD histogramme accelere baissier
        if snap.macd_hist < 0 and snap.macd_hist_prev < 0 and snap.macd_hist < snap.macd_hist_prev * 1.5:
            sell_score += 10
            sell_reasons.append("[S32] MACD momentum baissier accelere")

        # ==========================================================
        # CATEGORIE 7 : STOCHASTIQUE - Scenarios 33 a 35
        # ==========================================================

        # S33 : Stoch survendu + croisement
        if snap.stoch_k < 20 and snap.stoch_k > snap.stoch_d:
            buy_score += 15
            buy_reasons.append(f"[S33] Stoch survendu + croisement K={snap.stoch_k:.0f}")

        # S34 : Stoch surachete + croisement
        if snap.stoch_k > 80 and snap.stoch_k < snap.stoch_d:
            sell_score += 15
            sell_reasons.append(f"[S34] Stoch surachete + croisement K={snap.stoch_k:.0f}")

        # S35 : Double confirmation Stoch + RSI
        if snap.stoch_k < 25 and snap.rsi_7 < 30:
            buy_score += 20
            buy_reasons.append("[S35] Double survendu: Stoch+RSI")
        if snap.stoch_k > 75 and snap.rsi_7 > 70:
            sell_score += 20
            sell_reasons.append("[S35] Double surachete: Stoch+RSI")

        # ==========================================================
        # CATEGORIE 8 : VOLUME - Scenarios 36 a 38
        # ==========================================================

        # S36 : Volume spike
        if snap.volume_ratio > 1.5:
            buy_score += 8
            sell_score += 8
            buy_reasons.append(f"[S36] Volume x{snap.volume_ratio:.1f}")
            sell_reasons.append(f"[S36] Volume x{snap.volume_ratio:.1f}")

        # S37 : Volume explosion + direction
        if snap.volume_ratio > 2.5 and snap.price_change_1b > 0.2:
            buy_score += 15
            buy_reasons.append(f"[S37] Volume explosion HAUSSIER x{snap.volume_ratio:.1f}")
        if snap.volume_ratio > 2.5 and snap.price_change_1b < -0.2:
            sell_score += 15
            sell_reasons.append(f"[S37] Volume explosion BAISSIER x{snap.volume_ratio:.1f}")

        # S38 : OBV confirme direction
        if snap.obv_slope > 0 and snap.price_slope_5 > 0:
            buy_score += 8
            buy_reasons.append("[S38] OBV confirme flux acheteur")
        if snap.obv_slope < 0 and snap.price_slope_5 < 0:
            sell_score += 8
            sell_reasons.append("[S38] OBV confirme flux vendeur")

        # ==========================================================
        # CATEGORIE 9 : BOUGIES JAPONAISES - Scenarios 39 a 42
        # ==========================================================

        # S39 : Marteau en zone basse
        if snap.is_hammer and snap.rsi_7 < 40:
            buy_score += 15
            buy_reasons.append("[S39] Marteau + RSI bas retournement haussier")

        # S40 : Etoile filante en zone haute
        if snap.is_shooting_star and snap.rsi_7 > 60:
            sell_score += 15
            sell_reasons.append("[S40] Etoile filante + RSI haut retournement baissier")

        # S41 : Engulfing haussier
        if snap.is_engulfing_bull:
            buy_score += 15
            buy_reasons.append("[S41] Engulfing haussier renversement")

        # S42 : Engulfing baissier
        if snap.is_engulfing_bear:
            sell_score += 15
            sell_reasons.append("[S42] Engulfing baissier renversement")

        # ==========================================================
        # CATEGORIE 10 : EMA (BONUS SEULEMENT) - Scenarios 43 a 46
        # ==========================================================

        # S43 : Prix au-dessus EMA200
        if snap.prix > snap.ema_200:
            buy_score += 5
            buy_reasons.append("[S43] Prix > EMA200 (tendance)")

        # S44 : Prix en-dessous EMA200
        if snap.prix < snap.ema_200:
            sell_score += 5
            sell_reasons.append("[S44] Prix < EMA200 (tendance)")

        # S45 : Golden cross EMA9/21
        if snap.ema_9 > snap.ema_21 and snap.prix > snap.ema_9:
            buy_score += 8
            buy_reasons.append("[S45] EMA9 > EMA21 + Prix au-dessus")

        # S46 : Death cross EMA9/21
        if snap.ema_9 < snap.ema_21 and snap.prix < snap.ema_9:
            sell_score += 8
            sell_reasons.append("[S46] EMA9 < EMA21 + Prix en-dessous")

        # ==========================================================
        # CATEGORIE 11 : CONTEXTE / MOMENTUM - Scenarios 47 a 52
        # ==========================================================

        # S47 : Contre-tendance RSI fort
        if snap.rsi_7 >= 70 and snap.prix > snap.ema_200:
            sell_score += 8
            sell_reasons.append("[S47] Surachete malgre hausse retournement")
        if snap.rsi_7 <= 30 and snap.prix < snap.ema_200:
            buy_score += 8
            buy_reasons.append("[S47] Survendu malgre baisse rebond")

        # S48 : Sequence bougies rouges (epuisement vendeurs)
        if snap.consecutive_red >= 5:
            buy_score += 15
            buy_reasons.append(f"[S48] {snap.consecutive_red} bougies rouges epuisement vendeurs")

        # S49 : Sequence bougies vertes (epuisement acheteurs)
        if snap.consecutive_green >= 5:
            sell_score += 15
            sell_reasons.append(f"[S49] {snap.consecutive_green} bougies vertes epuisement acheteurs")

        # S50 : Prix proche support 2h
        if snap.distance_from_low_2h < 0.15 and not snap.is_new_low_2h:
            buy_score += 10
            buy_reasons.append(f"[S50] Prix proche support 2h ({snap.distance_from_low_2h:.2f}%)")

        # S51 : Prix proche resistance 2h
        if snap.distance_from_high_2h < 0.15 and not snap.is_new_high_2h:
            sell_score += 10
            sell_reasons.append(f"[S51] Prix proche resistance 2h ({snap.distance_from_high_2h:.2f}%)")

        # S52 : Bougie geante > 2x ATR
        if snap.move_1b_in_atr >= 2.0:
            if snap.price_change_1b > 0:
                if snap.rsi_7 < 65:
                    buy_score += 15
                    buy_reasons.append(f"[S52] Bougie geante haussiere {snap.move_1b_in_atr:.1f}xATR")
                else:
                    sell_score += 10
                    sell_reasons.append("[S52] Bougie geante haussiere + RSI haut epuisement")
            else:
                if snap.rsi_7 > 35:
                    sell_score += 15
                    sell_reasons.append(f"[S52] Bougie geante baissiere {snap.move_1b_in_atr:.1f}xATR")
                else:
                    buy_score += 10
                    buy_reasons.append("[S52] Bougie geante baissiere + RSI bas rebond")

        # ==========================================================
        # CATEGORIE 12 : COMBOS RARES - Scenarios 53 a 58
        # ==========================================================

        # S53 : COMBO ULTIME BUY
        if snap.rsi_7 <= 25 and snap.move_3b_in_atr >= 2.0 and snap.volume_ratio > 1.3:
            buy_score += 30
            buy_reasons.append(f"[S53] COMBO ULTIME: RSI extreme + Crash {snap.move_3b_in_atr:.1f}xATR + Volume")

        # S54 : COMBO ULTIME SELL
        if snap.rsi_7 >= 75 and snap.move_3b_in_atr >= 2.0 and snap.volume_ratio > 1.3:
            sell_score += 30
            sell_reasons.append(f"[S54] COMBO ULTIME: RSI extreme + Pump {snap.move_3b_in_atr:.1f}xATR + Volume")

        # S55 : Triple convergence BUY
        if snap.stoch_k < 20 and snap.rsi_7 < 30 and snap.prix <= snap.bollinger_lower * 1.005:
            buy_score += 25
            buy_reasons.append("[S55] Triple convergence BUY: Stoch+RSI+BB")

        # S56 : Triple convergence SELL
        if snap.stoch_k > 80 and snap.rsi_7 > 70 and snap.prix >= snap.bollinger_upper * 0.995:
            sell_score += 25
            sell_reasons.append("[S56] Triple convergence SELL: Stoch+RSI+BB")

        # S57 : Rebond sur EMA200
        if abs(snap.prix - snap.ema_200) / snap.ema_200 < 0.002:
            if snap.rsi_7 < 50 and snap.prix > snap.ema_200:
                buy_score += 15
                buy_reasons.append("[S57] Rebond sur EMA200 confirme")
            elif snap.rsi_7 > 50 and snap.prix < snap.ema_200:
                sell_score += 15
                sell_reasons.append("[S57] Rejet EMA200 confirme")

        # S58 : Doji apres tendance forte
        if snap.is_doji:
            if snap.consecutive_red >= 3:
                buy_score += 12
                buy_reasons.append(f"[S58] Doji apres {snap.consecutive_red} rouges retournement")
            if snap.consecutive_green >= 3:
                sell_score += 12
                sell_reasons.append(f"[S58] Doji apres {snap.consecutive_green} vertes retournement")

        # ==========================================================
        # CATEGORIE 13 : MULTI-TIMEFRAME M15 - Scenarios 59 a 62
        # ==========================================================

        # S59 : Tendance M15 confirme BUY
        if snap.trend_m15 == "UP":
            buy_score += 10
            buy_reasons.append("[S59] M15 tendance HAUSSIERE confirme")

        # S60 : Tendance M15 confirme SELL
        if snap.trend_m15 == "DOWN":
            sell_score += 10
            sell_reasons.append("[S60] M15 tendance BAISSIERE confirme")

        # S61 : RSI M15 survendu confirme achat M5
        if snap.rsi_14_m15 <= 35 and snap.rsi_7 <= 35:
            buy_score += 15
            buy_reasons.append(f"[S61] M15 RSI={snap.rsi_14_m15:.0f} confirme survendu")

        # S62 : RSI M15 surachete confirme vente M5
        if snap.rsi_14_m15 >= 65 and snap.rsi_7 >= 65:
            sell_score += 15
            sell_reasons.append(f"[S62] M15 RSI={snap.rsi_14_m15:.0f} confirme surachete")

        # S63 : MACD M15 confirme direction M5
        if snap.macd_hist_m15 > 0 and snap.macd_hist > 0:
            buy_score += 8
            buy_reasons.append("[S63] MACD M15+M5 haussier aligne")
        if snap.macd_hist_m15 < 0 and snap.macd_hist < 0:
            sell_score += 8
            sell_reasons.append("[S63] MACD M15+M5 baissier aligne")

        # S64 : Contre-tendance M15 (signal fort si M5 diverge de M15)
        # Si M15 est DOWN mais M5 donne un BUY fort = rebond puissant
        if snap.trend_m15 == "DOWN" and snap.rsi_7 <= 25:
            buy_score += 10
            buy_reasons.append("[S64] Contre-tendance M15: RSI extreme malgre baisse")
        if snap.trend_m15 == "UP" and snap.rsi_7 >= 75:
            sell_score += 10
            sell_reasons.append("[S64] Contre-tendance M15: RSI extreme malgre hausse")

        # ==========================================================
        # CATEGORIE 14 : MULTI-TIMEFRAME M30 (FOND) - Scenarios 65 a 68
        # ==========================================================

        # S65 : Triple alignement M5+M15+M30 HAUSSIER (tres fort)
        if snap.trend_m15 == "UP" and snap.trend_m30 == "UP":
            buy_score += 12
            buy_reasons.append("[S65] TRIPLE ALIGNEMENT M5+M15+M30 HAUSSIER")

        # S66 : Triple alignement M5+M15+M30 BAISSIER (tres fort)
        if snap.trend_m15 == "DOWN" and snap.trend_m30 == "DOWN":
            sell_score += 12
            sell_reasons.append("[S66] TRIPLE ALIGNEMENT M5+M15+M30 BAISSIER")

        # S67 : RSI M30 confirme zone extreme
        if snap.rsi_14_m30 <= 35 and snap.rsi_7 <= 35:
            buy_score += 12
            buy_reasons.append(f"[S67] M30 RSI={snap.rsi_14_m30:.0f} confirme survendu profond")
        if snap.rsi_14_m30 >= 65 and snap.rsi_7 >= 65:
            sell_score += 12
            sell_reasons.append(f"[S67] M30 RSI={snap.rsi_14_m30:.0f} confirme surachete profond")

        # S68 : Signal contre la tendance M30 (prudence ou opportunite)
        # Si M30 monte mais signal SELL sur M5 = trade plus risque, reduire score
        if snap.trend_m30 == "UP" and sell_score > buy_score:
            sell_score -= 5
            sell_reasons.append("[S68] -5 M30 haussier: SELL contre tendance fond")
        if snap.trend_m30 == "DOWN" and buy_score > sell_score:
            buy_score -= 5
            buy_reasons.append("[S68] -5 M30 baissier: BUY contre tendance fond")

        # ==========================================================
        # DECISION FINALE (avec filtre conflit)
        # ==========================================================
        buy_scenarios = len(buy_reasons)
        sell_scenarios = len(sell_reasons)

        # Filtre conflit : si BUY et SELL sont trop proches, ne pas entrer
        score_diff = abs(buy_score - sell_score)
        if score_diff < MIN_SCORE_DIFF and buy_score >= SCORE_MIN_ENTRY and sell_score >= SCORE_MIN_ENTRY:
            # Signal ambigu, on attend
            best = max(buy_score, sell_score)
            reasons = [f"[CONFLIT] BUY={buy_score} vs SELL={sell_score} (diff={score_diff} < {MIN_SCORE_DIFF})"]
            return Signal("NONE", best, reasons, False, 0)

        if buy_score >= SCORE_MIN_ENTRY and buy_score > sell_score:
            return Signal("BUY", buy_score, buy_reasons, buy_score >= SCORE_AGGRESSIVE, buy_scenarios)
        elif sell_score >= SCORE_MIN_ENTRY and sell_score > buy_score:
            return Signal("SELL", sell_score, sell_reasons, sell_score >= SCORE_AGGRESSIVE, sell_scenarios)
        else:
            best = max(buy_score, sell_score)
            reasons = buy_reasons if buy_score >= sell_score else sell_reasons
            count = buy_scenarios if buy_score >= sell_score else sell_scenarios
            return Signal("NONE", best, reasons, False, count)


# ==============================================================
# RISK MANAGER - DCA + Post-SL RSI Reset
# ==============================================================
class RiskManager:

    def __init__(self):
        self.entry_score = 0
        self.max_profit_seen = 0
        self.last_close_time = 0
        self.total_profit = 0
        self.trade_count = 0
        self.win_count = 0
        self.consecutive_losses = 0
        self.session_stopped = False
        self.last_signal_dir = ""
        self.signal_confirm_count = 0
        # DCA
        self.first_trade_score = 0
        self.first_trade_dir = ""
        self.dca_active = False
        # Post-SL : attendre que RSI revienne en zone neutre
        self.waiting_rsi_reset = False
        self.rsi_was_neutral = False

    def on_new_trade(self, signal: Signal, is_dca=False):
        """Appele quand un nouveau trade est ouvert"""
        self.entry_score = signal.score
        self.max_profit_seen = 0
        self.waiting_rsi_reset = False
        self.rsi_was_neutral = False
        if not is_dca:
            self.first_trade_score = signal.score
            self.first_trade_dir = signal.direction
            self.dca_active = False
        else:
            self.dca_active = True

    def on_trade_closed(self, profit: float, was_stop=False):
        """Appele quand un trade est ferme"""
        self.last_close_time = time.time()
        self.total_profit += profit
        self.trade_count += 1
        if profit > 0:
            self.win_count += 1
            self.consecutive_losses = 0
            self.waiting_rsi_reset = False
        else:
            self.consecutive_losses += 1
            if was_stop:
                self.waiting_rsi_reset = True
                self.rsi_was_neutral = False
                logger.info("POST-SL: attente RSI zone neutre avant prochain trade")

        if self.total_profit <= -DAILY_LOSS_LIMIT:
            self.session_stopped = True
            logger.warning(f"SESSION ARRETEE: perte totale {self.total_profit:.2f}$")

        try:
            header_needed = not os.path.exists(TRADE_HISTORY_FILE) or os.path.getsize(TRADE_HISTORY_FILE) == 0
            row = f"{datetime.now()},{profit:.2f},{self.total_profit:.2f},{self.trade_count},{self.win_count}\n"
            with open(TRADE_HISTORY_FILE, 'a') as f:
                if header_needed:
                    f.write("date,profit,cumul,trades,wins\n")
                f.write(row)
        except:
            pass

    def update_rsi_reset(self, rsi: float):
        """Apres un SL, verifie si RSI est revenu en zone neutre (40-60)"""
        if not self.waiting_rsi_reset:
            return
        if RSI_NEUTRAL_LOW <= rsi <= RSI_NEUTRAL_HIGH:
            self.rsi_was_neutral = True
            logger.info(f"POST-SL: RSI={rsi:.1f} zone neutre, nouveau cycle autorise")

    def is_rsi_reset_blocking(self) -> Tuple[bool, str]:
        """True si on attend encore le RSI reset apres un SL"""
        if not self.waiting_rsi_reset:
            return False, ""
        if self.rsi_was_neutral:
            self.waiting_rsi_reset = False
            return False, ""
        return True, "POST-SL: attente RSI zone neutre (40-60) = correction en cours"

    def get_adjusted_score_min(self) -> int:
        """Score minimum adaptatif"""
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return SCORE_MIN_ENTRY + 30
        elif self.consecutive_losses >= 2:
            return SCORE_MIN_ENTRY + 20
        return SCORE_MIN_ENTRY

    def needs_confirmation(self, signal: Signal) -> bool:
        """Apres pertes, exiger 2 lectures consecutives"""
        if self.consecutive_losses < 2:
            self.last_signal_dir = signal.direction
            return False
        if signal.direction == self.last_signal_dir:
            self.signal_confirm_count += 1
        else:
            self.signal_confirm_count = 0
            self.last_signal_dir = signal.direction
        return self.signal_confirm_count < 2

    def is_cooldown_active(self) -> bool:
        if self.last_close_time == 0:
            return False
        return (time.time() - self.last_close_time) < COOLDOWN_SECONDS

    def can_enter(self, signal: Signal, snap) -> Tuple[bool, str]:
        """1er trade : Score >= 150 + RSI extreme + tendance pas contre nous"""
        score_min = self.get_adjusted_score_min()
        if signal.direction == "NONE" or signal.score < score_min:
            return False, f"Score {signal.score} < seuil {score_min}"
        if signal.direction == "BUY" and snap.rsi_7 > RSI_BUY_1:
            return False, f"RSI {snap.rsi_7:.1f} > {RSI_BUY_1} (pas assez bas)"
        if signal.direction == "SELL" and snap.rsi_7 < RSI_SELL_1:
            return False, f"RSI {snap.rsi_7:.1f} < {RSI_SELL_1} (pas assez haut)"

        # FILTRE TENDANCE : ne pas SELL dans une forte tendance haussiere
        if signal.direction == "SELL" and snap.trend_m15 == "UP" and snap.trend_m30 == "UP":
            return False, f"BLOQUE SELL: M15+M30 haussiers (tendance forte contre SELL)"
        if signal.direction == "BUY" and snap.trend_m15 == "DOWN" and snap.trend_m30 == "DOWN":
            return False, f"BLOQUE BUY: M15+M30 baissiers (tendance forte contre BUY)"

        return True, f"Score {signal.score} + RSI {snap.rsi_7:.1f} EXTREME, ENTREE!"

    def can_open_dca(self, signal: Signal, positions, snap) -> Tuple[bool, str]:
        """2eme trade : Score >= 200 + RSI ultra-extreme"""
        if len(positions) != 1:
            return False, ""
        if self.dca_active:
            return False, "DCA deja actif"
        pos = positions[0]
        if pos.profit >= 0:
            return False, "En profit, pas besoin DCA"
        if signal.direction != self.first_trade_dir:
            return False, f"Direction diff ({signal.direction} vs {self.first_trade_dir})"
        if signal.score < SCORE_DCA_MIN:
            return False, f"Score {signal.score} < {SCORE_DCA_MIN} (DCA exige 200+)"
        if signal.direction == "BUY" and snap.rsi_7 > RSI_BUY_DCA:
            return False, f"RSI {snap.rsi_7:.1f} > {RSI_BUY_DCA} (pas assez bas pour DCA)"
        if signal.direction == "SELL" and snap.rsi_7 < RSI_SELL_DCA:
            return False, f"RSI {snap.rsi_7:.1f} < {RSI_SELL_DCA} (pas assez haut pour DCA)"
        return True, f"DCA OK: score {signal.score}>=200 + RSI {snap.rsi_7:.1f} ULTRA-EXTREME"

    def cooldown_remaining(self) -> int:
        if self.last_close_time == 0:
            return 0
        remaining = COOLDOWN_SECONDS - (time.time() - self.last_close_time)
        return max(0, int(remaining))

    def compute_lot(self, signal: Signal, atr: float, prix: float) -> float:
        return LOT

    def should_close_all(self, positions, snap: MarketSnapshot) -> Tuple[bool, str]:
        """Sortie: Stop / Break-even / RSI securise / Trade 100% perdu"""
        total_profit = sum(p.profit for p in positions)
        nb_pos = len(positions)

        if total_profit > self.max_profit_seen:
            self.max_profit_seen = total_profit

        # STOP
        if nb_pos >= 2 and total_profit <= -STOP_LOSS_TOTAL:
            return True, f"STOP TOTAL: {total_profit:.2f}$ (max -{STOP_LOSS_TOTAL}$)"
        if nb_pos == 1 and total_profit <= -STOP_LOSS_MAX:
            return True, f"STOP: {total_profit:.2f}$ (max -{STOP_LOSS_MAX}$)"

        # BREAK-EVEN
        if self.max_profit_seen >= 2 and total_profit <= 0:
            return True, f"Break-even: max +{self.max_profit_seen:.2f}$, retombe {total_profit:.2f}$"

        # TRADE 100% PERDU : RSI completement retourne + en perte
        # BUY entre a RSI bas, si RSI monte a 65+ = le mouvement est parti sans nous
        # SELL entre a RSI haut, si RSI descend a 35- = idem
        if total_profit < -3:
            if self.first_trade_dir == "BUY" and snap.rsi_7 >= 65:
                return True, f"TRADE PERDU: BUY mais RSI={snap.rsi_7:.0f}>=65 {total_profit:.2f}$"
            if self.first_trade_dir == "SELL" and snap.rsi_7 <= 35:
                return True, f"TRADE PERDU: SELL mais RSI={snap.rsi_7:.0f}<=35 {total_profit:.2f}$"

        # RSI REBONDIT = piege double-top/double-bottom
        # SELL: RSI etait a 90, corrige a 60, remonte a 70+ = le pump continue, sortir
        # BUY: RSI etait a 15, remonte a 40, redescend a 30- = la chute continue, sortir
        if total_profit < 0:
            if self.first_trade_dir == "SELL" and snap.rsi_7 >= 70:
                return True, f"RSI REBOND SELL: RSI={snap.rsi_7:.0f}>=70 pump continue {total_profit:.2f}$"
            if self.first_trade_dir == "BUY" and snap.rsi_7 <= 30:
                return True, f"RSI REBOND BUY: RSI={snap.rsi_7:.0f}<=30 chute continue {total_profit:.2f}$"

        # TP MINIMUM 4$ si solde < 200$ (securiser le gain)
        if total_profit >= 4:
            account = mt5.account_info()
            if account and account.balance < 200:
                # RSI doit aussi confirmer (au moins RSI > 40 pour BUY ou < 60 pour SELL)
                if self.first_trade_dir == "BUY" and snap.rsi_7 >= 40:
                    return True, f"TP MIN 4$: +{total_profit:.2f}$ (solde {account.balance:.0f}$<200)"
                if self.first_trade_dir == "SELL" and snap.rsi_7 <= 60:
                    return True, f"TP MIN 4$: +{total_profit:.2f}$ (solde {account.balance:.0f}$<200)"

        # RSI securise le profit (laisser courir au-dela de 4$ si RSI pas encore retourne)
        if total_profit > 0:
            if self.first_trade_dir == "BUY" and snap.rsi_7 >= 60:
                return True, f"RSI securise BUY: RSI={snap.rsi_7:.0f}, profit +{total_profit:.2f}$"
            if self.first_trade_dir == "SELL" and snap.rsi_7 <= 40:
                return True, f"RSI securise SELL: RSI={snap.rsi_7:.0f}, profit +{total_profit:.2f}$"

        return False, ""


# ==============================================================
# ORDER EXECUTOR
# ==============================================================

# Codes d'erreur MT5 courants
MT5_ERROR_CODES = {
    10004: "Requote (prix change)",
    10006: "Requete rejetee",
    10007: "Requete annulee par le trader",
    10010: "Ordre non supporte",
    10011: "Pas assez d'argent (marge insuffisante)",
    10013: "Trade desactive",
    10014: "Volume invalide",
    10015: "Prix invalide",
    10016: "Stops invalides",
    10017: "Trade desactive pour ce symbole",
    10018: "Marche ferme",
    10019: "Pas assez de marge",
    10020: "Requete modifiee mais resultat inconnu",
    10021: "Trop de requetes",
    10024: "Pas de changements",
    10026: "Autotrading desactive sur MT5",
    10027: "Autotrading desactive cote serveur",
    10030: "Type de filling invalide",
}


def get_filling_type():
    """Detecte le type de filling supporte par le symbole sur ce serveur"""
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    filling = info.filling_mode
    if filling & 1:  # FOK supporte
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:  # IOC supporte
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN


class OrderExecutor:
    def __init__(self):
        self.filling_type = None  # None = ne pas envoyer (comme le script initial qui marchait)
        logger.info("OrderExecutor: mode initial SANS type_filling")
        print(f"  Filling : AUTO (sans filling d'abord, comme script initial)")

    def _build_request(self, order_type, volume, price, position_ticket=None):
        """Construit la requete comme le script initial (SANS type_filling)"""
        # Calculer le SL
        point = mt5.symbol_info(SYMBOL).point if mt5.symbol_info(SYMBOL) else 1.0
        if order_type == mt5.ORDER_TYPE_BUY:
            sl = price - SL_POINTS * point
        else:
            sl = price + SL_POINTS * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "deviation": 20,
            "magic": MAGIC_NUMBER
        }
        # Ajouter filling SEULEMENT si on a trouve un qui marche
        if self.filling_type is not None:
            request["type_filling"] = self.filling_type
        if position_ticket is not None:
            request["position"] = position_ticket
        return request

    def _send_with_retry(self, request, label) -> object:
        """Envoie sans filling d'abord, puis retente avec si echec"""
        # Tentative 1 : SANS type_filling (comme le script initial)
        req1 = dict(request)
        req1.pop("type_filling", None)
        result = mt5.order_send(req1)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return result

        # Tentative 2 : avec RETURN
        code = result.retcode if result else 0
        logger.warning(f"{label} sans filling echoue (code={code}), essai RETURN")
        request["type_filling"] = mt5.ORDER_FILLING_RETURN
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.filling_type = mt5.ORDER_FILLING_RETURN
            return result

        # Tentative 3 : avec IOC
        code = result.retcode if result else 0
        logger.warning(f"{label} RETURN echoue (code={code}), essai IOC")
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.filling_type = mt5.ORDER_FILLING_IOC
            return result

        # Tentative 4 : avec FOK
        code = result.retcode if result else 0
        logger.warning(f"{label} IOC echoue (code={code}), essai FOK")
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.filling_type = mt5.ORDER_FILLING_FOK
            return result

        return result

    def open_order(self, direction: str, lot: float) -> bool:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error("Tick indisponible")
            print("  !!! Tick indisponible")
            return False

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "BUY" else tick.bid

        request = self._build_request(order_type, lot, price)

        # Si on a deja trouve un filling qui marche, envoyer direct
        if self.filling_type is not None:
            result = mt5.order_send(request)
        else:
            result = self._send_with_retry(request, f"OPEN {direction}")

        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"OUVERT {direction} | Lot={lot} | Prix={price:.2f} | Ticket={result.order}")
            return True

        # Echec final
        if result is None:
            err = mt5.last_error()
            logger.error(f"ECHEC {direction} | order_send=None | last_error={err}")
            print(f"  !!! ECHEC: order_send=None (erreur MT5: {err})")
        else:
            code = result.retcode
            desc = MT5_ERROR_CODES.get(code, "Erreur inconnue")
            comment = result.comment if result.comment else ""
            logger.error(f"ECHEC {direction} | Code={code} ({desc}) | {comment}")
            print(f"  !!! ECHEC {direction}: [{code}] {desc}")
            if comment:
                print(f"      MT5 dit: {comment}")
            if code == 10011:
                print(f"      -> Solde insuffisant pour ouvrir {lot} lot")
            elif code == 10026:
                print(f"      -> Activez AutoTrading dans MT5 (bouton en haut)")
            elif code == 10018:
                print(f"      -> Le marche est ferme (weekend?)")
        return False

    def close_position(self, pos, reason: str) -> bool:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return False

        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == 0 else tick.ask

        request = self._build_request(order_type, pos.volume, price, pos.ticket)

        if self.filling_type is not None:
            result = mt5.order_send(request)
        else:
            result = self._send_with_retry(request, "CLOSE")

        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"FERME | Raison: {reason} | Profit: {pos.profit:.2f}$")
            return True
        else:
            code = result.retcode if result else 0
            desc = MT5_ERROR_CODES.get(code, "Erreur inconnue")
            logger.error(f"ECHEC fermeture | Code={code} ({desc}) | {reason}")
            print(f"  !!! ECHEC fermeture: [{code}] {desc}")
            return False


# ==============================================================
# AFFICHAGE DASHBOARD
# ==============================================================
class Dashboard:
    @staticmethod
    def display(snap: MarketSnapshot, signal: Signal, positions, risk_mgr=None):
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 58)
        print("      BOT EXPERT TRADING BTC - 64 SCENARIOS")
        print("      Lot: 0.05 | 1 trade max")
        print("=" * 58)
        print(f"  Heure   : {datetime.now().strftime('%H:%M:%S')}")
        # Afficher spread en temps reel
        tick = mt5.symbol_info_tick(SYMBOL)
        spread_val = abs(tick.ask - tick.bid) if tick else 0
        spread_ok = "OK" if spread_val <= MAX_SPREAD_USD else "HAUT!"
        print(f"  Spread  : {spread_val:.1f}$ ({spread_ok})")
        print("-" * 58)
        print(f"  Prix    : {snap.prix:>10.2f} $")
        print(f"  RSI(7)  : {snap.rsi_7:>6.1f}  | RSI(14): {snap.rsi_14:>6.1f}")
        print(f"  Stoch K : {snap.stoch_k:>6.1f}  | Stoch D: {snap.stoch_d:>6.1f}")
        print(f"  EMA9/21 : {snap.ema_9:>10.2f} / {snap.ema_21:>10.2f}")
        print(f"  EMA50   : {snap.ema_50:>10.2f} | EMA200: {snap.ema_200:>10.2f}")
        print(f"  ATR     : {snap.atr:>8.2f}  ({snap.atr_percent:.2f}%)")
        print(f"  BB      : {snap.bollinger_lower:.0f} | {snap.bollinger_mid:.0f} | {snap.bollinger_upper:.0f}")
        print(f"  MACD H  : {snap.macd_hist:>+8.2f}")
        print(f"  Volume  : x{snap.volume_ratio:.1f}")
        print("-" * 58)
        print(f"  D Prix 1b: {snap.price_change_1b:>+6.2f}% | 3b: {snap.price_change_3b:>+6.2f}%")
        print(f"  D Prix 1h: {snap.price_change_12b:>+6.2f}% | 2h: {snap.price_change_24b:>+6.2f}%")
        print(f"  Move ATR : 1b={snap.move_1b_in_atr:.1f}x | 3b={snap.move_3b_in_atr:.1f}x | 5b={snap.move_5b_in_atr:.1f}x")
        print(f"  Bougies  : Vertes x{snap.consecutive_green} | Rouges x{snap.consecutive_red}")
        print(f"  M15 Trend: {snap.trend_m15} | M15 RSI: {snap.rsi_14_m15:.1f} | M15 MACD: {snap.macd_hist_m15:+.1f}")
        print(f"  M30 Trend: {snap.trend_m30} | M30 RSI: {snap.rsi_14_m30:.1f} | M30 MACD: {snap.macd_hist_m30:+.1f}")

        if snap.is_new_high_2h:
            print("  >>> NOUVEAU PIC HAUT 2H !")
        if snap.is_new_low_2h:
            print("  >>> NOUVEAU CREUX BAS 2H !")
        if snap.is_new_high_4h:
            print("  >>> NOUVEAU PIC HAUT 4H !")
        if snap.is_new_low_4h:
            print("  >>> NOUVEAU CREUX BAS 4H !")

        print("-" * 58)

        # Signal
        dir_text = f"BUY" if signal.direction == "BUY" else "SELL" if signal.direction == "SELL" else "ATTENTE"
        mode = ""
        if signal.is_aggressive:
            mode = " [MODE AGRESSIF - TP 25-30$]"
        elif signal.score >= 70:
            mode = " [MODE FORT - TP 10-15$]"
        elif signal.direction != "NONE":
            mode = " [MODE SAFE - TP 1-5$]"

        print(f"  SIGNAL: {dir_text} | Score: {signal.score}/100 | Scenarios: {signal.scenario_count}")
        if mode:
            print(f"  {mode}")

        print("-" * 58)
        print("  Raisons:")
        for r in signal.reasons[:8]:
            print(f"    > {r}")

        print("-" * 58)

        # Position en cours + details temps reel
        if positions:
            total_profit = sum(p.profit for p in positions)
            print(f"  POSITIONS: {len(positions)}/{MAX_POSITIONS} | Profit total: {total_profit:+.2f}$")
            for i, pos in enumerate(positions):
                pos_type = "BUY" if pos.type == 0 else "SELL"
                profit_sign = "+" if pos.profit >= 0 else ""
                duration = ""
                try:
                    open_time = datetime.fromtimestamp(pos.time)
                    elapsed = datetime.now() - open_time
                    minutes = int(elapsed.total_seconds() // 60)
                    seconds = int(elapsed.total_seconds() % 60)
                    duration = f" | {minutes}m{seconds:02d}s"
                except:
                    pass
                label = "1er" if i == 0 else "DCA"
                print(f"    [{label}] {pos_type} | Ouv: {pos.price_open:.2f} | P/L: {profit_sign}{pos.profit:.2f}${duration}")
            if risk_mgr and risk_mgr.max_profit_seen > 0:
                print(f"  Max vu  : +{risk_mgr.max_profit_seen:.2f}$ | Sortie: RSI change de sens")
        else:
            print("  Aucune position ouverte")

        print("-" * 58)

        # Solde et equity en temps reel
        account = mt5.account_info()
        if account:
            equity = account.equity
            balance = account.balance
            margin = account.margin
            free_margin = account.margin_free
            profit_total = account.profit
            credit = account.credit
            vrai_solde = balance  # sans le credit

            print(f"  Balance : {balance:.2f}$ | Equity: {equity:.2f}$")
            if credit > 0:
                print(f"  Credit  : {credit:.2f}$ (bonus XM, non retirable)")
                print(f"  Vrai $  : {balance:.2f}$ (sans bonus)")

            if positions:
                print(f"  Marge   : {margin:.2f}$ | Libre: {free_margin:.2f}$")
                print(f"  P/L     : {profit_total:+.2f}$")
                if margin > 0:
                    margin_level = equity / margin * 100
                    print(f"  Niveau  : {margin_level:.0f}%")
            else:
                pl_calc = equity - balance
                if abs(pl_calc) > 0.01 and credit == 0:
                    print(f"  Diff    : {pl_calc:+.2f}$ (swap/commission)")

        print("=" * 58)

    @staticmethod
    def display_data_warnings(warnings: List[str]):
        """Affiche les avertissements sur les donnees"""
        if warnings:
            print("  [DATA CHECK]")
            for w in warnings:
                print(f"    ! {w}")
            print("-" * 58)


# ==============================================================
# FONCTIONS UTILITAIRES
# ==============================================================
def check_spread() -> Tuple[bool, float]:
    """Verifie que le spread est acceptable"""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False, 0
    spread = abs(tick.ask - tick.bid)
    return spread <= MAX_SPREAD_USD, spread


def reconnect_mt5() -> bool:
    """Tentative de reconnexion a MT5"""
    logger.warning("Tentative de reconnexion MT5...")
    mt5.shutdown()
    time.sleep(5)
    if mt5.initialize():
        mt5.symbol_select(SYMBOL, True)
        logger.info("Reconnexion MT5 reussie")
        return True
    return False


# ==============================================================
# BOUCLE PRINCIPALE
# ==============================================================
def main():
    print("=" * 58)
    print("  DEMARRAGE BOT EXPERT TRADING BTC")
    print("=" * 58)
    print()

    # === CHECK 1 : Connexion MT5 ===
    print("  [1/6] Connexion MetaTrader 5...", end=" ")
    if not mt5.initialize():
        print("ECHEC")
        print("  ERREUR: MetaTrader 5 non demarre ou non installe")
        print("  -> Verifiez que MT5 est ouvert et connecte")
        return
    print("OK")

    # === CHECK 2 : Infos compte ===
    print("  [2/6] Verification compte...", end=" ")
    account_info = mt5.account_info()
    if account_info is None:
        print("ECHEC")
        print("  ERREUR: Impossible de lire les infos du compte")
        mt5.shutdown()
        return
    print(f"OK (Solde: {account_info.balance:.2f}$, Serveur: {account_info.server})")

    # === CHECK 3 : Symbole disponible ===
    print(f"  [3/6] Symbole {SYMBOL}...", end=" ")
    if not mt5.symbol_select(SYMBOL, True):
        print("ECHEC")
        print(f"  ERREUR: {SYMBOL} non disponible sur ce serveur")
        mt5.shutdown()
        return
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print("ECHEC")
        mt5.shutdown()
        return
    print(f"OK (Spread: {symbol_info.spread} pts)")

    # === CHECK 4 : Tick en temps reel ===
    print("  [4/6] Donnees temps reel...", end=" ")
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("ECHEC")
        print("  ERREUR: Pas de tick disponible (marche ferme?)")
        mt5.shutdown()
        return
    tick_age = int(time.time()) - tick.time
    if tick_age > 60:
        print(f"ATTENTION (tick ancien: {tick_age}s)")
        print("  Le marche est peut-etre ferme, le bot attendra des donnees")
    else:
        print(f"OK (Bid: {tick.bid:.2f} Ask: {tick.ask:.2f})")

    # === CHECK 5 : Donnees historiques ===
    print("  [5/6] Donnees historiques...", end=" ")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 250)
    if rates is None or len(rates) < 250:
        count = len(rates) if rates is not None else 0
        print(f"ECHEC ({count}/250 bougies)")
        print("  ERREUR: Pas assez de donnees historiques")
        mt5.shutdown()
        return
    print(f"OK ({len(rates)} bougies M5 disponibles)")

    # === CHECK 6 : Spread acceptable ===
    print("  [6/6] Spread...", end=" ")
    spread = abs(tick.ask - tick.bid)
    if spread > MAX_SPREAD_USD:
        print(f"ATTENTION ({spread:.1f}$ > max {MAX_SPREAD_USD}$)")
        print("  Le bot attendra que le spread baisse pour trader")
    else:
        print(f"OK ({spread:.1f}$ <= max {MAX_SPREAD_USD}$)")

    # === RESUME ===
    print()
    print("-" * 58)
    print(f"  Compte  : {account_info.login} ({account_info.server})")
    print(f"  Solde   : {account_info.balance:.2f}$")
    print(f"  Symbole : {SYMBOL}")
    print(f"  Lot     : {LOT} x {MAX_POSITIONS} max (DCA)")
    print(f"  Score   : 1er >= {SCORE_MIN_ENTRY} | DCA >= {SCORE_DCA_MIN}")
    print(f"  RSI BUY : 1er <= {RSI_BUY_1} | DCA <= {RSI_BUY_DCA}")
    print(f"  RSI SELL: 1er >= {RSI_SELL_1} | DCA >= {RSI_SELL_DCA}")
    print(f"  Stop    : -{STOP_LOSS_MAX}$/trade | -{STOP_LOSS_TOTAL}$ total (SL sur MT5)")
    print(f"  Post-SL : attente RSI zone {RSI_NEUTRAL_LOW}-{RSI_NEUTRAL_HIGH} avant re-entry")
    print(f"  TP      : RSI change de sens (profit libre)")
    print(f"  Boucle  : {LOOP_INTERVAL}s")
    print("-" * 58)
    print()

    data_engine = DataEngine(SYMBOL, TIMEFRAME, HIST_BOUGIES)
    signal_gen = SignalGenerator()
    risk_mgr = RiskManager()
    executor = OrderExecutor()
    dashboard = Dashboard()
    validator = DataValidator()

    print(f"  Seuil   : {SCORE_MIN_ENTRY} (warmup {WARMUP_LOOPS} boucles avant trading)")

    # Check initial du fichier historique
    hist_ok, hist_msg = validator.validate_trade_history()
    logger.info(f"Historique: {hist_msg}")
    print(f"  Historique: {hist_msg}")

    logger.info("=== BOT EXPERT V3 DEMARRE - 64 Scenarios ===")

    # === CHECK SECURITE RELANCE ===
    # Verifier s'il y a deja des positions ouvertes (script a pu crasher)
    existing_positions = mt5.positions_get(symbol=SYMBOL)
    if existing_positions and len(existing_positions) > 0:
        total_pl = sum(p.profit for p in existing_positions)
        nb_exist = len(existing_positions)
        print(f"\n  !!! ATTENTION: {nb_exist} position(s) DEJA ouvertes !!!")
        for p in existing_positions:
            ptype = "BUY" if p.type == 0 else "SELL"
            print(f"      {ptype} | Ouv: {p.price_open:.2f} | P/L: {p.profit:+.2f}$")
        print(f"      Profit total: {total_pl:+.2f}$")
        print(f"\n  Le bot va GERER ces positions (pas en ouvrir de nouvelles)")
        print(f"  Il appliquera les regles de sortie (RSI/Stop/Break-even)")
        # Marquer le DCA si 2 positions
        if nb_exist >= 2:
            risk_mgr.dca_active = True
        # Detecter la direction
        risk_mgr.first_trade_dir = "BUY" if existing_positions[0].type == 0 else "SELL"
        risk_mgr.first_trade_score = SCORE_MIN_ENTRY  # On ne connait pas le score initial
        logger.info(f"RELANCE: {nb_exist} positions existantes, P/L={total_pl:.2f}$")
    else:
        print("\n  Aucune position existante. Observation avant trading.")

    # Verifier les pertes recentes dans l'historique
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            df_hist = pd.read_csv(TRADE_HISTORY_FILE)
            if len(df_hist) >= 2:
                recent = df_hist.tail(3)
                recent_losses = sum(1 for _, r in recent.iterrows() if r['profit'] < 0)
                if recent_losses >= 2:
                    risk_mgr.consecutive_losses = recent_losses
                    new_seuil = risk_mgr.get_adjusted_score_min()
                    print(f"\n  !!! {recent_losses} pertes recentes detectees dans l'historique")
                    print(f"      Seuil adaptatif augmente: {new_seuil} (au lieu de {SCORE_MIN_ENTRY})")
                    logger.info(f"RELANCE: {recent_losses} pertes recentes, seuil={new_seuil}")
        except:
            pass

    print(f"\n  WARMUP: {WARMUP_LOOPS} boucles ({WARMUP_LOOPS * LOOP_INTERVAL}s) d'observation")
    print("  Le bot ne fera AUCUN trade pendant cette periode.")
    print("\n  TOUS LES CHECKS OK - Bot demarre !")
    print("  (Ctrl+C pour arreter)")
    print()
    time.sleep(2)

    consecutive_errors = 0
    loop_count = 0

    try:
        while True:
            # --- Reconnexion automatique ---
            snap = data_engine.get_snapshot()
            if snap is None:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print("Connexion perdue, reconnexion...")
                    if not reconnect_mt5():
                        print("Echec reconnexion, attente 30s...")
                        time.sleep(30)
                        continue
                    consecutive_errors = 0
                else:
                    print("En attente de donnees marche...")
                    time.sleep(LOOP_INTERVAL)
                continue

            consecutive_errors = 0
            loop_count += 1

            # --- Validation des donnees ---
            data_ok, data_warnings = validator.full_check(snap)
            if not data_ok:
                print("DONNEES INVALIDES:")
                for w in data_warnings:
                    print(f"  ! {w}")
                    logger.warning(f"DATA: {w}")
                time.sleep(LOOP_INTERVAL)
                continue

            # Check historique toutes les 50 boucles (~100s)
            if loop_count % 50 == 0:
                hist_ok, hist_msg = validator.validate_trade_history()
                if not hist_ok:
                    logger.warning(f"HISTORIQUE: {hist_msg}")

            signal = signal_gen.evaluate(snap)
            positions = mt5.positions_get(symbol=SYMBOL)

            # Mise a jour du RSI reset (apres SL)
            risk_mgr.update_rsi_reset(snap.rsi_7)

            # Affichage
            dashboard.display(snap, signal, positions, risk_mgr)

            # Afficher warnings data si present
            if data_warnings:
                dashboard.display_data_warnings(data_warnings)

            # Afficher stats session
            if risk_mgr.trade_count > 0:
                winrate = risk_mgr.win_count / risk_mgr.trade_count * 100
                print(f"  SESSION: {risk_mgr.trade_count} trades | "
                      f"Profit: {risk_mgr.total_profit:+.2f}$ | "
                      f"Winrate: {winrate:.0f}%")

            # --- Gestion positions existantes (DCA) ---
            if positions:
                # Verifier si on doit fermer TOUTES les positions
                should_close, reason = risk_mgr.should_close_all(positions, snap)
                if should_close:
                    total_profit = sum(p.profit for p in positions)
                    was_stop = "STOP" in reason
                    for pos in positions:
                        executor.close_position(pos, reason)
                    risk_mgr.on_trade_closed(total_profit, was_stop=was_stop)
                    risk_mgr.dca_active = False
                    risk_mgr.first_trade_score = 0
                    risk_mgr.first_trade_dir = ""
                    logger.info(f"FERMETURE TOTALE ({len(positions)} pos): {reason} | Profit={total_profit:.2f}$")

                # Sinon, verifier si on peut ouvrir le 2eme trade (DCA)
                elif signal.direction != "NONE" and len(positions) < MAX_POSITIONS:
                    can_dca, dca_msg = risk_mgr.can_open_dca(signal, positions, snap)
                    if can_dca:
                        spread_ok, spread_val = check_spread()
                        if spread_ok:
                            lot = LOT
                            success = executor.open_order(signal.direction, lot)
                            if success:
                                risk_mgr.on_new_trade(signal, is_dca=True)
                                print(f"  >>> DCA OUVERT: {signal.direction} | Score={signal.score} | RSI={snap.rsi_7:.1f}")
                                logger.info(f"DCA {signal.direction} | Score={signal.score} | RSI={snap.rsi_7:.1f}")
                    else:
                        total_profit = sum(p.profit for p in positions)
                        if len(positions) == 1 and positions[0].profit < 0:
                            print(f"\n  DCA ATTENTE: score {signal.score}/{SCORE_DCA_MIN} + RSI={snap.rsi_7:.1f}")
                        print(f"  PROFIT TOTAL: {total_profit:+.2f}$ ({len(positions)} pos)")

            # --- Ouverture premiere position ---
            elif signal.direction != "NONE":
                # Check 0: Warmup
                if loop_count <= WARMUP_LOOPS:
                    print(f"\n  WARMUP: observation {loop_count}/{WARMUP_LOOPS} (pas de trade)")

                # Check 1: Session arretee
                elif risk_mgr.session_stopped:
                    print(f"\n  SESSION ARRETEE: perte limite {DAILY_LOSS_LIMIT}$ atteinte")

                # Check 2: Post-SL RSI reset (correction en cours)
                elif risk_mgr.is_rsi_reset_blocking()[0]:
                    _, reset_msg = risk_mgr.is_rsi_reset_blocking()
                    print(f"\n  {reset_msg} | RSI={snap.rsi_7:.1f}")

                # Check 3: Cooldown
                elif risk_mgr.is_cooldown_active():
                    remaining = risk_mgr.cooldown_remaining()
                    print(f"\n  COOLDOWN: {remaining}s restantes")

                # Check 4: Spread
                elif not check_spread()[0]:
                    _, spread = check_spread()
                    print(f"\n  SPREAD TROP LARGE: {spread:.0f}$ > max {MAX_SPREAD_USD}$")

                # Check 5: Confirmation (apres pertes)
                elif risk_mgr.needs_confirmation(signal):
                    print(f"\n  CONFIRMATION: signal {signal.direction} vu "
                          f"{risk_mgr.signal_confirm_count}/2 fois (apres pertes)")

                # Check 6: Entree (Score >= 150 + RSI extreme)
                else:
                    can_enter, entry_msg = risk_mgr.can_enter(signal, snap)
                    print(f"\n  ENTRY: {entry_msg}")

                    if can_enter:
                        lot = risk_mgr.compute_lot(signal, snap.atr, snap.prix)
                        success = executor.open_order(signal.direction, lot)
                        if success:
                            risk_mgr.on_new_trade(signal)
                            risk_mgr.signal_confirm_count = 0
                            print(f"  >>> TRADE OUVERT: {signal.direction} | Lot={lot} | Score={signal.score} | RSI={snap.rsi_7:.1f}")
                            logger.info(
                                f"OUVERTURE {signal.direction} | Score={signal.score} | RSI={snap.rsi_7:.1f} | "
                                f"Scenarios={signal.scenario_count} | Lot={lot} | "
                                f"Raisons: {'; '.join(signal.reasons[:5])}"
                            )
                        else:
                            print(f"\n  !!! ECHEC OUVERTURE {signal.direction} (verifier MT5)")
            else:
                # Afficher info
                adj_min = risk_mgr.get_adjusted_score_min()
                extra = ""
                if risk_mgr.consecutive_losses >= 2:
                    extra = f" (seuil +{adj_min - SCORE_MIN_ENTRY} apres {risk_mgr.consecutive_losses} pertes)"
                rsi_info = f" | RSI={snap.rsi_7:.1f} (BUY<={RSI_BUY_1} SELL>={RSI_SELL_1})"
                if risk_mgr.is_cooldown_active():
                    print(f"\n  COOLDOWN: {risk_mgr.cooldown_remaining()}s | "
                          f"Score: {signal.score}/{adj_min}{extra}")
                else:
                    print(f"\n  En attente... Score: {signal.score}/{adj_min}{rsi_info}{extra}")

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        logger.info("=== BOT ARRETE PAR L'UTILISATEUR ===")
        mt5.shutdown()
        print(f"\nBot arrete. Session: {risk_mgr.trade_count} trades, "
              f"profit: {risk_mgr.total_profit:+.2f}$")
    except Exception as e:
        logger.error(f"ERREUR FATALE: {e}")
        mt5.shutdown()
        raise


if __name__ == "__main__":
    main()
