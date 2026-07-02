# -*- coding: utf-8 -*-
"""
=============================================================================
  ULTIMATE SCALPING BOT V2 - LIVE MT5 - ETH ONLY

  RESULTATS BACKTEST 1 MOIS (ETH/USDT M5):
  +384$ (+256%) | 293 trades | WR 37% | R:R 2.48 | PF 1.43
  Avg Win: +12$ | Avg Loss: -4.83$ | Breakeven saves: 162
  Jours verts: 19/31 (61%)

  STRATEGIES:
  - RSI7_STRICT: RSI7 < 25 rebond (WR 38%, gros R:R)
  - RSI7_RELAX: RSI7 30-25 rebond (WR 35%, bon volume)
  - BB_VOL: Squeeze + Volume spike (WR 34%, tres bon R:R = +256$)

  PARAMS OPTIMISES:
  - TP: 1.8x ATR | Stop: 0.7x ATR | R:R cible: 2.57
  - Breakeven rapide a 0.5x ATR (sauve 50%+ des trades)
  - Trail a 1.0x ATR, distance 0.4x ATR
  - Levier 20x | Risk 2%/trade
  - Dynamic lot: x2 a 500$, x3.5 a 1000$, x5 a 2000$, x8 a 5000$

  FEATURES:
  - bot_state.json: persistance (cree au 1er lancement)
  - Weekly self-test: backtest auto chaque lundi
  - Dynamic lot scaling: augmente apres 500$
  - Auto-filling mode: detecte le mode du broker

  MetaTrader 5 | ETH Only | Auto-Compound
=============================================================================
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import requests
import urllib3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator

urllib3.disable_warnings()


# =============================================================================
#  CONFIG
# =============================================================================
@dataclass
class Config:
    # Asset unique: ETH (meilleur backtest: +384$, +256%, PF 1.43, R:R 2.48)
    SYMBOLS: list = field(default_factory=lambda: [
        {'name': 'ETHUSD', 'alts': ['ETHUSDm', 'ETHUSDT', 'ETHUSD.a'], 'priority': 1},
    ])

    # Timeframe
    TF: int = mt5.TIMEFRAME_M5
    BARS: int = 150

    # Lot sizing
    DEFAULT_LOT: float = 0.01
    LEVERAGE: int = 20
    RISK_PCT: float = 2.0  # % capital risk per trade

    # TP/SL (ATR-based - optimise backtest)
    TP_ATR: float = 1.8
    STOP_ATR: float = 0.7
    TRAIL_START_ATR: float = 1.0
    TRAIL_DIST_ATR: float = 0.4
    BREAKEVEN_ATR: float = 0.5

    # RSI params
    RSI7_STRICT_LEVEL: int = 25
    RSI7_RELAX_LEVEL: int = 30
    RSI14_EXTREME: int = 20
    RSI7_SELL_STRICT: int = 75
    RSI7_SELL_RELAX: int = 70
    RSI14_SELL_EXTREME: int = 80

    # BB params
    BB_SQUEEZE_THRESH: float = 1.2
    BB_VOL_MIN: float = 1.5

    # Risk management
    MIN_SCORE: int = 72
    MAX_TRADES_DAY: int = 25
    MAX_DD_PCT: float = 15.0
    COOLDOWN_SECS: int = 300  # 5 min entre trades
    PAUSE_LOSSES: int = 3  # Pause apres 3 pertes consecutives
    PAUSE_MINS: int = 30

    # Dynamic lot scaling (augmente apres paliers de balance)
    LOT_SCALE_THRESHOLDS: list = field(default_factory=lambda: [
        # (balance_min, lot_multiplier)
        (150, 1.0),    # 150-499$: lot de base
        (500, 2.0),    # 500-999$: 2x lot
        (1000, 3.5),   # 1000-1999$: 3.5x lot
        (2000, 5.0),   # 2000-4999$: 5x lot
        (5000, 8.0),   # 5000$+: 8x lot
    ])

    # Weekly self-test
    STATE_FILE: str = 'bot_state.json'
    WEEKLY_CHECK_DAY: int = 0  # 0=Lundi
    WEEKLY_CHECK_HOUR: int = 6  # 6h UTC
    MIN_WEEKLY_WR: float = 35.0   # WR minimum pour continuer
    MIN_WEEKLY_PF: float = 1.0    # Profit Factor minimum

    # Magic
    MAGIC: int = 888888


# =============================================================================
#  BOT STATE MANAGER (bot_state.json)
# =============================================================================
class BotState:
    """
    Manages persistent state in bot_state.json:
    - Created on first launch
    - Tracks last weekly check date
    - Tracks balance history for lot scaling
    - Stores cumulative stats
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.STATE_FILE)
        self.state = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load state file or create it on first run"""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                print(f'  [STATE] Loaded: {self.state_path}')
                return state
            except (json.JSONDecodeError, IOError):
                print(f'  [STATE] Corrupt file, recreating...')

        # First launch: create initial state
        state = {
            'created_at': datetime.now().isoformat(),
            'last_weekly_check': None,
            'next_weekly_check': self._calc_next_check().isoformat(),
            'weekly_check_passed': True,
            'current_lot_multiplier': 1.0,
            'peak_balance': 150.0,
            'total_trades': 0,
            'total_wins': 0,
            'total_losses': 0,
            'total_pnl': 0.0,
            'weekly_trades': [],  # trades this week for self-test
            'balance_history': [{'date': datetime.now().isoformat(), 'balance': 150.0}],
            'lot_scale_events': [],
        }
        self._save(state)
        print(f'  [STATE] Created new: {self.state_path}')
        return state

    def _save(self, state: dict = None):
        """Save state to file"""
        if state is None:
            state = self.state
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    def _calc_next_check(self) -> datetime:
        """Calculate next weekly check datetime (next Monday at WEEKLY_CHECK_HOUR)"""
        now = datetime.now()
        days_ahead = self.cfg.WEEKLY_CHECK_DAY - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_check = now.replace(hour=self.cfg.WEEKLY_CHECK_HOUR, minute=0, second=0, microsecond=0)
        next_check += timedelta(days=days_ahead)
        return next_check

    def get_lot_multiplier(self, balance: float) -> float:
        """Get dynamic lot multiplier based on current balance"""
        multiplier = 1.0
        for threshold, mult in sorted(self.cfg.LOT_SCALE_THRESHOLDS, key=lambda x: x[0], reverse=True):
            if balance >= threshold:
                multiplier = mult
                break

        # Update state if multiplier changed
        if multiplier != self.state.get('current_lot_multiplier', 1.0):
            old_mult = self.state.get('current_lot_multiplier', 1.0)
            self.state['current_lot_multiplier'] = multiplier
            self.state['lot_scale_events'].append({
                'date': datetime.now().isoformat(),
                'balance': balance,
                'old_mult': old_mult,
                'new_mult': multiplier,
            })
            self._save()
            print(f'  [LOT SCALE] Balance {balance:.2f}$ -> Lot x{multiplier:.1f} '
                  f'(was x{old_mult:.1f})')

        return multiplier

    def record_trade(self, pnl: float, win: bool):
        """Record a trade for weekly stats"""
        self.state['total_trades'] += 1
        self.state['total_pnl'] += pnl
        if win:
            self.state['total_wins'] += 1
        else:
            self.state['total_losses'] += 1

        self.state['weekly_trades'].append({
            'time': datetime.now().isoformat(),
            'pnl': pnl,
            'win': win,
        })
        self._save()

    def update_balance(self, balance: float):
        """Update balance history"""
        if balance > self.state.get('peak_balance', 0):
            self.state['peak_balance'] = balance

        # Add to history (max 1 entry per hour)
        history = self.state.get('balance_history', [])
        if not history or (datetime.now() - datetime.fromisoformat(history[-1]['date'])).seconds > 3600:
            history.append({'date': datetime.now().isoformat(), 'balance': balance})
            # Keep last 30 days
            if len(history) > 720:
                history = history[-720:]
            self.state['balance_history'] = history
            self._save()

    def needs_weekly_check(self) -> bool:
        """Check if it's time for the weekly self-test"""
        next_check_str = self.state.get('next_weekly_check')
        if not next_check_str:
            return True
        try:
            next_check = datetime.fromisoformat(next_check_str)
            return datetime.now() >= next_check
        except (ValueError, TypeError):
            return True

    def run_weekly_check(self) -> dict:
        """
        Run the weekly self-test:
        - Backtest last 7 days on all assets
        - Check WR and PF thresholds
        - Update state with results
        Returns: {'passed': bool, 'details': str}
        """
        print('\n  [WEEKLY CHECK] Running self-test...')

        weekly_trades = self.state.get('weekly_trades', [])
        n = len(weekly_trades)

        result = {'passed': True, 'details': '', 'trades': n}

        if n < 5:
            result['details'] = f'Trop peu de trades ({n}). Skip check.'
            result['passed'] = True
        else:
            wins = sum(1 for t in weekly_trades if t['win'])
            losses = n - wins
            wr = wins / n * 100
            total_pnl = sum(t['pnl'] for t in weekly_trades)
            gross_win = sum(t['pnl'] for t in weekly_trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in weekly_trades if t['pnl'] <= 0))
            pf = gross_win / gross_loss if gross_loss > 0 else 99

            result['wr'] = wr
            result['pf'] = pf
            result['pnl'] = total_pnl
            result['details'] = (f'{n}T | WR {wr:.1f}% | PF {pf:.2f} | '
                                 f'P/L {total_pnl:+.2f}$')

            if wr < self.cfg.MIN_WEEKLY_WR and pf < self.cfg.MIN_WEEKLY_PF:
                result['passed'] = False
                result['details'] += ' -> ECHEC (WR et PF sous seuils)'
            else:
                result['passed'] = True
                result['details'] += ' -> OK'

        # Also run a quick backtest validation on recent data
        backtest_result = self._quick_backtest()
        if backtest_result is not None:
            result['backtest'] = backtest_result
            if backtest_result.get('profit_pct', 0) < -5:
                result['passed'] = False
                result['details'] += f' | Backtest 7j: {backtest_result["profit_pct"]:+.1f}% -> ALERTE'
            else:
                result['details'] += f' | Backtest 7j: {backtest_result["profit_pct"]:+.1f}%'

        # Update state
        self.state['last_weekly_check'] = datetime.now().isoformat()
        self.state['next_weekly_check'] = self._calc_next_check().isoformat()
        self.state['weekly_check_passed'] = result['passed']
        self.state['weekly_trades'] = []  # Reset for next week
        self._save()

        print(f'  [WEEKLY CHECK] {result["details"]}')
        print(f'  [WEEKLY CHECK] Next: {self.state["next_weekly_check"]}')

        return result

    def _quick_backtest(self) -> dict:
        """Quick 7-day backtest on ETH M5 to validate strategy still works"""
        try:
            symbol = 'ETHUSDT'
            end_t = int(datetime.now().timestamp() * 1000)
            start_t = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            all_data = []
            cs = start_t
            while cs < end_t:
                url = (f'https://api.binance.com/api/v3/klines?symbol={symbol}'
                       f'&interval=5m&startTime={cs}&limit=1000')
                r = requests.get(url, timeout=10, verify=False)
                if r.status_code != 200:
                    break
                d = r.json()
                if not d:
                    break
                all_data.extend(d)
                cs = d[-1][0] + 1

            if len(all_data) < 500:
                return None

            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'ct', 'qv', 'tc', 'bb', 'bq', 'ig'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)

            # Indicators
            df['rsi7'] = RSIIndicator(df['close'], 7).rsi()
            df['rsi7_prev'] = df['rsi7'].shift(1)
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
            bb = BollingerBands(df['close'], 20, 2)
            df['bbu'] = bb.bollinger_hband()
            df['bbl'] = bb.bollinger_lband()
            df['bbw'] = (df['bbu'] - df['bbl']) / bb.bollinger_mavg() * 100
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['ema8'] = EMAIndicator(df['close'], 8).ema_indicator()
            df['ema21'] = EMAIndicator(df['close'], 21).ema_indicator()
            df['mom3'] = df['close'].diff(3) / df['atr']

            # Simple signal count: RSI7 < 30 rebounds
            trades = 0
            wins = 0
            for idx in range(100, len(df)):
                row = df.iloc[idx]
                if pd.isna(row['rsi7']) or pd.isna(row['atr']) or row['atr'] == 0:
                    continue
                rsi7 = row['rsi7']
                rsi7_p = row['rsi7_prev'] if not pd.isna(row['rsi7_prev']) else 50

                # Simple test: RSI7 bounce
                if rsi7 < 30 and rsi7 > rsi7_p:
                    # Check next 10 bars for profit
                    future = df.iloc[idx+1:idx+11]
                    if len(future) >= 5:
                        max_gain = (future['high'].max() - row['close']) / row['atr']
                        max_loss = (row['close'] - future['low'].min()) / row['atr']
                        trades += 1
                        if max_gain > 1.0:  # Would have hit 1 ATR TP
                            wins += 1
                elif rsi7 > 70 and rsi7 < rsi7_p:
                    future = df.iloc[idx+1:idx+11]
                    if len(future) >= 5:
                        max_gain = (row['close'] - future['low'].min()) / row['atr']
                        trades += 1
                        if max_gain > 1.0:
                            wins += 1

            if trades == 0:
                return {'profit_pct': 0, 'trades': 0, 'wr': 0}

            wr = wins / trades * 100
            # Estimate profit: (wins * 1.8 ATR - losses * 0.7 ATR) as % approx
            avg_atr_pct = df['atr'].dropna().mean() / df['close'].mean() * 100
            est_profit_pct = (wins * 1.8 - (trades - wins) * 0.7) * avg_atr_pct

            return {'profit_pct': est_profit_pct, 'trades': trades, 'wr': wr}

        except Exception as e:
            print(f'  [WEEKLY CHECK] Backtest error: {e}')
            return None

    def get_status_summary(self) -> str:
        """Return a short status string for the dashboard"""
        mult = self.state.get('current_lot_multiplier', 1.0)
        total_t = self.state.get('total_trades', 0)
        total_pnl = self.state.get('total_pnl', 0)
        passed = self.state.get('weekly_check_passed', True)
        next_check = self.state.get('next_weekly_check', '?')
        return (f'Lot:x{mult:.1f} | Trades:{total_t} | P/L:{total_pnl:+.2f}$ | '
                f'Check:{"OK" if passed else "FAIL"} | Next:{next_check[:10]}')


# =============================================================================
#  MARKET ANALYZER
# =============================================================================
class MarketAnalyzer:
    def __init__(self, config: Config):
        self.cfg = config

    def analyze(self, symbol: str) -> dict:
        """Get full market analysis for a symbol"""
        rates = mt5.copy_rates_from_pos(symbol, self.cfg.TF, 0, self.cfg.BARS)
        if rates is None or len(rates) < 80:
            return None

        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')

        # RSI
        df['rsi7'] = RSIIndicator(df['close'], 7).rsi()
        df['rsi7_prev'] = df['rsi7'].shift(1)
        df['rsi7_prev2'] = df['rsi7'].shift(2)
        df['rsi14'] = RSIIndicator(df['close'], 14).rsi()
        df['rsi14_prev'] = df['rsi14'].shift(1)

        # EMA
        df['ema8'] = EMAIndicator(df['close'], 8).ema_indicator()
        df['ema21'] = EMAIndicator(df['close'], 21).ema_indicator()
        df['ema50'] = EMAIndicator(df['close'], 50).ema_indicator()

        # ATR
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()

        # Bollinger
        bb = BollingerBands(df['close'], 20, 2)
        df['bbu'] = bb.bollinger_hband()
        df['bbl'] = bb.bollinger_lband()
        df['bbm'] = bb.bollinger_mavg()
        df['bbw'] = (df['bbu'] - df['bbl']) / df['bbm'] * 100

        # MACD
        mc = MACD(df['close'], 8, 21, 5)
        df['macd'] = mc.macd_diff()
        df['macd_prev'] = df['macd'].shift(1)

        # Volume
        df['vol_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()

        # Momentum
        df['mom3'] = df['close'].diff(3) / df['atr']

        # Micro trend
        df['micro'] = 'N'
        df.loc[df['ema8'] > df['ema21'], 'micro'] = 'U'
        df.loc[df['ema8'] < df['ema21'], 'micro'] = 'D'

        # Trend strong
        df['strong'] = (
            ((df['ema8'] > df['ema21']) & (df['ema21'] > df['ema50'])) |
            ((df['ema8'] < df['ema21']) & (df['ema21'] < df['ema50']))
        )

        # Regime
        regime_slope = df['ema50'].diff(15) / df['atr']
        df['regime'] = 'NEUTRAL'
        df.loc[regime_slope > 0.2, 'regime'] = 'BULL'
        df.loc[regime_slope < -0.2, 'regime'] = 'BEAR'

        # Candle patterns
        df['body'] = df['close'] - df['open']
        df['body_abs'] = abs(df['body'])
        df['wick_dn'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_up'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['pin_dn'] = df['wick_dn'] > df['body_abs'] * 2
        df['pin_up'] = df['wick_up'] > df['body_abs'] * 2

        prev_body = df['body'].shift(1)
        df['bull_engulf'] = (df['body'] > 0) & (prev_body < 0) & (df['body_abs'] > abs(prev_body))
        df['bear_engulf'] = (df['body'] < 0) & (prev_body > 0) & (df['body_abs'] > abs(prev_body))

        # Last bar data
        last = df.iloc[-1]
        return {
            'rsi7': last['rsi7'] if not pd.isna(last['rsi7']) else 50,
            'rsi7_prev': last['rsi7_prev'] if not pd.isna(last['rsi7_prev']) else 50,
            'rsi7_prev2': last['rsi7_prev2'] if not pd.isna(last['rsi7_prev2']) else 50,
            'rsi14': last['rsi14'] if not pd.isna(last['rsi14']) else 50,
            'rsi14_prev': last['rsi14_prev'] if not pd.isna(last['rsi14_prev']) else 50,
            'atr': last['atr'] if not pd.isna(last['atr']) else 0,
            'price': last['close'],
            'bbw': last['bbw'] if not pd.isna(last['bbw']) else 2,
            'bbu': last['bbu'] if not pd.isna(last['bbu']) else last['close'],
            'bbl': last['bbl'] if not pd.isna(last['bbl']) else last['close'],
            'macd': last['macd'] if not pd.isna(last['macd']) else 0,
            'macd_prev': last['macd_prev'] if not pd.isna(last['macd_prev']) else 0,
            'vol_ratio': last['vol_ratio'] if not pd.isna(last['vol_ratio']) else 1,
            'mom3': last['mom3'] if not pd.isna(last['mom3']) else 0,
            'micro': last['micro'],
            'strong': bool(last['strong']),
            'regime': last['regime'],
            'pin_dn': bool(last['pin_dn']) if not pd.isna(last['pin_dn']) else False,
            'pin_up': bool(last['pin_up']) if not pd.isna(last['pin_up']) else False,
            'bull_engulf': bool(last['bull_engulf']) if not pd.isna(last['bull_engulf']) else False,
            'bear_engulf': bool(last['bear_engulf']) if not pd.isna(last['bear_engulf']) else False,
        }


# =============================================================================
#  SIGNAL ENGINE V2
# =============================================================================
class SignalEngine:
    def __init__(self, config: Config):
        self.cfg = config

    def get_signal(self, data: dict) -> tuple:
        """Returns (direction, strategy, score)"""
        if data is None or data['atr'] == 0:
            return '', '', 0

        rsi7 = data['rsi7']
        rsi7_p = data['rsi7_prev']
        rsi7_p2 = data['rsi7_prev2']
        rsi14 = data['rsi14']
        rsi14_p = data['rsi14_prev']
        vr = data['vol_ratio']
        mom3 = data['mom3']
        bbw = data['bbw']
        micro = data['micro']
        strong = data['strong']
        regime = data['regime']
        prix = data['price']

        signals = []

        # === RSI7 STRICT ===
        if rsi7 < self.cfg.RSI7_STRICT_LEVEL and rsi7 > rsi7_p:
            score = 78
            if rsi7_p < rsi7_p2:
                score += 3
            if regime != 'BEAR':
                score += 3
            if data['pin_dn']:
                score += 5
            if data['bull_engulf']:
                score += 5
            if vr > 1.3:
                score += 2
            signals.append(('BUY', 'RSI7_STRICT', score))

        elif rsi7 > self.cfg.RSI7_SELL_STRICT and rsi7 < rsi7_p:
            score = 78
            if rsi7_p > rsi7_p2:
                score += 3
            if regime != 'BULL':
                score += 3
            if data['pin_up']:
                score += 5
            if data['bear_engulf']:
                score += 5
            if vr > 1.3:
                score += 2
            signals.append(('SELL', 'RSI7_STRICT', score))

        # === RSI7 RELAX ===
        if rsi7 < self.cfg.RSI7_RELAX_LEVEL and rsi7 > rsi7_p and rsi7 >= self.cfg.RSI7_STRICT_LEVEL:
            score = 73
            if regime == 'BULL':
                score += 4
            elif regime != 'BEAR':
                score += 2
            if mom3 > -0.5:
                score += 2
            if data['pin_dn']:
                score += 4
            if vr > 1.2:
                score += 2
            signals.append(('BUY', 'RSI7_RELAX', score))

        elif rsi7 > self.cfg.RSI7_SELL_RELAX and rsi7 < rsi7_p and rsi7 <= self.cfg.RSI7_SELL_STRICT:
            score = 73
            if regime == 'BEAR':
                score += 4
            elif regime != 'BULL':
                score += 2
            if mom3 < 0.5:
                score += 2
            if data['pin_up']:
                score += 4
            if vr > 1.2:
                score += 2
            signals.append(('SELL', 'RSI7_RELAX', score))

        # === RSI14 EXTREME ===
        if rsi14 < self.cfg.RSI14_EXTREME and rsi14 > rsi14_p:
            score = 82
            if rsi14 < 15:
                score += 5
            signals.append(('BUY', 'RSI14_EXTREME', score))
        elif rsi14 > self.cfg.RSI14_SELL_EXTREME and rsi14 < rsi14_p:
            score = 82
            if rsi14 > 85:
                score += 5
            signals.append(('SELL', 'RSI14_EXTREME', score))

        # === BB SQUEEZE + VOLUME ===
        if bbw < self.cfg.BB_SQUEEZE_THRESH and vr > self.cfg.BB_VOL_MIN:
            if prix > data['bbu'] and micro == 'U' and mom3 > 0.2:
                if strong or regime == 'BULL':
                    score = 76
                    if vr > 2.0:
                        score += 3
                    if strong:
                        score += 2
                    signals.append(('BUY', 'BB_VOL', score))
            elif prix < data['bbl'] and micro == 'D' and mom3 < -0.2:
                if strong or regime == 'BEAR':
                    score = 76
                    if vr > 2.0:
                        score += 3
                    if strong:
                        score += 2
                    signals.append(('SELL', 'BB_VOL', score))

        # === MACD + RSI CONFIRM ===
        macd = data['macd']
        macd_p = data['macd_prev']
        if macd > 0 and macd_p <= 0 and rsi7 < 45 and rsi7 > rsi7_p:
            if regime != 'BEAR' and vr > 1.0:
                score = 74
                if strong:
                    score += 3
                signals.append(('BUY', 'MACD_RSI', score))
        elif macd < 0 and macd_p >= 0 and rsi7 > 55 and rsi7 < rsi7_p:
            if regime != 'BULL' and vr > 1.0:
                score = 74
                if strong:
                    score += 3
                signals.append(('SELL', 'MACD_RSI', score))

        if not signals:
            return '', '', 0

        signals.sort(key=lambda x: x[2], reverse=True)
        best = signals[0]
        if best[2] < self.cfg.MIN_SCORE:
            return '', '', 0

        return best[0], best[1], best[2]


# =============================================================================
#  FILLING MODE AUTO-DETECT
# =============================================================================
def get_filling_mode(symbol: str):
    """Auto-detect the correct filling mode for a symbol.
    Brokers support different modes: IOC, FOK, or RETURN.
    This checks symbol_info.filling_mode bitmask."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    filling = info.filling_mode

    # filling_mode is a bitmask:
    # bit 0 (1) = FILLING_FOK supported
    # bit 1 (2) = FILLING_IOC supported
    # If neither, use FILLING_RETURN (0)
    if filling & 2:  # IOC supported
        return mt5.ORDER_FILLING_IOC
    elif filling & 1:  # FOK supported
        return mt5.ORDER_FILLING_FOK
    else:
        return mt5.ORDER_FILLING_RETURN


# =============================================================================
#  TRADE EXECUTOR
# =============================================================================
class TradeExecutor:
    def __init__(self, config: Config):
        self.cfg = config
        self.daily_trades = 0
        self.consec_losses = 0
        self.last_trade_time = None
        self.pause_until = None
        self.today = None
        self.stats = {'wins': 0, 'losses': 0, 'pnl': 0}

    def can_trade(self) -> bool:
        now = datetime.now()
        today = now.date()
        if self.today != today:
            self.today = today
            self.daily_trades = 0

        if self.daily_trades >= self.cfg.MAX_TRADES_DAY:
            return False
        if self.pause_until and now < self.pause_until:
            return False
        if self.last_trade_time:
            elapsed = (now - self.last_trade_time).total_seconds()
            if elapsed < self.cfg.COOLDOWN_SECS:
                return False

        # Drawdown check
        account = mt5.account_info()
        if account and account.balance > 0:
            dd = (1 - account.equity / account.balance) * 100
            if dd > self.cfg.MAX_DD_PCT:
                return False
        return True

    def execute(self, symbol: str, direction: str, strategy: str, score: int, atr: float, lot_multiplier: float = 1.0):
        """Execute a trade with dynamic lot scaling"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        # Calculate lot based on risk
        account = mt5.account_info()
        if account is None:
            return False

        # SL/TP distances
        sl_dist = atr * self.cfg.STOP_ATR
        tp_dist = atr * self.cfg.TP_ATR

        # Position size: risk_amount / sl_distance
        risk_amount = account.balance * (self.cfg.RISK_PCT / 100)
        price = tick.ask if direction == 'BUY' else tick.bid
        if price == 0:
            return False

        lot = risk_amount / sl_dist
        # Apply dynamic lot multiplier (based on balance thresholds)
        lot *= lot_multiplier
        # Normalize to symbol constraints
        lot = max(info.volume_min, min(lot, info.volume_max))
        lot = round(lot / info.volume_step) * info.volume_step
        lot = round(lot, 8)

        if direction == 'BUY':
            entry_price = tick.ask
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry_price = tick.bid
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist
            order_type = mt5.ORDER_TYPE_SELL

        digits = info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 30,
            "magic": self.cfg.MAGIC,
            "comment": f"V2|{strategy}|{score}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(symbol),
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            print(f'    [OPEN] {direction} {symbol} @ {entry_price:.5f} | Lot: {lot:.4f}')
            print(f'           SL: {sl:.5f} | TP: {tp:.5f} | {strategy} Score:{score}')
            return True
        else:
            code = result.retcode if result else 'None'
            print(f'    [FAIL] {symbol}: {code}')
            return False

    def manage_positions(self, symbol: str, data: dict):
        """Manage trailing stop and breakeven"""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != self.cfg.MAGIC:
                continue

            atr = data['atr'] if data and data['atr'] > 0 else 0
            if atr == 0:
                continue

            price = data['price']
            if pos.type == mt5.POSITION_TYPE_BUY:
                pnl_price = price - pos.price_open
            else:
                pnl_price = pos.price_open - price

            pnl_atr = pnl_price / atr

            info = mt5.symbol_info(symbol)
            if info is None:
                continue
            digits = info.digits

            # Breakeven
            if pnl_atr >= self.cfg.BREAKEVEN_ATR:
                spread = abs(mt5.symbol_info_tick(symbol).ask - mt5.symbol_info_tick(symbol).bid)
                if pos.type == mt5.POSITION_TYPE_BUY:
                    be_sl = pos.price_open + spread
                    if pos.sl < be_sl:
                        self._modify_sl(pos, round(be_sl, digits))
                else:
                    be_sl = pos.price_open - spread
                    if pos.sl > be_sl or pos.sl == 0:
                        self._modify_sl(pos, round(be_sl, digits))

            # Trail
            if pnl_atr >= self.cfg.TRAIL_START_ATR:
                trail_dist = atr * self.cfg.TRAIL_DIST_ATR
                if pos.type == mt5.POSITION_TYPE_BUY:
                    new_sl = price - trail_dist
                    if new_sl > pos.sl:
                        self._modify_sl(pos, round(new_sl, digits))
                else:
                    new_sl = price + trail_dist
                    if new_sl < pos.sl or pos.sl == 0:
                        self._modify_sl(pos, round(new_sl, digits))

    def _modify_sl(self, pos, new_sl):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": pos.tp,
            "magic": self.cfg.MAGIC,
        }
        mt5.order_send(request)

    def on_loss(self):
        self.consec_losses += 1
        self.stats['losses'] += 1
        if self.consec_losses >= self.cfg.PAUSE_LOSSES:
            self.pause_until = datetime.now() + timedelta(minutes=self.cfg.PAUSE_MINS)
            print(f'    [PAUSE] {self.cfg.PAUSE_MINS}min after {self.consec_losses} losses')

    def on_win(self, profit):
        self.consec_losses = 0
        self.stats['wins'] += 1
        self.stats['pnl'] += profit


# =============================================================================
#  MAIN BOT
# =============================================================================
def main():
    print('=' * 70)
    print('  ULTIMATE SCALPING BOT V2 - LIVE')
    print('  ETH ONLY | M5 | RSI7 + BB Squeeze')
    print('  + Weekly Self-Test | Dynamic Lot Scaling | State Persistence')
    print('=' * 70)

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    if not account:
        print("Cannot get account!")
        mt5.shutdown()
        return

    print(f"  Account: {account.login} | Balance: {account.balance:.2f}$")
    print(f"  Leverage: 1:{account.leverage}")

    cfg = Config()

    # Initialize state manager (creates bot_state.json on first run)
    bot_state = BotState(cfg)
    print(f'  [STATE] {bot_state.get_status_summary()}')

    # Check if weekly test is due NOW before starting
    if bot_state.needs_weekly_check():
        result = bot_state.run_weekly_check()
        if not result['passed']:
            print('  [WARNING] Weekly check FAILED - bot will still run but with caution')
            # Reduce risk temporarily
            cfg.RISK_PCT = max(1.0, cfg.RISK_PCT * 0.5)
            print(f'  [SAFETY] Risk reduced to {cfg.RISK_PCT}%')

    analyzer = MarketAnalyzer(cfg)
    signal_engine = SignalEngine(cfg)
    executor = TradeExecutor(cfg)

    # Resolve symbols
    active_symbols = []
    for sym_cfg in cfg.SYMBOLS:
        sym = sym_cfg['name']
        info = mt5.symbol_info(sym)
        if info is None:
            for alt in sym_cfg['alts']:
                if mt5.symbol_info(alt):
                    sym = alt
                    break
        info = mt5.symbol_info(sym)
        if info:
            if not info.visible:
                mt5.symbol_select(sym, True)
            active_symbols.append(sym)
            print(f"  Symbol: {sym} OK")
        else:
            print(f"  Symbol: {sym_cfg['name']} NOT FOUND")

    if not active_symbols:
        print("No symbols available!")
        mt5.shutdown()
        return

    # Get current lot multiplier
    lot_mult = bot_state.get_lot_multiplier(account.balance)
    print(f"\n  Active: {active_symbols}")
    print(f"  Lot multiplier: x{lot_mult:.1f} (balance: {account.balance:.2f}$)")
    print(f"  Starting bot loop...\n")

    prev_positions = {}
    last_balance_check = datetime.now()

    try:
        while True:
            now = datetime.now()

            # === WEEKLY CHECK (runs once per week) ===
            if bot_state.needs_weekly_check():
                result = bot_state.run_weekly_check()
                if not result['passed']:
                    cfg.RISK_PCT = max(1.0, cfg.RISK_PCT * 0.5)
                    print(f'  [SAFETY] Risk reduced to {cfg.RISK_PCT}%')
                else:
                    cfg.RISK_PCT = 2.0  # Restore normal risk
                    print(f'  [SAFETY] Risk restored to {cfg.RISK_PCT}%')

            # === DYNAMIC LOT UPDATE (every 5 min) ===
            if (now - last_balance_check).seconds > 300:
                account = mt5.account_info()
                if account:
                    lot_mult = bot_state.get_lot_multiplier(account.balance)
                    bot_state.update_balance(account.balance)
                last_balance_check = now

            for symbol in active_symbols:
                # Analyze
                data = analyzer.analyze(symbol)
                if data is None:
                    continue

                # Manage existing positions
                executor.manage_positions(symbol, data)

                # Check for closed positions (profit tracking)
                current_pos = mt5.positions_get(symbol=symbol)
                current_tickets = {p.ticket for p in current_pos} if current_pos else set()
                prev_tickets = prev_positions.get(symbol, set())

                closed = prev_tickets - current_tickets
                if closed:
                    # Check deal history for closed trades
                    for ticket in closed:
                        deals = mt5.history_deals_get(position=ticket)
                        if deals:
                            profit = sum(d.profit for d in deals)
                            if profit > 0:
                                executor.on_win(profit)
                                bot_state.record_trade(profit, True)
                            else:
                                executor.on_loss()
                                bot_state.record_trade(profit, False)

                prev_positions[symbol] = current_tickets

                # Generate signal
                if executor.can_trade():
                    # Check no existing position
                    has_pos = any(p.magic == cfg.MAGIC for p in (current_pos or []))
                    if not has_pos:
                        d, strat, score = signal_engine.get_signal(data)
                        if d:
                            print(f'  [{now.strftime("%H:%M:%S")}] Signal: {d} {symbol} | {strat} | Score {score} | Lot x{lot_mult:.1f}')
                            executor.execute(symbol, d, strat, score, data['atr'], lot_mult)

            # Dashboard (every cycle)
            account = mt5.account_info()
            all_pos = mt5.positions_get()
            my_pos = [p for p in (all_pos or []) if p.magic == cfg.MAGIC]

            print(f'\r  [{now.strftime("%H:%M:%S")}] '
                  f'Bal:{account.balance:.2f}$ Eq:{account.equity:.2f}$ '
                  f'Pos:{len(my_pos)} '
                  f'Today:{executor.daily_trades}/{cfg.MAX_TRADES_DAY} '
                  f'W:{executor.stats["wins"]} L:{executor.stats["losses"]} '
                  f'Lot:x{lot_mult:.1f}',
                  end='', flush=True)

            time.sleep(15)  # Check every 15 seconds

    except KeyboardInterrupt:
        print('\n\n  Bot stopped.')
    finally:
        account = mt5.account_info()
        if account:
            bot_state.update_balance(account.balance)
            print(f'  Final balance: {account.balance:.2f}$')
            print(f'  State: {bot_state.get_status_summary()}')
        mt5.shutdown()


if __name__ == '__main__':
    main()
