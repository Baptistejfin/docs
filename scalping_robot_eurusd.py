"""
ROBOT DE SCALPING EURUSD - M1
Duree des trades: 30 secondes a 3 minutes max

Caracteristiques:
- Timeframe M1 pour reactions rapides
- Indicateurs optimises pour le scalping
- Gestion stricte du risque
- Stop loss et take profit serres
- Filtres de volatilite et spread
- Horaires de trading optimaux
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import time
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class ScalpingConfig:
    """Configuration du robot de scalping"""
    # Symbole et timeframe
    symbol: str = "EURUSD"
    timeframe: int = mt5.TIMEFRAME_M1

    # Objectifs en pips
    take_profit_pips: float = 8.0      # TP: 8 pips
    stop_loss_pips: float = 5.0        # SL: 5 pips
    trailing_stop_pips: float = 3.0    # Trailing: 3 pips (0 = desactive)
    breakeven_pips: float = 4.0        # Passer BE apres 4 pips de profit

    # Gestion du risque
    risk_percent: float = 1.0          # Risque 1% du capital par trade
    max_trades_per_day: int = 20       # Max trades/jour
    max_daily_loss_percent: float = 3.0  # Arret si -3% sur la journee
    max_daily_profit_percent: float = 5.0  # Arret si +5% (securiser gains)
    max_consecutive_losses: int = 3    # Pause apres 3 pertes consecutives

    # Filtres
    max_spread_pips: float = 1.5       # Spread max acceptable
    min_atr_pips: float = 3.0          # Volatilite minimum
    max_atr_pips: float = 20.0         # Volatilite maximum (eviter news)

    # Temps
    max_trade_duration_seconds: int = 180  # 3 minutes max
    min_time_between_trades: int = 30      # 30 sec entre trades

    # Horaires de trading (heures UTC)
    trading_hours_start: int = 7       # 7h UTC (ouverture Londres)
    trading_hours_end: int = 16        # 16h UTC (fin overlap NY)
    allow_24h_trading: bool = False    # True = ignore les horaires

    # Indicateurs
    ema_fast: int = 5
    ema_medium: int = 10
    ema_slow: int = 20
    rsi_period: int = 7                # RSI rapide
    rsi_overbought: float = 75
    rsi_oversold: float = 25
    stoch_period: int = 5
    stoch_smooth: int = 3
    atr_period: int = 14

    # Confirmation
    min_candles_trend: int = 3         # Min bougies dans la direction
    volume_spike_ratio: float = 1.5    # Volume > 1.5x moyenne

    # MT5
    magic_number: int = 123456
    deviation: int = 10                # Slippage max en points
    pip_value: float = 0.0001          # Pour EURUSD


class ScalpingIndicators:
    """Calcul des indicateurs pour scalping"""

    @staticmethod
    def calculate_all(df: pd.DataFrame, config: ScalpingConfig) -> pd.DataFrame:
        """Calcule tous les indicateurs necessaires"""

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=config.ema_fast, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=config.ema_medium, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=config.ema_slow, adjust=False).mean()

        # RSI rapide
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic rapide
        low_min = df['low'].rolling(window=config.stoch_period).min()
        high_max = df['high'].rolling(window=config.stoch_period).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=config.stoch_smooth).mean()

        # ATR pour volatilite
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=config.atr_period).mean()
        df['atr_pips'] = df['atr'] / config.pip_value

        # Volume
        df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']

        # Tendance des bougies
        df['bullish'] = df['close'] > df['open']
        df['bearish'] = df['close'] < df['open']

        # Momentum (ROC rapide)
        df['momentum'] = df['close'].pct_change(3) * 100

        # EMA Alignment
        df['ema_bullish'] = (df['ema_fast'] > df['ema_medium']) & (df['ema_medium'] > df['ema_slow'])
        df['ema_bearish'] = (df['ema_fast'] < df['ema_medium']) & (df['ema_medium'] < df['ema_slow'])

        # Distance au EMA (pour mean reversion)
        df['dist_ema_fast'] = (df['close'] - df['ema_fast']) / df['atr']

        return df


class RiskManager:
    """Gestion du risque et money management"""

    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_time: Optional[datetime] = None
        self.initial_balance = 0.0
        self.trading_day = None

    def reset_daily_stats(self):
        """Reset les stats journalieres"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trading_day = datetime.now().date()
        logger.info("Stats journalieres remises a zero")

    def update_balance(self, balance: float):
        """Met a jour le solde initial"""
        if self.initial_balance == 0:
            self.initial_balance = balance

    def check_new_day(self):
        """Verifie si c'est un nouveau jour"""
        today = datetime.now().date()
        if self.trading_day != today:
            self.reset_daily_stats()

    def can_trade(self, current_balance: float) -> Tuple[bool, str]:
        """Verifie si on peut ouvrir un nouveau trade"""
        self.check_new_day()

        # Max trades par jour
        if self.daily_trades >= self.config.max_trades_per_day:
            return False, f"Max trades atteint ({self.config.max_trades_per_day})"

        # Perte journaliere max
        if self.initial_balance > 0:
            daily_loss_pct = (self.daily_pnl / self.initial_balance) * 100
            if daily_loss_pct <= -self.config.max_daily_loss_percent:
                return False, f"Perte journaliere max atteinte ({daily_loss_pct:.1f}%)"

            # Profit journalier max (securiser les gains)
            if daily_loss_pct >= self.config.max_daily_profit_percent:
                return False, f"Profit journalier max atteint ({daily_loss_pct:.1f}%)"

        # Pertes consecutives
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Pause apres {self.consecutive_losses} pertes consecutives"

        # Temps entre trades
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.config.min_time_between_trades:
                return False, f"Attendre {self.config.min_time_between_trades - elapsed:.0f}s"

        return True, "OK"

    def calculate_lot_size(self, balance: float, stop_loss_pips: float) -> float:
        """Calcule la taille de lot basee sur le risque"""
        risk_amount = balance * (self.config.risk_percent / 100)

        # Valeur d'un pip pour 1 lot standard EURUSD = ~10 USD
        pip_value_per_lot = 10.0

        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Limites MT5
        lot_size = max(0.01, min(lot_size, 10.0))
        lot_size = round(lot_size, 2)

        return lot_size

    def record_trade_result(self, profit: float):
        """Enregistre le resultat d'un trade"""
        self.daily_trades += 1
        self.daily_pnl += profit
        self.last_trade_time = datetime.now()

        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        logger.info(f"Trade #{self.daily_trades} | P&L: {profit:.2f} | Daily: {self.daily_pnl:.2f}")


class SignalGenerator:
    """Generation des signaux de trading"""

    def __init__(self, config: ScalpingConfig):
        self.config = config

    def check_trading_hours(self) -> bool:
        """Verifie si on est dans les heures de trading"""
        now = datetime.now(timezone.utc)

        # Eviter le weekend
        if now.weekday() >= 5:  # Samedi = 5, Dimanche = 6
            return False

        # Mode 24h: ignore les horaires
        if self.config.allow_24h_trading:
            return True

        hour = now.hour
        return self.config.trading_hours_start <= hour < self.config.trading_hours_end

    def check_spread(self, symbol_info) -> Tuple[bool, float]:
        """Verifie si le spread est acceptable"""
        spread_points = symbol_info.spread
        spread_pips = spread_points / 10  # Pour EURUSD, 1 pip = 10 points

        is_ok = spread_pips <= self.config.max_spread_pips
        return is_ok, spread_pips

    def check_volatility(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Verifie si la volatilite est dans la plage acceptable"""
        current_atr = df['atr_pips'].iloc[-1]

        is_ok = self.config.min_atr_pips <= current_atr <= self.config.max_atr_pips
        return is_ok, current_atr

    def generate_signal(self, df: pd.DataFrame) -> TradeDirection:
        """
        Genere un signal de trading base sur plusieurs criteres

        Strategie de scalping:
        - Croisement EMAs + confirmation RSI/Stoch
        - Volume spike pour confirmer
        - Tendance des dernieres bougies
        """
        if len(df) < 50:
            return TradeDirection.NONE

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # ===== CONDITIONS BUY =====
        buy_conditions = []

        # 1. EMA Alignment haussier
        buy_conditions.append(current['ema_bullish'])

        # 2. Prix au-dessus EMA fast
        buy_conditions.append(current['close'] > current['ema_fast'])

        # 3. RSI pas en surachat et en hausse
        buy_conditions.append(current['rsi'] < self.config.rsi_overbought)
        buy_conditions.append(current['rsi'] > prev['rsi'])

        # 4. Stochastic en zone de survente OU croisement haussier
        stoch_buy = (current['stoch_k'] < 30) or \
                    (current['stoch_k'] > current['stoch_d'] and prev['stoch_k'] <= prev['stoch_d'])
        buy_conditions.append(stoch_buy)

        # 5. Volume au-dessus de la moyenne
        buy_conditions.append(current['volume_ratio'] >= 1.0)

        # 6. Derniere bougie haussiere
        buy_conditions.append(current['bullish'])

        # 7. Momentum positif
        buy_conditions.append(current['momentum'] > 0)

        # ===== CONDITIONS SELL =====
        sell_conditions = []

        # 1. EMA Alignment baissier
        sell_conditions.append(current['ema_bearish'])

        # 2. Prix en-dessous EMA fast
        sell_conditions.append(current['close'] < current['ema_fast'])

        # 3. RSI pas en survente et en baisse
        sell_conditions.append(current['rsi'] > self.config.rsi_oversold)
        sell_conditions.append(current['rsi'] < prev['rsi'])

        # 4. Stochastic en zone de surachat OU croisement baissier
        stoch_sell = (current['stoch_k'] > 70) or \
                     (current['stoch_k'] < current['stoch_d'] and prev['stoch_k'] >= prev['stoch_d'])
        sell_conditions.append(stoch_sell)

        # 5. Volume au-dessus de la moyenne
        sell_conditions.append(current['volume_ratio'] >= 1.0)

        # 6. Derniere bougie baissiere
        sell_conditions.append(current['bearish'])

        # 7. Momentum negatif
        sell_conditions.append(current['momentum'] < 0)

        # ===== DECISION =====
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)

        min_score = 5  # Au moins 5 conditions sur 7

        if buy_score >= min_score and buy_score > sell_score:
            logger.info(f"Signal BUY detecte (score: {buy_score}/7)")
            return TradeDirection.BUY

        if sell_score >= min_score and sell_score > buy_score:
            logger.info(f"Signal SELL detecte (score: {sell_score}/7)")
            return TradeDirection.SELL

        return TradeDirection.NONE


class TradeExecutor:
    """Execution et gestion des ordres"""

    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.current_position: Optional[int] = None
        self.entry_time: Optional[datetime] = None
        self.entry_price: float = 0.0

    def get_symbol_info(self):
        """Recupere les infos du symbole"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            logger.error(f"Symbole {self.config.symbol} non trouve")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(self.config.symbol, True):
                logger.error(f"Impossible d'activer {self.config.symbol}")
                return None

        return symbol_info

    def open_position(self, direction: TradeDirection, lot_size: float) -> bool:
        """Ouvre une position"""
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return False

        point = symbol_info.point
        price = mt5.symbol_info_tick(self.config.symbol)

        if direction == TradeDirection.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = price.ask
            sl = entry_price - (self.config.stop_loss_pips * self.config.pip_value)
            tp = entry_price + (self.config.take_profit_pips * self.config.pip_value)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = price.bid
            sl = entry_price + (self.config.stop_loss_pips * self.config.pip_value)
            tp = entry_price - (self.config.take_profit_pips * self.config.pip_value)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": "Scalper",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Erreur ouverture: {result.comment} (code: {result.retcode})")
            return False

        self.current_position = result.order
        self.entry_time = datetime.now()
        self.entry_price = entry_price

        logger.info(f"Position ouverte: {direction.value} {lot_size} lots @ {entry_price:.5f}")
        logger.info(f"SL: {sl:.5f} | TP: {tp:.5f}")

        return True

    def close_position(self, reason: str = "") -> Optional[float]:
        """Ferme la position actuelle"""
        positions = mt5.positions_get(symbol=self.config.symbol)

        if not positions:
            self.current_position = None
            return None

        for position in positions:
            if position.magic != self.config.magic_number:
                continue

            tick = mt5.symbol_info_tick(self.config.symbol)

            if position.type == mt5.ORDER_TYPE_BUY:
                close_price = tick.bid
                order_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = tick.ask
                order_type = mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": self.config.deviation,
                "magic": self.config.magic_number,
                "comment": f"Close: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erreur fermeture: {result.comment}")
                return None

            profit = position.profit
            logger.info(f"Position fermee: {reason} | Profit: {profit:.2f}")

            self.current_position = None
            self.entry_time = None

            return profit

        return None

    def has_position(self) -> bool:
        """Verifie si une position est ouverte"""
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.config.magic_number:
                    return True
        return False

    def manage_position(self) -> Optional[float]:
        """Gere la position ouverte (trailing stop, breakeven, timeout)"""
        positions = mt5.positions_get(symbol=self.config.symbol)

        if not positions:
            return None

        for position in positions:
            if position.magic != self.config.magic_number:
                continue

            current_profit_pips = position.profit / (position.volume * 10)  # Approximation

            # Timeout - ferme apres duree max
            if self.entry_time:
                elapsed = (datetime.now() - self.entry_time).total_seconds()
                if elapsed >= self.config.max_trade_duration_seconds:
                    return self.close_position("Timeout")

            # Breakeven
            if current_profit_pips >= self.config.breakeven_pips:
                tick = mt5.symbol_info_tick(self.config.symbol)

                if position.type == mt5.ORDER_TYPE_BUY:
                    new_sl = position.price_open + (0.5 * self.config.pip_value)  # BE + 0.5 pip
                    if position.sl < new_sl:
                        self._modify_sl(position.ticket, new_sl)
                else:
                    new_sl = position.price_open - (0.5 * self.config.pip_value)
                    if position.sl > new_sl:
                        self._modify_sl(position.ticket, new_sl)

            # Trailing Stop
            if self.config.trailing_stop_pips > 0 and current_profit_pips >= self.config.trailing_stop_pips:
                tick = mt5.symbol_info_tick(self.config.symbol)

                if position.type == mt5.ORDER_TYPE_BUY:
                    new_sl = tick.bid - (self.config.trailing_stop_pips * self.config.pip_value)
                    if new_sl > position.sl:
                        self._modify_sl(position.ticket, new_sl)
                else:
                    new_sl = tick.ask + (self.config.trailing_stop_pips * self.config.pip_value)
                    if new_sl < position.sl:
                        self._modify_sl(position.ticket, new_sl)

        return None

    def _modify_sl(self, ticket: int, new_sl: float):
        """Modifie le stop loss d'une position"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return

        position = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.config.symbol,
            "position": ticket,
            "sl": round(new_sl, 5),
            "tp": position.tp,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.debug(f"SL modifie: {new_sl:.5f}")


class ScalpingRobot:
    """Robot de scalping principal"""

    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.signal_generator = SignalGenerator(config)
        self.executor = TradeExecutor(config)
        self.running = False

    def connect(self, mt5_path: str, login: int, password: str, server: str) -> bool:
        """Connexion a MT5"""
        if not mt5.initialize(mt5_path):
            logger.error("Erreur initialisation MT5")
            return False

        if not mt5.login(login, password=password, server=server):
            logger.error(f"Erreur connexion: {mt5.last_error()}")
            mt5.shutdown()
            return False

        account_info = mt5.account_info()
        logger.info(f"Connecte: {account_info.name}")
        logger.info(f"Balance: {account_info.balance:.2f} {account_info.currency}")

        self.risk_manager.update_balance(account_info.balance)

        return True

    def get_data(self, num_candles: int = 100) -> Optional[pd.DataFrame]:
        """Recupere les donnees de marche"""
        rates = mt5.copy_rates_from_pos(
            self.config.symbol,
            self.config.timeframe,
            0,
            num_candles
        )

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = ScalpingIndicators.calculate_all(df, self.config)

        return df

    def run_once(self) -> bool:
        """Execute un cycle de trading"""
        # Verifier si on a une position ouverte
        if self.executor.has_position():
            # Gerer la position existante
            result = self.executor.manage_position()
            if result is not None:
                self.risk_manager.record_trade_result(result)
            return True

        # Verifier les heures de trading
        if not self.signal_generator.check_trading_hours():
            return True

        # Verifier si on peut trader
        account_info = mt5.account_info()
        can_trade, reason = self.risk_manager.can_trade(account_info.balance)
        if not can_trade:
            logger.debug(f"Trading bloque: {reason}")
            return True

        # Recuperer les donnees
        df = self.get_data()
        if df is None:
            logger.warning("Pas de donnees")
            return True

        # Verifier le spread
        symbol_info = self.executor.get_symbol_info()
        if symbol_info:
            spread_ok, spread = self.signal_generator.check_spread(symbol_info)
            if not spread_ok:
                logger.debug(f"Spread trop eleve: {spread:.1f} pips")
                return True

        # Verifier la volatilite
        vol_ok, atr = self.signal_generator.check_volatility(df)
        if not vol_ok:
            logger.debug(f"Volatilite hors range: {atr:.1f} pips")
            return True

        # Generer signal
        signal = self.signal_generator.generate_signal(df)

        if signal != TradeDirection.NONE:
            # Calculer la taille de lot
            lot_size = self.risk_manager.calculate_lot_size(
                account_info.balance,
                self.config.stop_loss_pips
            )

            # Ouvrir la position
            success = self.executor.open_position(signal, lot_size)

            if not success:
                logger.warning("Echec ouverture position")

        return True

    def run(self, interval_seconds: float = 1.0):
        """Boucle principale du robot"""
        logger.info("="*50)
        logger.info("DEMARRAGE DU ROBOT DE SCALPING")
        logger.info(f"Symbole: {self.config.symbol}")
        logger.info(f"TP: {self.config.take_profit_pips} pips | SL: {self.config.stop_loss_pips} pips")
        logger.info(f"Risque: {self.config.risk_percent}% par trade")
        logger.info("="*50)

        self.running = True

        try:
            while self.running:
                self.run_once()
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Arret demande par l'utilisateur")
        finally:
            # Fermer les positions ouvertes
            if self.executor.has_position():
                self.executor.close_position("Arret robot")

            self.running = False
            logger.info("Robot arrete")

    def stop(self):
        """Arrete le robot"""
        self.running = False


def main():
    """Point d'entree principal"""
    print("="*60)
    print("ROBOT DE SCALPING EURUSD")
    print("Trades de 30 secondes a 3 minutes")
    print("="*60)

    # Connexion MT5
    mt5_path = r"C:\Program Files\XM MT5\terminal64.exe"

    print("\n--- Connexion MT5 ---")
    login = input("Login: ")
    password = input("Password: ")
    server = input("Serveur: ")

    # Options de configuration
    print("\n--- Options ---")
    mode_24h = input("Activer trading 24h? (o/N): ").lower() == 'o'

    # Configuration personnalisable
    config = ScalpingConfig(
        take_profit_pips=8.0,
        stop_loss_pips=5.0,
        risk_percent=1.0,
        max_trades_per_day=20,
        max_spread_pips=1.5,
        allow_24h_trading=mode_24h,
    )

    robot = ScalpingRobot(config)

    if not robot.connect(mt5_path, int(login), password, server):
        print("Echec connexion")
        return

    print("\n--- Configuration ---")
    print(f"Take Profit: {config.take_profit_pips} pips")
    print(f"Stop Loss: {config.stop_loss_pips} pips")
    print(f"Risque par trade: {config.risk_percent}%")
    print(f"Max trades/jour: {config.max_trades_per_day}")
    if config.allow_24h_trading:
        print("Horaires: 24h/24 (mode actif)")
    else:
        print(f"Horaires: {config.trading_hours_start}h - {config.trading_hours_end}h UTC")

    # Affiche l'heure actuelle UTC
    now_utc = datetime.now(timezone.utc)
    print(f"\nHeure UTC actuelle: {now_utc.strftime('%H:%M:%S')}")

    print("\nAppuyez sur Ctrl+C pour arreter le robot")
    print("="*60)

    # Demarrer le robot
    robot.run(interval_seconds=1.0)

    mt5.shutdown()


if __name__ == "__main__":
    main()
