"""
ANALYSE DES MEILLEURS POINTS D'ENTREE/SORTIE - VERSION OPTIMISEE
EURUSD 2023-2025 - Recherche de patterns recurrents

Ameliorations par rapport a la version originale:
1. Correction du bug idxmax() (retournait datetime au lieu d'index)
2. Detection correcte des swing points avec algorithme vectorise
3. Performance O(n) au lieu de O(n^2)
4. Elimination des doublons de mouvements
5. Structure modulaire avec classes
6. Analyse statistique plus robuste
7. Support multi-timeframe optionnel
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configuration
@dataclass
class AnalysisConfig:
    symbol: str = "EURUSD"
    timeframe: int = mt5.TIMEFRAME_H4
    start_date: datetime = field(default_factory=lambda: datetime(2023, 1, 1))
    end_date: datetime = field(default_factory=datetime.now)
    min_move_pips: float = 150
    swing_lookback: int = 10  # Bougies pour detecter swing high/low
    pip_multiplier: int = 10000  # Pour EURUSD


class TechnicalIndicators:
    """Calcul vectorise des indicateurs techniques"""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed DM
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower


class SwingDetector:
    """Detection optimisee des swing points (hauts et bas locaux)"""

    @staticmethod
    def detect_swing_highs(high: pd.Series, lookback: int = 10) -> pd.Series:
        """
        Detecte les swing highs - points ou High est le max local
        Retourne une Series booleenne
        """
        # Un swing high est un point ou le high est >= aux 'lookback' bougies avant et apres
        is_swing_high = pd.Series(False, index=high.index)

        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i - lookback:i + lookback + 1]
            if high.iloc[i] == window.max():
                is_swing_high.iloc[i] = True

        return is_swing_high

    @staticmethod
    def detect_swing_lows(low: pd.Series, lookback: int = 10) -> pd.Series:
        """
        Detecte les swing lows - points ou Low est le min local
        """
        is_swing_low = pd.Series(False, index=low.index)

        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i - lookback:i + lookback + 1]
            if low.iloc[i] == window.min():
                is_swing_low.iloc[i] = True

        return is_swing_low

    @staticmethod
    def detect_swings_vectorized(high: pd.Series, low: pd.Series, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
        """
        Version vectorisee - beaucoup plus rapide pour grandes series
        """
        # Swing highs: high == rolling max
        rolling_max = high.rolling(window=2*lookback+1, center=True).max()
        is_swing_high = high == rolling_max

        # Swing lows: low == rolling min
        rolling_min = low.rolling(window=2*lookback+1, center=True).min()
        is_swing_low = low == rolling_min

        return is_swing_high, is_swing_low


@dataclass
class TradeMove:
    """Represente un mouvement de trading detecte"""
    trade_type: str  # 'BUY' ou 'SELL'
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    move_pips: float
    duration_hours: float

    # Indicateurs a l'entree
    entry_rsi: float = 0.0
    entry_adx: float = 0.0
    entry_macd_hist: float = 0.0
    entry_stoch_k: float = 0.0
    entry_stoch_d: float = 0.0
    entry_atr_pct: float = 0.0
    entry_volume_ratio: float = 0.0

    # Position vs EMAs
    entry_vs_ema20: float = 0.0
    entry_vs_ema50: float = 0.0
    entry_vs_ema200: float = 0.0
    entry_ema_aligned: bool = False

    # Bollinger
    entry_bb_position: float = 0.0

    # ROC
    entry_roc_10: float = 0.0
    entry_roc_20: float = 0.0


class MoveDetector:
    """Detection optimisee des mouvements significatifs"""

    def __init__(self, df: pd.DataFrame, config: AnalysisConfig):
        self.df = df
        self.config = config
        self.moves: List[TradeMove] = []

    def find_moves_optimized(self) -> List[TradeMove]:
        """
        Algorithme optimise pour trouver les mouvements:
        1. Detecte tous les swing highs et lows
        2. Connecte les swings consecutifs
        3. Filtre par taille minimale
        """
        print("   Detection des swing points...")

        # Detection vectorisee des swings
        is_swing_high, is_swing_low = SwingDetector.detect_swings_vectorized(
            self.df['High'],
            self.df['Low'],
            self.config.swing_lookback
        )

        # Indices des swings
        swing_high_idx = self.df.index[is_swing_high].tolist()
        swing_low_idx = self.df.index[is_swing_low].tolist()

        print(f"   Swing Highs: {len(swing_high_idx)} | Swing Lows: {len(swing_low_idx)}")

        # Combine et trie tous les swings
        all_swings = []
        for idx in swing_high_idx:
            all_swings.append(('HIGH', idx, self.df.loc[idx, 'High']))
        for idx in swing_low_idx:
            all_swings.append(('LOW', idx, self.df.loc[idx, 'Low']))

        all_swings.sort(key=lambda x: x[1])

        print(f"   Total swings: {len(all_swings)}")
        print("   Detection des mouvements significatifs...")

        # Parcours des swings pour trouver les mouvements
        moves = []
        i = 0

        while i < len(all_swings) - 1:
            current_swing = all_swings[i]
            next_swing = all_swings[i + 1]

            # Mouvement LOW -> HIGH = BUY
            if current_swing[0] == 'LOW' and next_swing[0] == 'HIGH':
                move_pips = (next_swing[2] - current_swing[2]) * self.config.pip_multiplier

                if move_pips >= self.config.min_move_pips:
                    move = self._create_move(
                        'BUY',
                        current_swing[1],  # entry at low
                        next_swing[1],     # exit at high
                        current_swing[2],  # entry price
                        next_swing[2],     # exit price
                        move_pips
                    )
                    if move:
                        moves.append(move)

            # Mouvement HIGH -> LOW = SELL
            elif current_swing[0] == 'HIGH' and next_swing[0] == 'LOW':
                move_pips = (current_swing[2] - next_swing[2]) * self.config.pip_multiplier

                if move_pips >= self.config.min_move_pips:
                    move = self._create_move(
                        'SELL',
                        current_swing[1],  # entry at high
                        next_swing[1],     # exit at low
                        current_swing[2],  # entry price
                        next_swing[2],     # exit price
                        move_pips
                    )
                    if move:
                        moves.append(move)

            i += 1

        self.moves = moves
        return moves

    def _create_move(self, trade_type: str, entry_idx, exit_idx,
                     entry_price: float, exit_price: float, move_pips: float) -> Optional[TradeMove]:
        """Cree un TradeMove avec toutes les caracteristiques a l'entree"""
        try:
            entry_data = self.df.loc[entry_idx]

            duration = (exit_idx - entry_idx).total_seconds() / 3600

            return TradeMove(
                trade_type=trade_type,
                entry_date=entry_idx,
                exit_date=exit_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                move_pips=move_pips,
                duration_hours=duration,

                entry_rsi=entry_data.get('rsi', 0),
                entry_adx=entry_data.get('adx', 0),
                entry_macd_hist=entry_data.get('macd_hist', 0),
                entry_stoch_k=entry_data.get('stoch_k', 0),
                entry_stoch_d=entry_data.get('stoch_d', 0),
                entry_atr_pct=entry_data.get('atr_pct', 0),
                entry_volume_ratio=entry_data.get('volume_ratio', 0),

                entry_vs_ema20=((entry_price - entry_data.get('ema_20', entry_price)) / entry_price * 100) if entry_data.get('ema_20') else 0,
                entry_vs_ema50=((entry_price - entry_data.get('ema_50', entry_price)) / entry_price * 100) if entry_data.get('ema_50') else 0,
                entry_vs_ema200=((entry_price - entry_data.get('ema_200', entry_price)) / entry_price * 100) if entry_data.get('ema_200') else 0,

                entry_ema_aligned=self._check_ema_alignment(entry_data, trade_type),

                entry_bb_position=self._calc_bb_position(entry_price, entry_data),

                entry_roc_10=entry_data.get('roc_10', 0),
                entry_roc_20=entry_data.get('roc_20', 0)
            )
        except Exception as e:
            print(f"   Warning: Erreur creation move: {e}")
            return None

    def _check_ema_alignment(self, data: pd.Series, trade_type: str) -> bool:
        """Verifie l'alignement des EMAs"""
        try:
            ema20 = data.get('ema_20', 0)
            ema50 = data.get('ema_50', 0)
            ema200 = data.get('ema_200', 0)

            if trade_type == 'BUY':
                return ema20 > ema50 > ema200
            else:
                return ema20 < ema50 < ema200
        except:
            return False

    def _calc_bb_position(self, price: float, data: pd.Series) -> float:
        """Calcule la position dans les Bollinger Bands (0=lower, 1=upper)"""
        try:
            bb_upper = data.get('bb_upper', 0)
            bb_lower = data.get('bb_lower', 0)

            if bb_upper and bb_lower and bb_upper != bb_lower:
                return (price - bb_lower) / (bb_upper - bb_lower)
            return 0.5
        except:
            return 0.5


class DataLoader:
    """Chargement et preparation des donnees MT5"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None

    def connect_mt5(self, mt5_path: str, login: int, password: str, server: str) -> bool:
        """Connexion a MetaTrader 5"""
        if not mt5.initialize(mt5_path):
            print("Erreur initialisation MT5")
            return False

        if not mt5.login(login, password=password, server=server):
            print(f"Echec connexion: {mt5.last_error()}")
            mt5.shutdown()
            return False

        return True

    def load_data(self) -> pd.DataFrame:
        """Telecharge les donnees depuis MT5"""
        print(f"Telechargement {self.config.symbol} H4...")

        rates = mt5.copy_rates_range(
            self.config.symbol,
            self.config.timeframe,
            self.config.start_date,
            self.config.end_date
        )

        if rates is None or len(rates) == 0:
            raise ValueError("Pas de donnees disponibles")

        print(f"{len(rates)} bougies telechargees")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)

        mt5.shutdown()
        self.df = df
        return df

    def calculate_indicators(self) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques"""
        if self.df is None:
            raise ValueError("Donnees non chargees")

        df = self.df
        ti = TechnicalIndicators

        print("Calcul des indicateurs...")

        # EMAs
        for period in [9, 20, 50, 100, 200]:
            df[f'ema_{period}'] = ti.ema(df['Close'], period)

        # RSI
        df['rsi'] = ti.rsi(df['Close'], 14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = ti.macd(df['Close'])

        # Stochastic
        df['stoch_k'], df['stoch_d'] = ti.stochastic(df['High'], df['Low'], df['Close'])

        # ADX
        df['adx'], df['adx_pos'], df['adx_neg'] = ti.adx(df['High'], df['Low'], df['Close'])

        # ATR
        df['atr'] = ti.atr(df['High'], df['Low'], df['Close'])
        df['atr_pct'] = (df['atr'] / df['Close']) * 100

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ti.bollinger_bands(df['Close'])

        # Volume
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']

        # ROC
        df['roc_10'] = df['Close'].pct_change(10) * 100
        df['roc_20'] = df['Close'].pct_change(20) * 100

        # Supprime les NaN
        df.dropna(inplace=True)

        print("Indicateurs calcules")
        self.df = df
        return df


class AnalysisReport:
    """Generation des rapports d'analyse"""

    def __init__(self, moves: List[TradeMove]):
        self.moves = moves
        self.moves_df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convertit la liste de moves en DataFrame"""
        if not self.moves:
            return pd.DataFrame()

        data = []
        for m in self.moves:
            data.append({
                'type': m.trade_type,
                'entry_date': m.entry_date,
                'exit_date': m.exit_date,
                'entry_price': m.entry_price,
                'exit_price': m.exit_price,
                'move_pips': m.move_pips,
                'duration_hours': m.duration_hours,
                'entry_rsi': m.entry_rsi,
                'entry_adx': m.entry_adx,
                'entry_macd_hist': m.entry_macd_hist,
                'entry_stoch_k': m.entry_stoch_k,
                'entry_atr_pct': m.entry_atr_pct,
                'entry_volume_ratio': m.entry_volume_ratio,
                'entry_vs_ema20': m.entry_vs_ema20,
                'entry_vs_ema50': m.entry_vs_ema50,
                'entry_vs_ema200': m.entry_vs_ema200,
                'entry_ema_aligned': m.entry_ema_aligned,
                'entry_bb_position': m.entry_bb_position,
                'entry_roc_10': m.entry_roc_10,
                'entry_roc_20': m.entry_roc_20
            })

        return pd.DataFrame(data)

    def print_yearly_stats(self):
        """Affiche les statistiques par annee"""
        print("=" * 70)
        print("REPARTITION PAR ANNEE")
        print("=" * 70)

        if self.moves_df.empty:
            print("Aucun mouvement detecte")
            return

        self.moves_df['year'] = pd.to_datetime(self.moves_df['entry_date']).dt.year

        for year in sorted(self.moves_df['year'].unique()):
            year_moves = self.moves_df[self.moves_df['year'] == year]
            buy_count = len(year_moves[year_moves['type'] == 'BUY'])
            sell_count = len(year_moves[year_moves['type'] == 'SELL'])
            avg_pips = year_moves['move_pips'].mean()

            print(f"\n{year}:")
            print(f"  Total: {len(year_moves)} | BUY: {buy_count} | SELL: {sell_count}")
            print(f"  Mouvement moyen: {avg_pips:.1f} pips")

    def print_pattern_analysis(self, trade_type: str):
        """Analyse des patterns pour un type de trade"""
        df = self.moves_df[self.moves_df['type'] == trade_type]

        if df.empty:
            print(f"Aucun mouvement {trade_type}")
            return

        direction = "HAUSSIERS (BUY)" if trade_type == 'BUY' else "BAISSIERS (SELL)"
        print(f"\nMOUVEMENTS {direction} - {len(df)} trades")
        print("-" * 70)

        # RSI
        print(f"\nRSI a l'entree:")
        print(f"   Moyenne: {df['entry_rsi'].mean():.1f} | Mediane: {df['entry_rsi'].median():.1f}")
        print(f"   Zone optimale (Q1-Q3): {df['entry_rsi'].quantile(0.25):.1f} - {df['entry_rsi'].quantile(0.75):.1f}")

        # ADX
        adx_strong = (df['entry_adx'] > 25).sum()
        print(f"\nADX a l'entree:")
        print(f"   Moyenne: {df['entry_adx'].mean():.1f}")
        print(f"   ADX > 25: {adx_strong}/{len(df)} ({adx_strong/len(df)*100:.1f}%)")

        # Stochastic
        if trade_type == 'BUY':
            stoch_condition = (df['entry_stoch_k'] < 30).sum()
            stoch_label = "Stoch < 30"
        else:
            stoch_condition = (df['entry_stoch_k'] > 70).sum()
            stoch_label = "Stoch > 70"

        print(f"\nStochastic K:")
        print(f"   Moyenne: {df['entry_stoch_k'].mean():.1f}")
        print(f"   {stoch_label}: {stoch_condition}/{len(df)} ({stoch_condition/len(df)*100:.1f}%)")

        # EMA Alignment
        ema_aligned = df['entry_ema_aligned'].sum()
        print(f"\nEMA Alignment:")
        print(f"   Parfait: {ema_aligned}/{len(df)} ({ema_aligned/len(df)*100:.1f}%)")

        # Bollinger Position
        if trade_type == 'BUY':
            bb_condition = (df['entry_bb_position'] < 0.3).sum()
            bb_label = "Lower 30%"
        else:
            bb_condition = (df['entry_bb_position'] > 0.7).sum()
            bb_label = "Upper 30%"

        print(f"\nPosition Bollinger:")
        print(f"   Moyenne: {df['entry_bb_position'].mean():.2f}")
        print(f"   {bb_label}: {bb_condition}/{len(df)} ({bb_condition/len(df)*100:.1f}%)")

        # Volume
        high_vol = (df['entry_volume_ratio'] > 1.0).sum()
        print(f"\nVolume Ratio:")
        print(f"   Moyenne: {df['entry_volume_ratio'].mean():.2f}x")
        print(f"   > moyenne: {high_vol}/{len(df)} ({high_vol/len(df)*100:.1f}%)")

        # Duration
        print(f"\nDuree moyenne: {df['duration_hours'].mean():.0f}h ({df['duration_hours'].mean()/24:.1f}j)")
        print(f"Mouvement moyen: {df['move_pips'].mean():.1f} pips")

    def print_top_moves(self, n: int = 10):
        """Affiche les N meilleurs mouvements"""
        print("\n" + "=" * 70)
        print(f"TOP {n} MEILLEURS MOUVEMENTS")
        print("=" * 70)

        top = self.moves_df.nlargest(n, 'move_pips')

        for i, (_, move) in enumerate(top.iterrows(), 1):
            aligned = "OUI" if move['entry_ema_aligned'] else "NON"
            print(f"\n#{i}. {move['type']} - {move['move_pips']:.0f} pips")
            print(f"   {move['entry_date'].strftime('%Y-%m-%d')} -> {move['exit_date'].strftime('%Y-%m-%d')}")
            print(f"   Duree: {move['duration_hours']:.0f}h | RSI: {move['entry_rsi']:.1f} | ADX: {move['entry_adx']:.1f}")
            print(f"   EMA Aligned: {aligned} | BB Pos: {move['entry_bb_position']:.2f}")

    def print_optimal_rules(self):
        """Deduit et affiche les regles optimales"""
        print("\n" + "=" * 70)
        print("REGLES OPTIMALES DEDUITES")
        print("=" * 70)

        for trade_type in ['BUY', 'SELL']:
            df = self.moves_df[self.moves_df['type'] == trade_type]

            if df.empty:
                continue

            direction = "LONG (BUY)" if trade_type == 'BUY' else "SHORT (SELL)"
            print(f"\nPOUR LES ENTREES {direction}:")

            # RSI
            rsi_q1 = df['entry_rsi'].quantile(0.25)
            rsi_q3 = df['entry_rsi'].quantile(0.75)
            print(f"   1. RSI entre {rsi_q1:.0f} et {rsi_q3:.0f}")

            # ADX
            adx_median = df['entry_adx'].median()
            print(f"   2. ADX > {adx_median:.0f}")

            # Stochastic
            if trade_type == 'BUY':
                stoch_pct = (df['entry_stoch_k'] < 30).sum() / len(df) * 100
                print(f"   3. Stochastic < 30 ({stoch_pct:.0f}% des cas)")
            else:
                stoch_pct = (df['entry_stoch_k'] > 70).sum() / len(df) * 100
                print(f"   3. Stochastic > 70 ({stoch_pct:.0f}% des cas)")

            # EMA
            ema_pct = df['entry_ema_aligned'].sum() / len(df) * 100
            if trade_type == 'BUY':
                print(f"   4. EMA 20 > 50 > 200 ({ema_pct:.0f}% des cas)")
            else:
                print(f"   4. EMA 20 < 50 < 200 ({ema_pct:.0f}% des cas)")

            # Bollinger
            if trade_type == 'BUY':
                bb_pct = (df['entry_bb_position'] < 0.3).sum() / len(df) * 100
                print(f"   5. Prix dans lower 30% BB ({bb_pct:.0f}% des cas)")
            else:
                bb_pct = (df['entry_bb_position'] > 0.7).sum() / len(df) * 100
                print(f"   5. Prix dans upper 30% BB ({bb_pct:.0f}% des cas)")

            # Volume
            vol_pct = (df['entry_volume_ratio'] > 1).sum() / len(df) * 100
            print(f"   6. Volume > moyenne ({vol_pct:.0f}% des cas)")

    def save_to_csv(self, filename: str):
        """Sauvegarde l'analyse en CSV"""
        self.moves_df.to_csv(filename, index=False)
        print(f"\nAnalyse sauvegardee: '{filename}'")


def main():
    """Point d'entree principal"""
    print("=" * 70)
    print("ANALYSE DES MEILLEURS POINTS D'ENTREE/SORTIE - VERSION OPTIMISEE")
    print("EURUSD 2023-2025 - Recherche de patterns recurrents")
    print("=" * 70)

    # Configuration
    config = AnalysisConfig()

    # Connexion MT5
    mt5_path = r"C:\Program Files\XM MT5\terminal64.exe"

    if not mt5.initialize(mt5_path):
        print("Erreur MT5")
        input("Quitter...")
        return

    login = input("\nLogin : ")
    password = input("Password : ")
    server = input("Serveur : ")

    # Chargement des donnees
    loader = DataLoader(config)

    if not loader.connect_mt5(mt5_path, int(login), password, server):
        return

    print(f"\nConnecte\n")

    try:
        df = loader.load_data()
        df = loader.calculate_indicators()

        print(f"\n{len(df)} bougies avec indicateurs")

        # Detection des mouvements
        print("\nRecherche des mouvements significatifs...\n")
        detector = MoveDetector(df, config)
        moves = detector.find_moves_optimized()

        print(f"\n{len(moves)} mouvements majeurs detectes (>= {config.min_move_pips} pips)\n")

        # Rapport
        report = AnalysisReport(moves)
        report.print_yearly_stats()

        print("\n" + "=" * 70)
        print("PATTERNS COMMUNS DES MEILLEURS POINTS D'ENTREE")
        print("=" * 70)

        report.print_pattern_analysis('BUY')
        report.print_pattern_analysis('SELL')
        report.print_top_moves(10)
        report.print_optimal_rules()

        report.save_to_csv('best_moves_analysis_optimized.csv')

    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    input("\nAppuyez sur Entree pour quitter...")


if __name__ == "__main__":
    main()
