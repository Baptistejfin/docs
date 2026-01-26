# Robot de Scalping EURUSD - Documentation

## Caracteristiques

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| Timeframe | M1 | 1 minute |
| Duree trade | 30s - 3min | Timeout automatique |
| Take Profit | 8 pips | Objectif de gain |
| Stop Loss | 5 pips | Perte maximum |
| Ratio R:R | 1.6:1 | Risk/Reward |

## Strategie de Trading

### Conditions d'entree LONG (BUY)

1. **EMA Alignment**: EMA5 > EMA10 > EMA20
2. **Prix**: Au-dessus de EMA5
3. **RSI(7)**: < 75 et en hausse
4. **Stochastic(5,3)**: < 30 OU croisement haussier
5. **Volume**: >= moyenne
6. **Bougie**: Haussiere
7. **Momentum**: Positif

**Score minimum**: 5/7 conditions

### Conditions d'entree SHORT (SELL)

1. **EMA Alignment**: EMA5 < EMA10 < EMA20
2. **Prix**: En-dessous de EMA5
3. **RSI(7)**: > 25 et en baisse
4. **Stochastic(5,3)**: > 70 OU croisement baissier
5. **Volume**: >= moyenne
6. **Bougie**: Baissiere
7. **Momentum**: Negatif

## Gestion du Risque

### Protection du Capital

| Regle | Valeur | Action |
|-------|--------|--------|
| Risque par trade | 1% | Calcul automatique du lot |
| Perte journaliere max | 3% | Arret trading |
| Profit journalier max | 5% | Securise les gains |
| Pertes consecutives | 3 | Pause trading |
| Max trades/jour | 20 | Limite d'operations |

### Gestion de Position

- **Breakeven**: Apres 4 pips de profit, SL -> entree + 0.5 pip
- **Trailing Stop**: 3 pips (suit le prix)
- **Timeout**: Fermeture apres 3 minutes

## Filtres de Securite

### Spread
- Maximum: 1.5 pips
- Si spread > 1.5 pips: pas de trade

### Volatilite (ATR)
- Minimum: 3 pips (assez de mouvement)
- Maximum: 20 pips (evite les news)

### Horaires (UTC)
- Debut: 7h (ouverture Londres)
- Fin: 16h (fin overlap New York)
- Weekend: Pas de trading

## Utilisation

### Demarrage

```bash
python scalping_robot_eurusd.py
```

### Arret
Appuyez sur `Ctrl+C` pour arreter proprement.
Le robot fermera automatiquement les positions ouvertes.

## Configuration Personnalisee

```python
config = ScalpingConfig(
    # Objectifs
    take_profit_pips=8.0,
    stop_loss_pips=5.0,
    trailing_stop_pips=3.0,

    # Risque
    risk_percent=1.0,
    max_trades_per_day=20,
    max_daily_loss_percent=3.0,

    # Filtres
    max_spread_pips=1.5,
    min_atr_pips=3.0,
    max_atr_pips=20.0,

    # Horaires
    trading_hours_start=7,
    trading_hours_end=16,
)
```

## Indicateurs Utilises

| Indicateur | Periode | Usage |
|------------|---------|-------|
| EMA | 5, 10, 20 | Tendance |
| RSI | 7 | Momentum |
| Stochastic | 5, 3 | Surachat/Survente |
| ATR | 14 | Volatilite |
| Volume MA | 20 | Confirmation |

## Exemple de Trade

```
09:15:32 | INFO | Signal BUY detecte (score: 6/7)
09:15:32 | INFO | Position ouverte: BUY 0.05 lots @ 1.08542
09:15:32 | INFO | SL: 1.08492 | TP: 1.08622
09:16:45 | INFO | Position fermee: TP atteint | Profit: 4.00
```

## Risques et Avertissements

1. **Testez d'abord en demo** avant le compte reel
2. **Le scalping est risque** - pertes rapides possibles
3. **Surveillez le robot** - ne laissez pas tourner sans surveillance prolongee
4. **Adaptez les parametres** a votre capital et tolerance au risque
5. **Les performances passees** ne garantissent pas les resultats futurs

## Ameliorations Possibles

- [ ] Ajout filtre news economiques
- [ ] Detection de patterns (pin bar, engulfing)
- [ ] Multi-paires
- [ ] Backtesting integre
- [ ] Interface graphique
- [ ] Notifications Telegram/Discord
