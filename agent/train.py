#!/usr/bin/env python3
"""Entrena el modelo ML del BTC Trading Agent con 1 año de datos de Binance.

Para cada vela del histórico simula una entrada larga y una corta con el
TP/SL del agente y etiqueta si el take profit se alcanza antes que el stop.
Sobre esas ~17.000 muestras (1h) entrena:
  - Regresión logística estandarizada -> se exporta a data/model.json y el
    navegador/Node la evalúan en vivo (P de que la señal alcance TP).
  - HistGradientBoosting como referencia de techo de rendimiento (solo métrica).

Validación walk-forward honesta: entrena con el primer 75% del año y se
evalúa con el 25% final, que el modelo nunca ha visto. Sin barajar: en series
temporales barajar es hacerse trampas al solitario.

IMPORTANTE: los indicadores y features espejan 1:1 lib/agent-core.js
(featurePrep / FEATURES). Si cambias algo aquí, cámbialo allí.
"""
import csv, gzip, json, math, os, sys, datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TF_PARAMS = {"1h": (0.012, 0.020), "4h": (0.020, 0.033)}  # tp, sl (= agent-core TF)
LOOKAHEAD = 400   # velas máximas para resolver TP/SL; sin resolver -> se descarta
FEATURES = ["dir", "rsi", "ema_spread_atr", "close_ema21_atr", "atr_pct",
            "vol_ratio", "ret5", "ret20", "pos_range20", "hour_sin", "hour_cos"]


# ── indicadores: espejo exacto de lib/agent-core.js ──
def ema(values, period):
    k = 2 / (period + 1)
    out, prev = [], values[0]
    for i, v in enumerate(values):
        prev = values[0] if i == 0 else v * k + prev * (1 - k)
        out.append(prev)
    return out


def rsi(closes, period=14):
    out = [None] * len(closes)
    gain = loss = 0.0
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        g, l = max(d, 0), max(-d, 0)
        if i <= period:
            gain += g; loss += l
            if i == period:
                gain /= period; loss /= period
                out[i] = 100 - 100 / (1 + (1e9 if loss == 0 else gain / loss))
        else:
            gain = (gain * (period - 1) + g) / period
            loss = (loss * (period - 1) + l) / period
            out[i] = 100 - 100 / (1 + (1e9 if loss == 0 else gain / loss))
    return out


def atr(cs, period=14):
    out = [None] * len(cs)
    prev = None
    for i in range(1, len(cs)):
        tr = max(cs[i]["high"] - cs[i]["low"],
                 abs(cs[i]["high"] - cs[i - 1]["close"]),
                 abs(cs[i]["low"] - cs[i - 1]["close"]))
        prev = (((prev or 0) * (i - 1)) + tr) / i if i <= period \
            else (prev * (period - 1) + tr) / period
        out[i] = prev
    return out


def sma(values, period):
    out = [None] * len(values)
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= period:
            acc -= values[i - period]
        if i >= period - 1:
            out[i] = acc / period
    return out


def feature_prep(cs):
    closes = [c["close"] for c in cs]
    e9, e21, r, a = ema(closes, 9), ema(closes, 21), rsi(closes, 14), atr(cs, 14)
    vol_s = sma([c["volume"] for c in cs], 20)
    def row(i, direction):
        if i < 30 or i >= len(cs) or a[i] is None or r[i] is None:
            return None
        c, at = cs[i], a[i] or 1e-9
        win = closes[i - 19:i + 1]
        min20, max20 = min(win), max(win)
        hr = datetime.datetime.fromtimestamp(c["time"], datetime.timezone.utc).hour
        ang = 2 * math.pi * hr / 24
        vr = min(5, c["volume"] / vol_s[i]) if (vol_s[i] and vol_s[i] > 0) else 1
        return [direction, r[i] / 100, (e9[i] - e21[i]) / at, (c["close"] - e21[i]) / at,
                at / c["close"] * 100, vr,
                (c["close"] / closes[i - 5] - 1) * 100, (c["close"] / closes[i - 20] - 1) * 100,
                (c["close"] - min20) / (max20 - min20) if max20 > min20 else 0.5,
                math.sin(ang), math.cos(ang)]
    return row


def label(cs, i, direction, tp, sl):
    """1 si TP antes que SL, 0 al revés (SL primero, como el backtest JS), None sin resolver."""
    entry = cs[i]["close"]
    for j in range(i + 1, min(i + 1 + LOOKAHEAD, len(cs))):
        c = cs[j]
        if direction == 1:
            if c["low"] <= entry * (1 - sl):  return 0
            if c["high"] >= entry * (1 + tp): return 1
        else:
            if c["high"] >= entry * (1 + sl): return 0
            if c["low"] <= entry * (1 - tp):  return 1
    return None


def load(tf):
    path = os.path.join(DATA, f"btc-{tf}.csv.gz")
    with gzip.open(path, "rt") as f:
        return [{"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
                 "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"])}
                for r in csv.DictReader(f)]


def train_tf(tf):
    tp, sl = TF_PARAMS[tf]
    cs = load(tf)
    row = feature_prep(cs)
    X, y = [], []
    for i in range(30, len(cs) - 1):
        for d in (1, -1):
            x = row(i, d)
            if x is None:
                continue
            lb = label(cs, i, d, tp, sl)
            if lb is None:
                continue
            X.append(x); y.append(lb)
    X, y = np.array(X), np.array(y)
    cut = int(len(X) * 0.75)  # walk-forward: pasado entrena, futuro examina
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(scaler.transform(Xtr), ytr)
    p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]

    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                         learning_rate=0.06).fit(Xtr, ytr)
    p_gbm = gbm.predict_proba(Xte)[:, 1]

    metrics = {
        "n_train": int(cut), "n_test": int(len(X) - cut),
        "base_rate": round(float(yte.mean()), 4),
        "auc_lr": round(float(roc_auc_score(yte, p_lr)), 4),
        "acc_lr": round(float(accuracy_score(yte, p_lr > 0.5)), 4),
        "auc_gbm": round(float(roc_auc_score(yte, p_gbm)), 4),
        "acc_gbm": round(float(accuracy_score(yte, p_gbm > 0.5)), 4),
    }
    print(f"{tf}: {metrics}", file=sys.stderr)
    return {
        "features": FEATURES, "tp": tp, "sl": sl,
        "mu": [round(float(v), 8) for v in scaler.mean_],
        "sigma": [round(float(v), 8) for v in scaler.scale_],
        "w": [round(float(v), 8) for v in lr.coef_[0]],
        "b": round(float(lr.intercept_[0]), 8),
        "metrics": metrics,
    }


def main():
    model = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days": 365, "symbol": "BTCUSDT",
        "tfs": {tf: train_tf(tf) for tf in TF_PARAMS},
    }
    out = os.path.join(DATA, "model.json")
    with open(out, "w") as f:
        json.dump(model, f, indent=1)
    print(f"modelo -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
