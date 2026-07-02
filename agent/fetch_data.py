#!/usr/bin/env python3
"""Descarga 1 año de velas BTCUSDT de Binance (API pública, sin clave) y lo
guarda en data/btc-<tf>.csv.gz. Pensado para ejecutarse en GitHub Actions.

Usa data-api.binance.vision (espejo público de solo-datos de Binance, sin
geobloqueo en los runners de GitHub) con api.binance.com como respaldo.
Límite: 1000 velas por petición -> se pagina (1 año de 1h = 9 peticiones).
"""
import csv, gzip, io, json, os, sys, time, urllib.request

HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
SYMBOL = "BTCUSDT"
DAYS = 365
TFS = {"1h": 3600, "4h": 14400}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-trading-agent/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(io.TextIOWrapper(r, encoding="utf-8"))


def fetch_tf(tf, secs):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - DAYS * 86400 * 1000
    rows, cursor = [], start_ms
    while cursor < end_ms:
        last_err = None
        for host in HOSTS:
            url = (f"{host}/api/v3/klines?symbol={SYMBOL}&interval={tf}"
                   f"&startTime={cursor}&limit=1000")
            try:
                batch = get(url)
                break
            except Exception as e:  # noqa: BLE001 - probamos el siguiente host
                last_err = e
        else:
            raise SystemExit(f"sin acceso a Binance: {last_err}")
        if not batch:
            break
        for r in batch:
            rows.append((r[0] // 1000, r[1], r[2], r[3], r[4], r[5]))
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + secs * 1000
        time.sleep(0.3)  # cortesía con la API
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tf, secs in TFS.items():
        rows = fetch_tf(tf, secs)
        path = os.path.join(OUT_DIR, f"btc-{tf}.csv.gz")
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "open", "high", "low", "close", "volume"])
            w.writerows(rows)
        print(f"{tf}: {len(rows)} velas -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
