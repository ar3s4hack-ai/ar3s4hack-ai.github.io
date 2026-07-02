#!/usr/bin/env node
/* Analizador autónomo del BTC Trading Agent (GitHub Actions, cada 30 min).
   - Descarga 30 días de velas 1h y 4h de Binance (espejo público de datos).
   - Calcula señales (cruces EMA + rupturas de consolidación) con lib/agent-core.
   - Puntúa cada señal con el modelo ML (data/model.json) si existe.
   - Escribe data/signals.json (lo lee el dashboard) y, si hay señal nueva
     desde la última ejecución, la envía a Telegram (secrets del repo).
   Sin dependencias: Node 18+ (fetch global). */
'use strict';
const fs = require('fs');
const path = require('path');
const core = require('../lib/agent-core.js');

const HOSTS = ['https://data-api.binance.vision', 'https://api.binance.com'];
const SYMBOL = 'BTCUSDT';
const TFS = ['1h', '4h'];
const DATA = path.join(__dirname, '..', 'data');
const OUT = path.join(DATA, 'signals.json');

async function fetchCandles(tf){
  const {binance, secs, count} = core.TF[tf];
  const end = Date.now();
  let cursor = end - count*secs*1000;
  const out = [];
  while(cursor < end){
    let batch = null, lastErr = null;
    for(const host of HOSTS){
      const url = `${host}/api/v3/klines?symbol=${SYMBOL}&interval=${binance}&startTime=${cursor}&limit=1000`;
      try{
        const res = await fetch(url, {headers:{'User-Agent':'btc-trading-agent/1.0'}});
        if(!res.ok) throw new Error(`${host} ${res.status}`);
        batch = await res.json();
        break;
      }catch(e){ lastErr = e; }
    }
    if(!batch) throw lastErr;
    if(!batch.length) break;
    for(const r of batch) out.push({time:Math.floor(r[0]/1000), open:+r[1], high:+r[2], low:+r[3], close:+r[4], volume:+r[5]});
    if(batch.length < 1000) break;
    cursor = batch[batch.length-1][0] + secs*1000;
  }
  return out;
}

async function telegram(text){
  const token = process.env.TELEGRAM_BOT_TOKEN, chat = process.env.TELEGRAM_CHAT_ID;
  if(!token || !chat){ console.error('telegram: secrets no configurados, alerta omitida'); return; }
  const res = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({chat_id: chat, text, parse_mode:'HTML'}),
  });
  if(!res.ok) console.error('telegram:', res.status, await res.text());
}

function loadJSON(p){ try{ return JSON.parse(fs.readFileSync(p,'utf8')); }catch(e){ return null; } }
const fmt$ = n => '$'+n.toLocaleString('es-ES',{maximumFractionDigits:0});

async function main(){
  const model = loadJSON(path.join(DATA, 'model.json'));
  const prev = loadJSON(OUT);
  const state = {generated_at:new Date().toISOString(), symbol:SYMBOL, tfs:{}};
  const newAlerts = [];

  for(const tf of TFS){
    // la última vela está sin cerrar: se descarta para que las señales sean firmes
    const candles = (await fetchCandles(tf)).slice(0, -1);
    if(candles.length < 60) throw new Error(`histórico insuficiente para ${tf}: ${candles.length} velas`);
    const an = core.computeSignals(candles);
    const bt = core.backtest(candles, an.sigs, core.TF[tf].tp, core.TF[tf].sl);
    const mtf = model && model.tfs && model.tfs[tf];
    const rowFn = mtf ? core.featurePrep(candles) : null;
    const score = s => {
      if(!rowFn) return null;
      const x = rowFn(s.i, s.type==='BUY' ? 1 : -1);
      return x ? Math.round(core.scoreProb(mtf, x)*1000)/10 : null;
    };
    const lastZone = an.zones[an.zones.length-1];
    const active = lastZone && lastZone.active ? lastZone : null;
    const recent = an.sigs.slice(-5).map(s=>({
      time:s.time, kind:s.kind, type:s.type, price:s.price,
      strong:s.strong||false, zone:s.zone||null, outcome:s.outcome, prob:score(s),
    }));
    state.tfs[tf] = {
      price: candles[candles.length-1].close,
      candles: candles.length,
      state: active ? 'range' : (recent.length ? recent[recent.length-1].type : 'neutral'),
      zone: active ? {top:active.top, bottom:active.bottom} : null,
      winRate: bt.winRate!=null ? Math.round(bt.winRate*10)/10 : null,
      byKind: bt.byKind, signals: recent,
    };
    // señal nueva = posterior a la última vista en la ejecución anterior
    const prevLast = prev && prev.tfs && prev.tfs[tf] && prev.tfs[tf].signals.length
      ? prev.tfs[tf].signals[prev.tfs[tf].signals.length-1].time : 0;
    for(const s of recent){
      if(s.time <= prevLast) continue;
      const buy = s.type==='BUY';
      const head = s.kind==='break'
        ? `${buy?'🟢':'🔴'} <b>RUPTURA ${buy?'ALCISTA':'BAJISTA'}</b>${s.strong?' (vol. alto)':''}`
        : `${buy?'🟢':'🔴'} <b>${buy?'COMPRA':'VENTA'}</b>`;
      const zone = s.zone ? `\nRango: ${fmt$(s.zone.bottom)}–${fmt$(s.zone.top)}` : '';
      const prob = s.prob!=null ? `\nProb. ML (TP antes que SL): <b>${s.prob}%</b>` : '';
      newAlerts.push(`${head} — BTC/USDT ${tf}\nPrecio: ${fmt$(s.price)}${zone}${prob}`);
    }
  }

  fs.mkdirSync(DATA, {recursive:true});
  fs.writeFileSync(OUT, JSON.stringify(state, null, 1));
  console.error(`signals.json actualizado · ${newAlerts.length} alertas nuevas`);
  for(const msg of newAlerts) await telegram(msg + '\n\n<i>BTC Trading Agent · análisis educativo, no es asesoramiento financiero</i>');
}

main().catch(e=>{ console.error(e); process.exit(1); });
