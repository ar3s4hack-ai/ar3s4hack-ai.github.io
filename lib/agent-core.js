/* Núcleo del BTC Trading Agent — compartido entre el dashboard (navegador),
   el analizador autónomo (Node, GitHub Actions) y espejado 1:1 en Python
   (agent/train.py). Cualquier cambio en indicadores o features debe
   replicarse en train.py o el modelo ML puntuará mal. */
(function(root, factory){
  if(typeof module==='object' && module.exports) module.exports = factory();
  else root.AgentCore = factory();
})(typeof self!=='undefined' ? self : this, function(){
'use strict';

// count = velas que cubren 30 días (1D usa 200: con 30 no hay histórico para los indicadores)
const TF = {
  '15m': {binance:'15m', cb:900,   secs:900,   count:2880, tp:0.006, sl:0.010},
  '1h' : {binance:'1h',  cb:3600,  secs:3600,  count:720,  tp:0.012, sl:0.020},
  '4h' : {binance:'4h',  cb:14400, secs:14400, count:180,  tp:0.020, sl:0.033},
  '1d' : {binance:'1d',  cb:86400, secs:86400, count:200,  tp:0.035, sl:0.058},
};
const ZONE_MIN = 10, ZONE_WIDTH = 2.2, VOL_CONFIRM = 1.5;

/* ── indicadores ── */
function ema(values, period){
  const k = 2/(period+1), out = [];
  let prev = values[0];
  for(let i=0;i<values.length;i++){
    prev = i===0 ? values[0] : values[i]*k + prev*(1-k);
    out.push(prev);
  }
  return out;
}
function rsi(closes, period=14){
  const out = new Array(closes.length).fill(null);
  let gain=0, loss=0;
  for(let i=1;i<closes.length;i++){
    const d = closes[i]-closes[i-1];
    const g = Math.max(d,0), l = Math.max(-d,0);
    if(i<=period){
      gain+=g; loss+=l;
      if(i===period){ gain/=period; loss/=period; out[i]=100-100/(1+(loss===0?1e9:gain/loss)); }
    } else {
      gain=(gain*(period-1)+g)/period;
      loss=(loss*(period-1)+l)/period;
      out[i]=100-100/(1+(loss===0?1e9:gain/loss));
    }
  }
  return out;
}
function atr(cs, period=14){
  const out = new Array(cs.length).fill(null);
  let prev = null;
  for(let i=1;i<cs.length;i++){
    const tr = Math.max(cs[i].high-cs[i].low,
      Math.abs(cs[i].high-cs[i-1].close), Math.abs(cs[i].low-cs[i-1].close));
    prev = i<=period ? (((prev??0)*(i-1))+tr)/i : (prev*(period-1)+tr)/period;
    out[i] = prev;
  }
  return out;
}
function sma(values, period){
  const out = new Array(values.length).fill(null);
  let acc = 0;
  for(let i=0;i<values.length;i++){
    acc += values[i];
    if(i>=period) acc -= values[i-period];
    if(i>=period-1) out[i] = acc/period;
  }
  return out;
}

/* ── consolidación y rupturas ──
   Un rango nace cuando ZONE_MIN velas caben en una banda < ZONE_WIDTH×ATR(14).
   La caja absorbe mechas mientras siga estrecha; una vela que CIERRA fuera
   es la ruptura (volumen > VOL_CONFIRM×media del rango = confirmada). */
function detectZones(cs, a){
  const zones = [];
  let i = ZONE_MIN + 14;
  while(i < cs.length){
    const s = i - ZONE_MIN + 1;
    const win = cs.slice(s, i+1);
    let hi = Math.max(...win.map(c=>c.high));
    let lo = Math.min(...win.map(c=>c.low));
    if(a[i] && (hi-lo) < ZONE_WIDTH*a[i]){
      const maxW = ZONE_WIDTH*a[i];
      let j = i+1, breakout = null;
      while(j < cs.length){
        const c = cs[j];
        if(c.close > hi){ breakout = {i:j, dir:'up'}; break; }
        if(c.close < lo){ breakout = {i:j, dir:'down'}; break; }
        const nh = Math.max(hi, c.high), nl = Math.min(lo, c.low);
        if(nh-nl < maxW){ hi = nh; lo = nl; }
        j++;
      }
      const end = breakout ? breakout.i : cs.length-1;
      const avgVol = win.reduce((t,c)=>t+(c.volume||0),0) / win.length;
      if(breakout) breakout.strong = avgVol>0 && (cs[breakout.i].volume||0) > VOL_CONFIRM*avgVol;
      zones.push({start:s, end, top:hi, bottom:lo, breakout, active:!breakout});
      i = end + 1;
    } else i++;
  }
  return zones;
}

/* ── motor de señales (solo velas cerradas) ── */
function computeSignals(cs){
  const closes = cs.map(c=>c.close);
  const e9 = ema(closes,9), e21 = ema(closes,21), r = rsi(closes,14);
  const a = atr(cs,14);
  const zones = detectZones(cs, a);
  const sigs = [];
  for(let i=30;i<cs.length;i++){
    const up = e9[i-1]<=e21[i-1] && e9[i]>e21[i];
    const dn = e9[i-1]>=e21[i-1] && e9[i]<e21[i];
    if(up && r[i]!=null && r[i]>=45 && r[i]<=72) sigs.push({i, kind:'ema', type:'BUY',  price:closes[i], time:cs[i].time});
    if(dn && r[i]!=null && r[i]<=55 && r[i]>=28) sigs.push({i, kind:'ema', type:'SELL', price:closes[i], time:cs[i].time});
  }
  for(const z of zones){
    if(!z.breakout) continue;
    const b = z.breakout;
    sigs.push({i:b.i, kind:'break', type:b.dir==='up'?'BUY':'SELL', strong:b.strong,
      price:cs[b.i].close, time:cs[b.i].time, zone:{top:z.top, bottom:z.bottom}});
  }
  sigs.sort((x,y)=>x.i-y.i || (x.kind<y.kind?-1:1));
  return {sigs, zones, e9, e21, r, a};
}

/* ── backtest TP/SL sobre velas posteriores (SL se comprueba primero) ── */
function backtest(cs, sigs, tp, sl){
  let wins=0, losses=0;
  const byKind = {};
  for(const s of sigs){
    s.outcome = 'open';
    for(let j=s.i+1;j<cs.length;j++){
      const c = cs[j];
      if(s.type==='BUY'){
        if(c.low  <= s.price*(1-sl)){ s.outcome='loss'; break; }
        if(c.high >= s.price*(1+tp)){ s.outcome='win';  break; }
      } else {
        if(c.high >= s.price*(1+sl)){ s.outcome='loss'; break; }
        if(c.low  <= s.price*(1-tp)){ s.outcome='win';  break; }
      }
    }
    if(s.outcome==='win') wins++;
    else if(s.outcome==='loss') losses++;
    if(s.outcome!=='open'){
      const k = byKind[s.kind] || (byKind[s.kind]={w:0,l:0});
      if(s.outcome==='win') k.w++; else k.l++;
    }
  }
  const closed = wins+losses;
  return {wins, losses, winRate: closed ? wins/closed*100 : null, byKind};
}

/* ── features para el modelo ML ──
   El orden es un contrato: espejado en agent/train.py (FEATURES). */
const FEATURES = ['dir','rsi','ema_spread_atr','close_ema21_atr','atr_pct',
                  'vol_ratio','ret5','ret20','pos_range20','hour_sin','hour_cos'];
function featurePrep(cs){
  const closes = cs.map(c=>c.close);
  const e9 = ema(closes,9), e21 = ema(closes,21), r = rsi(closes,14), a = atr(cs,14);
  const volS = sma(cs.map(c=>c.volume||0), 20);
  return function(i, dir){ // dir: +1 largo (BUY), -1 corto (SELL)
    if(i<30 || i>=cs.length || a[i]==null || r[i]==null) return null;
    const c = cs[i], at = a[i] || 1e-9;
    let min20=Infinity, max20=-Infinity;
    for(let k=i-19;k<=i;k++){ if(closes[k]<min20)min20=closes[k]; if(closes[k]>max20)max20=closes[k]; }
    const hr = new Date(c.time*1000).getUTCHours(), ang = 2*Math.PI*hr/24;
    const vr = (volS[i]&&volS[i]>0) ? Math.min(5,(c.volume||0)/volS[i]) : 1;
    return [dir, r[i]/100, (e9[i]-e21[i])/at, (c.close-e21[i])/at, at/c.close*100,
            vr, (c.close/closes[i-5]-1)*100, (c.close/closes[i-20]-1)*100,
            max20>min20 ? (c.close-min20)/(max20-min20) : 0.5,
            Math.sin(ang), Math.cos(ang)];
  };
}
/* Puntúa una fila de features con un modelo exportado por train.py:
   regresión logística sobre features estandarizadas. Devuelve P(TP antes que SL). */
function scoreProb(model, x){
  let z = model.b;
  for(let k=0;k<x.length;k++) z += model.w[k] * ((x[k]-model.mu[k]) / (model.sigma[k]||1));
  return 1/(1+Math.exp(-z));
}

return {TF, ZONE_MIN, ZONE_WIDTH, VOL_CONFIRM, FEATURES,
        ema, rsi, atr, sma, detectZones, computeSignals, backtest,
        featurePrep, scoreProb};
});
