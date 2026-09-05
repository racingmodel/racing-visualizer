/* A single immutable, hash-checked snapshot for tables, cards and notifications. */
(() => {
  'use strict';
  const names=['value_bets.json','placepot_picks.json','ai_bets.json','racing_products.json','shortlist_backtest.json','p5_shadow_predictions.json','historic_systems.json','daily_decisions.json','p6_predictions.json','race_finder_data.json','race_simulation_insights.json','extra_places.txt'];
  const events=new EventTarget(); let snapshot=null, inflight=null, error=null;
  const hash=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))), b=>b.toString(16).padStart(2,'0')).join('');
  async function read(url){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),45000);
    try{const r=await fetch(url,{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return await r.text();}
    catch(e){if(e.name==='AbortError')throw new Error(`Loading timed out: ${url.split('?')[0]}. Retrying automatically.`);throw e;}
    finally{clearTimeout(timer);}
  }
  async function refresh(){
    if(inflight)return inflight;
    inflight=(async()=>{
      try {
        const manifest=JSON.parse(await read(`release_manifest.json?t=${Date.now()}`));
        if(!manifest.files || !manifest.run_id)throw new Error('Release manifest invalid');
        const revision=JSON.stringify(manifest.files);
        if(snapshot?.revision===revision){error=null;return snapshot;}
        const data={};
        await Promise.all(names.map(async name=>{
          if(!manifest.files[name]){data[name]=name.endsWith('.txt')?'':[];return;}
          const text=await read(`${name}?release=${encodeURIComponent(manifest.run_id)}`);
          if(await hash(text)!==manifest.files[name])throw new Error(`Release still updating: ${name}`);
          data[name]=name.endsWith('.txt')?text:JSON.parse(text);
        }));
        if(!Array.isArray(data['value_bets.json']) || !Array.isArray(data['daily_decisions.json']?.rows))throw new Error('Required data contract missing');
        if(data['daily_decisions.json'].run_id!==manifest.run_id)throw new Error('Decision release mismatch');
        snapshot=Object.freeze({manifest,data,revision,receivedAt:new Date().toISOString()});error=null;
        events.dispatchEvent(new Event('change'));return snapshot;
      } catch(e){error=e;events.dispatchEvent(new Event('error'));throw e;}
      finally {inflight=null;}
    })();return inflight;
  }
  function ukDate(text){
    if(!text)return new Date(NaN);
    if(/(?:Z|[+-]\d\d:\d\d)$/.test(String(text)))return new Date(text);
    let m=String(text).match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?/);
    if(!m){let d=String(text).match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?/);if(!d)return new Date(NaN);m=[d[0],d[3],d[2],d[1],d[4],d[5],d[6]];}
    const target=Date.UTC(+m[1],+m[2]-1,+m[3],+m[4],+m[5],+(m[6]||0));let guess=target;
    const fmt=new Intl.DateTimeFormat('en-GB',{timeZone:'Europe/London',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'});
    for(let i=0;i<3;i++){const p=Object.fromEntries(fmt.formatToParts(new Date(guess)).map(x=>[x.type,x.value]));guess=target-(Date.UTC(+p.year,+p.month-1,+p.day,+p.hour,+p.minute,+p.second)-guess);}
    return new Date(guess);
  }
  function decisionState(row, now=new Date()){
    const result={...row};
    const start=ukDate(row?.race_start_utc||row?.race_datetime).getTime();
    const timestamp=ukDate(row?.quote_timestamp||row?.execution_quote_timestamp).getTime();
    const current=Number(now),age=current-timestamp;
    if(Number.isFinite(start) && start<=current){result.execution_status='closed';result.execution_reason='race_started';}
    else if(result.execution_status==='ready'){
      const reason=!Number.isFinite(start)?'invalid_race_time':!Number.isFinite(timestamp)?'missing_quote_timestamp':age < -5000?'future_timestamp':age > 300000?'stale_quote':null;
      if(reason){result.execution_status='watch';result.execution_reason=reason;}
    }
    return result;
  }
  const ready=refresh();ready.catch(()=>{});
  window.RacingStore={ready,refresh,get snapshot(){return snapshot;},get error(){return error;},ukDate,decisionState,
    async get(name){if(!snapshot)await refresh();return snapshot.data[name]??(name.endsWith('.txt')?'':[]);},
    subscribe(fn){events.addEventListener('change',fn);return()=>events.removeEventListener('change',fn);},
    onError(fn){events.addEventListener('error',fn);},
    isReady(row){return decisionState(row).execution_status==='ready';}
  };
  setInterval(()=>refresh().catch(()=>{}),60000);
})();
