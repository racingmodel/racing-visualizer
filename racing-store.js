/* A single immutable, hash-checked snapshot for tables, cards and notifications. */
(() => {
  'use strict';
  const events=new EventTarget(); let snapshot=null, inflight=null, error=null;
  const files=new Map();
  const hash=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))), b=>b.toString(16).padStart(2,'0')).join('');
  async function read(url, cache='no-store'){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),45000);
    try{const r=await fetch(url,{cache,signal:controller.signal});if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return await r.text();}
    catch(e){if(e.name==='AbortError')throw new Error(`Loading timed out: ${url.split('?')[0]}. Retrying automatically.`);throw e;}
    finally{clearTimeout(timer);}
  }
  async function load(manifest,name){
    const digest=manifest.files[name];
    if(!digest)return name.endsWith('.txt')?'':[];
    const key=`${name}:${digest}`;
    if(!files.has(key)){
      const request=(async()=>{
        // Verify every response, including resources reused from browser cache.
        const text=await read(`${name}?release=${encodeURIComponent(digest)}`,'no-cache');
        if(await hash(text)!==digest)throw new Error(`Release still updating: ${name}`);
        const data=name.endsWith('.txt')?text:JSON.parse(text);
        if(name==='race_finder_data.json'){
          const universe=manifest.race_universe||{};
          if(data?.run_id!==manifest.run_id || Number(data?.summary?.races)!==Number(universe.races)
             || Number(data?.summary?.runners)!==Number(universe.runners)){
            throw new Error('Published race universe mismatch');
          }
        }
        return data;
      })();
      files.set(key,request);
      request.catch(()=>{if(files.get(key)===request)files.delete(key);});
    }
    return files.get(key);
  }
  async function refresh(){
    if(inflight)return inflight;
    inflight=(async()=>{
      try {
        const manifest=JSON.parse(await read(`release_manifest.json?t=${Date.now()}`));
        if(!manifest.files || !manifest.run_id)throw new Error('Release manifest invalid');
        if(manifest.integrity?.status!=='passed')throw new Error('Release integrity was not verified');
        const revision=JSON.stringify(manifest.files);
        if(snapshot?.revision===revision){
          const recovered=error;error=null;
          if(recovered)events.dispatchEvent(new Event('change'));
          return snapshot;
        }
        const data={};
        const core=['release_integrity.json','daily_decisions.json'];
        await Promise.all(core.map(async name=>{data[name]=await load(manifest,name);}));
        if(!Array.isArray(data['daily_decisions.json']?.rows))throw new Error('Required data contract missing');
        if(data['daily_decisions.json'].run_id!==manifest.run_id)throw new Error('Decision release mismatch');
        const integrity=data['release_integrity.json'];
        if(integrity?.status!=='passed'||integrity?.run_id!==manifest.run_id)throw new Error('Release integrity report mismatch');
        for(const name of Object.keys(snapshot?.data||{})){
          if(!core.includes(name) && manifest.files[name] && manifest.files[name]===snapshot.manifest.files[name]){
            data[name]=snapshot.data[name];
          }
        }
        snapshot=Object.freeze({manifest,data,revision,receivedAt:new Date().toISOString()});error=null;
        const keep=new Set(Object.entries(manifest.files).map(([name,digest])=>`${name}:${digest}`));
        for(const key of files.keys())if(!keep.has(key))files.delete(key);
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
      const reason=!Number.isFinite(start)?'invalid_race_time':!Number.isFinite(timestamp)?'missing_quote_timestamp':age < -5000?'future_timestamp':age > 1000*Number(row?.quote_max_age_seconds||300)?'stale_quote':null;
      if(reason){result.execution_status='watch';result.execution_reason=reason;}
    }
    return result;
  }
  function isModelSelection(row, now=new Date()){
    const start=ukDate(row?.race_start_utc||row?.race_datetime).getTime();
    const stamp=ukDate(row?.model_snapshot_at).getTime();
    return row?.shortlist_status==='qualified' && row?.current_model_qualified===true
      && !!row?.route && row?.input_universe_verified===true && !row?.field_changed_after_model
      && !row?.non_runner && !['REMOVED','REMOVED_VACANT','NON_RUNNER','NR','N/R','NON RUNNER','WITHDRAWN'].includes(String(row?.bf_runner_status||'').toUpperCase())
      && Number.isFinite(start) && start>Number(now) && Number.isFinite(stamp) && stamp<=Number(now) && stamp<start;
  }
  function currentShortlistRows(payload, now=new Date()){
    const selected=new Map();
    const identity=r=>`${ukDate(r.race_start_utc||r.race_datetime).toISOString()}|${String(r.course||'').trim().toLowerCase()}|${String(r.horse||'').trim().toLowerCase()}`;
    for(const row of payload?.model_shortlist_rows||[]){
      if(payload.run_id && row.run_id===payload.run_id && isModelSelection(row,now))selected.set(identity(row),{...row});
    }
    const age=Number(now)-ukDate(payload?.generated_at_utc).getTime();
    if(Number.isFinite(age) && age>=-5000 && age<=1000*Number(payload?.risk_policy?.quote_max_age_seconds||300)){
      for(const row of payload?.rows||[])if(decisionState(row,now).execution_status==='ready')selected.set(identity(row),{...row});
    }
    return [...selected.values()].sort((a,b)=>ukDate(a.race_start_utc||a.race_datetime)-ukDate(b.race_start_utc||b.race_datetime));
  }
  const ready=refresh();ready.catch(()=>{});
  window.RacingStore={ready,refresh,get snapshot(){return snapshot;},get error(){return error;},ukDate,decisionState,isModelSelection,currentShortlistRows,
    async get(name){
      const pinned=snapshot||await refresh();
      if(Object.hasOwn(pinned.data,name))return pinned.data[name];
      const data=await load(pinned.manifest,name);
      if(snapshot!==pinned)return this.get(name);
      pinned.data[name]=data;
      return data;
    },
    subscribe(fn){events.addEventListener('change',fn);return()=>events.removeEventListener('change',fn);},
    onError(fn){events.addEventListener('error',fn);},
    isReady(row){return decisionState(row).execution_status==='ready';}
  };
  setInterval(()=>refresh().catch(()=>{}),60000);
  if(typeof window.addEventListener==='function')window.addEventListener('focus',()=>refresh().catch(()=>{}));
})();
