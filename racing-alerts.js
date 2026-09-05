/* Alerts use precisely the same backend decisions and release as the page. */
(() => {
 let previous=null;
 const update=()=>{
  const payload=RacingStore.snapshot?.data['daily_decisions.json'];if(!payload)return;
  const age=Date.now()-Date.parse(payload.generated_at_utc);
  const fresh=age>=-5000 && age<=300000;
  const rows=fresh?payload.rows.filter(RacingStore.isReady):[];
  const current=new Set(rows.map(r=>`${r.race_start_utc}|${r.course}|${r.horse}|${r.route}`));
  if(previous && localStorage.getItem('racing_notifications')==='on' && 'Notification'in window && Notification.permission==='granted'){
   const added=rows.filter(r=>!previous.has(`${r.race_start_utc}|${r.course}|${r.horse}|${r.route}`));
   if(added.length)new Notification('Racing shortlist updated',{body:added.map(r=>`${r.horse} — ${r.route}`).join('\n'),tag:'racing-shortlist'});
  }
  previous=current;
 };
 RacingStore.subscribe(update);RacingStore.ready.then(update).catch(()=>{});
})();
