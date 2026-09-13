#!/usr/bin/env python3
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/"agent_os"; S=O/"state.json"; D=O/"runs"; P=O/"reports"
def now(): return datetime.now(timezone.utc).isoformat()
def load(): return json.loads(S.read_text()) if S.exists() else {"task_types":{}}
def classify(x):
    return "refused" if x.get("refusal") or x.get("stop_reason") in ("refusal","content_filter") else ("blocked" if x.get("blocked") else ("fail" if x.get("error") else "pass"))
def main():
    a=argparse.ArgumentParser(); a.add_argument("--response-json"); z=a.parse_args(); D.mkdir(parents=True,exist_ok=True); P.mkdir(parents=True,exist_ok=True)
    kind=os.getenv("CORTEX_OS_TASK_TYPE","health_check"); rid=os.getenv("CORTEX_OS_TASK_ID","daily-"+str(datetime.now(timezone.utc).date())); x=json.loads(Path(z.response_json).read_text()) if z.response_json else {}; result=classify(x) if z.response_json else "pass"
    st=load(); b=st["task_types"].setdefault(kind,{"runs":[]}); b["runs"].append({"at":now(),"id":rid,"outcome":result}); done=[q for q in b["runs"] if q["outcome"] in ("pass","fail")]; good=sum(q["outcome"]=="pass" for q in done); rate=good/len(done) if done else 0
    old=b.get("privilege","supervised"); priv="autonomous" if len(done)>=20 and rate>=.95 else ("supervised_revoked" if old=="autonomous" and rate<.90 else old); b.update(completed=len(done),success_rate=round(rate,4),privilege=priv); S.write_text(json.dumps(st,indent=2)+"\n")
    out={"at":now(),"task_type":kind,"task_id":rid,"outcome":result,"http_status_observed":x.get("http_status"),"grade":{"completed":len(done),"success_rate":rate,"privilege":priv}}; text=json.dumps(out,indent=2)+"\n"; (D/(rid+".json")).write_text(text); (P/"latest.json").write_text(text); print(text,end="")
if __name__=="__main__": main()
