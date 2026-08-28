"""Pack the Emscripten runtime and LWM models into one offline HTML file."""
from __future__ import annotations
import argparse, base64, json
from pathlib import Path
def enc(path: Path) -> str: return base64.b64encode(path.read_bytes()).decode("ascii")
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--template",type=Path,required=True); p.add_argument("--runtime",type=Path,required=True); p.add_argument("--det",type=Path,required=True); p.add_argument("--cls",type=Path,required=True); p.add_argument("--rec",type=Path,required=True); p.add_argument("--dictionary",type=Path,required=True); p.add_argument("--sponsor",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    values={"__LW_DET_MODEL_BASE64__":enc(a.det),"__LW_CLS_MODEL_BASE64__":enc(a.cls),"__LW_REC_MODEL_BASE64__":enc(a.rec),"__LW_DICTIONARY_BASE64__":enc(a.dictionary),"__LW_SPONSOR_IMAGE_BASE64__":enc(a.sponsor),"__LW_RUNTIME_JS_JSON__":json.dumps(a.runtime.read_text(encoding="utf-8"))}
    out=a.template.read_text(encoding="utf-8")
    for k,v in values.items(): out=out.replace(k,v)
    if "__LW_" in out: raise SystemExit("unresolved HTML placeholder")
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(out,encoding="utf-8",newline="\n"); print(f"wrote {a.output} ({a.output.stat().st_size} bytes)"); return 0
if __name__=="__main__": raise SystemExit(main())
