#!/usr/bin/env python3
"""Pin/download LFM3 and expose custom source files without Hub symlink issues."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi,snapshot_download

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/lfm3-evaluation'
OUT.mkdir(parents=True,exist_ok=True)
record=OUT/'model-revision.json'
MODEL_ID='LiquidAI/LFM3-1.7B'
REVISION='1da06bec6a819da10b54c67e5c90972b46d81871'
if record.exists():
    metadata=json.loads(record.read_text())
else:
    info=HfApi().model_info(MODEL_ID,revision=REVISION)
    metadata={'model_id':info.id,'revision':info.sha,'files':[s.rfilename for s in info.siblings]}
    record.write_text(json.dumps(metadata,indent=2)+'\n')
assert metadata['model_id']==MODEL_ID and metadata['revision']==REVISION
snapshot=Path(snapshot_download(metadata['model_id'],revision=metadata['revision'],
    allow_patterns=['*.json','*.jinja','*.py','*.safetensors','LICENSE','README.md'],max_workers=2))
directory=ROOT/'models/LFM3-1.7B'
if directory.is_symlink():directory.unlink()
directory.mkdir(exist_ok=True)
hashes={}
for source in snapshot.iterdir():
    destination=directory/source.name
    if source.suffix=='.py':
        # Transformers 5.12 hashes relative imports using resolved source paths.
        # Real Python files preserve their shared directory; weights stay linked.
        if destination.is_symlink():destination.unlink()
        shutil.copyfile(source,destination)
        hashes[source.name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    else:
        if destination.is_symlink() and not destination.exists():
            destination.unlink()
        if not destination.exists():
            destination.symlink_to(os.path.relpath(source.resolve(),directory.resolve()))
(OUT/'custom-code-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'model_id':metadata['model_id'],'revision':metadata['revision'],
                  'local_path':str(directory),'custom_code_sha256':hashes},indent=2))
