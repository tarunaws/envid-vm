#!/usr/bin/env bash
set -euo pipefail

URL="https://download.openmmlab.com/mmcv/dist/cpu/torch2.2.0/index.html"

echo "Probing mmcv wheel from: $URL"

docker run --rm --platform linux/amd64 python:3.10-slim-bookworm bash -lc "
set -euo pipefail
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 >/dev/null
rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir -q -f ${URL} mmcv==2.2.0

python - <<'PY'
import importlib.util
import pkgutil
import mmcv

print('mmcv_version', getattr(mmcv, '__version__', None))

spec_mmcv_ext = importlib.util.find_spec('mmcv._ext')
spec_mmcv_ops_ext = importlib.util.find_spec('mmcv.ops._ext')
print('spec_mmcv__ext', spec_mmcv_ext is not None)
print('spec_mmcv_ops__ext', spec_mmcv_ops_ext is not None)

mods = [m.name for m in pkgutil.iter_modules(mmcv.__path__) if 'ext' in m.name.lower()]
print('submods_with_ext', mods)

try:
    import mmcv.ops  # noqa
    print('import_mmcv_ops', True)
except Exception as exc:
    print('import_mmcv_ops', False, repr(exc))
PY
"
