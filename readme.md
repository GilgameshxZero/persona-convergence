```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade -r requirements/pre.txt
pip install --upgrade -r requirements/gpu.txt

cd models
git submodule add git@hf.co:gpt2

git submodule init
git submodule update
```
