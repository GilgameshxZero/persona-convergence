```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade -r requirements/pre.txt
pip install --upgrade -r requirements/gpu.txt

# git lfs install
# cd models
# git submodule add https://git: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@hf.co/gpt2
# cd datasets
# git submodule add https://git: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@hf.co/roneneldan/TinyStories

git submodule init
git submodule update
```
