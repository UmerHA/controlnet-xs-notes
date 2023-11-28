#!/bin/bash

rm -rf diffusers
git clone https://github.com/UmerHA/diffusers.git --branch controlnet-xs -qq
echo "Cloned cnxs branch"

echo "Installing diffusers..."
pip install -e "diffusers[test]" -qq
echo "Installed diffusers!"

echo "Installing transformers..."
pip install -Uqq transformers
echo "Installed transformers!"

cd diffusers

# Check for --slow flag
if [[ "$1" == "--slow" ]]; then
    echo "Also slow tests"
    RUN_SLOW=1 python -m pytest ./tests/pipelines/controlnetxs/test_controlnetxs_sdxl.py ./tests/pipelines/controlnetxs/test_controlnetxs.py
else
    python -m pytest -n auto --dist=loadfile -s -v ./tests/pipelines/controlnetxs/test_controlnetxs_sdxl.py
fi
