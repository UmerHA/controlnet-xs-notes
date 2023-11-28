#!/bin/bash

rm -rf diffusers
git clone https://github.com/UmerHA/diffusers.git --branch controlnet-xs -qq
echo "Cloned cnxs branch"

echo "Installing diffusers..."
pip install -e "diffusers[test]" -qq
echo "Installed diffusers!"

echo "Installing transformers..."
pip install -Uqq transformers
echo "Installing transformers!"

cd diffusers

make test
