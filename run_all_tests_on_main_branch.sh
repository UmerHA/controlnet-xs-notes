#!/bin/bash

rm -rf diffusers
git clone https://github.com/UmerHA/diffusers.git -qq
echo "Cloned main branch"

echo "Installing diffusers..."
pip install -e "diffusers[test]" -qq
echo "Installed diffusers!"

echo "Installing transformers..."
pip install -Uqq transformers
echo "Installing transformers!"

cd diffusers

make test
