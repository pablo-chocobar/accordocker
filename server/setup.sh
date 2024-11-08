#!/bin/bash

# Exit script on any error
set -e

# Run Ollama's install script
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Install Flask and Flask-Cors ignoring any previously installed versions
echo "Installing Flask and Flask-Cors..."
pip install --ignore-installed Flask Flask_Cors

# Install requirements from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r "requirements.txt"
else
    echo "requirements.txt file not found!"
    exit 1
fi

echo "Installation completed successfully."
