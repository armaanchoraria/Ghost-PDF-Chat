#!/bin/zsh
# Run this script from the project folder (or point an Automator action at it).
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
exec python3 -m streamlit run app.py
