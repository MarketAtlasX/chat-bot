@echo off
set HF_HOME=%~dp0.cache\huggingface
set TRANSFORMERS_CACHE=%~dp0.cache\huggingface\transformers
set TORCH_HOME=%~dp0.cache\torch
set SENTENCE_TRANSFORMERS_HOME=%~dp0.cache\sentence_transformers
set PIP_CACHE_DIR=%~dp0.cache\pip
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    call .venv\Scripts\Activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\Activate.bat
)
python main.py
