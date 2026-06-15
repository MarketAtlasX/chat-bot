@echo off
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    call .venv\Scripts\Activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\Activate.bat
)
python main.py
