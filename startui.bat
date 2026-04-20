@echo off
title Whisper AI Dashboard
cd /d "%~dp0"

echo Starte Whisper AI Dashboard...
echo ===============================

powershell -NoProfile -Command "Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like '*whisper_ui.py*' } | ForEach-Object { Write-Host ('Beende Instanz PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

powershell -NoProfile -WindowStyle Hidden -Command "Start-Process '.\venv\Scripts\python.exe' -ArgumentList 'whisper_ui.py' -WindowStyle Hidden"
timeout /t 3 /nobreak >nul
start http://localhost:8080
