@echo off
cd /d "%~dp0"
if not exist zerolm.exe (
  gcc -O2 -std=c11 -Wall -Wextra -o zerolm.exe main.c -lm
  if errorlevel 1 exit /b 1
)
zerolm.exe %*
