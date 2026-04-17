@echo off

REM Start frontend
start cmd /k "cd /d D:\Project\web\frontend && npm run dev"

REM Start backend
start cmd /k "cd /d D:\Project\web\backend && D:\Project\venv\Scripts\activate && python -m uvicorn main:app --reload"

exit