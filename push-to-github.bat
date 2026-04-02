@echo off
REM Batch Script to Push Crypto Forecasting Project to GitHub
REM Usage: push-to-github.bat <GitHub_URL>
REM Example: push-to-github.bat https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git

setlocal enabledelayedexpansion

cls
echo.
echo === Git Push to GitHub Script ===
echo Project: Crypto Forecasting Model
echo.

if "%1"=="" (
    echo Usage: push-to-github.bat ^<GitHub_URL^>
    echo Example: push-to-github.bat https://github.com/your-username/Crypto-Forecasting.git
    echo.
    pause
    exit /b 1
)

set "GITHUB_URL=%1"

REM Step 1: Initialize git repository if not already done
echo [1] Checking git repository...
if not exist ".git" (
    echo Initializing git repository...
    call git init
    echo. OK: Git repository initialized
) else (
    echo. OK: Git repository already exists
)

REM Step 2: Add .gitignore and all files
echo.
echo [2] Adding files to git...
call git add .
echo. OK: Files staged for commit

REM Step 3: Show what will be committed
echo.
echo [3] Files to be committed:
call git diff --cached --name-status

REM Step 4: Create initial commit
echo.
echo [4] Creating commit...
call git commit -m "Initial commit: Crypto Forecasting Model with ANFIS, LSTM, and ANN"
echo. OK: Commit created

REM Step 5: Add remote origin
echo.
echo [5] Adding remote origin...
call git remote | find "origin" >nul
if !errorlevel! equ 0 (
    echo Remote origin already exists. Updating...
    call git remote remove origin
)
call git remote add origin %GITHUB_URL%
echo. OK: Remote origin configured: %GITHUB_URL%

REM Step 6: Set default branch to main and push
echo.
echo [6] Pushing to GitHub...
call git branch -M main
call git push -u origin main --force

if !errorlevel! equ 0 (
    echo.
    echo. OK: Successfully pushed to GitHub!
    echo Repository: %GITHUB_URL%
    echo.
    echo Your project is now on GitHub!
    echo.
) else (
    echo.
    echo ERROR: Push failed. Please check:
    echo   - GitHub URL is correct
    echo   - You have Git configured with GitHub credentials
    echo   - Your GitHub token/SSH key is set up
    echo.
)

pause
endlocal
