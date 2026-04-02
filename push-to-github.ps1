# PowerShell Script to Push Crypto Forecasting Project to GitHub
# Usage: .\push-to-github.ps1 "<GitHub_URL>"
# Example: .\push-to-github.ps1 "https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git"

param(
    [string]$GitHubURL = ""
)

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"

Write-Host "`n=== Git Push to GitHub Script ===" -ForegroundColor $Yellow
Write-Host "Project: Crypto Forecasting Model`n" -ForegroundColor $Yellow

# Check if GitHub URL is provided
if ([string]::IsNullOrEmpty($GitHubURL)) {
    Write-Host "Usage: .\push-to-github.ps1 `"<GitHub_URL>`"" -ForegroundColor $Red
    Write-Host "Example: .\push-to-github.ps1 `"https://github.com/your-username/Crypto-Forecasting.git`"`n" -ForegroundColor $Red
    exit 1
}

# Step 1: Initialize git repository if not already done
Write-Host "[1] Checking git repository..." -ForegroundColor $Yellow
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor $Yellow
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor $Green
} else {
    Write-Host "✓ Git repository already exists" -ForegroundColor $Green
}

# Step 2: Add .gitignore and all files
Write-Host "`n[2] Adding files to git..." -ForegroundColor $Yellow
git add .
Write-Host "✓ Files staged for commit" -ForegroundColor $Green

# Step 3: Show what will be committed
Write-Host "`n[3] Files to be committed:" -ForegroundColor $Yellow
git diff --cached --name-status | ForEach-Object { Write-Host "  $_" }

# Step 4: Create initial commit
Write-Host "`n[4] Creating commit..." -ForegroundColor $Yellow
$CommitMessage = "Initial commit: Crypto Forecasting Model with ANFIS, LSTM, and ANN"
git commit -m $CommitMessage
Write-Host "✓ Commit created" -ForegroundColor $Green

# Step 5: Add remote origin
Write-Host "`n[5] Adding remote origin..." -ForegroundColor $Yellow
$RemoteExists = git remote | Select-String origin
if ($RemoteExists) {
    Write-Host "Remote origin already exists. Updating..." -ForegroundColor $Yellow
    git remote remove origin
}
git remote add origin $GitHubURL
Write-Host "✓ Remote origin configured: $GitHubURL" -ForegroundColor $Green

# Step 6: Set default branch to main and push
Write-Host "`n[6] Pushing to GitHub..." -ForegroundColor $Yellow
git branch -M main
git push -u origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Successfully pushed to GitHub!" -ForegroundColor $Green
    Write-Host "Repository: $GitHubURL" -ForegroundColor $Green
    Write-Host "`nYour project is now on GitHub!`n" -ForegroundColor $Green
} else {
    Write-Host "`n✗ Push failed. Please check:" -ForegroundColor $Red
    Write-Host "  - GitHub URL is correct" -ForegroundColor $Red
    Write-Host "  - You have Git configured with GitHub credentials" -ForegroundColor $Red
    Write-Host "  - Your GitHub token/SSH key is set up`n" -ForegroundColor $Red
}
