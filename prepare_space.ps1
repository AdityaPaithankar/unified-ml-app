
param(
    [string]$Source = "E:\Projects\Final Models for internship",
    [string]$Target = "E:\Projects\Final Models for internship\hf-space"
)

if (-not (Test-Path (Join-Path $Target ".git"))) {
    Write-Error "Target is not a cloned Git repository: $Target"
    exit 1
}

robocopy $Source $Target /E `
    /XD ".git" "hf-space" "__pycache__" ".pytest_cache" "outputs" "_backup_before_final" "tests" ".venv" ".venv-app311" "wav2vec2_checkpoints" ".cache" `
    /XF "*.pyc" "audit_models.py" "audit_keras3.py" "requirements-keras3.txt" "requirements-ui.txt" |
    Out-Null

if ($LASTEXITCODE -ge 8) {
    Write-Error "Robocopy failed with exit code $LASTEXITCODE"
    exit 1
}

$readme = Join-Path $Target "README.md"
$body = Get-Content $readme -Raw

if ($body -notmatch "(?s)^---\s") {
    $header = @"
---
title: Unified ML Internship App
sdk: docker
app_port: 7860
---

"@
    Set-Content -Path $readme -Value ($header + $body) -NoNewline
}

Write-Host "Space folder ready at $Target"
