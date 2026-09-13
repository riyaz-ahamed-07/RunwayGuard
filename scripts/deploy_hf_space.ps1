# Assemble and upload the Gradio Space (run from RunwayGuard root).
# Requires: hf auth login  (account that can create Spaces)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Stage = Join-Path $Root "deploy\hf_space_build"
$User = if ($env:HF_USER) { $env:HF_USER } else { "DarkKnight1217" }
$Space = "$User/RunwayGuard-demo"

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

Copy-Item (Join-Path $Root "deploy\hf_space\README.md") $Stage -Force
Copy-Item (Join-Path $Root "deploy\hf_space\requirements.txt") $Stage -Force
Copy-Item (Join-Path $Root "deploy\hf_space\app.py") $Stage -Force
Copy-Item (Join-Path $Root "app") (Join-Path $Stage "app") -Recurse -Force
Copy-Item (Join-Path $Root "config") (Join-Path $Stage "config") -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "deploy") | Out-Null
Copy-Item (Join-Path $Root "deploy\gradio_app.py") (Join-Path $Stage "deploy\gradio_app.py") -Force
Copy-Item (Join-Path $Root "deploy\samples") (Join-Path $Stage "deploy\samples") -Recurse -Force
New-Item -ItemType File -Force -Path (Join-Path $Stage "deploy\__init__.py") | Out-Null
New-Item -ItemType File -Force -Path (Join-Path $Stage "app\__init__.py") -ErrorAction SilentlyContinue | Out-Null

Write-Host "Staging at $Stage"
hf repos create "$Space" --type space --space-sdk gradio --exist-ok
hf upload $Space $Stage . --repo-type=space --commit-message "Deploy RunwayGuard reviewer UI"
Write-Host "Space: https://huggingface.co/spaces/$Space"
