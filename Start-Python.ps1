$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\python\.venv\Scripts\python.exe" "$PSScriptRoot\python\run.py"
    if ($LASTEXITCODE -ne 0) { throw 'Python-Anwendung fehlgeschlagen.' }
} finally { Pop-Location }
