$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    dotnet run --no-restore -c Release --project "$PSScriptRoot\.NET\RetrievalAugmentedGeneration\RetrievalAugmentedGeneration.csproj"
    if ($LASTEXITCODE -ne 0) { throw '.NET-Anwendung fehlgeschlagen.' }
} finally { Pop-Location }
