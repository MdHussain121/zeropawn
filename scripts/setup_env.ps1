# setup_env.ps1 — Prepend standard C++ compiler to PATH for current PowerShell session.
# Usage: . .\scripts\setup_env.ps1

$candidateDirs = @(
    if ($env:LOCALAPPDATA) {
        Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
    },
    "C:\mingw64\bin",
    "C:\msys64\mingw64\bin"
)

$found = $false
foreach ($dir in $candidateDirs) {
    if ($dir -and (Test-Path "$dir\g++.exe")) {
        $env:Path = "$dir;$env:Path"
        Write-Host "C++ compiler found and prepended to PATH: $dir"
        & "$dir\g++.exe" --version 2>&1 | Select-Object -First 1
        $found = $true
        break
    }
}

if (-not $found) {
    $existing = Get-Command g++.exe, clang++.exe, cl.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing) {
        Write-Host "Found existing compiler on PATH: $($existing.Source)"
    } else {
        Write-Warning "No C++ compiler found on PATH or standard directories."
        Write-Host "Tip: Install MinGW via winget: winget install BrechtSanders.WinLibs.POSIX.UCRT"
    }
}
