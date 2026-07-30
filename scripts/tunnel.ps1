# Starts a local dev server + a Cloudflare quick tunnel, prints the public
# URL, and stops both when you Ctrl+C this script. Reusable across projects —
# only -Command/-Port are project-specific, everything else is generic.
#
# Usage (this project):
#   powershell -File scripts/tunnel.ps1
#   powershell -File scripts/tunnel.ps1 -TimeoutMinutes 60
#
# Usage (any other project — override the start command and port):
#   powershell -File tunnel.ps1 -Command "npm run dev" -Port 3000
#   powershell -File tunnel.ps1 -Command "python manage.py runserver 8000" -Port 8000

param(
    [string]$Command = ".venv\Scripts\python.exe -m uvicorn app.main:app --port 5000",  # command that starts your dev server
    [int]$Port = 5000,                                       # port that command listens on
    [string]$WorkingDirectory = (Split-Path -Parent $PSScriptRoot),
    [int]$TimeoutMinutes = 0,   # 0 = run until Ctrl+C
    [string]$Cloudflared = "$env:USERPROFILE\cloudflared.exe"
)

$ErrorActionPreference = "Stop"
$tunnelLog = Join-Path $env:TEMP "cloudflared_tunnel.log"

if (-not (Test-Path $Cloudflared)) {
    Write-Host "cloudflared.exe not found at $Cloudflared" -ForegroundColor Red
    Write-Host 'Download it with: Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "$env:USERPROFILE\cloudflared.exe"'
    exit 1
}

$parts = $Command -split " ", 2
Write-Host "Starting dev server: $Command" -ForegroundColor Cyan
$devServer = Start-Process -FilePath $parts[0] -ArgumentList $parts[1] `
    -WorkingDirectory $WorkingDirectory -PassThru -WindowStyle Hidden

Write-Host "Starting Cloudflare tunnel..." -ForegroundColor Cyan
Remove-Item $tunnelLog -ErrorAction SilentlyContinue
$tunnel = Start-Process -FilePath $Cloudflared -ArgumentList "tunnel --url http://localhost:$Port" `
    -RedirectStandardError $tunnelLog -PassThru -WindowStyle Hidden

try {
    $url = $null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $tunnelLog) {
            $match = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($match) { $url = $match.Matches[0].Value; break }
        }
    }

    if (-not $url) {
        Write-Host "Tunnel URL did not appear in time - check $tunnelLog" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Live at: $url" -ForegroundColor Green
    if ($TimeoutMinutes -gt 0) {
        Write-Host "Will auto-stop in $TimeoutMinutes minute(s). Ctrl+C to stop sooner."
        Start-Sleep -Seconds ($TimeoutMinutes * 60)
    } else {
        Write-Host "Ctrl+C to stop."
        while ($true) { Start-Sleep -Seconds 5 }
    }
}
finally {
    Write-Host "Stopping tunnel and dev server..." -ForegroundColor Cyan
    # dev servers with a reloader/watcher (Flask debug, nodemon, etc.) fork a
    # child that outlives Stop-Process on the parent alone; /T kills the tree.
    taskkill /PID $tunnel.Id /T /F 2>$null | Out-Null
    taskkill /PID $devServer.Id /T /F 2>$null | Out-Null
}
