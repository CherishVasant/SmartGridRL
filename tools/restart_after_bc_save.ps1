# Restarts given notebooks onto the current code without losing work: waits until each one's seed folder writes a fresh
# bc_state.json (saved after every DAgger iteration), then stops that exact notebook's process; the supervisor relaunches
# it, and it resumes from the saved state. Matches "run_notebook.py <name>.ipynb" exactly (not the __s<seed> notebooks).
# Usage: powershell -File restart_after_bc_save.ps1 -Specs "M83_bc_3x512_o3_dagger15:42,M84_bc_3x512_o3_dagger20:42"
param([string]$Specs)
$results = Join-Path (Split-Path $PSScriptRoot -Parent) 'results'
$start = Get-Date
$pending = $Specs -split ','
while ($pending.Count -gt 0) {
    foreach ($spec in @($pending)) {
        $nb, $seed = $spec -split ':'
        $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match [regex]::Escape("run_notebook.py $nb.ipynb") }
        if (-not $procs) { "$(Get-Date -Format HH:mm) $nb not running"; $pending = @($pending | Where-Object { $_ -ne $spec }); continue }
        $state = Get-Item "$results\$nb\seed_$seed\bc_state.json" -ErrorAction SilentlyContinue
        if ($state -and $state.LastWriteTime -gt $start) {
            Start-Sleep -Seconds 2
            $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            "$(Get-Date -Format HH:mm) RESTARTED $nb after DAgger save (seed $seed)"
            $pending = @($pending | Where-Object { $_ -ne $spec })
        }
    }
    if ($pending.Count -gt 0) { Start-Sleep -Seconds 10 }
}
"$(Get-Date -Format HH:mm) ALL RESTARTED"
