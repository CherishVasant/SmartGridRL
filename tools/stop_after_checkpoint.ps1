# Stops each running notebook right after it writes its next checkpoint (or finishes a seed).
# checkpoint.json is written last in save_checkpoint, so once it changes the checkpoint is complete.
# Usage: powershell -File stop_after_checkpoint.ps1 M7_multikpi_adaptive_norm,M8_multikpi_fixed_norm
param([string]$Methods = 'M7_multikpi_adaptive_norm,M8_multikpi_fixed_norm,M10_per_building_dynamic_district,M11_per_building_adaptive_district')
$results = Join-Path (Split-Path $PSScriptRoot -Parent) 'results'
$start = Get-Date
$pending = $Methods -split ','
while ($pending.Count -gt 0) {
    foreach ($m in @($pending)) {
        $procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match "run_notebook.py $m" }
        if (-not $procs) { "$(Get-Date -Format HH:mm) $m was not running"; $pending = @($pending | Where-Object { $_ -ne $m }); continue }
        # checkpoint.json (RL), bc_state.json (behaviour cloning, after each DAgger iteration) or summary.json (seed done)
        $fresh = Get-ChildItem "$results\$m\seed_*\checkpoint.json", "$results\$m\seed_*\bc_state.json", "$results\$m\seed_*\summary.json" -ErrorAction SilentlyContinue |
                 Where-Object { $_.LastWriteTime -gt $start } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($fresh) {
            Start-Sleep -Seconds 2   # let the rename finish
            $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            "$(Get-Date -Format HH:mm) STOPPED $m after $($fresh.Name) ($($fresh.Directory.Name))"
            $pending = @($pending | Where-Object { $_ -ne $m })
        }
    }
    if ($pending.Count -gt 0) { Start-Sleep -Seconds 5 }
}
"$(Get-Date -Format HH:mm) ALL STOPPED"
