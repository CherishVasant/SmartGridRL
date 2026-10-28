# Stops per-seed notebooks (<method>__s<seed>) right after their seed writes its next checkpoint (or finishes).
# Each one saves into results/<method>/seed_<seed>/, which stop_after_checkpoint.ps1 does not watch.
# Usage: powershell -File stop_seed_after_checkpoint.ps1 -Notebooks "M131_rl_tuned_gbm_mpc__s100,M135_rl_gbm_mpc_sac__s2026"
param([string]$Notebooks)
$results = Join-Path (Split-Path $PSScriptRoot -Parent) 'results'
$start = Get-Date
$pending = $Notebooks -split ','
while ($pending.Count -gt 0) {
    foreach ($nb in @($pending)) {
        $method, $seed = $nb -split '__s'
        $procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match [regex]::Escape("run_notebook.py $nb") }
        if (-not $procs) { "$(Get-Date -Format HH:mm) $nb was not running"; $pending = @($pending | Where-Object { $_ -ne $nb }); continue }
        $fresh = Get-ChildItem "$results\$method\seed_$seed\checkpoint.json", "$results\$method\seed_$seed\summary.json" -ErrorAction SilentlyContinue |
                 Where-Object { $_.LastWriteTime -gt $start } | Select-Object -First 1
        if ($fresh) {
            Start-Sleep -Seconds 2   # let the rename finish
            $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            "$(Get-Date -Format HH:mm) STOPPED $nb after $($fresh.Name) (seed_$seed)"
            $pending = @($pending | Where-Object { $_ -ne $nb })
        }
    }
    if ($pending.Count -gt 0) { Start-Sleep -Seconds 5 }
}
"$(Get-Date -Format HH:mm) ALL STOPPED"
