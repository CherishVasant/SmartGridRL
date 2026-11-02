# Paper: Phase III scoring of the extra seeds (Table II rows at 3 seeds, the proposed method at 5).
# Skips seeds already in results\phase3\scores.csv; at most $MaxJobs scoring jobs at once, at normal priority
# (Windows starved BelowNormal jobs for hours, and High priority on every job froze the laptop).
param([int]$MaxJobs = 2)
$root = 'D:\DEV\Projects\1_ResumeWorthy\SmartGridRL'; Set-Location $root
$py = "$root\venv\Scripts\python.exe"
$jobs = @(@('M129_rl_tuned_mpc',100,'M129'), @('M129_rl_tuned_mpc',2026,'M129'),
          @('M144_rl_mpc_tuned_h36',100,'M144'), @('M144_rl_mpc_tuned_h36',2026,'M144'),
          @('M139_rl_mpc_gbm_tuned',100,'M139'), @('M139_rl_mpc_gbm_tuned',2026,'M139'),
          @('M152_rl_mpc_tuned_h30',7,'M152'), @('M152_rl_mpc_tuned_h30',1234,'M152'))
$scores = "$root\results\phase3\scores.csv"
function Scored($name) { (Test-Path $scores) -and (Select-String -Path $scores -Pattern "^$name," -Quiet) }
function Running { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'phase3_eval\.py P3_RLTUNED_' }).Count }
$left = [System.Collections.ArrayList]@($jobs | Where-Object { -not (Scored "P3_RLTUNED_$($_[2])_s$($_[1])") })
while ($left.Count -gt 0) {
    foreach ($j in @($left)) {
        if ((Running) -ge 2 * $MaxJobs) { break }   # each job is a launcher plus its python child
        if (Test-Path "$root\results\$($j[0])\seed_$($j[1])\summary.json") {
            $name = "P3_RLTUNED_$($j[2])_s$($j[1])"
            Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$py`" tools\phase3_eval.py $name rltuned $($j[0]) $($j[1]) > results\logs\p3_$name.log 2>&1" -WindowStyle Hidden
            Start-Sleep 10
            Add-Content "$root\results\logs\seed_phase3_queue.log" "$(Get-Date -Format 'HH:mm') started $name"
            $left.Remove($j)
        }
    }
    Start-Sleep 60
}
