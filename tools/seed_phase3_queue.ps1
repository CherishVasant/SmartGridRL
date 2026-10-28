$root = 'D:\DEV\Projects\1_ResumeWorthy\SmartGridRL'; Set-Location $root
$py = "$root\venv\Scripts\python.exe"
$jobs = @(@('M152_rl_mpc_tuned_h30',100,'M152'), @('M152_rl_mpc_tuned_h30',2026,'M152'), @('M155_rl_mpc_tuned_h36_sac',100,'M155'), @('M155_rl_mpc_tuned_h36_sac',2026,'M155'))
$left = [System.Collections.ArrayList]@($jobs)
while ($left.Count -gt 0) {
    foreach ($j in @($left)) {
        if (Test-Path "$root\results\$($j[0])\seed_$($j[1])\summary.json") {
            $name = "P3_RLTUNED_$($j[2])_s$($j[1])"
            $p = Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$py`" tools\phase3_eval.py $name rltuned $($j[0]) $($j[1]) > results\logs\p3_$name.log 2>&1" -WindowStyle Hidden -PassThru
            Start-Sleep 10
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $name } | ForEach-Object { (Get-Process -Id $_.ProcessId).PriorityClass = 'BelowNormal' }
            Add-Content "$root\results\logs\seed_phase3_queue.log" "$(Get-Date -Format 'HH:mm') started $name"
            $left.Remove($j)
        }
    }
    Start-Sleep 60
}
