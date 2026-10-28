$root = 'D:\DEV\Projects\1_ResumeWorthy\SmartGridRL'; Set-Location $root
$py = "$root\venv\Scripts\python.exe"
$jobs = @(@('P3_GBMTUNED_MPC_g3 mpcfc gbm_tuned 24 1 3','p3_gbmtuned_mpc'), @('P3_RLTUNED_M144 rltuned M144_rl_mpc_tuned_h36 42','p3_M144'), @('P3_RLTUNED_M154 rltuned M154_rl_mpc_tuned_h36_grid15 42','p3_M154'), @('P3_RLTUNED_M155 rltuned M155_rl_mpc_tuned_h36_sac 42','p3_M155'))
foreach ($j in $jobs) {
    while (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'phase3_eval' }).Count -ge 3 -or (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB -lt 2.5) { Start-Sleep 60 }
    Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$py`" tools\phase3_eval.py $($j[0]) > results\logs\$($j[1]).log 2>&1" -WindowStyle Hidden
    Start-Sleep 120
}

