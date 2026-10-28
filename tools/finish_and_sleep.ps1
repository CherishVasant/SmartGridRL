$log = 'D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\results\logs\finish_and_sleep.log'
Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') waiting for all jobs to finish"
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_notebook|load_lab|forecast_value|phase3_eval|tune_forecaster|stop_after_checkpoint' })
    $busy += @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'stop_after_checkpoint\.ps1' -and $_.Name -eq 'powershell.exe' })
    if ($busy.Count -eq 0) { break }
    Start-Sleep 60
}
Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') all jobs finished; stopping helpers and going to sleep"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'keep_awake|battery_guard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep 10
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false) | Out-Null
