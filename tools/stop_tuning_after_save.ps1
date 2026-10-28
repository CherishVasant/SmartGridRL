$sum = 'D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\results\forecast\tuned\tuning_summary.csv'
$log = 'D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\results\logs\stop_all_20261022.log'
$start = (Get-Item $sum).LastWriteTime
while ($true) {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'tune_forecaster' }
    if (-not $p) { Add-Content $log "$(Get-Date -Format HH:mm) tuning already stopped"; break }
    if ((Get-Item $sum).LastWriteTime -gt $start) {
        Start-Sleep 5
        $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Add-Content $log "$(Get-Date -Format HH:mm) STOPPED forecaster tuning after saving its current model"
        break
    }
    Start-Sleep 20
}
