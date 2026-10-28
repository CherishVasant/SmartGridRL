# Battery guard: stops the runs gracefully before the laptop's battery runs out.
# Every minute: if the laptop is on battery and the charge is <= $GracefulAt %, it stops the supervisor (so nothing new
# starts or restarts), gracefully stops every training run right after its next checkpoint (stop_after_checkpoint.ps1 /
# stop_seed_after_checkpoint.ps1) and stops the jobs that have no checkpoints (forecaster tuning, Phase III scoring).
# If the charge then reaches <= $ForceAt %, whatever is still running is stopped at once (the last checkpoint stays).
# Usage: powershell -File tools\battery_guard.ps1 [-GracefulAt 30] [-ForceAt 12]   (log: results\logs\battery_guard.log)
param([int]$GracefulAt = 30, [int]$ForceAt = 12)
$root = Split-Path $PSScriptRoot -Parent
$log = Join-Path $root 'results\logs\battery_guard.log'
function Log($msg) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') $msg" }
function OnBattery {
    $s = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($s) { return -not $s.PowerOnline }
    $b = Get-CimInstance Win32_Battery | Select-Object -First 1
    return $b -and $b.BatteryStatus -ne 2
}
function Charge { (Get-CimInstance Win32_Battery | Measure-Object EstimatedChargeRemaining -Average).Average }
function RunNames {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
        if ($_.CommandLine -match 'run_notebook\.py (\S+?)(\.ipynb)?(\s|$)') { $matches[1] } } | Sort-Object -Unique
}
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [UInt16]0 }
function StartHidden($cmd) { Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $root; ProcessStartupInformation = $si } | Out-Null }

Log "battery guard started (graceful stop at <= $GracefulAt% on battery, immediate stop at <= $ForceAt%)"
$stage = 0
while ($true) {
    $onBattery, $charge = (OnBattery), (Charge)
    if ($stage -eq 0 -and $onBattery -and $charge -le $GracefulAt) {
        Log "on battery at $charge% -> graceful stop"
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'supervisor\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'tune_forecaster|phase3_eval|forecast_lab|mpc_eval|transfer_eval' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        $names = @(RunNames)
        $main = @($names | Where-Object { $_ -notmatch '__s\d+$' })
        $seeds = @($names | Where-Object { $_ -match '__s\d+$' })
        if ($main.Count) { StartHidden "cmd.exe /c powershell -NoProfile -ExecutionPolicy Bypass -File `"$root\tools\stop_after_checkpoint.ps1`" -Methods $($main -join ',') >> `"$root\results\logs\battery_stop.log`" 2>&1" }
        if ($seeds.Count) { StartHidden "cmd.exe /c powershell -NoProfile -ExecutionPolicy Bypass -File `"$root\tools\stop_seed_after_checkpoint.ps1`" -Notebooks $($seeds -join ',') >> `"$root\results\logs\battery_stop.log`" 2>&1" }
        Log "  graceful stop requested for: $($names -join ', ')"
        $stage = 1
    }
    if ($stage -ge 1 -and $charge -le $ForceAt -and $onBattery) {
        $left = @(RunNames)
        if ($left.Count) {
            Log "at $charge% -> stopping the remaining runs now: $($left -join ', ')"
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_notebook' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        }
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'stop_after_checkpoint|stop_seed_after_checkpoint|keep_awake' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Log 'all runs stopped; guard exits'
        break
    }
    if ($stage -ge 1 -and -not (RunNames)) {
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'keep_awake' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Log "all runs stopped gracefully at $charge%; guard exits"
        break
    }
    Start-Sleep -Seconds 60
}
