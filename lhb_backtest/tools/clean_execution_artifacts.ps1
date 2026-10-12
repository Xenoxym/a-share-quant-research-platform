param(
    [switch]$Apply,
    [string[]]$ExcludeNames = @()
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$appRoot = Join-Path $projectRoot 'lhb_backtest'
$reportRoot = Join-Path $appRoot 'research\execution_cleanup'
$started = [DateTime]::UtcNow.ToString('o')
$candidates = [System.Collections.Generic.List[System.IO.DirectoryInfo]]::new()
foreach ($base in @($projectRoot, $appRoot)) {
    foreach ($entry in (Get-ChildItem -LiteralPath $base -Directory -Force)) {
        if ($entry.Name -like '.qa_*' -or
            $entry.Name -in @('.pytest_cache', '.ruff_cache', '__pycache__') -or
            ($base -eq $appRoot -and ($entry.Name -eq 'build' -or $entry.Name -like '*.egg-info'))) {
            $candidates.Add($entry)
        }
    }
}
$delivery = Join-Path $projectRoot '.local_delivery'
if (Test-Path -LiteralPath $delivery) {
    foreach ($entry in (Get-ChildItem -LiteralPath $delivery -Directory -Force)) {
        if ($entry.Name -like '*html_profile*' -or $entry.Name -like 'wheels*') {
            $candidates.Add($entry)
        }
    }
}
$processes = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } |
    Select-Object -ExpandProperty CommandLine)
$rows = [System.Collections.Generic.List[object]]::new()
foreach ($target in $candidates) {
    $absolute = (Resolve-Path -LiteralPath $target.FullName).Path
    if (-not $absolute.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escaped project: $absolute"
    }
    $status = 'preview'
    $size = 0L
    $files = 0
    $reason = $null
    try {
        if ($target.Name -in $ExcludeNames) {
            $status = 'skipped'; $reason = 'explicit exclusion'
        } elseif ($target.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            $status = 'skipped'; $reason = 'linked directory'
        } elseif (@($processes | Where-Object {
            $_.IndexOf($absolute, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $_.IndexOf($target.Name, [StringComparison]::OrdinalIgnoreCase) -ge 0
        }).Count -gt 0) {
            $status = 'skipped'; $reason = 'mentioned by a live process'
        } else {
            $entries = @(Get-ChildItem -LiteralPath $absolute -Recurse -Force -ErrorAction Stop)
            if (@($entries | Where-Object {
                $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint
            }).Count -gt 0) {
                $status = 'skipped'; $reason = 'contains linked entries'
            } else {
                foreach ($file in $entries) {
                    if (-not $file.PSIsContainer) { $size += $file.Length; $files++ }
                }
                if ($Apply) {
                    # Native PowerShell, verified absolute target, no other shell.
                    Remove-Item -LiteralPath $absolute -Recurse -Force -ErrorAction Stop
                    $status = 'deleted'
                }
            }
        }
    } catch {
        $status = 'failed'; $reason = $_.Exception.Message
    }
    $rows.Add([ordered]@{ path=$absolute; status=$status; bytes=$size; files=$files; reason=$reason })
}
$deletedBytes = 0L
foreach ($row in $rows) {
    if ($row.status -eq 'deleted') { $deletedBytes += [long]$row.bytes }
}
$result = [ordered]@{
    schema='execution-cleanup-v1'; project=$projectRoot; apply=[bool]$Apply
    started_at_utc=$started; ended_at_utc=[DateTime]::UtcNow.ToString('o')
    candidates=$rows.Count; deleted=@($rows | Where-Object status -eq 'deleted').Count
    deleted_bytes=$deletedBytes
    skipped=@($rows | Where-Object status -eq 'skipped').Count
    failures=@($rows | Where-Object status -eq 'failed').Count; entries=@($rows)
}
if ($Apply) {
    New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
    $name = 'cleanup-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff') + '.json'
    $reportPath = Join-Path $reportRoot $name
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8
    $result.Remove('entries')
    $result['report'] = $reportPath
}
$result | ConvertTo-Json -Depth 6 -Compress
if ($result.failures -gt 0) { exit 1 }
