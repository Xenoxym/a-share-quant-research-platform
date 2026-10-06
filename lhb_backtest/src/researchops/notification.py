"""Best-effort local Windows alert after a durable important-result stop.

No chat, email or third-party messaging. Tests do not enable this callback.
"""
import base64
import os
from pathlib import Path
import subprocess

from ..technical.artifacts import utc_now, write_json


def desktop_notice(folder, notice):
    folder = Path(folder)
    marker = folder / "notification_attempt.json"
    if marker.exists():
        return dict(status="already_attempted", marker=str(marker))
    if os.name != "nt":
        return dict(status="unsupported_platform")
    # Never interpolate model-generated text into an executable shell script.
    # Detailed findings belong in the pinned report; this is a fixed local alert.
    title = "量化研究发现重要成果或问题"
    script = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$researchNotice = New-Object System.Windows.Forms.NotifyIcon
try {
    $researchNotice.Icon = [System.Drawing.SystemIcons]::Information
    $researchNotice.Visible = $true
    $researchNotice.BalloonTipTitle = '量化研究发现重要成果或问题'
    $researchNotice.BalloonTipText = '自动循环已停止。请查看研究状态页中的重要发现报告；明确要求继续后才会接续。'
    $researchNotice.ShowBalloonTip(12000)
    $researchNoticeDeadline = [DateTime]::UtcNow.AddSeconds(20)
    while ([DateTime]::UtcNow -lt $researchNoticeDeadline) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 100
    }
} finally { $researchNotice.Dispose() }
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    write_json(marker, dict(at=utc_now(), status="attempting", title=title,
                           report=str(folder / "REPORT.md"), delivery_confirmed=False))
    try:
        with (folder / "notification.log").open("ab") as log:
            process = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        result = dict(at=utc_now(), status="process_started", pid=process.pid, delivery_confirmed=False)
        write_json(marker, result)
        return result
    except Exception as exc:
        write_json(marker, dict(at=utc_now(), status="failed", error=str(exc), delivery_confirmed=False))
        raise
