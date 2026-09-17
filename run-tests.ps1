[CmdletBinding()]
param()

$showDetails = $VerbosePreference -eq 'Continue'

# Keep successful builds quiet; retain diagnostics if the build fails.
$buildLog = [System.Collections.Generic.List[string]]::new()
cmake --build "$PSScriptRoot/build" --config Release 2>&1 |
    ForEach-Object {
        $buildLog.Add("$_")
        if ($showDetails) { Write-Host "$_" }
    }
$buildExitCode = $LASTEXITCODE

if ($buildExitCode -ne 0) {
    if (-not $showDetails) {
        $buildLog | ForEach-Object { Write-Host $_ }
    }
    Write-Host 'Build failed. Tests were not run.' -ForegroundColor Red
    exit $buildExitCode
}

$ctestArgs = @('--test-dir', "$PSScriptRoot/build", '-C', 'Release')
if ($showDetails) {
    $ctestArgs += '-V'
} else {
    $ctestArgs += '--output-on-failure'
}

# Preserve the complete output so errors cannot be hidden by the short view.
$testLog = [System.Collections.Generic.List[string]]::new()
ctest @ctestArgs 2>&1 |
    ForEach-Object {
        $line = "$_"
        $testLog.Add($line)

        if ($line -match '^\s*\d+/\d+\s+Test\s+#\d+:') {
            if ($line -match '\bPassed\b') {
                $line = $line -replace '\bPassed\b', "$([char]0x2713) PASS"
                Write-Host $line -ForegroundColor Green
            } else {
                Write-Host $line -ForegroundColor Red
            }
        } elseif ($line -match '^100% tests passed') {
            Write-Host $line -ForegroundColor Green
        } elseif ($line -match '^Total Test time') {
            Write-Host $line -ForegroundColor Green
        } elseif ($showDetails) {
            Write-Host $line
        }
    }

$testExitCode = $LASTEXITCODE
if ($testExitCode -ne 0 -and -not $showDetails) {
    Write-Host 'CTest failed. Full diagnostics:' -ForegroundColor Red
    $testLog | ForEach-Object { Write-Host $_ }
}
exit $testExitCode
