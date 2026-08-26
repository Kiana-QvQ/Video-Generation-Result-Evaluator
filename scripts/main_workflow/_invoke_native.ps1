# Helpers for long-running workflow scripts on Windows PowerShell 5.x.
# Python/torch often writes FutureWarning to stderr; with $ErrorActionPreference
# = Stop that becomes a terminating error even when exit code is 0.

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [scriptblock]$OnLine
    )

    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Python @ArgumentList 2>&1
        $exit = $LASTEXITCODE
        foreach ($line in @($output)) {
            if ($null -eq $line) { continue }
            if ($OnLine) {
                & $OnLine $line
            }
            else {
                Write-Host $line
            }
        }
    }
    finally {
        $ErrorActionPreference = $previousEap
    }
    if ($exit -ne 0) {
        throw "${Stage} failed with exit code ${exit}"
    }
}

function Invoke-ScriptChecked {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$Stage,
        [hashtable]$BoundParameters = @{},
        [string]$LogPath = $null
    )

    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($LogPath) {
            & $ScriptPath @BoundParameters *>> $LogPath
        }
        else {
            & $ScriptPath @BoundParameters
        }
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousEap
    }
    if ($exit -ne 0) {
        throw "${Stage} failed with exit code ${exit}"
    }
}
