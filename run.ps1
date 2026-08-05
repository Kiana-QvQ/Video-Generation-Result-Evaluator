param(
    [switch]$Public,
    [switch]$WithGrpc,
    [switch]$WithVlm,
    [switch]$WithoutVlm,
    [ValidateSet("2b", "2.5-3b")]
    [string]$VlmModel = "2b",
    [ValidateSet("local", "docker")]
    [string]$VlmBackend = "local",
    [string]$BindHost,
    [int]$Port = 7860,
    [int]$GrpcPort = 50051,
    [string]$ApiKey,
    [string]$TlsCertfile,
    [string]$TlsKeyfile,
    [switch]$AllowInsecurePublic
)

$ErrorActionPreference = "Stop"

if ($WithVlm -and $WithoutVlm) {
    throw "Use either -WithVlm or -WithoutVlm, not both."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}

$hostAddress = if ($BindHost) {
    $BindHost
} elseif ($Public) {
    "0.0.0.0"
} else {
    "127.0.0.1"
}

if ($ApiKey) {
    $env:FRAME_AUDIT_API_KEY = $ApiKey
}
if ($TlsCertfile) {
    $env:EVALUATOR_TLS_CERTFILE = $TlsCertfile
}
if ($TlsKeyfile) {
    $env:EVALUATOR_TLS_KEYFILE = $TlsKeyfile
}
if ($Public) {
    if (-not $env:FRAME_AUDIT_API_KEY) {
        throw "Public binding requires -ApiKey or FRAME_AUDIT_API_KEY."
    }
    if ((-not $TlsCertfile -or -not $TlsKeyfile) -and -not $AllowInsecurePublic) {
        throw "Public binding requires -TlsCertfile/-TlsKeyfile or -AllowInsecurePublic."
    }
    $env:FRAME_AUDIT_REQUIRE_AUTH = "1"
}
if ($AllowInsecurePublic) {
    $env:EVALUATOR_ALLOW_INSECURE_PUBLIC = "1"
}

$env:PYTHONNOUSERSITE = "1"
$env:EVALUATOR_FACE_DEVICE = "auto"
$env:EVALUATOR_IQA_DEVICE = "auto"
$env:EVALUATOR_SEMANTIC_DEVICE = "auto"
$env:EVALUATOR_HOST = $hostAddress
$env:EVALUATOR_PORT = [string]$Port

Set-Location $root
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $pids = ($listeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "Port $Port is already in use by PID $pids. Stop the existing service before starting another instance."
}
$startArguments = @()
if ($WithGrpc) {
    $startArguments += "--with-grpc"
    $startArguments += "--grpc-port"
    $startArguments += [string]$GrpcPort
    Write-Host "Starting HTTP on ${hostAddress}:${Port} and gRPC on ${hostAddress}:${GrpcPort}"
} else {
    Write-Host "Starting HTTP on http://${hostAddress}:${Port}"
}
if ($TlsCertfile) {
    $startArguments += "--tls-certfile"
    $startArguments += $TlsCertfile
}
if ($TlsKeyfile) {
    $startArguments += "--tls-keyfile"
    $startArguments += $TlsKeyfile
}
if ($WithVlm) {
    $startArguments += "--with-vlm"
    $startArguments += "--vlm-model"
    $startArguments += $VlmModel
    $startArguments += "--vlm-backend"
    $startArguments += $VlmBackend
} elseif ($WithoutVlm) {
    $startArguments += "--without-vlm"
}
& $python (Join-Path $root "start.py") @startArguments
exit $LASTEXITCODE
