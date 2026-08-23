$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Pipeline = Join-Path $Root "scripts\web_forensics\run_web_forensics_v2.py"

Push-Location $Root
try {
    Write-Host "[Web v4.4] Explicit 85% expression + 15% face-crop." -ForegroundColor Cyan
    & $Python $Pipeline all `
      --dataset-root data/test/web_forensics_v44 `
      --split-manifest outputs/vedio_pred/wangxing_dual_pt_split_res1k.json `
      --profile-exclusion data/forensics/web_forensics_v44_profile_exclusion.json `
      --base-profile-exclusion data/forensics/web_forensics_v2_profile_exclusion.json `
      --forensics-profile outputs/forensics/forensics_profiles_web_v44_test_excluded.json `
      --source-profile outputs/forensics/wangxing_source_profile_web_v44_test_excluded.json `
      --calibrator outputs/forensics/forensics_authenticity_calibrator_web_v44.json `
      --fusion-head outputs/forensics/web_forensics_fusion_v44_monotonic.json `
      --feature-cache outputs/forensics/web_forensics_v44_monotonic_feature_cache.npz `
      --weighted-policy outputs/forensics/wangxing_authenticity_weighted_policy_v44.json `
      --ranking-root C:\Users\zhanghaotian\Desktop\ppt_video `
      --ranking-cache-root outputs/forensics/ppt_video_wangxing_policy_v44_cache `
      --ranking-output-root outputs/forensics/ppt_video_wangxing_policy_v44 `
      --test-set "25+25" data/test/web_forensics_v44/single_video/manifest_25x25.json `
      --test-set "32+32" data/test/wangxing_32x32/single_video/manifest.json `
      --expression-only `
      --face-crop-features `
      --monotonic-crop `
      --transition-features `
      --device cuda `
      --wangxing-device cuda `
      --profile-max-videos 120 `
      --output-root outputs/forensics/web_forensics_v44_results
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
