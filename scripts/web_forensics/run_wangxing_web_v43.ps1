$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\web_forensics\run_web_forensics_v2.py"

Push-Location $ProjectRoot
try {
    Write-Host "[Web v4.3] Expression + face-crop candidate; current web unchanged." -ForegroundColor Cyan
    & $Python $Pipeline all `
      --dataset-root data/test/web_forensics_v43 `
      --split-manifest outputs/vedio_pred/wangxing_dual_pt_split_res1k.json `
      --profile-exclusion data/forensics/web_forensics_v43_profile_exclusion.json `
      --base-profile-exclusion data/forensics/web_forensics_v2_profile_exclusion.json `
      --forensics-profile outputs/forensics/forensics_profiles_web_v43_test_excluded.json `
      --source-profile outputs/forensics/wangxing_source_profile_web_v43_test_excluded.json `
      --calibrator outputs/forensics/forensics_authenticity_calibrator_web_v43.json `
      --fusion-head outputs/forensics/web_forensics_fusion_v43_face_crop.json `
      --feature-cache outputs/forensics/web_forensics_v43_face_crop_feature_cache.npz `
      --weighted-policy outputs/forensics/wangxing_authenticity_weighted_policy_v43.json `
      --ranking-root C:\Users\zhanghaotian\Desktop\ppt_video `
      --ranking-cache-root outputs/forensics/ppt_video_wangxing_policy_v43_cache `
      --ranking-output-root outputs/forensics/ppt_video_wangxing_policy_v43 `
      --test-set "25+25" data/test/web_forensics_v43/single_video/manifest_25x25.json `
      --test-set "32+32" data/test/wangxing_32x32/single_video/manifest.json `
      --expression-only `
      --face-crop-features `
      --transition-features `
      --device cuda `
      --wangxing-device cuda `
      --profile-max-videos 120 `
      --output-root outputs/forensics/web_forensics_v43_results
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
