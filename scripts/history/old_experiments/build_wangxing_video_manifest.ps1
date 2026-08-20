param(
    [string]$AccessRoot = '\\192.168.22.84\minio\4Real\Project\4D\WangXing',
    [string]$CanonicalRoot = '\\smb.ceres.digi-sky.com\minio\4Real\Project\4D\WangXing',
    [string]$Username = 'minio',
    [Parameter(Mandatory = $true)]
    [string]$Password,
    [string]$OutputDir = 'outputs\wangxing_csv_dataset',
    [int]$MaxRetries = 2,
    [switch]$OnlyWangXing
)

$ErrorActionPreference = 'Stop'

$coreRecordProperties = @(
    'record_id', 'source_type', 'generator', 'subject', 'subject_status',
    'capture_date', 'top_level', 'emotion_pinyin', 'emotion_class',
    'source_folder', 'semantic_label', 'action_labels', 'device_labels',
    'modality', 'quality_status',
    'variant_type', 'needs_review', 'duplicate_key', 'size_bytes',
    'file_name', 'relative_path', 'full_path', 'duplicate_count'
)

function Join-Labels {
    param([System.Collections.Generic.List[string]]$Values)

    if ($null -eq $Values -or $Values.Count -eq 0) {
        return ''
    }
    return (($Values | Select-Object -Unique) -join ';')
}

function Get-Sha256Text {
    param([string]$Text)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $digest = $sha.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant().Substring(0, 16)
    }
    finally {
        $sha.Dispose()
    }
}

function Get-EmotionLabels {
    param([string]$Text)

    $definitions = @(
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])xi[_-]?nu[_-]?ai(?=$|[^a-z0-9])'
            Pinyin = 'Xi_Nu_Ai'
            Class = 'multi_emotion'
            Label = 'multi_emotion'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])beishang\d*(?=$|[^a-z0-9])'
            Pinyin = 'BeiShang'
            Class = 'sadness'
            Label = 'sadness'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])fennu\d*(?=$|[^a-z0-9])'
            Pinyin = 'FenNu'
            Class = 'anger'
            Label = 'anger'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])shengqi\d*(?=$|[^a-z0-9])'
            Pinyin = 'ShengQi'
            Class = 'anger'
            Label = 'anger'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])jingya\d*(?=$|[^a-z0-9])'
            Pinyin = 'JingYa'
            Class = 'surprise'
            Label = 'surprise'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])kongju\d*(?=$|[^a-z0-9])'
            Pinyin = 'KongJu'
            Class = 'fear'
            Label = 'fear'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])kaixin\d*(?=$|[^a-z0-9])'
            Pinyin = 'KaiXin'
            Class = 'happiness'
            Label = 'happiness'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])xiao\d*(?=$|[^a-z0-9])'
            Pinyin = 'Xiao'
            Class = 'smile'
            Label = 'smile'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])neutral(?=$|[^a-z0-9])'
            Pinyin = 'Neutral'
            Class = 'neutral'
            Label = 'neutral'
        }
    )

    $pinyin = [System.Collections.Generic.List[string]]::new()
    $classes = [System.Collections.Generic.List[string]]::new()
    $labels = [System.Collections.Generic.List[string]]::new()
    foreach ($definition in $definitions) {
        if ($Text -match $definition.Pattern) {
            $pinyin.Add($definition.Pinyin)
            $classes.Add($definition.Class)
            $labels.Add($definition.Label)
        }
    }

    return [PSCustomObject]@{
        Pinyin = Join-Labels $pinyin
        Class = Join-Labels $classes
        Label = Join-Labels $labels
    }
}

function Get-ActionLabels {
    param([string]$Text)

    $definitions = @(
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])facs\d*(?=$|[^a-z0-9])'
            Label = 'facial_action'
            Name = 'FACS'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])(headmove|yaotou)(?=$|[^a-z0-9])'
            Label = 'head_motion'
            Name = 'HeadMove'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])(fayin|fuyin|gouyi\d*|raokouling\d*|yuanyin)(?=$|[^a-z0-9])'
            Label = 'articulation'
            Name = 'Articulation'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])(xinwengao\d*|yingwenzimu|yanwu)(?=$|[^a-z0-9])'
            Label = 'speech'
            Name = 'Speech'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])shengyin(?=$|[^a-z0-9])'
            Label = 'speech_audio'
            Name = 'ShengYin'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])wenshengshipin(?=$|[^a-z0-9])'
            Label = 'text_to_video'
            Name = 'WenShengShiPin'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])biaoyan\d*(?=$|[^a-z0-9])'
            Label = 'performance'
            Name = 'BiaoYan'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])biaoding\d*(?=$|[^a-z0-9])'
            Label = 'calibration'
            Name = 'BiaoDing'
        },
        [PSCustomObject]@{
            Pattern = '(?i)(^|[^a-z0-9])biaoqing(?=$|[^a-z0-9])'
            Label = 'facial_expression'
            Name = 'BiaoQing'
        }
    )

    $labels = [System.Collections.Generic.List[string]]::new()
    $names = [System.Collections.Generic.List[string]]::new()
    foreach ($definition in $definitions) {
        if ($Text -match $definition.Pattern) {
            $labels.Add($definition.Label)
            $names.Add($definition.Name)
        }
    }

    return [PSCustomObject]@{
        Labels = Join-Labels $labels
        Names = Join-Labels $names
    }
}

function Get-SourceInfo {
    param(
        [string]$Text,
        [string]$TopLevel
    )

    if ($Text -match '(?i)(seedance|wenshengshipin|text[-_ ]?to[-_ ]?video|synthetic|generated)') {
        return [PSCustomObject]@{
            Type = 'ai_generated'
            Generator = if ($Text -match '(?i)seedance') { 'Seedance' } else { 'text_to_video' }
            Confidence = 'high'
        }
    }

    if ($TopLevel -match '^\d{4}_\d{2}_\d{2}$') {
        return [PSCustomObject]@{
            Type = 'real_capture'
            Generator = ''
            Confidence = 'medium'
        }
    }

    return [PSCustomObject]@{
        Type = 'unknown'
        Generator = ''
        Confidence = 'low'
    }
}

function Get-SubjectInfo {
    param(
        [string]$Text,
        [string]$TopLevel
    )

    if ($Text -match '(?i)(^|[^a-z0-9])wangxing(?=$|[^a-z0-9])') {
        return [PSCustomObject]@{
            Subject = 'WangXing'
            Status = 'wangxing'
        }
    }

    $subjects = @(
        @{ Pattern = '(?i)(^|[^a-z0-9])qianliuying(?=$|[^a-z0-9])'; Name = 'QianLiuYing' },
        @{ Pattern = '(?i)(^|[^a-z0-9])qingkangzhi(?=$|[^a-z0-9])'; Name = 'QingKangZhi' },
        @{ Pattern = '(?i)(^|[^a-z0-9])suntengfei(?=$|[^a-z0-9])'; Name = 'SunTengFei' },
        @{ Pattern = '(?i)(^|[^a-z0-9])tianjiudata(?=$|[^a-z0-9])'; Name = 'TianJiuData' },
        @{ Pattern = '(?i)(^|[^a-z0-9])xuhuan(?=$|[^a-z0-9])'; Name = 'XuHuan' },
        @{ Pattern = '(?i)(^|[^a-z0-9])xiaoxiao(?=$|[^a-z0-9])'; Name = 'XiaoXiao' },
        @{ Pattern = '(?i)(^|[^a-z0-9])xiaoyue1(?=$|[^a-z0-9])'; Name = 'XiaoYue1' },
        @{ Pattern = '(?i)(^|[^a-z0-9])xiaoyue2(?=$|[^a-z0-9])'; Name = 'XiaoYue2' },
        @{ Pattern = '(?i)(^|[^a-z0-9])zhouzhicheng(?=$|[^a-z0-9])'; Name = 'ZhouZhicheng' }
    )
    foreach ($subject in $subjects) {
        if ($Text -match $subject.Pattern) {
            return [PSCustomObject]@{
                Subject = $subject.Name
                Status = 'other_named'
            }
        }
    }

    return [PSCustomObject]@{
        Subject = 'WangXing'
        Status = 'wangxing'
    }
}

function Get-ExclusionInfo {
    param([string]$RelativePath)

    $rules = @(
        @{ Pattern = '(?i)^2026_07_09/Canon_Video2_grid45_9x5\.mp4$'; Reason = 'sample_025' },
        @{ Pattern = '(?i)^2026_07_09/Canon_Video2/'; Reason = 'sample_026_folder' },
        @{ Pattern = '(?i)^2026_07_09/Canon_Video3/'; Reason = 'sample_027_folder' },
        @{ Pattern = '(?i)^2026_07_13/5335/'; Reason = 'sample_029_folder' },
        @{ Pattern = '(?i)^2026_07_15/Canon/Background/'; Reason = 'sample_031_folder' },
        @{ Pattern = '(?i)^2026_07_10/Canon_OLAT1/'; Reason = 'not_wangxing_2026_07_10_olat1' },
        @{ Pattern = '(?i)^2026_07_10/Canon_OLAT2/'; Reason = 'not_wangxing_2026_07_10_olat2' },
        @{ Pattern = '(?i)^2026_07_10/Canon_OLAT3/'; Reason = 'not_wangxing_2026_07_10_olat3' },
        @{ Pattern = '(?i)^2026_07_30/Video/Canon2/Marker/'; Reason = 'sample_124_folder' },
        @{ Pattern = '(?i)^2026_07_30/Video/MD/Marker/'; Reason = 'sample_126_folder' },
        @{ Pattern = '(?i)^2026_08_04/Data/MD/BeiJing/BeiJing/'; Reason = 'sample_158_folder' }
    )
    foreach ($rule in $rules) {
        if ($RelativePath -match $rule.Pattern) {
            return [PSCustomObject]@{
                Excluded = 'yes'
                Reason = $rule.Reason
            }
        }
    }
    return [PSCustomObject]@{
        Excluded = 'no'
        Reason = ''
    }
}

function Get-QualityInfo {
    param(
        [string]$Text,
        [string]$FileName
    )

    if ($Text -match '(?i)(data[_-]?problem|data[_-]?broken|broken|problem)') {
        return [PSCustomObject]@{
            Status = 'problem'
            Variant = 'problem'
        }
    }
    if ($FileName -match '(?i)grid45[_-]?9x5') {
        return [PSCustomObject]@{
            Status = 'preview'
            Variant = 'grid_preview'
        }
    }
    if ($Text -match '(?i)(^|[^a-z0-9])(test|testcanon|testtrim|test_video|old_rc)([^a-z0-9]|$)') {
        return [PSCustomObject]@{
            Status = 'test'
            Variant = 'test'
        }
    }
    if ($Text -match '(?i)(resize|process|ppfix|jibian|distort|disort|sync|trim|background)') {
        $variant = 'derived'
        if ($Text -match '(?i)resize') { $variant = 'resize' }
        elseif ($Text -match '(?i)process') { $variant = 'processed' }
        elseif ($Text -match '(?i)ppfix') { $variant = 'postprocessed' }
        elseif ($Text -match '(?i)jibian') { $variant = 'corrected' }
        elseif ($Text -match '(?i)(distort|disort)') { $variant = 'distortion_corrected' }
        elseif ($Text -match '(?i)sync') { $variant = 'synchronized' }
        elseif ($Text -match '(?i)trim') { $variant = 'trimmed' }
        elseif ($Text -match '(?i)background') { $variant = 'background' }
        return [PSCustomObject]@{
            Status = 'derived'
            Variant = $variant
        }
    }

    return [PSCustomObject]@{
        Status = 'usable'
        Variant = 'source'
    }
}

function Get-DeviceLabels {
    param([string]$Text)

    $labels = [System.Collections.Generic.List[string]]::new()
    $definitions = @(
        @{ Pattern = '(?i)(^|[^a-z0-9])canon([^a-z0-9]|$)'; Label = 'Canon' },
        @{ Pattern = '(?i)(^|[^a-z0-9])ximea([^a-z0-9]|$)'; Label = 'Ximea' },
        @{ Pattern = '(?i)(^|[^a-z0-9])(daheng|dh)([^a-z0-9]|$)'; Label = 'DH' },
        @{ Pattern = '(?i)(^|[^a-z0-9])md([^a-z0-9]|$)'; Label = 'MD' },
        @{ Pattern = '(?i)(^|[^a-z0-9])hk([^a-z0-9]|$)'; Label = 'HK' },
        @{ Pattern = '(?i)(^|[^a-z0-9])hw([^a-z0-9]|$)'; Label = 'HW' },
        @{ Pattern = '(?i)olat'; Label = 'OLAT' },
        @{ Pattern = '(?i)body3d'; Label = 'Body3D' },
        @{ Pattern = '(?i)(^|[^a-z0-9])4d([^a-z0-9]|$)'; Label = '4D' }
    )
    foreach ($definition in $definitions) {
        if ($Text -match $definition.Pattern) {
            $labels.Add($definition.Label)
        }
    }
    return @($labels | Select-Object -Unique)
}

function Get-Modality {
    param([string]$Text)

    $modalities = [System.Collections.Generic.List[string]]::new()
    $map = @(
        @{ Pattern = '(?i)infrared'; Value = 'infrared' },
        @{ Pattern = '(?i)(^|[^a-z0-9])gray([^a-z0-9]|$)'; Value = 'gray' },
        @{ Pattern = '(?i)olat'; Value = 'OLAT' },
        @{ Pattern = '(?i)(^|[^a-z0-9])cl([^a-z0-9]|$)'; Value = 'CL' },
        @{ Pattern = '(?i)(^|[^a-z0-9])bs([^a-z0-9]|$)'; Value = 'BS' },
        @{ Pattern = '(?i)body3d'; Value = 'Body3D' },
        @{ Pattern = '(?i)(^|[^a-z0-9])4d([^a-z0-9]|$)'; Value = '4D' }
    )
    foreach ($item in $map) {
        if ($Text -match $item.Pattern) {
            $modalities.Add($item.Value)
        }
    }
    return Join-Labels $modalities
}

function Get-Mp4FilesWithRetry {
    param(
        [string]$Path,
        [int]$Retries
    )

    $filesByPath = @{}
    $lastErrors = @()
    for ($attempt = 1; $attempt -le ($Retries + 1); $attempt++) {
        $attemptErrors = @()
        $items = @(
            Get-ChildItem -LiteralPath $Path -Recurse -File -Force `
                -Filter '*.mp4' -ErrorAction SilentlyContinue `
                -ErrorVariable attemptErrors
        )
        foreach ($item in $items) {
            $filesByPath[$item.FullName] = $item
        }
        if ($attemptErrors.Count -eq 0) {
            $lastErrors = @()
            break
        }
        $lastErrors = $attemptErrors
        if ($attempt -lt ($Retries + 1)) {
            Start-Sleep -Seconds 1
        }
    }

    return [PSCustomObject]@{
        Files = @($filesByPath.Values | Sort-Object FullName)
        Errors = @($lastErrors)
    }
}

function Export-RecordsCsv {
    param(
        [object[]]$Rows,
        [string]$Path,
        [object]$Template
    )

    if ($Rows.Count -eq 0) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
        return
    }

    $Rows |
        Select-Object $coreRecordProperties |
        Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
}

function Select-MultiLabelRows {
    param(
        [object[]]$Rows,
        [string]$Property,
        [string]$Label,
        [switch]$SelectBlank
    )

    if ($SelectBlank) {
        return @($Rows | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.$Property) })
    }
    return @(
        $Rows | Where-Object {
            $value = [string]$_.$Property
            if ([string]::IsNullOrWhiteSpace($value)) {
                return $false
            }
            return @($value.Split(';')) -contains $Label
        }
    )
}

function Get-CategoryStats {
    param(
        [object[]]$Rows,
        [string]$Dimension,
        [string]$Category
    )

    $size = ($Rows | Measure-Object size_bytes -Sum).Sum
    $usable = @($Rows | Where-Object quality_status -eq 'usable').Count
    $review = @($Rows | Where-Object needs_review -eq 'yes').Count
    return [PSCustomObject]@{
        dimension = $Dimension
        category = $Category
        count = $Rows.Count
        size_bytes = if ($null -eq $size) { 0 } else { $size }
        size_gb = if ($null -eq $size) { 0 } else { [math]::Round($size / 1GB, 2) }
        usable_count = $usable
        review_count = $review
    }
}

$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$securePassword = ConvertTo-SecureString $Password -AsPlainText -Force
$credential = New-Object System.Management.Automation.PSCredential($Username, $securePassword)
$driveName = 'WXManifest'

$records = [System.Collections.Generic.List[object]]::new()
$errors = [System.Collections.Generic.List[object]]::new()
$branchSummary = [System.Collections.Generic.List[object]]::new()

$emotionTaxonomy = @(
    [PSCustomObject]@{ pinyin = 'BeiShang'; class = 'sadness'; label = 'sadness'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'FenNu'; class = 'anger'; label = 'anger'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'ShengQi'; class = 'anger'; label = 'anger'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'JingYa'; class = 'surprise'; label = 'surprise'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'KongJu'; class = 'fear'; label = 'fear'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'KaiXin'; class = 'happiness'; label = 'happiness'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'Xiao'; class = 'smile'; label = 'smile'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'Neutral'; class = 'neutral'; label = 'neutral'; is_emotion = 'yes' },
    [PSCustomObject]@{ pinyin = 'Xi_Nu_Ai'; class = 'multi_emotion'; label = 'multi_emotion'; is_emotion = 'yes' }
)

try {
    New-PSDrive -Name $driveName -PSProvider FileSystem -Root $AccessRoot `
        -Credential $credential -Scope Global -ErrorAction Stop | Out-Null
    $basePath = "${driveName}:\"
    $topDirectories = @(Get-ChildItem -LiteralPath $basePath -Directory -Force -ErrorAction Stop)

    foreach ($topDirectory in $topDirectories) {
        $scan = Get-Mp4FilesWithRetry -Path $topDirectory.FullName -Retries $MaxRetries
        $branchMp4Count = 0
        $branchSize = [int64]0

        foreach ($scanError in $scan.Errors) {
            $errors.Add([PSCustomObject]@{
                    top_level = $topDirectory.Name
                    error_path = [string]$scanError.TargetObject
                    error_message = [string]$scanError.Exception.Message
                })
        }

        foreach ($file in $scan.Files) {
            $relativeWindows = $file.FullName.Substring($AccessRoot.Length).TrimStart([char]'\')
            $relativePath = $relativeWindows.Replace('\', '/')
            $parts = $relativeWindows.Split([char]'\')
            $topLevel = if ($parts.Count -gt 0) { $parts[0] } else { '' }
            $directoryChain = if ($parts.Count -gt 1) {
                ($parts[0..($parts.Count - 2)] -join '\')
            }
            else {
                ''
            }
            $nameText = $relativeWindows.ToLowerInvariant()
            $emotion = Get-EmotionLabels $nameText
            $action = Get-ActionLabels $nameText
            $source = Get-SourceInfo -Text $nameText -TopLevel $topLevel
            $subject = Get-SubjectInfo -Text $nameText -TopLevel $topLevel
            $quality = Get-QualityInfo -Text $nameText -FileName $file.Name
            $exclusion = Get-ExclusionInfo $relativePath
            $sourceFolder = $file.Directory.Name
            $semanticLabel = if ($emotion.Pinyin) {
                $emotion.Pinyin
            }
            elseif ($action.Labels) {
                $action.Labels
            }
            else {
                $sourceFolder
            }
            $deviceLabels = @(Get-DeviceLabels $nameText)
            $device = if ($deviceLabels.Count -gt 0) { $deviceLabels[0] } else { '' }
            $modality = Get-Modality $nameText
            $recordId = Get-Sha256Text $relativePath.ToLowerInvariant()
            $duplicateKey = [System.IO.Path]::GetFileNameWithoutExtension($file.Name).ToLowerInvariant()
            $nameCategories = [System.Collections.Generic.List[string]]::new()

            if ($source.Type -eq 'ai_generated') { $nameCategories.Add('ai_generated') }
            if ($source.Type -eq 'real_capture') { $nameCategories.Add('real_capture') }
            if ($emotion.Class) { $nameCategories.Add('emotion') }
            if ($action.Labels) { $nameCategories.Add($action.Labels) }
            if ($quality.Status -ne 'usable') { $nameCategories.Add($quality.Status) }
            if ($device) { $nameCategories.Add('capture_device') }

            $usable = if ($quality.Status -eq 'usable' -and $source.Type -ne 'unknown') { 'yes' } else { 'no' }
            $needsReview = if (
                $source.Type -eq 'unknown' -or
                $quality.Status -ne 'usable' -or
                ($emotion.Class -eq '' -and $action.Labels -eq '')
            ) { 'yes' } else { 'no' }

            $records.Add([PSCustomObject]@{
                    record_id = $recordId
                    source_type = $source.Type
                    generator = $source.Generator
                    subject = $subject.Subject
                    subject_status = $subject.Status
                    capture_date = if ($topLevel -match '^\d{4}_\d{2}_\d{2}$') { $topLevel.Replace('_', '-') } else { '' }
                    top_level = $topLevel
                    emotion_pinyin = $emotion.Pinyin
                    emotion_class = $emotion.Class
                    source_folder = $sourceFolder
                    semantic_label = $semanticLabel
                    action_labels = $action.Labels
                    device_labels = ($deviceLabels -join ';')
                    modality = $modality
                    quality_status = $quality.Status
                    variant_type = $quality.Variant
                    needs_review = $needsReview
                    duplicate_key = $duplicateKey
                    size_bytes = $file.Length
                    file_name = $file.Name
                    relative_path = $relativePath
                    full_path = ($CanonicalRoot.TrimEnd('\') + '\' + $relativeWindows)
                    excluded = $exclusion.Excluded
                    exclusion_reason = $exclusion.Reason
                })
            $branchMp4Count++
            $branchSize += $file.Length
        }

        $branchSummary.Add([PSCustomObject]@{
                top_level = $topDirectory.Name
                mp4_count = $branchMp4Count
                size_bytes = $branchSize
                size_gb = [math]::Round($branchSize / 1GB, 2)
                read_error_count = $scan.Errors.Count
            })
    }
}
finally {
    if (Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue) {
        Remove-PSDrive -Name $driveName -Force -ErrorAction SilentlyContinue
    }
}

$sourceRecords = @(
    $records |
        Where-Object excluded -eq 'no' |
        Sort-Object relative_path
)
$namedRecords = @(
    $sourceRecords |
        Where-Object subject_status -in @('wangxing', 'other_named')
)
$allRecords = @($sourceRecords)
if ($OnlyWangXing) {
    $allRecords = @($allRecords | Where-Object subject_status -eq 'wangxing')
    $filteredSummary = [System.Collections.Generic.List[object]]::new()
    foreach ($group in @($allRecords | Group-Object top_level)) {
        $size = ($group.Group | Measure-Object size_bytes -Sum).Sum
        $filteredSummary.Add([PSCustomObject]@{
                top_level = $group.Name
                mp4_count = $group.Count
                size_bytes = $size
                size_gb = [math]::Round($size / 1GB, 2)
                read_error_count = @(
                    $errors | Where-Object top_level -eq $group.Name
                ).Count
            })
    }
    $branchSummary = $filteredSummary
}
$duplicateCounts = @{}
foreach ($record in $allRecords) {
    if (-not $duplicateCounts.ContainsKey($record.duplicate_key)) {
        $duplicateCounts[$record.duplicate_key] = 0
    }
    $duplicateCounts[$record.duplicate_key]++
}
foreach ($record in $allRecords) {
    $record | Add-Member -NotePropertyName duplicate_count `
        -NotePropertyValue $duplicateCounts[$record.duplicate_key] -Force
}

$realRecords = @($allRecords | Where-Object source_type -eq 'real_capture')
$aiRecords = @($allRecords | Where-Object source_type -eq 'ai_generated')
$reviewRecords = @($allRecords | Where-Object needs_review -eq 'yes')

$folderReviewRows = @(
    $sourceRecords |
        Where-Object subject_status -eq 'unknown' |
        ForEach-Object {
            $relativeFolder = Split-Path `
                -Path $_.relative_path.Replace('/', '\') `
                -Parent
            [PSCustomObject]@{
                relative_folder = $relativeFolder.Replace('\', '/')
                top_level = ($relativeFolder.Replace('\', '/') -split '/')[0]
                subject_hint = $_.subject
                subject_status = $_.subject_status
                suggested_status = if ($_.subject_status -eq 'not_wangxing') {
                    'known_excluded'
                }
                else {
                    'pending_user_judgment'
                }
                file_name = $_.file_name
            }
        } |
        Group-Object relative_folder |
        ForEach-Object {
            $first = $_.Group[0]
            $relativeFolder = $_.Name
            [PSCustomObject]@{
                relative_folder = $relativeFolder
                full_folder_path = (
                    $CanonicalRoot.TrimEnd('\') + '\' +
                    $relativeFolder.Replace('/', '\')
                )
                top_level = $first.top_level
                subject_hint = $first.subject_hint
                subject_status = $first.subject_status
                suggested_status = $first.suggested_status
                mp4_count = $_.Count
                sample_files = (
                    $_.Group |
                        Select-Object -First 5 -ExpandProperty file_name
                ) -join ';'
            }
        } |
        Sort-Object relative_folder
)

$allRecords |
    Select-Object $coreRecordProperties |
    Export-Csv -LiteralPath (Join-Path $outputPath 'manifest_all.csv') -NoTypeInformation -Encoding UTF8
$emotionTaxonomy | Export-Csv -LiteralPath (Join-Path $outputPath 'emotion_taxonomy.csv') -NoTypeInformation -Encoding UTF8
@($errors | ForEach-Object { $_ }) | Export-Csv `
    -LiteralPath (Join-Path $outputPath 'scan_errors.csv') `
    -NoTypeInformation -Encoding UTF8
@($branchSummary | Sort-Object top_level) | Export-Csv `
    -LiteralPath (Join-Path $outputPath 'directory_summary.csv') `
    -NoTypeInformation -Encoding UTF8
if ($folderReviewRows.Count -gt 0) {
    $folderReviewRows | Export-Csv `
        -LiteralPath (Join-Path $outputPath 'folder_review.csv') `
        -NoTypeInformation -Encoding UTF8
    $pendingFolderRows = @(
        $folderReviewRows |
            Where-Object suggested_status -eq 'pending_user_judgment'
    )
    $pendingFolderPaths = @(
        $pendingFolderRows |
            Select-Object full_folder_path
    )
    $pendingFolderPaths | Export-Csv `
        -LiteralPath (Join-Path $outputPath 'folder_review_pending.csv') `
        -NoTypeInformation -Encoding UTF8
    @(
        $pendingFolderRows |
            Group-Object top_level |
            ForEach-Object {
                [PSCustomObject]@{
                    top_level = $_.Name
                    folder_count = $_.Count
                    mp4_count = (
                        ($_.Group | Measure-Object -Property mp4_count -Sum).Sum
                    )
                }
            } |
            Sort-Object top_level
    ) | Export-Csv `
        -LiteralPath (Join-Path $outputPath 'folder_review_summary_by_date.csv') `
        -NoTypeInformation -Encoding UTF8
}
else {
    foreach ($reviewFile in @(
            'folder_review.csv',
            'folder_review_pending.csv',
            'folder_review_summary_by_date.csv'
        )) {
        $reviewPath = Join-Path $outputPath $reviewFile
        if (Test-Path -LiteralPath $reviewPath) {
            Remove-Item -LiteralPath $reviewPath -Force
        }
    }
}

$summary = @(
    [PSCustomObject]@{
        metric = 'all_mp4'
        count = $allRecords.Count
        size_bytes = ($allRecords | Measure-Object size_bytes -Sum).Sum
    },
    [PSCustomObject]@{
        metric = 'real_capture'
        count = $realRecords.Count
        size_bytes = ($realRecords | Measure-Object size_bytes -Sum).Sum
    },
    [PSCustomObject]@{
        metric = 'ai_generated'
        count = $aiRecords.Count
        size_bytes = ($aiRecords | Measure-Object size_bytes -Sum).Sum
    },
    [PSCustomObject]@{
        metric = 'needs_review'
        count = $reviewRecords.Count
        size_bytes = ($reviewRecords | Measure-Object size_bytes -Sum).Sum
    },
    [PSCustomObject]@{
        metric = 'scan_errors'
        count = $errors.Count
        size_bytes = 0
    }
)
$summary | Export-Csv -LiteralPath (Join-Path $outputPath 'summary.csv') -NoTypeInformation -Encoding UTF8

$viewStats = [System.Collections.Generic.List[object]]::new()
$templateRecord = $allRecords | Select-Object -First 1
$viewDefinitions = @(
    @{
        Directory = 'by_emotion'
        Property = 'emotion_pinyin'
        Categories = @(
            'BeiShang', 'FenNu', 'ShengQi', 'JingYa', 'KongJu',
            'KaiXin', 'Xiao', 'Neutral', 'Xi_Nu_Ai'
        )
        BlankName = 'NoEmotion'
        Dimension = 'emotion'
    },
    @{
        Directory = 'by_device'
        Property = 'device_labels'
        Categories = @(
            'Canon', 'MD', 'HK', 'DH', 'HW', 'Ximea', 'OLAT',
            'Body3D', '4D'
        )
        BlankName = 'Unknown'
        Dimension = 'device'
    }
)

foreach ($definition in $viewDefinitions) {
    $viewDirectory = Join-Path $outputPath $definition.Directory
    New-Item -ItemType Directory -Path $viewDirectory -Force | Out-Null
    foreach ($category in $definition.Categories) {
        $rows = @(Select-MultiLabelRows -Rows $allRecords -Property $definition.Property -Label $category)
        $fileName = ($category -replace '[\\/:*?"<>|]', '_') + '.csv'
        Export-RecordsCsv -Rows $rows -Path (Join-Path $viewDirectory $fileName) -Template $templateRecord
        $viewStats.Add((Get-CategoryStats -Rows $rows -Dimension $definition.Dimension -Category $category))
    }

    $blankRows = @(Select-MultiLabelRows -Rows $allRecords -Property $definition.Property -Label '' -SelectBlank)
    Export-RecordsCsv `
        -Rows $blankRows `
        -Path (Join-Path $viewDirectory ($definition.BlankName + '.csv')) `
        -Template $templateRecord
    $viewStats.Add((Get-CategoryStats `
            -Rows $blankRows `
            -Dimension $definition.Dimension `
            -Category $definition.BlankName))
}

$sourceDirectory = Join-Path $outputPath 'by_source'
New-Item -ItemType Directory -Path $sourceDirectory -Force | Out-Null
foreach ($category in @('real_capture', 'ai_generated', 'unknown')) {
    $rows = @($allRecords | Where-Object source_type -eq $category)
    Export-RecordsCsv `
        -Rows $rows `
        -Path (Join-Path $sourceDirectory ($category + '.csv')) `
        -Template $templateRecord
    $viewStats.Add((Get-CategoryStats -Rows $rows -Dimension 'source' -Category $category))
}

$subjectDirectory = Join-Path $outputPath 'by_subject'
New-Item -ItemType Directory -Path $subjectDirectory -Force | Out-Null
$subjectCategories = @(
    'WangXing', 'LiYou', 'QianLiuYing', 'QingKangZhi',
    'SunTengFei', 'TianJiuData', 'XuHuan', 'XiaoXiao',
    'XiaoYue1', 'XiaoYue2', 'ZhouZhicheng'
)
foreach ($category in $subjectCategories) {
    $rows = @($namedRecords | Where-Object subject -eq $category)
    Export-RecordsCsv `
        -Rows $rows `
        -Path (Join-Path $subjectDirectory ($category + '.csv')) `
        -Template $templateRecord
    $viewStats.Add((Get-CategoryStats -Rows $rows -Dimension 'subject' -Category $category))
}
$otherPeopleRows = @(
    $namedRecords | Where-Object subject_status -eq 'other_named'
)
$otherPeopleRows | Select-Object $coreRecordProperties | Export-Csv `
    -LiteralPath (Join-Path $outputPath 'other_people_manifest.csv') `
    -NoTypeInformation -Encoding UTF8
$otherPeopleSummary = @(
    $otherPeopleRows |
        Group-Object subject |
        ForEach-Object {
            [PSCustomObject]@{
                subject = $_.Name
                mp4_count = $_.Count
                size_bytes = (
                    ($_.Group | Measure-Object -Property size_bytes -Sum).Sum
                )
            }
        } |
        Sort-Object subject
)
$otherPeopleSummary | Export-Csv `
    -LiteralPath (Join-Path $outputPath 'other_people_summary.csv') `
    -NoTypeInformation -Encoding UTF8

$notWangXingRows = @()
Export-RecordsCsv `
    -Rows $notWangXingRows `
    -Path (Join-Path $subjectDirectory 'NotWangXing.csv') `
    -Template $templateRecord
$viewStats.Add((Get-CategoryStats `
        -Rows $notWangXingRows `
        -Dimension 'subject' `
        -Category 'NotWangXing'))

$unknownSubjectRows = @()
Export-RecordsCsv `
    -Rows $unknownSubjectRows `
    -Path (Join-Path $subjectDirectory 'Unknown.csv') `
    -Template $templateRecord
$viewStats.Add((Get-CategoryStats `
        -Rows $unknownSubjectRows `
        -Dimension 'subject' `
        -Category 'Unknown'))

$viewStats | Export-Csv `
    -LiteralPath (Join-Path $outputPath 'statistics.csv') `
    -NoTypeInformation -Encoding UTF8

Write-Output ("output_dir=" + $outputPath)
Write-Output ("all_mp4=" + $allRecords.Count)
Write-Output ("real_capture=" + $realRecords.Count)
Write-Output ("ai_generated=" + $aiRecords.Count)
Write-Output ("needs_review=" + $reviewRecords.Count)
Write-Output ("scan_errors=" + $errors.Count)
