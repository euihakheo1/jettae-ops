<#
수요 실측치를 공개 사이트에서 받아 docs/demand/<날짜>.md · .json 으로 남깁니다.
숫자(사전 신청 인원 · 이벤트별 횟수)만 담고, 이메일 · 회사명 같은 개인정보는 담지 않습니다.

쓰는 법 (jettae-ops 폴더에서):
  powershell -ExecutionPolicy Bypass -File scripts\publish_demand.ps1          # 파일만 만들기
  powershell -ExecutionPolicy Bypass -File scripts\publish_demand.ps1 -Push    # 커밋 · 푸시까지

- 토큰은 lead 또는 admin 토큰입니다. 환경 변수 JETTAE_ADMIN_TOKEN이 없으면 물어보고,
  입력은 화면에 보이지 않으며 어떤 파일에도 남기지 않습니다.
- -FromJson demand.json: 이미 내려받은 수요 지표 파일로 문서만 만듭니다(서버에 접속하지 않음).
- -Push는 이번에 만든 두 파일만 커밋합니다. 다른 수정 사항은 건드리지 않습니다.
#>
param(
  [string]$App = "https://jettae-ops.onrender.com",
  [string]$FromJson = "",
  [string]$OutDir = "",
  [switch]$Push
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root "docs\demand" }

if ($FromJson) {
  $raw = Get-Content -Raw -Encoding UTF8 $FromJson
} else {
  $token = $env:JETTAE_ADMIN_TOKEN
  if (-not $token) {
    $sec = Read-Host "lead 또는 admin 토큰 (입력해도 화면에 보이지 않습니다)" -AsSecureString
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
  }
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  Write-Host "수요 지표를 받는 중입니다 (무료 서버라 처음엔 1분 가까이 걸릴 수 있습니다)..."
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "$App/api/v1/metrics/demand" -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 180
  } catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 401) { throw "토큰이 틀렸습니다(401). Render의 JETTAE_API_TOKENS에 있는 lead/admin 토큰인지 확인하세요." }
    if ($code -eq 403) { throw "토큰 역할이 부족합니다(403). lead 또는 admin 토큰이어야 합니다." }
    throw
  } finally {
    Remove-Variable token -ErrorAction SilentlyContinue
  }
  $raw = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray())
}

$m = $raw | ConvertFrom-Json
function N([string]$name) {
  $v = $m.events.$name
  if ($null -eq $v) { return 0 } else { return [int]$v }
}
$kst = [DateTime]::UtcNow.AddHours(9)
$date = $kst.ToString("yyyy-MM-dd")
$stamp = $kst.ToString("yyyy-MM-dd HH:mm")

$md = @"
# 수요 실측치 ($stamp KST 기준)

신청서에서 "본선에서 실측치 그대로 공개"를 약속한 숫자입니다.
공개 베타(jettae-ops.onrender.com) 서버가 센 값을 고치지 않고 옮겼습니다.

| 지표 | 값 | 세는 방법 |
|---|---:|---|
| 방문자(브라우저당 첫 방문) | $(N 'landing_first_view') | ``landing_first_view`` |
| 랜딩 페이지 조회(재방문 포함) | $(N 'landing_view') | ``landing_view`` |
| **무료 검진 업로드(실행)** | **$(N 'runbook_upload')** | ``runbook_upload`` · 서버가 직접 기록 |
| 사전 신청 버튼 클릭 | $(N 'signup_click') | ``signup_click`` |
| **사전 신청 완료(이메일 등록 · 개인정보 동의)** | **$([int]$m.signups)** | 서버가 직접 기록 · 같은 이메일은 1번 |
| 예제 데모 시작 | $(N 'demo_start') | ``demo_start`` |
| 리포트 내려받기 | $(N 'report_download') | ``report_download`` |

- **기간:** 공개 베타 측정을 시작한 2026-10-17부터 위 시각까지의 누적입니다.
- **섞여 있는 값:** 창업자의 확인용 방문 · 업로드가 섞여 있을 수 있고, 걸러 내지 않았습니다.
- **믿을 만한 값:** 방문 · 클릭은 브라우저가 보내는 값이라 부풀릴 수 있습니다. 판단은 서버가 직접 세는
  **무료 검진 업로드**와 **사전 신청 완료**를 중심으로 봐 주세요.
- **개인정보:** 이메일 · 회사명은 올리지 않습니다. 서버 응답 원본은 [$date.json](./$date.json)입니다.
"@

New-Item -ItemType Directory -Force $OutDir | Out-Null
$utf8 = New-Object Text.UTF8Encoding $false
$mdPath = Join-Path $OutDir "$date.md"
$jsonPath = Join-Path $OutDir "$date.json"
[IO.File]::WriteAllText($mdPath, $md.Replace("`r`n", "`n") + "`n", $utf8)
[IO.File]::WriteAllText($jsonPath, $raw.Trim() + "`n", $utf8)

Write-Host ""
Write-Host "만들었습니다: $mdPath"
Write-Host ("  방문자 {0} · 무료 검진 업로드 {1} · 사전 신청 완료 {2} · 사전 신청 클릭 {3}" -f (N 'landing_first_view'), (N 'runbook_upload'), ([int]$m.signups), (N 'signup_click'))

if ($Push) {
  Push-Location $root
  try {
    git add -- $mdPath $jsonPath
    git commit -m "수요 실측치 공개 ($stamp KST)" -- $mdPath $jsonPath
    git push
    Write-Host "GitHub에 올렸습니다."
  } finally {
    Pop-Location
  }
} else {
  Write-Host "GitHub에 올리려면 -Push를 붙여 다시 실행하세요."
}
