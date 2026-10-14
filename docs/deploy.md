# 공개 베타 배포 가이드 (Render)

제때옵스를 **공개 웹사이트**로 띄우는 순서입니다. 운영 경험이 없어도 따라 할 수 있게 썼고, 명령은 모두
**Windows PowerShell**에 그대로 붙여 넣을 수 있습니다(PowerShell에서는 `curl`이 아니라 꼭 `curl.exe`라고 씁니다).

배포가 끝나면 누구나 다음을 할 수 있습니다.

- 랜딩 페이지 보기, 사전 신청 남기기(개인정보 수집·이용 동의 필수)
- 별빛상점(가상 회사) **예제 진단** 둘러보기(서버가 시작할 때 한 번 진단해 둔 결과를 보여 줍니다)
- 로그인 없이 **무료 런북 검진**: 런북 파일을 올리면 비공개 워크스페이스가 만들어지고, 결과는 그 검진을 한
  브라우저에 저장된 **접근 키**로만 다시 열립니다(서버에는 키의 해시만 저장). 결과는 **7일 뒤 자동 삭제**됩니다.

운영자(나)는 관리자 토큰으로 **수요 지표**와 **사전 신청 명단(CSV)** 을 내려받고, 삭제 요청을 처리합니다.

> 공개 사이트에서는 생성형 AI(LLM)를 **끕니다**(`JETTAE_LLM_MODE=off`). 이미지에 AI SDK도 넣지 않습니다.
> API 비용이 들지 않고, 방문자가 올린 런북 내용이 외부 AI로 나가지 않습니다. 판정·점수·리허설은 원래 LLM 없이
> 고정 규칙으로 동작합니다.
>
> 올린 파일(런북·명부·당직표·카탈로그) 자체는 저장하지 않습니다. 런북 본문과 카탈로그의 설명·담당·위치, 명부의
> 직급·팀·별칭 같은 자유 글자는 연락처·주민번호 등을 길이를 보존한 채 가린 뒤 씁니다. 명부의 이름·재직 상태·
> 역할·연락 수단 종류(매체 이름만, 번호 없음)는 진단에 필요해서 결과(운영 그래프·보고서)에 그대로 들어가고,
> 결과와 함께 7일 뒤 지워집니다.

## 준비물

- GitHub 계정, [Render](https://render.com) 계정(결제 수단 등록 — 웹 서비스는 가장 작은 유료 사양을 씁니다)
- 내 컴퓨터의 Python(관리자 토큰을 만들 때 한 번 씁니다)
- 비밀번호 관리자(토큰 보관용)

## 1. GitHub에 올리기

저장소 루트에 `render.yaml`, `Dockerfile`, `constraints.txt`가 있어야 합니다(이미 들어 있습니다).

```powershell
git init                      # 이미 git 저장소라면 건너뜁니다
git add .
git commit -m "제때옵스 공개 베타"
git branch -M main
git remote add origin https://github.com/<내 계정>/jettae-ops.git
git push -u origin main
```

비공개(private) 저장소여도 됩니다. Render가 GitHub 저장소를 읽을 수 있게 연결만 허락하면 됩니다.

## 2. 관리자 토큰 만들기

토큰은 관리자 비밀번호 같은 것입니다. 무작위 문자열을 새로 만듭니다(문서·예제에 나온 문자열을 그대로 쓰지 마세요).

```powershell
py -c "import secrets; print(secrets.token_urlsafe(32))"
```

(`py`가 없다는 오류가 나면 `py` 대신 `python`이라고 씁니다.) 출력된 43자 문자열을 비밀번호 관리자에 저장합니다.
그다음 아래 JSON에서 `<여기에 토큰>`을 꺾쇠(`<` `>`)까지 통째로 지우고 그 자리에 `<py -c 로 만든 43자 문자열>`,
즉 방금 출력된 문자열을 넣습니다. 이 JSON 한 줄이 `JETTAE_API_TOKENS` 값입니다.

```json
{"<여기에 토큰>": {"user": "heo", "role": "admin"}}
```

- `user`는 감사 로그에 남는 이름입니다(영문·한글 모두 가능).
- 토큰은 어디에도 커밋하지 마세요. 팀원이 생기면 토큰을 하나 더 추가합니다
  (`{"<토큰1>": {...}, "<토큰2>": {"user": "kim", "role": "lead"}}`; `lead`는 수요 지표까지, 명단은 `admin`만).

## 3. Render에서 Blueprint로 배포

1. Render 대시보드 오른쪽 위 **New → Blueprint**를 누릅니다.
2. GitHub 저장소 목록에서 `jettae-ops`를 고릅니다. Render가 `render.yaml`을 읽어 다음 두 가지를 보여 줍니다.
   - 웹 서비스 `jettae-ops` (Docker, 싱가포르, `0.5c-512mb`)
   - PostgreSQL `jettae-db` (싱가포르, `free`, 인터넷에서의 외부 접속 차단)
3. 비밀 값 입력 칸에 `JETTAE_API_TOKENS`가 나오면 2단계의 JSON 한 줄을 그대로 붙여 넣습니다.
4. **Apply**(또는 Deploy Blueprint)를 누릅니다. 첫 배포는 이미지 빌드 때문에 5~10분쯤 걸립니다.

컨테이너는 뜰 때마다 ① DB 스키마를 올리고(`jettae db upgrade`) ② 별빛상점 예제를 다시 불러온 뒤
③ 서버를 엽니다. 다시 시작해도 예제 워크스페이스는 하나로 유지됩니다.

이미지에 들어가는 파이썬 패키지는 `constraints.txt`에 버전이 고정되어 있어, 자동 배포 때마다 시험하지 않은 새
버전이 섞이지 않습니다(웹 화면은 커밋된 `web/package-lock.json` 그대로 `npm ci`로 설치합니다).

> 웹 서비스를 `free` 플랜으로 바꾸지 않는 이유: 15분 동안 방문이 없으면 잠들고, 다음 방문자는 1분 가까이
> 기다려야 합니다. 그 사이에 떠난 방문자 때문에 방문·전환 수치가 왜곡됩니다.

## 4. 배포 직후 확인

### 4-1. 서버와 관리자 토큰

- Render가 알려 주는 주소(`https://jettae-ops.onrender.com` — 이름이 겹치면 뒤에 문자가 붙습니다)를 엽니다.
- `https://<앱 주소>/healthz` 가 `{"status":"ok", ...}`를 돌려주면 서버가 살아 있는 것입니다(Render도 이 경로로
  상태를 확인합니다). `https://<앱 주소>/readyz` 는 DB 연결까지 확인합니다.
- 로그는 Render 대시보드 → 서비스 → **Logs**에서 봅니다.

이제 PowerShell에서 관리자 토큰이 잘 먹는지 확인합니다. 먼저 비밀번호 관리자에서 토큰을 **복사**한 뒤, 같은
창에서 아래 두 줄을 실행합니다(토큰을 명령에 직접 붙여 넣지 않으므로 PowerShell 기록 파일에 남지 않습니다).

```powershell
$token = (Get-Clipboard).Trim()
$app = "https://jettae-ops.onrender.com"     # 내 앱 주소로 바꿉니다(끝에 / 없이)
```

```powershell
curl.exe -H "Authorization: Bearer $token" "$app/api/v1/me"
```

`"role":"admin"`이 보이면 됩니다. `401`이면 토큰이 틀린 것입니다(아래 「자주 묻는 것」).
이후 명령은 모두 이 `$token`·`$app`을 씁니다. PowerShell 창을 새로 열었다면 위 두 줄을 다시 실행하세요.

배포가 실패하고 로그에 `공개 모드(JETTAE_AUTH_MODE=public)에도 JETTAE_API_TOKENS에 관리자 토큰이 하나 이상
있어야 합니다`가 보이면 3단계의 토큰 입력이 빠진 것입니다. 서비스 → **Environment**에서 `JETTAE_API_TOKENS`를
넣고 저장하면 다시 배포됩니다.

### 4-2. 방문자 주소(IP)를 어디서 읽을지 확인 — 꼭 하세요

무료 검진(하루 5회)·화면 이벤트(1분 60회)·사전 신청(1시간 10회)의 **접속 주소당 한도**는 방문자의 공인 IP로
셉니다. 그런데 요청은 Render 앞단(Cloudflare)을 거쳐 들어오고, 진짜 방문자 주소가 어느 헤더에 실려 오는지는 Render
문서에 나와 있지 않습니다. `render.yaml`은 Cloudflare가 넣는 헤더를 읽는 설정
`JETTAE_CLIENT_IP_SOURCE=header:CF-Connecting-IP`로 시작합니다. 이 값이 맞지 않으면

- 방문자가 헤더를 꾸며 보내 주소별 한도를 피하거나,
- 모든 방문자가 한 주소로 세어져, 한 사람 몫의 한도(무료 검진 하루 5회)에 사이트 전체가 걸립니다.

그래서 첫 배포 직후 한 번, 서버가 실제로 받은 값을 보고 맞는 설정을 고릅니다. 아래 확인용 주소는 관리자 토큰으로만
열리고, 보여 준 값은 저장하거나 로그에 남기지 않습니다.

**① 내 공인 IP 알아보기.** 이 PowerShell 창을 연 컴퓨터의 브라우저에서 네이버나 구글에 `내 IP 주소`를 검색합니다
(IP 확인 사이트를 열어도 됩니다). VPN·회사 프록시는 끄고 확인합니다. 나온 주소를 적어 둡니다(아래 예에서는
`198.51.100.23`). IPv4와 IPv6(`2001:db8:…`처럼 `:`가 섞인 주소)가 함께 나오면 둘 다 적습니다. 서버는 둘 중
하나로 봅니다.

**② 서버가 본 값 보기.**

```powershell
curl.exe -H "Authorization: Bearer $token" "$app/api/v1/admin/request-info"
```

응답 모양은 이렇습니다(주소는 예시).

```json
{"peer": "198.51.100.23", "x_forwarded_for": "198.51.100.23, 10.214.3.9", "cf_connecting_ip": "198.51.100.23", "true_client_ip": null, "x_real_ip": null, "client_ip_source": "header:CF-Connecting-IP", "resolved_client": "198.51.100.23"}
```

| 칸 | 뜻 |
|---|---|
| `resolved_client` | 지금 설정으로 고른 방문자 주소. 접속 주소당 한도에 실제로 쓰는 값입니다. IPv6는 같은 집·사무실 기기를 함께 세도록 앞 네 묶음까지의 대역(`2001:db8:1:2::/64` 모양)으로 보입니다 |
| `client_ip_source` | 지금 설정(`JETTAE_CLIENT_IP_SOURCE`) |
| `cf_connecting_ip` | `CF-Connecting-IP` 헤더(Cloudflare가 넣는 방문자 주소). `null`이면 그 헤더가 오지 않은 것 |
| `true_client_ip` | `True-Client-IP` 헤더 |
| `x_real_ip` | `X-Real-IP` 헤더 |
| `x_forwarded_for` | `X-Forwarded-For` 헤더 그대로. 프록시를 하나 지날 때마다 **오른쪽 끝**에 주소가 붙고, 왼쪽 부분은 방문자가 마음대로 써 보낼 수 있습니다 |
| `peer` | uvicorn이 알려 준 접속 주소. `FORWARDED_ALLOW_IPS="*"` 때문에 `X-Forwarded-For`의 맨 왼쪽 값이라 꾸밀 수 있으므로 Render에서는 고르지 않습니다 |

고른 헤더가 오지 않았거나(`null`), 두 번 왔거나, 주소가 아닌 값이면 `resolved_client`는 `peer` 값으로 셉니다.
`invalid`는 읽은 자리(`xff:<n>`의 n번째 값이나 `peer`)에 주소가 아닌 값이 있다는 뜻이고, `unknown`은 주소를 전혀
받지 못했다는 뜻입니다.

**③ 설정 고르기.** 표를 위에서부터 보고, ①에서 적은 내 IP가 **처음으로** 보이는 줄을 따릅니다.

| 내 IP가 보이는 칸 | `JETTAE_CLIENT_IP_SOURCE` |
|---|---|
| `cf_connecting_ip` | `header:CF-Connecting-IP` — 이미 이 값이므로 바꿀 것이 없습니다. `resolved_client`도 내 IP인지만 봅니다 |
| `true_client_ip` | `header:True-Client-IP` |
| `x_real_ip` | `header:X-Real-IP` |
| `x_forwarded_for`에만 | `xff:<n>` — 목록의 **오른쪽 끝에서 몇 번째**인지 셉니다(맨 오른쪽이 1). 예: `203.0.113.7, 198.51.100.23, 10.214.3.9`에서 내 IP가 `198.51.100.23`이면 `xff:2` |
| 어디에도 없음 | 아래 「내 IP가 어디에도 없을 때」 |

**④ 바꾸기**(③에서 다른 값이 나왔을 때만). Render 대시보드 → `jettae-ops` 서비스 → **Environment**에서
`JETTAE_CLIENT_IP_SOURCE`의 값을 고치고 **Save and deploy**를 누릅니다. 몇 분 뒤 배포가 끝나면 ② 명령을 다시
실행해 `client_ip_source`가 새 값이고 `resolved_client`가 내 IP인지 확인합니다. 같은 값으로 `render.yaml`도 고쳐
커밋·푸시해 둡니다(대시보드에서만 바꾼 값은 다음 Blueprint 동기화 때 `render.yaml` 값으로 돌아갈 수 있습니다).

**⑤ 꾸민 헤더로 시험하기**(설정을 바꿨든 안 바꿨든 마지막에 꼭). 방문자가 헤더를 꾸며 보내는 상황을 흉내 냅니다.
`203.0.113.7`은 문서용으로 정해 둔, 실제로는 쓰이지 않는 주소입니다.

```powershell
curl.exe -H "Authorization: Bearer $token" -H "X-Forwarded-For: 203.0.113.7" -H "CF-Connecting-IP: 203.0.113.7" -H "True-Client-IP: 203.0.113.7" -H "X-Real-IP: 203.0.113.7" "$app/api/v1/admin/request-info"
```

`resolved_client`가 그대로 내 IP(IPv6면 내 주소가 속한 `/64` 대역)면 확인이 끝났습니다. `203.0.113.7`이 나오면
지금 고른 헤더는 방문자가 꾸밀 수 있다는 뜻입니다. ③ 표에서 그 아래 줄로 내려가(대개 `xff:<n>`, n은 ②의 결과로
셉니다) ④·⑤를 다시 합니다.

**내 IP가 어디에도 없을 때.** VPN·회사 프록시를 끄거나 휴대폰 테더링으로 인터넷 연결을 바꿔 ①·②를 다시 합니다.
그래도 없으면 Render 앞단 구성이 이 안내와 다른 것이고, 지금은 모든 방문자가 한 주소로 세어질 수 있습니다. 원인을
찾을 때까지 **Environment**에서 접속 주소당 한도를 넉넉히 올려 두세요: `JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY`는
`100`(사이트 전체 하루 한도와 같게), `JETTAE_SIGNUPS_PER_CLIENT_PER_HOUR`는 `200`,
`JETTAE_EVENTS_PER_CLIENT_PER_MINUTE`는 `600`. 사이트 전체 하루 한도·이벤트 하루 상한·동시 진단 수·크기 한도는
주소와 상관없이 그대로 지켜집니다.

## 5. 수요 지표와 사전 신청 명단 보기

수요 지표(사전 신청 인원, 이벤트별 횟수, `lead` 이상):

```powershell
curl.exe -H "Authorization: Bearer $token" "$app/api/v1/metrics/demand"
```

```json
{"signups": 12, "events": {"landing_first_view": 340, "landing_view": 512, "runbook_upload": 41, "signup_click": 30}}
```

| 이벤트 | 뜻 |
|---|---|
| `landing_first_view` | 브라우저당 첫 방문(대략 순방문자 수) |
| `landing_view` | 랜딩 페이지 조회(재방문 포함) |
| `runbook_upload` | 무료 검진 실행 횟수(서버가 직접 기록 — 브라우저가 보낼 수 없음) |
| `upload_click` · `signup_click` · `demo_start` · `report_download` | 버튼 클릭 등 화면 이벤트 |

- 이벤트는 정해진 이름, 이벤트마다 정해진 몇 가지 값, 정해진 화면 경로(`/`·`/app`·`/app/report`·`/privacy`) 중
  하나만 받습니다. 자유 글자 칸이 없어 이메일·이름 같은 개인정보가 들어갈 자리가 없습니다.
- 같은 접속 주소에서 너무 자주 보내면 거절하고, 사이트 전체로도 하루(한국 시간) 20,000건(`JETTAE_EVENTS_PER_DAY`)까지만
  저장합니다(넘은 이벤트에도 응답은 정상으로 하므로 화면에는 영향이 없습니다). 그래도 화면 이벤트는 브라우저가 보내는
  값이라 마음먹으면 부풀릴 수 있습니다. 판단은 서버가 직접 세는 `signups`와 `runbook_upload`를 중심으로 하세요.

사전 신청 명단(엑셀용 CSV, **관리자 토큰만**):

```powershell
curl.exe -f -H "Authorization: Bearer $token" -o signups.csv "$app/api/v1/signups?format=csv"
```

- `-f`: 토큰이 틀리면 오류 내용을 `signups.csv`에 쓰지 않고 `curl: (22) ... 401`처럼 화면에 알립니다.
- 열: `created_at, email, organization, team_size, consent_marketing, source, message`
- UTF-8(BOM 포함)이라 엑셀에서 바로 열어도 한글이 깨지지 않습니다. `=`, `+`, `-`, `@`로 시작하는 칸은 수식으로
  실행되지 않게 앞에 `'`를 붙여 둡니다.
- 명단을 조회할 때마다 누가 몇 건을 받았는지 감사 로그에 남습니다. 마케팅 메일은 `consent_marketing`이 `true`인
  사람에게만 보내세요.
- JSON이 필요하면 `?format=csv`를 빼면 됩니다.
- **내려받은 파일도 개인정보입니다.** 안전한 곳에만 두고, 아래 7단계의 보관 기간·삭제 요청을 내 컴퓨터의
  사본에도 똑같이 적용하세요.

## 6. 남용 방지 한도(이미 켜져 있음)

공개 사이트는 0.5 CPU · 512 MB 서버 한 대와 1 GB 무료 DB로 돌아갑니다. 한 사람이 사이트를 멈추거나 DB를 채우지
못하게 `render.yaml`에 다음 한도를 넣어 두었습니다. 바꾸려면 `render.yaml`의 값을 고쳐 푸시하세요(대시보드
**Environment**에서만 바꾼 값은 다음 Blueprint 동기화 때 `render.yaml` 값으로 돌아갈 수 있습니다).

| 무엇 | 한도 | 설정(환경 변수) | 넘으면 |
|---|---|---|---|
| 무료 검진 — 사이트 전체 | 하루 50회(한국 시간 자정에 초기화) | `JETTAE_DAILY_FREE_SCAN_LIMIT` | `429` `free_scan_limit` |
| 무료 검진 — 접속 주소 하나 | 하루 5회(한국 시간) | `JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY` | `429` `free_scan_limit` |
| 무료 검진 시도 — 접속 주소 하나 | 1분 10회(실패한 시도 포함) | 고정 | `429` `rate_limited` |
| 동시에 도는 방문자 진단 | 1개(줄 세우지 않고 바로 거절) | `JETTAE_MAX_CONCURRENT_SCANS` | `503` `server_busy` |
| 업로드 한 번의 크기 | 2 MB | `JETTAE_MAX_UPLOAD_MB` | `413` |
| 런북 텍스트 길이(파일 하나, 검진 한 번의 합계 모두) | 50,000자 | `JETTAE_MAX_DOCUMENT_CHARS` | `422`, 핵심 런북만 골라 나눠 올리라는 안내 |
| 명부·당직표·카탈로그(무료 검진) | 각 1,000행 | 고정 | `422` |
| 화면 이벤트 — 접속 주소 하나 | 1분 60회(기본값) | `JETTAE_EVENTS_PER_CLIENT_PER_MINUTE` | `429` `rate_limited` |
| 화면 이벤트 — 사이트 전체 | 하루 20,000건(한국 시간) | `JETTAE_EVENTS_PER_DAY` | 응답은 `202` 그대로, 그날은 더 저장하지 않음 |
| 사전 신청 — 접속 주소 하나 | 1시간 10회(기본값) | `JETTAE_SIGNUPS_PER_CLIENT_PER_HOUR` | `429` `rate_limited` |

- 이 밖에 파일 자체를 부풀리는 공격(YAML 별칭 폭탄, 압축 폭탄 DOCX, 쪽수·글자가 부풀려진 PDF, 아주 깊은 HTML)은
  읽는 단계에서 바로 거절합니다(조정할 설정 없음).
- 검진 시간은 글자 수보다 빠르게 늘어납니다(데스크톱에서 41,000자 약 3초, 81,000자 12초). 0.5 CPU에서는 몇 배 더
  걸리고 그동안 다른 방문자는 `503`을 받으므로 글자 수 한도를 50,000자(A4 20쪽 안팎)로 작게 둡니다.
- 횟수는 서버 메모리에서 셉니다. 재배포·재시작하면 0부터 다시 셉니다.
- 접속 주소는 4-2에서 고른 곳(`JETTAE_CLIENT_IP_SOURCE`)에서 읽고, 횟수를 세는 데만 씁니다(로그·DB에 남기지 않음).
  회사 사무실처럼 여러 사람이 한 주소를 쓰면 함께 셉니다. 주소별 한도는 한 사람의 반복을 막는 장치이고, 마지막
  방어선은 주소와 상관없는 사이트 전체 하루 한도·이벤트 하루 상한·동시 진단 수입니다.
- 무료 검진 `429`가 자주 보이면 수요가 있다는 신호입니다. 서버 사양을 올린 뒤 한도를 올리세요(한도만 올리면 DB와
  CPU가 먼저 찹니다).
- 검진 결과는 7일 동안 남으므로 DB에는 많아야 350건(하루 50회 × 7일)이 쌓이고, 한 건은 올린 런북 크기에 따라
  수백 KB~수 MB입니다. 일주일에 한 번 대시보드의 `jettae-db` 화면(Metrics)에서 저장 공간 사용량을 보고, 1 GB의
  70%를 넘으면 `JETTAE_DAILY_FREE_SCAN_LIMIT`·`JETTAE_MAX_DOCUMENT_CHARS`를 낮추거나 DB 플랜을 올리세요. DB가 가득
  차면 사전 신청도 저장되지 않습니다.

## 7. 보관 기간과 삭제 요청(개인정보)

방문자에게는 사이트의 **개인정보 처리방침** `https://<앱 주소>/privacy`가 보입니다(수집 항목·목적·보유 기간,
Render(미국 회사)의 싱가포르 서버로의 국외 이전, 권리 행사 방법). 이 페이지가 약속한 대로 운영해야 합니다.

| 데이터 | 보관 | 지우는 방법 |
|---|---|---|
| 무료 검진 결과(워크스페이스 전체: 그래프·보고서·접근 키 해시 — 명부를 올렸다면 이름·직급·팀·역할·재직 상태·연락 수단 종류 포함) | 7일(`JETTAE_FREE_SCAN_RETENTION_DAYS`) | 서버가 스스로 지움(시작할 때와 6시간마다 정리) |
| 사전 신청(이메일·조직명·팀 규모·남긴 말) | 1년(`JETTAE_SIGNUP_RETENTION_DAYS`) | 서버가 스스로 지움(같은 정리 작업) · 요청 시 아래 명령 |
| 이벤트(이름·정해진 화면 경로·정해진 값 — 자유 글자 없음) | DB가 있는 동안 | — |
| 서버 로그(메서드·경로·상태·처리 시간 — IP 없음, 경로 속 이메일은 가려서 기록) | Render의 로그 보관 기간 | — |

- 자동 정리는 공개 모드(`JETTAE_AUTH_MODE=public`)에서만 돕니다. `render.yaml`이 공개 모드이므로 따로 할 일은
  없습니다(내 컴퓨터의 개발용 서버는 아무것도 지우지 않습니다).
- 서버 로그에 방문자 IP가 남지 않는 이유: uvicorn의 접속 로그는 꺼 두었고, 앱이 직접 남기는 요청 로그에는 주소를
  쓰지 않습니다. 삭제 요청처럼 경로에 이메일이 들어가는 요청도 로그에는 이메일을 가린 경로만 남습니다.

**삭제(동의 철회) 요청이 메일로 오면** — 처리방침에 "본인 확인 후 10일 이내"라고 약속했습니다.

1. 요청 메일이 신청한 그 이메일 주소에서 왔는지 확인합니다(다른 주소라면 신청한 주소로 확인 메일을 보냅니다).
2. 서버에서 지웁니다(`admin` 토큰). 첫 줄의 따옴표 안 `kim@example.com`을 그 사람의 이메일로 바꿉니다. 첫 줄이
   이메일을 주소에 넣을 수 있는 모양(`@` → `%40`, `+` → `%2B`)으로 바꿔 주므로 그대로 실행하면 됩니다. 대소문자는
   신경 쓰지 않아도 됩니다.

   ```powershell
   $email = [uri]::EscapeDataString("kim@example.com")
   curl.exe -f -X DELETE -H "Authorization: Bearer $token" "$app/api/v1/signups/$email"
   ```

   `{"deleted":true}`가 나오면 지운 것이고, `{"deleted":false}`면 서버에 그 주소로 된 신청이 없습니다(이미
   지워졌거나 다른 주소로 신청함). 감사 로그에는 이메일 대신 짧은 해시만 남습니다.
3. 내 컴퓨터에 내려받아 둔 CSV·JSON 사본과 메일 도구의 주소록에서도 지웁니다.
4. 처리했다고 답장합니다.

마케팅 수신만 거부하는 경우에는 신청을 지우지 않아도 됩니다. 메일 도구의 수신 거부 목록에 넣고, 다음에 명단을
받을 때 그 사람을 빼고 보내세요.

## 8. 무료 DB 만료 전에 꼭 할 일(만든 날로부터 30일)

Render의 무료 PostgreSQL은 **만든 날로부터 30일 뒤 만료되어 앱이 쓸 수 없게 되고**, 그 뒤 14일 안에 유료로
올리지 않으면 **데이터와 함께 삭제**됩니다(백업 없음, 용량 1 GB 고정). 만료되면 사이트가 사전 신청을 받지 못하고
서버도 제대로 뜨지 않습니다.

1. Blueprint로 배포한 날을 기준으로 달력에 **25일째** 알림을 걸어 둡니다.
2. 그날 수요 지표와 명단을 내려받습니다.

   ```powershell
   curl.exe -f -H "Authorization: Bearer $token" -o demand.json "$app/api/v1/metrics/demand"
   curl.exe -f -H "Authorization: Bearer $token" -o signups.csv "$app/api/v1/signups?format=csv"
   ```

3. 두 파일을 열어 내용이 있는지 확인합니다(`signups.csv`는 엑셀로, `demand.json`은 메모장으로).
4. 계속 운영하려면 만료 전에 대시보드 → `jettae-db`에서 플랜을 `0.1c-256mb`로 올리거나, `render.yaml`의
   데이터베이스 `plan`을 `0.1c-256mb`로 바꿔 푸시하세요. 데이터는 그대로 이어집니다. 베타를 끝낼 거라면 웹 서비스도
   함께 정지(Suspend)합니다.
5. 내려받은 `signups.csv`도 개인정보입니다. 각 줄의 `created_at`에서 1년이 지나면 그 줄을 지우세요(서버는 스스로
   지우지만 내 컴퓨터의 사본은 지우지 못합니다).

## 9. DB 외부 접속 차단(ipAllowList)

`render.yaml`의 `jettae-db`에는 `ipAllowList: []`(빈 목록)가 들어 있어 **인터넷에서 DB로 바로 붙는 접속을 모두
막습니다**. 웹 서비스는 같은 리전의 내부 주소로 붙으므로 영향이 없습니다. DB 비밀번호가 새더라도 밖에서는 붙을 수
없습니다.

내 컴퓨터에서 `psql`로 직접 봐야 할 때만 대시보드의 `jettae-db` 화면에서 접근 제어(Access Control) 목록에 내 IP를
잠깐 추가하고, 일이 끝나면 바로 지우세요. 평소의 명단·지표 확인은 위의 `curl.exe` 명령으로 충분합니다.

## 10. (선택) 내 도메인 연결

Render 대시보드 → 서비스 → **Settings → Custom Domains**에서 도메인(예: `www.jettae.kr`)을 추가하고, 도메인
업체의 DNS에 Render가 알려 주는 CNAME 레코드를 넣습니다. HTTPS 인증서는 Render가 자동으로 붙입니다.
도메인을 붙인 뒤에는 위 명령의 `$app`도 새 주소로 바꾸고, 새 주소로 4-2의 ②·⑤를 한 번 더 확인합니다(앞단
구성이 달라지면 방문자 주소가 실려 오는 헤더도 달라질 수 있습니다).

## 자주 묻는 것

| 증상 | 원인·해결 |
|---|---|
| 로그에 `오류: 환경 변수 설정이 올바르지 않습니다: JETTAE_API_TOKENS` | 토큰 JSON 형식 오류입니다(작은따옴표, 빠진 중괄호, 빈 값 등). 2단계의 모양 그대로, 따옴표는 큰따옴표(`"`)만 씁니다. |
| 로그에 `JETTAE_API_TOKENS의 1번째 토큰은(는) 공백 없이 16자 이상이어야 합니다.` | 토큰이 너무 짧거나 공백·줄바꿈이 섞였습니다. 2단계 명령으로 새로 만드세요. |
| 로그에 `JETTAE_API_TOKENS의 1번째 토큰의 설정이 올바르지 않습니다: 알 수 없는 역할입니다` | `role`은 `viewer`·`editor`·`lead`·`admin` 중 하나입니다. |
| `401` · `unauthorized` | 토큰이 틀렸거나 `Bearer ` 뒤에 공백·줄바꿈이 섞였습니다. 토큰을 다시 복사하고 4-1의 `$token` 줄을 다시 실행하세요. |
| `403` · `forbidden` | 그 토큰의 역할이 부족합니다. 사전 신청 명단·삭제·방문자 주소 확인은 `admin`, 수요 지표는 `lead` 이상입니다. |
| `429` · `free_scan_limit` | 오늘(한국 시간) 무료 검진 한도(사이트 전체 또는 그 접속 주소 몫)를 다 썼습니다. 6단계 표를 보세요. |
| `429` · `rate_limited` | 같은 접속 주소에서 요청이 너무 잦습니다. 응답의 `Retry-After`(초)만큼 기다리면 풀립니다. |
| 서로 다른 방문자들이 금방 `429`(`rate_limited`·`free_scan_limit`)를 받음 | 방문자 주소를 읽는 곳이 맞지 않아 모두 한 주소로 세어지고 있습니다. 4-2를 다시 하세요. |
| `503` · `server_busy` | 다른 방문자의 진단이 도는 중입니다. 잠시 후 다시 시도하면 됩니다. |
| PowerShell에서 `curl`이 이상한 오류를 냄 | PowerShell의 `curl`은 다른 명령입니다. 꼭 `curl.exe`라고 쓰세요. |
| 방문자가 접근 키를 잃어버림 | 키는 해시만 저장하므로 되살릴 수 없습니다. 무료 검진을 다시 하면 됩니다(옛 결과는 7일 뒤 지워짐). |
| 토큰을 바꾸고 싶음 | 서비스 → **Environment** → `JETTAE_API_TOKENS` 수정 → 저장(자동 재배포). 옛 토큰은 바로 무효가 됩니다. |
| 토큰이 샜을 수 있음 | 위처럼 즉시 새 토큰으로 바꾸고, Render 로그에서 그 사이 `/api/v1/signups` 요청이 있었는지 확인하세요. |
