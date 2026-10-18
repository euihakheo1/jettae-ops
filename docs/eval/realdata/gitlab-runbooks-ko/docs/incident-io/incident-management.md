# GitLab 프로덕션 Incident.io 온보딩

이 문서는 incident.io의 기본 온보딩 단계를 안내합니다.

**@ahanselka의 incident.io 워크스루를 [여기](https://youtu.be/JeQu1UAxJE4)에서 시청하세요.**

## 트리아지 인시던트

- 트리아지 인시던트는 `#incidents-dotcom-triage` Slack 채널에 자동으로 생성됩니다
- 인시던트를 수락하거나 거절하려면 생성된 인시던트 채널에 참여하세요. 이 채널에서 인시던트를 "Accept"하거나, 열려 있는 다른 인시던트로 "Merge"하거나, 실제 인시던트가 아니라면 "Decline"할 수 있습니다.
- 인시던트가 수락되면 [`#incidents-dotcom`](https://gitlab.enterprise.slack.com/archives/incidents-dotcom) 및 [`#incidents`](https://gitlab.enterprise.slack.com/archives/incidents-dedicated)에 인시던트로 공지됩니다.

## 인시던트를 수동으로 생성하는 방법

- `/incident declare` 명령을 입력합니다.
- 인시던트를 선언하는 팝업이 열립니다.
- 인시던트 이름을 지정하고 인시던트 유형을 `Gitlab.com`으로 선택합니다.
- 해당 드롭다운 메뉴에서 `yes`를 선택하여 배포/피처 플래그를 차단할지 선택할 수 있습니다
- 드롭다운 메뉴에서 `Yes`를 선택하여 이슈를 비공개로 유지하도록 선택할 수도 있습니다

<details>
  <summary>클릭하여 이미지 펼치기</summary>
  ![인시던트 선언](img/declare.png)
</details>

## 소급 인시던트 생성

소급 인시던트를 사용하면 알림, Slack 공지, 호출을 전혀 발생시키지 않고 incident.io에서 인시던트의 날짜를 과거로 소급해 등록할 수 있습니다. 이는 이미 발생한 인시던트를 문서화할 때 유용합니다.

### Slack에서

- Slack에서 `/inc retro` 명령을 입력합니다.
- 소급 인시던트를 선언하는 팝업이 열립니다.
- 평소와 같이 인시던트 세부 정보(이름, 심각도, 인시던트 유형 등)를 입력합니다.

### 대시보드에서

1. [incident.io 대시보드](https://incident.io/)로 이동하여 **Declare Incident**를 클릭합니다.
2. **Active incident** 대신 **Retrospective incident** 탭을 선택합니다.
3. 평소와 같이 인시던트 세부 정보(이름, 심각도, 인시던트 유형 등)를 입력합니다.

### 기본 동작

기본적으로 소급 인시던트에는 다음 옵션이 설정됩니다:

- **Announce this retrospective incident** → No
- **Enter post-incident flow** → No

이는 다음을 의미합니다:

- Slack 채널 공지가 전송되지 않습니다.
- EOC나 다른 누구에게도 호출이나 알림이 발생하지 않습니다.
- 기본적으로 자동화된 워크플로가 트리거되지 않습니다. (현재 워크플로는 **Active** 또는 **Triage** 인시던트에서만 트리거되도록 구성되어 있습니다 — 이 동작에 의존한다면 여전히 그런지 확인하세요.)

필요한 경우 소급 인시던트를 선언할 때 공지나 사후 인시던트 플로를 선택적으로 활성화할 수 있습니다.

## 인시던트 탐색하기

- 인시던트를 생성하면 인시던트 Slack 채널로 리디렉션됩니다.

<details>
  <summary>클릭하여 이미지 펼치기</summary>
  ![인시던트 Slack 채널](img/incidentio-slackchannel.png)
</details>

- 인시던트를 생성할 때마다 [여기](https://gitlab.com/gitlab-com/gl-infra/production/-/issues/?sort=created_date&state=opened&first_page_size=100)에 GitLab 이슈가 생성되며, Zoom 인시던트 통화도 함께 생성됩니다.
- Slack에서 `/incident` 명령을 사용하거나 인시던트 대시보드를 통해 인시던트를 업데이트할 수 있습니다. 인시던트 대시보드로 이동하려면 `overview`를 클릭한 다음 `incident homepage`를 클릭합니다.

<details>
  <summary>클릭하여 이미지 펼치기</summary>
  ![인시던트 홈페이지](img/incidentio-dashboard.png)
</details>

- 대시보드는 인시던트의 개요를 잘 보여줍니다. 대시보드를 보려면 먼저 incident.io에 [로그인](#login-to-the-incidentio-dashboard)해야 합니다.
- Incident.io는 Slack에서 인시던트를 업데이트할 수 있는 다양한 명령을 제공합니다.

<details>
  <summary>클릭하여 이미지 펼치기</summary>
  ![인시던트 명령](img/incidentio-options.png)
</details>

- 인시던트 Slack 채널로 이동하여 디버깅을 시작할 수 있습니다.
- 자신의 slack 메시지에 :mega: 이모지로 반응하여 인시던트를 지켜보는 사람들에게 주기적으로 업데이트를 제공하세요. 이렇게 하면 공개 GitLab 인시던트 이슈도 업데이트됩니다. 나중에 분석할 때 기록해 두면 유용할 데이터는 `:pushpin:` 또는 `:star:` 이모지로 기록할 수 있습니다. `:pushpin:`을 사용하면 메시지가 GitLab 인시던트 이슈에 공개 댓글로 게시되고, `:star:`를 사용하면 내부 댓글로 게시됩니다. Slack 메시지에 첨부된 이미지는 GitLab 이슈에 게시되지 않습니다. 각 핀 활동은 대시보드에 타임스탬프가 기록되므로 타임스탬프를 추적하려 할 때 유용합니다. 고정된 메시지를 보려면 Post-Incident 탭을 보거나 Overview 탭에서 "All Activity"를 선택해야 합니다.
- incident.io가 시작한 Zoom 통화는 [Scribe](https://incident.io/changelog/scribe)를 사용하여 자동으로 요약됩니다. 통화에 Scribe를 두고 싶지 않다면 회의에서 제거하거나 다른 zoom 통화를 시작할 수 있습니다. 회의에서 scribe를 제거하려면 호스트 권한을 가져와야 합니다. 호스트 PIN은 1password의 `incidentio-svc@gitlab.com` 항목과, 인시던트가 활성 상태일 때 slack 채널 상단에 모두 나와 있습니다.
- 인시던트가 완화되거나 해결되면 사후 인시던트 워크플로를 진행하거나, 인시던트와 관련하여 작업할 후속 조치 항목을 첨부하도록 선택할 수 있습니다.

## incident.io 대시보드에 로그인

- [incident.io](https://incident.io/)로 이동합니다.
- Log in -> Sign-in with SAML SSO를 클릭합니다
- 축하합니다. incident.io 대시보드에 성공적으로 로그인했습니다.

로그인에 문제가 있으면 #it_help에 문의하세요

## 향후 개선 사항

저희 에픽 [Implement incident.io - Improvements](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1489)를 팔로우하세요
