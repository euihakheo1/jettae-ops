# 온콜

이 문서는 대응자가 incident.io를 통해 에스컬레이션된 인시던트를 확인하는 방법과, 온콜 엔지니어가 incident.io를 통해 팀에 에스컬레이션하는 방법에 대한 단계를 설명합니다.

## 대응자 빠른 시작 가이드

- <b>이제 Infrastructure 팀으로의 모든 [수동 에스컬레이션](#manual-escalation)은 incident.io를 통해 이루어집니다 , 팀이 아직 PagerDuty의 온콜 로테이션에 있다면 , 알림에 대해서는 여전히 PagerDuty로 호출되므로 PagerDuty에서 호출을 확인해 주세요. 하지만 본인/본인 팀에 대한 모든 수동 에스컬레이션은 incident.io 온콜을 통해 이루어집니다.</b>
- 편의를 위해 모바일 기기에 [incident.io 앱](https://help.incident.io/articles/3472064049-get-started-as-an-on-call-responder)을 설치하세요
  - 앱 설치는 권장 사항이지만 필수는 아닙니다. 앱이 없어도 SMS와 전화로 호출을 받을 수 있습니다.
  - SAML로 로그인합니다. 이때 Okta를 사용합니다.
  - 알림이 작동하는지 테스트하려면:
    1. Okta 타일을 통해 incident.io를 엽니다.
    1. 왼쪽 상단의 :newtanuki: 메뉴에 있는 사용자 환경설정으로 이동합니다.
    1. `Contact methods` 섹션에는 각 방법 옆에 테스트 알림을 보내는 데 사용할 수 있는 종 아이콘이 있습니다.

> [!note]
> 어떤 이유로든 앱을 설치할 수 없다면, Incident IO가 내 전화번호로 전화/SMS를 보내도록 구성할 수 있습니다. 방해 금지 설정을 무시하도록 [인시던트 연락처 카드](https://app.incident.io/api/mobile/responder_vcard)를 추가하는 것이 좋을 수 있습니다.

- incident.io를 통해 호출을 받으면 incident.io 앱 , 또는 Slack을 통해 인시던트를 확인할 수 있으며 , 거기서 인시던트 slack 채널로 쉽게 이동할 수 있습니다

  ![인시던트 확인](img/ack.png)

- 수신된 호출을 보여주는 incident.app 모바일 앱의 다음 스크린샷을 참조하세요. 알림을 보려면 _Alerts_ 탭으로 전환하면 됩니다.

  ![모바일 애플리케이션을 통해](img/1_phone.jpg)
  ![모바일 애플리케이션을 통해](img/2_phone.jpg)
  ![모바일 애플리케이션을 통해](img/3_phone.jpeg)

- 인시던트 slack 채널에 들어가면 인시던트 트러블슈팅을 시작하고 , 인시던트 zoom 통화에 참여하고, 사용 가능한 명령으로 여러 작업을 수행할 수 있습니다.

### 온콜 알림 처리 프로세스

이 프로세스는 계속 반복 개선 중입니다. 프로세스는 다음과 같습니다:

1. 응답 시간 내에 호출을 확인 처리합니다. 즉, 15분 이내 , Customer Emergency On-Call Rotation의 경우 10분 이내
1. [프로덕션 프로젝트](https://gitlab.com/gitlab-com/gl-infra/production/-/issues/?sort=created_date&state=opened&first_page_size=100)의 활성 인시던트와 관련이 있는지 확인합니다
1. `/incident declare` 명령을 사용하여 인시던트를 선언할 수 있습니다. 모든 인시던트는 일반적으로 `#incidents` 채널에 공지됩니다. `Gitlab.com` 유형 인시던트의 공지만 보려면 `#incidents-dotcom`으로 이동하세요.
1. 인시던트를 선언하면 인시던트 Slack 채널과 Zoom 통화가 생성됩니다.
1. 인시던트 Slack 채널에서 `/incident field` 명령으로 해당 사용자 지정 필드를 업데이트하여 **Block Deployments**, **Feature Flags**, 또는 **Keep the issue confidential**을 선택할 수 있습니다

   ![사용자 지정 필드](img/custom.png)

1. [인시던트 관리 프로젝트](https://gitlab.com/gitlab-com/gl-infra/production) 이슈가 자동으로 생성되어 인시던트에 연결됩니다.
   - 자세한 단계는 [인시던트를 생성하는 방법](incident-management.md#how-to-manually-raise-an-incident)을 참조하세요.
1. 모든 S3 및 S4 인시던트에서는 EOC가 자동으로 인시던트 리드로 지정되고 , 모든 S1 및 S2 인시던트에서는 IMOC가 자동으로 리드로 지정됩니다 , 어떤 이유로 인시던트 리드를 변경하려면 `/incident lead` 명령을 사용하세요
1. [알림 작업](#working-on-the-alert)을 시작합니다

#### 알림 작업하기

알림을 처리할 때는 [incident.io](incident-management.md#navigating-through-an-incident)가 생성한 인시던트 전용 Slack 채널에서 협업합니다. 모든 조사 메모와 업데이트에는 인시던트의 Slack 채널을 사용하세요. 주요 정보는 :pushpin: 이모지를 사용하여 고정할 수 있으며, 이는 incident.io 대시보드와 GitLab 이슈에 자동으로 반영됩니다.

문서화해야 할 중요한 조사 세부 사항:

1. 스크린샷 및 증거: 스크린샷으로 정보를 제공하는 경우 각 스크린샷에 반드시 `Source` 링크를 추가하세요. 반드시 절대 시간을 사용하세요 - `Last x hours`나 `now-x hours`는 사용하지 마세요.
   1. Grafana에서는 `Share` > `Copy`를 클릭하여 링크를 가져옵니다. 이렇게 하면 링크가 선택한 시간에 고정됩니다. **참고**: `Shorten URL`은 작동하지 않습니다.
   1. Kibana에서는 `Share` > `Get Links` > `Short URL` 토글 > `Copy link`를 클릭합니다. 이렇게 하면 링크가 올바르게 작동하고 필터가 유지됩니다.
1. 실시간 협업이 필요할 때는 Zoom 통화를 사용하고, incident.io의 Scribe 기능을 활용하여 통화 논의 내용을 인시던트 채널에 바로 요약하세요.
1. 정기적인 상태 업데이트는 `/incident` 명령을 통해 incident.io의 업데이트 기능으로 게시할 수 있습니다.

Slack에서 `:boom:` 이모지를 사용하여 인시던트와 관련된 액션을 할당할 수 있습니다 , 액션을 할당해도 Gitlab 이슈나 MR은 생성되지 않습니다.

조사를 진행하면서 후속 조치가 필요한 모든 ~"corrective actions"s (레이블: `~corrective action`)를 추적하는 것이 중요합니다. 이를 `Investigation Thread`에 기록하거나, 참고용으로 별도의 `Corrective Actions` 스레드를 만드는 방법이 있습니다. 인시던트가 끝나면 EOC는 [사후 인시던트 책임](#post-incident-responsibilities)에 자세히 설명된 대로 ~"corrective actions"s에 대한 이슈를 생성해야 합니다.

## 수동 에스컬레이션

- 이 단계는 인시던트 중에 팀 또는 개인 , CMOC , IMOC에게 에스컬레이션하는 방법을 설명합니다.

- 인시던트 채널에서 팀 / 개인에게 에스컬레이션하려면 `/incident escalate` 명령을 사용하세요 , 그러면 에스컬레이션 양식 팝업이 표시됩니다

![에스컬레이션 양식](img/escalate.png)

- 드롭다운 메뉴에서 에스컬레이션할 온콜 팀 / 에스컬레이션할 개인을 선택하세요 , 대응자에게 보여주고 싶은 알림 메시지를 선택적으로 입력할 수 있습니다

- Security 호출 : 중간/높은 심각도 인시던트의 경우 [SEOC](https://handbook.gitlab.com/handbook/security/security-operations/sirt/engaging-security-on-call/#engage-the-security-engineer-on-call)를 참여시키는 방법을 참조하세요. 낮은 심각도 인시던트의 경우 [인시던트 심각도 표](https://handbook.gitlab.com/handbook/security/security-operations/sirt/engaging-security-on-call/#incident-severity)를 참조하여 적절한 조치 방향을 결정하세요

- Dev 호출 : #dev-escalation에 /devoncall incident-issue-url을 입력합니다. 핸드북

**_참고- woodhouse가 생성한 Slack 채널에서 에스컬레이션할 때는 Notification Message에 인시던트 slack 채널을 언급하세요 (( 권장하지 않음 ))_**

![Slack 채널 알림](img/notify.png)

### 인시던트 수명 주기

저희는 [incident.io](incident-management.md)를 사용하여 인시던트를 관리합니다. Slack에서 `/incident create`를 사용하여 인시던트를 생성하면 전용 Slack 채널과 GitLab 이슈가 모두 자동으로 생성됩니다. 수명 주기는 [여기](https://app.incident.io/gitlab/settings/lifecycle?tab=lifecycles)에 정의되어 있습니다. 인시던트에는 특히 주목할 만한 몇 가지 단계가 있습니다:

#### 활성

활성 인시던트는 다음을 통해 관리됩니다:

- incident.io가 생성한 인시던트 전용 Slack 채널
- [인시던트 관리 프로젝트](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/incident-management/-/issues)에 자동으로 생성된 GitLab 이슈
- incident.io 대시보드 - [인시던트 탐색하기](incident-management.md#navigating-through-an-incident) 참조

[현재 활성 인시던트 보기](https://app.incident.io/gitlab/incidents?incident_type%5Bone_of%5D=01JK9KV07AV2WZKHYZD1GTK5KB).

#### 모니터링

다음과 같은 경우 인시던트는 `monitoring` 상태로 간주됩니다 :

1. 문제가 (적어도 일시적으로) 완화되거나 해결되었지만 재발할 수 있는 경우.
1. 문제가 더 이상 활성 상태는 아니지만 알림이 해제되기를 아직 기다리고 있는 경우.

완화 상황을 반영하려면 인시던트 Slack 채널에서 `/incident` 명령을 사용하여 인시던트 상태를 업데이트하세요. 그러면 incident.io에서 인시던트 상태가 업데이트됩니다. 인시던트가 해결되면 인시던트 이슈는 별도로 종료해야 합니다.

[현재 완화된 인시던트 보기](https://app.incident.io/gitlab/incidents?incident_type%5Bone_of%5D=01JK9KV07AV2WZKHYZD1GTK5KB&status%5Bone_of%5D=01JH40E0GPQQZNK1DJHH2TCTEG)

#### 종료

인시던트는 완전히 처리되었을 때 해결된 것입니다. incident.io의 [사후 인시던트 워크플로](https://docs.incident.io/post-incident/post-incident-flow)를 사용하여 인시던트를 적절히 마무리하세요. 여기에는 [사후 인시던트 리뷰 작성](https://docs.incident.io/post-incident/postmortems-overview)이 포함될 수 있습니다. 인시던트 리뷰는 일반적으로 모든 S1 및 S2 인시던트에 대해, 또는 [GitLab의 인시던트 리뷰 프로세스](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/incident-review/#the-criteria-which-triggers-a-review)에 따라 인시던트 리뷰가 요청된 경우에 수행됩니다.

[현재 해결된 인시던트 보기](https://app.incident.io/gitlab/incidents?incident_type%5Bone_of%5D=01JK9KV07AV2WZKHYZD1GTK5KB&status%5Bone_of%5D=01JH40E0GPYDJ8B3AERA6RZSY8).

### 사후 인시던트 책임

- 후속 조치는 인시던트가 종료된 후 해야 할 일을 팀이 기록하는 방법입니다. 후속 조치는 임시 우회 방법을 되돌리는 것 같은 긴급한 작업부터 새 클라우드 제공업체로 이전하는 방법을 조사하는 것 같은 낮은 우선순위 작업까지 다양한 형태와 규모로 나타납니다. 각 후속 조치에 우선순위를 지정하여 incident.io 내에서 이러한 중요도를 기록할 수 있습니다.

- `/incident follow-up` 명령을 사용하여 후속 조치를 생성하세요 , 양식이 열리면 , 필수 항목인 제목을 입력하고 필요에 따라 본인 또는 다른 사람에게 할당하세요

**_참고 : 후속 조치는 Gitlab 이슈를 생성합니다 _**

![후속 조치](img/followup.png)

- 또는 Slack의 메시지에 :fast_forward:로 반응하여 후속 조치를 생성할 수도 있습니다 , 이렇게 하면 Gitlab 이슈가 생성됩니다

  ![모바일 애플리케이션을 통해](img/followup2.png)
  ![모바일 애플리케이션을 통해](img/gitlabissue.png)
