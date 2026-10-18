# GCP 지원 및 Rackspace로 도움 받기

**온콜 엔지니어 - 이제 지원 티켓은 GCP 콘솔에서 Google에 먼저 문의합니다**

1. GCP 이슈에 대한 지원 케이스는 [Google 지원 케이스 시스템](https://console.cloud.google.com/support/cases?organizationId=769164969568&supportedpurview=project)으로 접수합니다. 자세한 내용과 추가 에스컬레이션 경로는 [벤더 트래커의 readme](https://gitlab.com/gitlab-com/gl-infra/google-gitlab-tracker)를 참조하세요.
1. GCP 청구는 Rackspace를 통해 처리합니다.  청구 관련 문의는 Rackspace 포털에서 열 수 있습니다.

## 지원 포털 및 번호

Google:

* GCP에서 발생하고 있을 수 있는 이슈를 문의하고 지원을 요청할 수 있는 전용 slack 채널도 있습니다: [#ext-google-cloud](https://gitlab.enterprise.slack.com/archives/C01KPV0V3SM)

Rackspace:

* 포털 링크 <https://mycloud.rackspace.com>
* SRE 매니저와 인프라 애널리스트는 rackspace 포털에 접근할 수 있습니다

## GitLab 이슈 트래커

**Google 벤더 트래커**:<https://gitlab.com/gitlab-com/gl-infra/google-gitlab-tracker>

**Rackspace 벤더 트래커**: <https://gitlab.com/gitlab-com/gl-infra/rackspace/-/issues>

## 연락처

Google:

* [벤더 트래커의 readme](https://gitlab.com/gitlab-com/gl-infra/google-gitlab-tracker)를 참조하세요. 벤더와의 기밀 소통 때문에 이 프로젝트는 비공개입니다.

### Rackspace에서 지원 티켓 만들기 (청구 관련 문의 전용)

Rackspace Ticketing System, 프로젝트 ID: mgcp-1173105-ticket-escalation을 선택합니다.  이렇게 하면 심각도를 선택할 수 있습니다.

간단한 배경을 설명하자면, Rackspace 측의 자동화 요구 때문에 gitlab-production에 서비스 계정을 만들어야 했는데, 보안 관점에서 우리는 이를 받아들일 수 없었습니다.  이 임시 프로젝트는 티켓의 심각도를 선택할 수 있게 해 주는 우회 방법이며, 그곳의 우리 지원 담당자가 지켜보게 됩니다.

GitLab-production 프로젝트는 우리의 GitLab.com SaaS 애플리케이션 프로젝트이며, 그곳의 케이스는 가장 높은 우선순위를 갖습니다.

Rackspace의 티켓 SLA는 티켓 응답까지 4 hours입니다.
