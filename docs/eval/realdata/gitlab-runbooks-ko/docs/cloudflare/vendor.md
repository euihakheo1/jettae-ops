# CloudFlare 접근 및 사용

프로비저닝된 사용자는 Cloudflare에
`https://dash.cloudflare.com`에서 직접 접근할 수 있습니다.

## 접근 권한 프로비저닝 안내

1. 다음 Cloudflare 계정에 대해 Lumos로 전환했습니다

   - Gitlab
   - Runway
   - Gitlab Dedicated Production
   - Gitlab Dedicated Non-production

1. 이제 접근 요청은 셀프 서비스로 처리해야 하며, 요청은 계정 단위로 올립니다. 예를 들어 누군가 GitLab과 Runway 계정 모두에 접근하려면 Lumos 요청을 두 건 올려야 합니다.
1. 요청을 올릴 때 승인자에게 충분한 맥락을 제공할 수 있도록 코멘트 섹션에 관련 이슈와 맥락을 반드시 추가하세요
1. Lumos 접근 요청은 2단계 승인 절차입니다. 먼저 매니저의 승인이 필요하며, 그다음 접근을 요청하는 계정에 따라 다음과 같은 승인자 집합이 있습니다

   - Gitlab : 네트워크 및 인시던트 관리 팀
   - Runway : Runway 팀
   - Gitlab Dedicated Production : `@denhams , @nitinduttsharma and @o-lluch`
   - Gitlab Dedicated Non-production : `@denhams , @nitinduttsharma and @o-lluch`

1. 승인자는 적절한 사유를 들어 Lumos 요청을 취소할 수 있습니다.

### 디프로비저닝

1. 현재 Lumos를 통한 디프로비저닝은 사용할 수 없으며, 이 사용 사례에는 IT에 AR을 올려야 합니다
1. Cloudflare UI의 사용자 그룹에서 Cloudflare 멤버를 수동으로 제거하지 마세요

#### Gitlab Dedicated FedRAMP Cloudflare 계정 접근

**Gitlab Dedicated FedRAMP 샌드박스 Cloudflare 계정**에 대한 접근은 여전히 수동으로 관리됩니다 (곧 Lumos로 전환 예정)

**프로비저닝 단계:**

1. 사용자를 [`okta-cloudflare-users` google 그룹](https://groups.google.com/a/gitlab.com/g/okta-cloudflare-users/members)에 추가하세요
1. [Dedicated Cloudflare 조직](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/dedicated-organization-cloudflare/-/blob/main/environments/cloudflare_fedramp_sandbox/main.tfvars)에 MR을 생성하세요
**Gitlab Dedicated FedRAMP Cloudflare** 계정에 대한 접근은 별도의 [온보딩 이슈](https://gitlab.com/gitlab-com/gl-security/security-assurance/fedramp/fedramp-certification/-/blob/main/.gitlab/issue_templates/FedRAMP_Onboarding-New_Employees.md)를 통해 프로비저닝됩니다.

## 구성

### 사용자 지정 규칙 생성 또는 편집

[Cloudflare: 개요](./intro.md)

#### 트래픽 관리 (차단, 허용 목록 및 어뷰즈 완화)

[Cloudflare: 트래픽 관리](./managing-traffic.md)

### Workers 관리

[Cloudflare Workers](https://ops.gitlab.net/gitlab-com/gl-infra/terraform-modules/cloudflare_workers#configuration)

## Cloudflare 지원 받기

### 지원팀에 연락하기

### 연락처 전화번호

Cloudflare에 전화해야 할 경우를 대비해, 도움을 요청할 수 있는 이 번호들을 받았습니다.

해당 번호는 [내부 핸드북](https://internal.gitlab.com/handbook/engineering/infrastructure/vendor-contact/) 또는 내부 Cloudflare 지원 Slack 채널에 문서화되어 있습니다

## 기타 참고 자료

- 구현 에픽: <https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/94>
- 준비성 검토: <https://gitlab.com/gitlab-com/gl-infra/readiness/-/blob/master/cloudflare/README.md>
- 평가용 이슈 트래커: <https://gitlab.com/gitlab-com/gl-infra/cloudflare/-/issues>
- 진행 중인 Cloudflare 에픽: <https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1131>
- 한도 관리: <https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/rate-limiting/managing-limits/>
- Cloudflare terraform 구성: <https://gitlab.com/gitlab-com/gl-infra/terraform-modules/cloudflare>
