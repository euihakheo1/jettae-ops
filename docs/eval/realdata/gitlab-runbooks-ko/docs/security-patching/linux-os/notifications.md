# 패치 알림

TOC

## patching-notifier

[patching-notifier](https://gitlab.com/gitlab-com/gl-infra/ops-team/toolkit/patching-notifier) 프로젝트는 우리 인프라 전반에서 탐지된 보안 문제에 대해 GitLab 이슈를 생성하는 역할을 합니다. 설정과 의도된 동작에 대한 더 자세한 개요는 프로젝트의 [README](https://gitlab.com/gitlab-com/gl-infra/ops-team/toolkit/patching-notifier/-/blob/main/README.md)를 참고하세요.

### 파이프라인

ops.gitlab.net에는 patching-notifier의 주요 기능을 수행하기 위해 정기적으로 실행되는 [예약된 파이프라인](https://ops.gitlab.net/gitlab-com/gl-infra/ops-team/toolkit/patching-notifier/-/pipeline_schedules)이 두 개 있습니다.

- `refresh_cache`는 Wiz에서 모든 보안 취약점 정보를 다운로드하고 그 결과를 JSON 형식으로 디스크에 저장하는 역할을 합니다. 이 JSON 파일은 GitLab에 의해 캐시되며 다른 모든 파이프라인과 공유됩니다. 이 작업은 6시간마다 실행되며, Wiz의 rate limit에 걸리지 않도록 하기 위한 것입니다.
- `run`은 patching-notifier 내에서 `patch-notification` 명령을 실행하는 역할을 합니다. 이 파이프라인은 GitLab 이슈를 생성하기 위해 [설정된 서비스](https://gitlab.com/gitlab-com/gl-infra/ops-team/toolkit/patching-notifier/-/blob/59e6cbf25b2c171392e4cb3d28d181389a421fb7/.gitlab-ci.yml#L87-96)마다 병렬 작업을 실행합니다. 이 파이프라인은 매시간 실행되며, `refresh_cache` 작업에서 생성된 Wiz 캐시가 있으면 이를 사용합니다. **참고**: Wiz 캐시가 없으면 `patch-notification` 명령이 자동으로 Wiz에서 모든 취약점 정보를 다운로드하려고 시도하며, 이로 인해 rate limit에 걸릴 수 있습니다.

### 서비스 추가

서비스를 추가할 때 따라야 할 일반적인 단계는 다음과 같습니다:

- [gl-infra](https://gitlab.com/groups/gitlab-com/gl-infra/-/labels) GitLab 그룹 아래에 `Service::` 네임스페이스 레이블이 있는지 확인하세요. 이것이 되어 있지 않으면 patching-notifier가 특정 서비스에 대해 이미 이슈를 생성했는지 알 수 없어, 스팸성 이슈가 생성됩니다.
- 프로젝트의 README에 설명된 대로 서비스 설정을 추가하세요. 설정은 `configs/service-config.yml`에 있습니다.
  - `label`을 설정할 때, 이는 생성한 `Service::` 레이블을 가리키며, 레이블 네임스페이스는 생략합니다.
  - 작업이 실행될 때 service-config.yml은 저장소의 main 브랜치에서 직접 다운로드됩니다. 따라서 서비스를 추가하거나 수정할 때 컨테이너 이미지를 다시 빌드하고 새 릴리스를 만들 필요가 없습니다.
- `.gitlab-ci.yaml`의 작업 매트릭스 목록에 새 서비스의 이름을 추가하세요.
