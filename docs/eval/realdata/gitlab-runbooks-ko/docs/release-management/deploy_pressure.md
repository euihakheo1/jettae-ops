# 높은 배포 압력

## 배경

배포 압력은 아직 배포되지 않은 커밋의 수입니다.

Staging-canary (`gstg-cny`)는 배포 파이프라인이 패키지를 배포하는 첫 번째 환경입니다.
`gstg-cny`의 배포 압력이 높다면 배포 파이프라인에 문제가 있다는 뜻입니다.

## 문제 해결

다음은 `gstg-cny`의 배포 압력이 높아지는 일반적인 원인 몇 가지입니다.

1. 새 패키지가 생성되지 않음

   - 다음 위치에 있는 최신 커밋의 커밋 시간을 보고 최신 패키지가 언제 생성되었는지 확인하세요:
     <https://ops.gitlab.net/gitlab-org/release/metadata/-/commits/master>.

   - <https://ops.gitlab.net/gitlab-org/release/tools/-/settings/ci_cd#js-cicd-variables-settings>에서 `AUTO_DEPLOY_SCHEDULE` 변수를 확인하세요.
     이 변수에는 하루 중 새 패키지가 생성될 시각(UTC 기준)이 나열되어 있습니다.

   - 위의 두 가지 정보를 바탕으로, 더 새로운 패키지가 있었어야 했는지 판단하세요.

1. 새 배포 파이프라인이 생성되지 않음

   - 다음 위치의 배포 파이프라인을 보고 마지막 배포 파이프라인이 언제 생성되었는지 확인하세요:
   <https://ops.gitlab.net/gitlab-org/release/tools/-/pipelines?page=1&scope=all&username=gitlab-release-tools-bot&source=api>.

   - 빌드가 완료되어 배포할 수 있는 더 새로운 패키지가 있나요?

   - 최신 파이프라인이 `gprd-cny`에 배포를 완료하고
     빌드가 완료된 새 패키지가 있으면 새 배포 파이프라인이 자동으로 생성되어야 합니다.

1. gstg-cny로의 배포가 실패함

   - `gstg-cny`로의 최신 배포를 확인하세요.

## 알려진 상황

알려진 이유로 배포가 차단되어 있고 `gstg-cny`의 높은 배포 압력이 예상된 것이라면,
알림을 사일런스 처리할 수 있습니다([알림을 사일런스 처리하는 방법](../monitoring/alerts_manual.md#silencing)).

계획된 PCL이나 그 밖에 배포 차단이 계획된 시기에는 알림을 미리 사일런스 처리하여
불필요한 통지를 방지할 수 있습니다.
