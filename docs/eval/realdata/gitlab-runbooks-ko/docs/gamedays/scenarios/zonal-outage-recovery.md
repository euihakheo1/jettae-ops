# 존 전체 장애 복구 절차

## 실험

실험 단계는 change_zonal_recovery [템플릿](https://gitlab.com/gitlab-com/gl-infra/production/-/blob/master/.gitlab/issue_templates/change_zonal_recovery.md)에 포함되어 있습니다

## 안건

- [카나리 단계](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/deploy/canary.md#overview)를 이해하고 있으면 유용합니다. 게임데이 동안에는 카나리를 비활성화하며, 이로 인해 배포에 일부 지장이 생길 수 있습니다.

## 절차

- 다음 명령을 실행하여 slack의 [#production](https://gitlab.enterprise.slack.com/archives/C101F3796) 채널에서 변경 이슈를 선언합니다
  - `/change declare`를 실행하고 `change_zonal_recovery` 템플릿을 선택하여 게임데이 변경 이슈를 생성합니다.
