## 요약

이 문서는 GitLab.com의 재해 복구에 관한 정보 중 테스트, 검증, 그리고 복구를 가로막을 수 있는 현재의 미비점과 관련된 정보를 담고 있습니다.

[GitLab 백업](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/production/#backup-and-restore)은 데이터를 글로벌(멀티 리전) 오브젝트 스토리지에 저장함으로써 존 단위 및 리전 단위 장애 모두에 견딜 수 있도록 설계되었습니다.

SaaS를 위한 [DR 전략](https://internal-handbook.gitlab.io/handbook/engineering/disaster-recovery/)은 현재의 백업 전략을 기반으로 합니다:

- [WAL-G를 사용한 Postgresql 백업](../patroni/postgresql-backups-wale-walg.md)
- [GCP 디스크 스냅샷](gcp-snapshots.md)

복원 검증은 Postgresql 데이터베이스와 디스크 스냅샷 모두에 대해 CI 파이프라인에서 이루어집니다:

- [Postgresql 복원 테스트](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/database/disaster-recovery)
- [GitLab 프로덕션 스냅샷 복원](https://gitlab.com/gitlab-com/gl-infra/gitlab-restore/gitlab-production-snapshots)

## 복구 가이드

리전 또는 존 단위 장애(또는 성능 저하)가 의심되면 [복구 가이드](./recovery.md)를 꼼꼼히 읽어 주세요.

## 테스트

### 테스트 환경

스냅샷 복구 테스트에는 [`dr-testing`](https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/tree/main/environments/dr-testing) 환경을 사용할 수 있으며, 이 환경에는 Gitaly 스냅샷 복구를 포함한 다양한 복구 유형의 예시가 있습니다.

### 가용 영역에 대한 네트워크 트래픽 거부

인그레스와 이그레스 트래픽을 모두 차단하는 방화벽 규칙을 설정하여 존 장애를 시뮬레이션하는 데 도움이 되는 헬퍼 스크립트가 있으며, 현재 이 스크립트는 비프로덕션 환경에서 `us-east1-b` 및 `us-east1-d` 존에 대해 실행할 수 있습니다.
`us-east1-c` 존에는 [배포 및 콘솔 노드와 같은 SPOF](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/16251#us-east1-c-outage)가 있으므로, [존 장애 관련 핵심 작업을 추적하는 에픽](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/800)에서 이것들이 해결될 때까지 이 존에서는 테스트 실행을 피해야 합니다.

#### 방화벽 규칙 설정

**참고**: 이 스크립트는 주의해서 실행하세요! 비프로덕션 환경이라도 모든 변경은 [변경 관리](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/change-management/)를 거쳐야 합니다!

```
$ ./zone-denier -h
Usage: ./zone-denier [-e <environment> (gstg|pre) -a <action> (deny|allow) -z <zone> -d]

  -e : Environment to target, must be a non-prod env
  -a : deny or allow traffic for the specified zone
  -z : availability zone to target
  -d (optional): run in dry-run mode

Examples:

  # Use the dry-run option to see what infra will be denied
  ./zone-denier -e pre -z us-east1-b -a deny -d

  # Deny both ingress and egress traffic in us-east1-b in PreProd
  ./zone-denier -e pre -z us-east1-b -a deny

  # Revert the deny to allow traffic
  ./zone-denier -e pre -z us-east1-b -a allow
```

이 스크립트는 각 환경별로 알려진 SPOF의 정적 목록을 제외하도록 구성되어 있습니다.
