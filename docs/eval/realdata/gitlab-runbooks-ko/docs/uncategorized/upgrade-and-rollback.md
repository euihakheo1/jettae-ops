# 애플리케이션 코드의 업그레이드와 롤백

## 개요

업그레이드와 롤백은 보통 [delivery 팀](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/gitlab-delivery/delivery/)이
GitLab에서 담당합니다. 일반적인 상황에서 애플리케이션 코드 업그레이드는 월요일부터
금요일까지, Delivery 팀이 지원할 수 있을 때 진행됩니다. 예외적인
상황에서는 주말에 배포하거나 롤백해야 할 수도 있습니다. 이러한
경우에는 GitLab.com에 상당한 영향을 줄 수 있으므로 MOC를 호출해야 합니다.

## 애플리케이션 업그레이드

애플리케이션 업그레이드는 GitLab Chatops에서 트리거됩니다. 업그레이드 동작 방식에 대한
자세한 내용은 다음
[업그레이드 관련 릴리스 문서](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/deploy/gitlab-com-deployer.md#creating-a-new-deployment-for-upgrading-gitlab)를 참조하세요.

## 애플리케이션 롤백

애플리케이션 롤백은 GitLab Chatops에서 트리거됩니다. 롤백 동작 방식에 대한
자세한 내용은 다음
[롤백 관련 릴리스 문서](https://gitlab.com/gitlab-org/release/docs/-/blob/master/runbooks/rollback-a-deployment.md)를 참조하세요.

**GitLab.com에서 애플리케이션을 롤백하기 전에 다음 단계를
반드시 수행해야 합니다**

- [ ] ~S1 / ~S2 이슈가 있는지, 또는
  GitLab.com을 사용할 수 없는 경우에는 google doc이 있는지 확인합니다
- [ ] IMOC를 호출하여 곧 롤백이 진행될 것임을 알립니다

```
/pd-moc We are initiating a rollback of GitLab.com, for more information see <Issue or doc link>
```

- [ ] DBRE 온콜을 호출하여 롤백이 데이터베이스에 미치는 영향을 검토하도록 합니다

```
/pd-db We are initiating a rollback of GitLab.com, for more information see <Issue or doc link>
```

- [ ] DBRE는 다음을 검토해야 합니다
  [롤백 시 db 고려 사항](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/deploy/gitlab-com-deployer.md#rollback-considerations-for-database-migrations)
  온콜 DBRE가 부재 중이거나 롤백의 데이터베이스 영향을 검토할 수 없는 경우,
  IMOC가 롤백을 승인했는지 확인합니다.
- [ ] delivery 팀과 협력하여 GitLab Chatops로 롤백을 시작합니다.
  자세한 내용은 다음
  [롤백 관련 릴리스 문서](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/deploy/gitlab-com-deployer.md#creating-a-new-deployment-for-rolling-back-gitlab)를 참조하세요.
