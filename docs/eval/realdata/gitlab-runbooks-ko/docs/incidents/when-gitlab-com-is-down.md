# GitLab.com이 다운되었을 때

GitLab.com이 다운되면 인시던트를 열거나, 구성을 변경하거나, 변경 사항을 배포하지 못할 수 있습니다.

아래에서는 다음과 같이 지칭합니다:

- `ops`는 ops.gitlab.net을 의미하며, .com을 사용할 수 없을 때에도 프로젝트를 사용할 수 있도록 프로젝트를 미러링하는 곳입니다.
- `canonical`은 gitlab.com을 의미하며, .com이 정상일 때 코드 변경 사항을 푸시하는 곳입니다.

## Rails 애플리케이션 코드 업데이트 또는 되돌리기

GitLab.com이 완전히 다운되면 변경 사항을 푸시하거나 롤백하기 어려울 수 있습니다. 현재 코드 수정을 적용할 수 있는 유일한 방법은 [post-deployment patcher](https://ops.gitlab.net/gitlab-org/release/docs/-/blob/master/general/deploy/post-deployment-patches.md)를 사용하는 것입니다.

## 구성 변경하기

애플리케이션 구성 변경은 [chef-repo](https://ops.gitlab.net/gitlab-com/gl-infra/chef-repo) 또는 [k8s-workloads/gitlab-com](https://ops.gitlab.net/gitlab-com/gl-infra/k8s-workloads/gitlab-com)의 `ops.gitlab.net` 미러에서 CI 파이프라인으로 변경 사항을 적용하여 수행합니다.

### 리모트를 업데이트하고 ops.gitlab.net에서 MR 생성하기

이 프로젝트들의 canonical 소스를 사용할 수 없을 때는 미러에 직접 푸시하고 MR을 생성해야 합니다.

- `ops`의 프로젝트에 MR이 활성화되어 있는지 확인합니다:
  - <https://ops.gitlab.net/gitlab-com/gl-infra/chef-repo>
  - <https://ops.gitlab.net/gitlab-com/gl-infra/k8s-workloads/gitlab-com>
- `ops`에 대한 리모트가 있는지 확인합니다:
  - chef-repo: `git remote add ops git@ops.gitlab.net:gitlab-com/gl-infra/chef-repo.git`
  - k8s-workloads/gitlab-com: `git remote add ops git@ops.gitlab.net:gitlab-com/gl-infra/k8s-workloads/gitlab-com.git`
- 새 브랜치를 만들고 `ops`에 변경 사항을 푸시합니다:

```
git checkout -b <name>
# Make changes
git commit -m ' ... '
git push ops
```

MR이 병합되면 평소처럼 `ops` CI 파이프라인에서 적용합니다.

### gitlab.com을 사용할 수 있게 되면 변경 사항을 canonical에 반영하기

`ops`가 `canonical`보다 앞서게 되는 즉시 미러링이 작동하지 않게 됩니다. 변경 사항을 반영하려면 새 MR을 생성합니다:

```
git checkout -b update-canonical
git push origin master
```

`canonical`에서 MR을 병합하고, 병합 후 저장소 설정에서 저장소 미러링이 여전히 작동하는지 확인할 수 있습니다.

## 인시던트 중 공유 문서

GitLab.com에 더해 Incident.io도 다운된 경우, 현재 상태, 조치 항목, 메모를 위한 임시 장소로 Google Doc 사용을 고려하세요.

- <https://drive.google.com/>로 이동합니다
- 새 Google Doc을 만듭니다
- 오른쪽 상단의 "Share"를 클릭합니다
- 모달의 "Get link" 섹션에서 "Change link to GitLab"을 클릭하여
  회사 전체와 문서를 공유할 수 있게 합니다.
- "Anyone with the link in GitLab" 권한을 "Editor"로 변경합니다
- 완료를 클릭합니다.
- Slack에 문서 링크를 게시합니다
- 행운을 빕니다!
