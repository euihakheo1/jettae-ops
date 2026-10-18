## 확인 단계

1. 네트워크를 확인하고, [GitLab.com](https://gitlab.com)을 열어 봅니다. 본인 쪽에서 정상이라면, 네트워크 장애일 수밖에 없습니다.
1. [트리아지 대시보드](https://dashboards.gitlab.net/d/RZmbBr7mk/gitlab-triage?orgId=1)를 확인합니다.
1. [플릿 개요](https://dashboards.gitlab.net/dashboard/db/fleet-overview)를 확인합니다.
1. [데이터베이스 개요](https://dashboards.gitlab.net/d/000000144/postgresql-overview?orgId=1)를 확인합니다.
1. [pgbouncer 개요](https://dashboards.gitlab.net/d/PwlB97Jmk/pgbouncer-overview?orgId=1)를 확인합니다.

## 살펴볼 사항

- pgbouncer 커넥션의 리소스 고갈 여부를 살펴봅니다.
