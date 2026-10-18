# CDN으로 사용되어 높은 부하를 받는 워커

## 증상

* 워커 부하의 급격한 증가
  * ![워커 높은 부하 예시](img/workers-high-load.png)
* 연결 수 감소
  * ![낮은 HTTP 연결 수 예시](img/low-connections.png)
* 데이터베이스 부하 감소
  * ![낮은 데이터베이스 부하 예시](img/low-database-load.png)
* git http 프로세스 증가
  * ![많은 git http 프로세스 수 예시](img/high-http-git-processes.png)
* 이 모든 그래프를 찾을 수 있는 관련 대시보드
  * <http://dashboards.gitlab.net/dashboard/db/fleet-overview>

## 확인해 볼 수 있는 사항

* cat-file blob인 모든 http ssh 프로세스의 수를 셉니다

```
knife ssh 'role:<cluster-role>' "ps -U git -o cmd | grep 'cat-file blob' | grep -v grep | sort | uniq -c -u"
```

## 해결 방법

* 요청이 git 계층에 도달하지 않도록 프로젝트를 논리적으로 삭제된 상태로 설정합니다.

```
sudo gitlab-rails console
proj = Project.find_with_namespace('group/project_name') # the path without the .git
proj.pending_delete = true
proj.save
```

* 모두 종료합니다

```
knife ssh 'role:<cluster-role>' "for p in \$(ps -U git -o pid,cmd | grep <object-id> | grep -v grep | awk '{ print \$1 }'); do kill -9 \$p; done"
```

## 사후 점검

프로세스 수를 다시 세고 처음에 언급한 대시보드를 모니터링합니다
