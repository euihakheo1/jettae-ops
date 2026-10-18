# 프라이머리 무응답 또는 Leader 없음: 데이터 손실 없이 복구하기

**다음 경우에 사용하세요:** 클러스터에 `Leader`가 없거나, 프라이머리에 SSH로 접근할 수 없거나 프라이머리의 메트릭
스크레이프가 실패하는 경우. **먼저 페일오버하지 마세요.**

페일오버는 레플리카가 끝내 받지 못한 쓰기가 *없는* 상태로 그 레플리카를 승격합니다. 기존 프라이머리의 Postgres가
아직 실행 중이라면(예를 들어 OS 디스크나 Patroni만 멈춘 경우), 기존 프라이머리가
**가장 완전한 데이터 사본**입니다. 이 런북은 기존 프라이머리를 **제자리에서** 복구하므로 커밋된 쓰기가 손실되지 않습니다.

> **이것은 일반적인 장애 상황이 아닙니다.** 보통은 Patroni와 Consul이 장애가 난 프라이머리를
> 자체적으로 처리합니다: 프라이머리는 Consul에서 리더 락을 갱신하고, 노드, Postgres 또는 Patroni가
> 다운되면 락이 만료되어 Patroni가 가장 최신 상태의 레플리카를 승격합니다. 이것이 올바른 결과인 것은
> 기존 프라이머리가 쓰기 수신을 멈춘 경우입니다. 이 런북은 드물게 발생하는 **반쯤 죽은** 프라이머리를 위한 것입니다:
> 프라이머리의 Patroni 또는 Consul 에이전트가 멈춰서(예를 들어 OS 디스크가 멈춘 상태에서) 리더
> 락을 잃었지만, Postgres는 계속 실행되면서 레플리카가 더 이상 받지 못하는 쓰기를 커밋하고 있는 경우입니다.

제한 시간: 평가 ≤ 10 min.

## 시작하기 전에: 워치독이 활성화되어 있나요?

클러스터 멤버 중 아무 곳에서나(설정은 클러스터 전체에서 동일합니다):

```sh
ls -l /dev/watchdog; lsmod | grep softdog; sudo grep -A3 '^watchdog:' /var/opt/gitlab/patroni/patroni.yml
```

- **비활성** (디바이스 없음, 모듈 없음, `watchdog:` 블록 없음): 이 런북을 따르세요. 프라이머리를 자동으로
  펜싱해 주는 것은 없습니다. 직접 해야 합니다.
- **활성:** 워치독은 리더 락을 더 이상 갱신할 수 없는 프라이머리를, 마지막 갱신 성공 후 Patroni TTL에서
  `safety_margin`을 뺀 시간이 지났을 때 리셋합니다(현재 설정으로는 85 s). 이는
  락이 만료되기 직전으로, 이때 레플리카는 보통 아직 스트리밍 중이며, 그 후 레플리카가
  여러분이 일시 중지하기도 전에 자동으로 승격됩니다. 테스트에서는 그 순간 처리 중이던 쓰기만
  위험에 노출되었습니다. [5단계](#step-5-after-recovery)로 이동하고, 기존 프라이머리의 데이터 디스크는 보존하세요.

## 1단계: chef 비활성화 후 Patroni 일시 중지

먼저 **정상** 멤버에서 chef를 비활성화합니다. chef가 실행될 때마다 클러스터의 Patroni
설정이 role의 설정으로 교체되며, 이로 인해 **일시 중지가 해제됩니다**. 멈춘 프라이머리는 제외하세요: `knife ssh`가
그 노드에서 멈춰 버립니다.

```sh
knife ssh 'roles:<cluster-role> AND NOT name:<primary-fqdn>' 'sudo chef-client-disable "INC-xxxx: paused for recovery"'
```

그런 다음 아무 정상 멤버에서 Patroni를 일시 중지합니다. 일시 중지하면 어떤 레플리카도 승격되지 않습니다. Postgres는 전혀 건드리지 않습니다.

```sh
sudo gitlab-patronictl pause --wait
sudo gitlab-patronictl list | grep 'Maintenance mode'
```

> 일시 중지 중에는 **어떤 이유로도 자동 페일오버가 일어나지 않으며**, 워치독도 비활성 상태입니다.
> 일시 중지는 평가와 복구에 걸리는 시간만큼만 짧게 유지하세요.

## 2단계: 평가 (5분)

Grafana 쿼리는 데이터소스 `mimir-gitlab-gprd`를 사용하며 **main** 클러스터 기준으로 작성되어 있습니다. 다른
클러스터의 경우 `type` (`patroni-ci`, `patroni-sec`, `patroni-registry`)을 변경하고,
PgBouncer 쿼리에서는 `fqdn`을 해당 클러스터의 PgBouncer 호스트로 변경하세요. gstg의 경우 `env="gstg"`와
`mimir-gitlab-gstg`를 사용하세요.

점검 항목을 **한꺼번에** 열거나([Explore에서 전체 보기, main, 최근 6h](https://dashboards.gitlab.net/goto/ffzngg4vfe2o0b?orgId=1)),
표에 있는 링크로 **하나씩** 여세요.

| # | 질문 | 점검 | 해석 방법 |
| --- | --- | --- | --- |
| 0 | Consul이 프라이머리를 잃었나요? | 아무 노드에서: `consul members \| grep -E 'failed\|left'` 및 `consul kv get service/<scope>/leader` (`<scope>`는 `patroni.yml`의 `scope:`) | 장애 상태의 프라이머리와 함께 리더 키가 없다면 락이 만료되었다는 뜻입니다. 그래서 Leader가 없는 것입니다. 보통 이것이 첫 번째 신호입니다. |
| 1 | 레플리카가 WAL을 받고 있나요? | [Grafana](https://dashboards.gitlab.net/goto/cfznhkhnxnw8wb?orgId=1): `count(pg_stat_wal_receiver_status{env="gprd", type="patroni"}) or vector(0)` 대 `count(pg_replication_is_replica{env="gprd", type="patroni"} == 1)`. 레플리카에서: `select status, sender_host, flushed_lsn, last_msg_receipt_time from pg_stat_wal_receiver;` | 분리된 레플리카의 시리즈는 사라지므로 첫 번째 카운트가 두 번째보다 작아집니다(또는 0이 됩니다). 레플리카에서 행이 없으면 분리된 것입니다. **이때부터 쓰기가 프라이머리에만 존재하기 시작합니다.** |
| 2 | 기존 프라이머리가 아직 쓰기를 받고 있나요? | **직접(권장)**, 아무 레플리카에서, 10 s 간격으로 두 번: `sudo gitlab-psql -h <primary-fqdn> -Atc "select now(), pg_current_wal_lsn()"`. 프라이머리에 SSH할 필요는 없습니다. **대체 방법(불완전):** [Grafana](https://dashboards.gitlab.net/goto/dfznhkl7ek83kb?orgId=1) `sum(rate(pgbouncer_stats_sql_transactions_pooled_total{env="gprd", fqdn=~"pgbouncer-0[0-9]-db-gprd.*", database="gitlabhq_production"}[2m]))`는 web과 API만 포함합니다. | LSN이 움직이면 프라이머리가 아직 쓰고 있는 것입니다. PgBouncer 처리율이 낮다고 해서 쓰기가 멈췄다는 뜻은 **아닙니다**. **확인할 수 없다면 쓰기가 계속되고 있다고 가정하세요.** |
| 3 | 레플리카가 얼마나 뒤처져 있나요? | [Grafana](https://dashboards.gitlab.net/goto/ffznhkp6y5atcb?orgId=1): `max by (fqdn) (time() - pg_stat_wal_receiver_last_msg_receipt_time{env="gprd", type="patroni"})`. 각 레플리카에서: `select pg_last_wal_receive_lsn(), pg_last_xact_replay_timestamp();` | 프라이머리에만 존재하는 양. Prometheus로 바이트 지연을 계산하지 마세요. 스크레이프 타이밍 때문에 신뢰할 수 없습니다. |
| 4 | WAL이 GCS에 도달하고 있나요? | [Grafana](https://dashboards.gitlab.net/goto/afznhkrwm6dj4c?orgId=1): `max by (fqdn) (pg_archiver_pending_wal_count{env="gprd", type="patroni"})` | 보통 < 20이며, 수천이면 아카이빙이 멈춘 것이므로 **어떤 레플리카도 아카이브에서 따라잡을 수 없습니다**. |
| 5 | 프라이머리의 Patroni가 반쯤 죽은 상태인가요? | `curl -s --max-time 3 http://<primary>:8009/patroni \| jq '{state, role, lock_age_s: ((now\|floor) - .dcs_last_seen)}'`. TTL은 `sudo gitlab-patronictl show-config \| grep ttl`에 있습니다 | `role`이 `master`/`primary`이고 `lock_age_s`가 TTL **미만**: 정상. TTL **초과**: **반쯤 죽은 상태**. 쓰기를 하고 있고, 락을 잃었으며, 아무것도 이를 펜싱하지 않습니다. 타임아웃이 나면 Patroni가 죽었거나 노드가 멈춘 것입니다. 점검 2에 의존하세요. |

인시던트에 기록하세요: 레플리카가 분리된 시점, 프라이머리가 아직 쓰고 있는지 여부, 레플리카가 얼마나 뒤처져 있는지.

## 3단계: 기존 프라이머리를 제자리에서 복구

1. **가능하면 증거를 저장하세요.** 리셋하면 디스크에 기록되지 않은 내용은 사라집니다:
   `gcloud compute instances get-serial-port-output <vm> --project=<project> --zone=<zone> > <vm>-serial.log`
2. **VM을 리셋하세요.** `gcloud compute instances reset <vm> --project=<project> --zone=<zone>`를 사용하거나
   GCP 콘솔을 사용합니다. 또는 노드가 아직 응답한다면 Postgres를 재시작하세요. 커밋된 트랜잭션은
   데이터 디스크의 WAL에 안전하게 남아 있습니다. 일시 중지 중에는 Postgres가 멈추면 Patroni가 리더 락을
   제거하지만, **어떤 레플리카도 승격되지 않습니다**.

   VM을 리셋하는 대신 응답하는 노드에서 Postgres를 재시작했다면 항목 3과 4를 건너뛰세요:
   노드에서 chef를 비활성화하고(항목 5), Postgres가 이미 실행 중이 아니라면 수동으로 시작한 다음(항목 6),
   항목 7부터 계속 진행하세요.
3. **노드가 돌아올 때까지 기다리세요(리셋 시점부터 최대 10분).** 워크스테이션에서
   각각 별도의 터미널로:

   ```sh
   # VM state: expect RUNNING
   gcloud compute instances describe <vm> --project=<project> --zone=<zone> --format='value(status)'
   # Boot progress, live (Ctrl-C to stop)
   gcloud compute instances tail-serial-port-output <vm> --project=<project> --zone=<zone>
   # Or scan for trouble: hung tasks, disk errors, emergency mode
   gcloud compute instances get-serial-port-output <vm> --project=<project> --zone=<zone> \
     | grep -iE 'blocked for more than|hung_task|I/O error|EXT4-fs (error|warning)|nvme.*(timeout|reset|error)|emergency mode'
   ```

   시리얼 출력에는 syslog도 함께 나오므로 노이즈가 많고, 롤링 버퍼입니다: 리셋 후에는
   새 부팅 메시지가 끝부분에 있습니다. 리셋 이전의 매치가 아직 남아 있을 수 있으므로
   타임스탬프를 확인하세요.

   다른 아무 클러스터 멤버에서 Consul 에이전트가 돌아오는지 지켜보세요(`alive`가 예상됨).
   `<primary-short-name>`은 프라이머리의 `hostname -s`입니다:

   ```sh
   watch -n 5 'consul members | grep <primary-short-name>'
   ```

   GCP 콘솔에서 VM의 디스크 I/O를 지켜볼 수도 있습니다(VM → Observability → Disk).
   SSH가 되면 노드가 돌아온 것입니다: `ssh -o ConnectTimeout=10 <primary-fqdn> true`.

   - **10분 이내에 돌아온 경우:** 다음 단계로 계속 진행하세요.
   - **10분이 지나도 돌아오지 않는 경우:** [노드가 돌아오지 않는 경우](#step-4-if-the-node-doesnt-come-back)로 이동하세요.
4. **Postgres를 시작하기 전에 노드가 정상인지 확인하세요** (노드에서):

   ```sh
   uptime                                    # confirms it rebooted
   findmnt /var/opt/gitlab                   # data disk mounted, rw
   df -h /var/opt/gitlab
   sudo dmesg -T | grep -iE 'blocked for more than|I/O error|EXT4-fs (error|warning)|nvme.*(timeout|reset|error)'   # expect nothing
   sudo journalctl -b -p err --no-pager | tail -30
   systemctl is-active consul patroni
   sudo tail -5 /var/log/gitlab/patroni/patroni.log     # expect "PAUSE: postgres is not running"
   sudo -u gitlab-psql /usr/lib/postgresql/<ver>/bin/pg_ctl -D /var/opt/gitlab/postgresql/data<ver> status   # expect "no server running"
   ```

   다음 경우에는 Postgres를 시작하지 마세요: 데이터 디스크가 읽기-쓰기로 마운트되어 있지 않거나, 커널이 아직도
   hung task 또는 디스크 오류를 보고하는 경우. 대신 [노드가 돌아오지 않는 경우](#step-4-if-the-node-doesnt-come-back)로 이동하세요.
5. **Postgres를 시작하기 전에 노드에서 chef를 비활성화하세요.** chef 실행은 일시 중지를 해제하지만,
   이 노드가 `running`으로 표시된 이후에만 그렇습니다. 따라서 Postgres를 수동으로 시작한 후 chef가 실행되면
   이 노드가 Leader가 되기 전에 클러스터의 일시 중지가 해제되어 레플리카가 승격될 수 있습니다. chef를 비활성화한 다음,
   부팅 이후의 chef 실행 내역을 확인하세요:

   ```sh
   sudo chef-client-disable "INC-xxxx: paused for recovery"
   sudo chef-client-is-enabled
   sudo journalctl -b -u chef-client --no-pager | grep -E 'Starting Cinc Client|Cinc Client finished' | tail -4
   ```

   출력이 없으면 부팅 이후 chef가 실행되지 않은 것입니다. 마지막 줄이 `Starting Cinc Client`이고 그 뒤에
   `Cinc Client finished`가 없다면 실행이 진행 중인 것입니다: Postgres를 시작하기 전에 실행이 끝날 때까지 기다리세요.
6. **Postgres를 수동으로 시작하세요.** 일시 중지 중에는 Patroni가 Postgres를 시작하지 않습니다:

   ```sh
   sudo -u gitlab-psql /usr/lib/postgresql/<ver>/bin/pg_ctl -D /var/opt/gitlab/postgresql/data<ver> start
   ```

   크래시 후에는 `pg_ctl: another server might be running`이 나오는 것이 예상된 동작입니다(오래된 `postmaster.pid`).
   Postgres 로그에서 크래시 복구를 지켜보세요. 바쁜 프라이머리에서는 몇 분이 걸릴 수 있습니다:

   ```sh
   sudo tail -F /var/log/gitlab/postgresql/postgresql.csv | grep -a --line-buffered -E \
     'was interrupted|redo starts|redo in progress|redo done|ready to accept connections|FATAL|PANIC'
   ```

   `database system was interrupted`가 나오고, 이어서 `redo starts at …`, 복구가 길면 `redo in progress`가 매
   10초마다, 그리고 `redo done at …`, 마지막으로 `database system is ready to accept connections`가 나와야 합니다.
7. **같은 타임라인에서 Leader인지 확인하세요:**

   ```sh
   grep -E "PAUSE: acquired session lock|PAUSE: no action" /var/log/gitlab/patroni/patroni.log | tail -3
   sudo gitlab-patronictl list      # the old primary = Leader, TL unchanged
   ```

   재개하기 **전에** 레플리카가 다시 이 노드로부터 스트리밍을 시작할 수 있습니다. 이는 예상된 동작입니다.
8. **그런 다음에만 재개하고**, 모든 멤버에서 chef를 다시 활성화하세요:

   ```sh
   sudo gitlab-patronictl resume --wait
   knife ssh 'roles:<cluster-role>' 'sudo chef-client-enable'
   ```

> ⚠️ **Leader가 없는 동안에는 절대 재개하지 마세요.** 레플리카가 즉시 승격되며, 기존 프라이머리에만
> 있는 모든 쓰기가 손실됩니다.

## 4단계: 노드가 돌아오지 않는 경우

**일시 중지 상태를 유지하세요.** 클러스터가 일시 중지된 동안에는 어떤 레플리카도 승격되지 않으며 데이터는 기존
프라이머리의 디스크에 남아 있습니다. 문제를 해결하는 동안 아무것도 손실되지 않습니다.

- 두 번째 DBRE를 합류시키세요.
- 시리얼 출력과 3단계의 점검 항목으로 계속 문제를 해결하세요: hung task, 디스크 오류,
  emergency 모드에서 멈춘 부팅, 마운트되지 않는 데이터 디스크.
- **기존 프라이머리의 디스크를 삭제하거나 분리하지 마세요.**
- 다음 조치는 두 번째 DBRE와 Incident Manager가 결정합니다.
- 노드가 돌아오면 [3단계](#step-3-recover-the-old-primary-in-place)로 돌아가 노드 상태 점검부터 계속 진행하세요.

> **페일오버는 이 런북의 범위를 벗어납니다.** 페일오버는 기존 프라이머리에만 있는 모든 쓰기를
> 잃게 하므로 데이터 손실을 수반하는 결정입니다: 혼자 결정하지 마세요. 두 번째 DBRE가 검토하고
> 동의해야 하며, Incident Manager에게도 알려야 합니다.

## 5단계: 복구 후

- **지연 레플리카와 아카이브 레플리카:** 분기 지점 이후의 기존 프라이머리 WAL이 GCS에 도달하면, 이 레플리카들은
  기존 타임라인을 따라가다가 다음 메시지와 함께 멈출 수 있습니다:
  `new timeline N forked off current database system timeline N-1 before current recovery point`.
  그러면 재구축이 필요합니다. 레플리카의 위치(`select pg_last_wal_replay_lsn();`)를
  새 타임라인의 `.history` 파일에 있는 분기 LSN과 비교하세요. 분기 지점에 도달하기 **전에** 레플리카를 재시작하면,
  그 `.history` 파일이 GCS에 올라간 이후라면 새 타임라인을 따라가게 될 것입니다. 이 마지막 단계는 아직
  테스트되지 않았으므로 먼저 DBRE와 확인하세요. 다음 문서도 참고하세요:
  [GCS의 분기된 타임라인 WAL](patroni-management.md#diverged-timeline-wal-segments-in-gcs-after-failover).
- 애플리케이션 트래픽이 회복되었는지 확인하세요: 오류율, 그리고 PgBouncer 풀이 프라이머리를 가리키는지.

## 참고 사항 및 제한 사항

- **비동기 복제를 가정합니다.** 동기 복제에서는 페일오버가 보통 확인된 쓰기를
  잃지 않으므로 제자리 복구의 중요성이 덜합니다. 이것이 바뀌면 이 런북을 업데이트하세요.
- **일시 중지 중에는 아무것도 자동으로 이루어지지 않습니다.** *어떤* 장애에도 페일오버가 일어나지 않으며, 워치독도
  비활성 상태입니다. 일시 중지는 짧게 유지하고, 그동안 내내 누군가가 클러스터를 지켜보게 하세요.
- **VM 리셋은 비정상 종료입니다.** Postgres는 시작 시 크래시 복구를 실행하며, 바쁜 프라이머리에서는
  몇 분이 걸릴 수 있습니다. 커밋되지 않은 트랜잭션은 롤백되고, 클라이언트 연결은 끊깁니다.
  커밋된 트랜잭션은 영향을 받지 않습니다.
- **레플리카는 프라이머리의 복제 슬롯으로부터 따라잡습니다.** 슬롯이 없거나, 슬롯이
  필요로 하던 WAL이 제거되었다면(`max_slot_wal_keep_size`), 해당 레플리카는 재구축이 필요합니다. 프라이머리의 데이터는
  영향을 받지 않습니다.
- **지금까지의 테스트:** 2노드 테스트 클러스터에서 검증되었습니다. 크래시 복구 시간과 따라잡기 시간은
  프로덕션 규모에서 아직 측정되지 않았습니다.

## 검증

제자리 복구 절차는 db-benchmarking 환경에서 테스트되었습니다: 복제를 끊은 상태로, 쓰기 부하 중인
프라이머리의 VM을 리셋했습니다. 손실된 행은 없었으며 타임라인 변경도 없었습니다.
스테이징에서 게임 데이가 계획되어 있습니다.
