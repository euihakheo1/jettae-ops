# 출처: GitLab 운영 런북 한국어 번역판 (실증 ① 언어 대조 실험)

- 원문 저장소: https://gitlab.com/gitlab-com/runbooks (GitLab.com 운영팀 런북)
- 라이선스: MIT (원문 라이선스: 이 폴더의 `LICENSE`) — 번역판도 같은 조건, 출처 GitLab Inc.
- 원문 고정 커밋: `844a2b882687492cf117cc2fc553f7fbf9a02197`
- 대상: 실증 ①의 test 분할 20건만 (분할 파일 docs/eval/gitlab-runbooks-split.json, digest 441efb8bb3f3)
- 번역: 2026-10-18, Claude(Claude Code 세션의 작업 에이전트 5개)가 줄 단위로 대응시켜 충실 번역.
  코드 블록·인라인 코드·URL·명령어·서비스/도구/사람 이름·채널 이름·숫자는 원문 그대로 두고,
  내용 추가·삭제·요약·오류 수정은 하지 않음(원문의 오타·닫히지 않은 코드 블록도 그대로).
  회사·도구를 한국식으로 바꾸는 번안은 하지 않음 — 언어만 바뀐 대조 실험이 되도록.
- 검증: 구조 검사기(줄 수·코드 블록·줄 앞 마크다운 구조·빈 줄·URL·링크 대상·인라인 코드·숫자가
  줄마다 원문과 같은지) 실패 0건, 번역하지 않은 별도 에이전트 5개가 줄마다 대조해 찾은 7건 수정.
  남은 경고: 제목을 번역해 문서 안 앵커 링크 12개가 번역판에서는 안 맞음, 원문의 닫히지 않은 코드
  블록 안 문장 2줄은 규칙대로 영어로 둠.
- 줄 번호가 원문과 같으므로 결과의 원문 링크(GitLab, 같은 줄 번호)로 바로 대조할 수 있다.

| 파일 | 줄 | 원문 SHA-256 | 번역 SHA-256 |
|---|---:|---|---|
| `docs/cloudflare/troubleshooting.md` | 98 | e00ab0fa93da98f9… | 9c4d553f6d98cf8d… |
| `docs/cloudflare/vendor.md` | 72 | 94046297dd5630ad… | 956027cfc0fe2ae0… |
| `docs/disaster-recovery/gameday.md` | 115 | 0b378c2b33650ead… | 13d00f8694bc5b02… |
| `docs/disaster-recovery/README.md` | 57 | 20ae9354016a1d34… | beac71bf5b0df77a… |
| `docs/frontend/asset-proxy-down.md` | 25 | 4d276868f00e0384… | 271fa5b2f26c4917… |
| `docs/frontend/gitlab-com-is-down.md` | 11 | 4570609bff0120f1… | 1423830228db5825… |
| `docs/frontend/high-error-rate.md` | 17 | b48a30ced158553b… | 9d549564ab66edb3… |
| `docs/frontend/ssh-maxstartups-breach.md` | 73 | ad61379c26e29291… | 6dca84b8aa5e0649… |
| `docs/gamedays/scenarios/zonal-outage-recovery.md` | 14 | dfc84efb3f676545… | f17291fee1ec34c8… |
| `docs/incident-io/incident-management.md` | 97 | c66d86c685cb40bd… | 303945d3dae49c9e… |
| `docs/incident-io/oncall.md` | 126 | d44371b0038f6909… | d984382cf83c259e… |
| `docs/incidents/when-gitlab-com-is-down.md` | 62 | 18c04fbdbf0854e8… | b4d2d78b9d45421e… |
| `docs/patroni/primary-unresponsive-assess-before-failover.md` | 226 | 1361146715aae52c… | c8bc67af1f9644d3… |
| `docs/release-management/deploy_pressure.md` | 44 | d9c19cd0be6d9c2b… | e9887590c1dd35c7… |
| `docs/security-patching/linux-os/notifications.md` | 24 | 5bcec0776e3a511d… | 19e9804a7064eb5f… |
| `docs/uncategorized/block-high-load-project.md` | 35 | 87615e3f1af173a8… | aa848eb82adfc533… |
| `docs/uncategorized/externalvendors/GCP-rackspace-support.md` | 39 | 46be1d8831626048… | 76b61d031cdcb0e7… |
| `docs/uncategorized/namespace-restore.md` | 227 | 258fd1181c454453… | 26302b59c2dedd64… |
| `docs/uncategorized/upgrade-and-rollback.md` | 46 | 9fdae356a2d44f67… | 10a78745918e6695… |
| `docs/uncategorized/workers-high-load.md` | 43 | 2ebccf2b1ad49e48… | 3c642c5eda6867e3… |
