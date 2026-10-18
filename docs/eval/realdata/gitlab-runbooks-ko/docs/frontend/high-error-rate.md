# 오류율 증가

## 증상

- prometheus-alerts의 *Increased Error Rate Across Fleet* 메시지
- PagerDuty 알림: `HighRailsErrorRate`

## 문제 해결

### Kibana

- [rails의 모든 5xx 상태](https://log.gprd.gitlab.net/goto/c0d8ed2d964e4a792838e77a4ac1f942)
- [컨트롤러별 모든 5xx 상태](https://log.gprd.gitlab.net/goto/19bccd903f408085535df92734176cec)
- [특정 IP에서의 어뷰징 확인](https://log.gprd.gitlab.net/goto/d4c6a0d68a565a0ac70b3840306f8eca)
- [트리아지 개요](https://dashboards.gitlab.net/d/RZmbBr7mk/gitlab-triage) 대시보드에서 백엔드별 5xx 오류를 확인합니다.
- [Sentry](https://new-sentry.gitlab.net/organizations/gitlab/issues/?project=3)에서 새로운 500 오류나 증가 추세가 있는지 확인합니다.
- 문제가 지속되면 [#development](https://gitlab.enterprise.slack.com/archives/development)에 채널 전체 알림을 보냅니다.
