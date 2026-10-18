# CloudFlare 트러블슈팅

## 링크

- [Cloudflare Grafana 대시보드](https://dashboards.gitlab.net/d/sPqgMv9Zk/cloudflare-traffic-overview?orgId=1)
- [Cloudflare 대시보드](https://dash.cloudflare.net)
- [GitLab.com 방화벽 개요](https://dash.cloudflare.com/852e9d53d0f8adbd9205389356f2303d/gitlab.com/firewall)
- [GitLab.com 트레이스백 도구](https://gitlab.com/cdn-cgi/trace)

## 증상

CloudFlare 고유의 문제임을 나타내는 특정 조건들이 있습니다.
예를 들어, CloudFlare 오류는 증가했지만 프로덕션 오류는 증가하지 않았다면,
문제는 CloudFlare 내부에 있는 것이 분명합니다.

다음은 잠재적인 오류 원인 목록입니다

### 정적 객체 캐시

프로덕션용 [정적 객체 캐시][static-objects-cache-howto]는
gitlab.net 존에 CloudFlare 워커로 배포되어 있습니다. 받은 알림이
gitlab.net 존을 가리키고 있고, `/raw/` 또는 `/-/archive` 엔드포인트에 대한 요청이
실패하고 있다면 워커가 어떻게 동작하고 있는지 확인해 볼 가치가 있습니다. 트러블슈팅 정보는
해당 [런북][static-objects-cache-troubleshooting]을 참고하세요.

[static-objects-cache-howto]: ../web/static-repository-objects-caching.md
[static-objects-cache-troubleshooting]: ../web/static-objects-caching.md

## 오탐 트리아지 절차

다음 정보는 WAF 적용으로 인한 Cloudflare 차단에 대한 사용자 보고를 진단하고
해결하는 과정을 돕기 위한 것입니다. 어떤
WAF 제품이든 사용자에게 영향을 주는 오탐이 소수 발생하기 마련입니다. 우리의
목표는 GitLab.com에 호스팅되는 콘텐츠의 특성을 감안하여 이를 가능한 한 줄이면서도
Cloudflare WAF 제품으로부터 어느 정도의 이점은 계속 얻는 것입니다.

### 수집할 보조 자료

- 많은 수의 보고로 인해 이미 인시던트가 생성된 경우:
  - 아래의 [일반 트레이스 템플릿](#generic-trace-template)을 복사하고
      사용자에게 결과를 보고해 달라고 요청하세요.
  - 추가 세부 정보가 필요하면, 사용자에게 비공개
      [Cloudflare 트러블슈팅 이슈](https://gitlab.com/gitlab-com/gl-infra/-/issues/new?issuable_template=Cloudflare%20Troubleshooting.md)를 생성하고 이를
      인시던트 이슈에 연결하도록 안내하세요.
- 특정 URI 또는 요청의 문제인 경우:
  - 필요하면 비공개로 설정하여 [Cloudflare 트러블슈팅 이슈](https://gitlab.com/gitlab-com/gl-infra/-/issues/new?issuable_template=Cloudflare%20Troubleshooting.md)를 생성하도록 안내하세요.

### Cloudflare 및 기타 서비스 연결 확인

1. [Cloudflare Grafana 대시보드](https://dashboards.gitlab.net/d/sPqgMv9Zk/cloudflare-traffic-overview?orgId=1)를 검토하여 Cloudflare와
   `haproxy` 간 반환 코드에 큰 불일치가 있는지 확인하세요.
1. [https://dash.cloudflare.com](https://dash.cloudflare.com)에 로그인하여
   예상대로 동작하지 않는 요청을 검색하세요. 해당 요청이 차단되고 있거나
   Cloudflare 서비스 중 어느 것에 의해 다른 방식으로 조치되고 있습니까?
1. `workhorse` 및 `rails` [프로덕션 로그](https://log.gprd.gitlab.net)를 검색하여
   대응하는 요청을 확인하고, 해당 요청이
   GitLab의 서비스까지 도달하고 있는지 검증하세요.
1. 연결 문제가 발생하는 호스트에서 `gitlab.com`을 `/etc/hosts`
   파일에 오리진의 IP와 함께 추가하고 요청을 다시 시도하여
   문제가 Cloudflare와 GCP 사이에 있을 수 있는지 판단하세요.
   1. `gitlab.com`에 대해 DNS가 제공하는 주소와
      하드코딩된 오리진 주소를 모두 사용하여, 서로 다른 GCP 리전 및/또는 다른
      클라우드 제공업체에서 동일한 연결을 시도해 문제가 나타나는 특정 경로를 더 좁혀 보세요.

### 일반 트레이스 템플릿

```
<p>
<details>
<summary>`curl http://gitlab.com/cdn-cgi/trace`</summary>

<pre><code>PASTE OUTPUT HERE</code></pre>

</details>
</p>

<p>
<details>
<summary>`curl https://gitlab.com/cdn-cgi/trace`</summary>

<pre><code>PASTE OUTPUT HERE</code></pre>

</details>
</p>

<p>
<details>
<summary>`curl -svo /dev/null https://gitlab.com`</summary>

<pre><code>PASTE OUTPUT HERE</code></pre>

</details>
</p>

## GeoIP Troubleshooting

We use CloudFlare rules to block access to gitlab.com from various locations.  When we need to torubleshoot these rules with CloudFlare support they will ask for a trace from the user being blocked.  The user simply has to visit [`/cdn-cgi/trace`](https://gitlab.com/cdn-cgi/trace) and then we provide the output in the support ticket.
```
