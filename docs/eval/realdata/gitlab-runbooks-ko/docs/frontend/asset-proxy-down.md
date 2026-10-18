# `asset_proxy`가 `DOWN` 상태임

## 증상

```shell
steve@haproxy-main-06-lb-gprd.c.gitlab-production.internal:~$ echo show stat | sudo socat stdio /run/haproxy/admin.sock | grep 'asset_proxy'
asset_proxy,asset-bucket,0,0,0,3,,162,337432,1765043,,0,,0,0,0,0,DOWN,1,1,0,3,1,213485,213485,,1,12,1,,162,,2,0,,6,L4TOUT,,2000,0,82,0,80,0,0,,,,162,0,0,,,,,213544,,,0,2,27,220,,,,Layer4 timeout,,2,3,0,,,,74.125.31.128:443,,http,,,,,,,,0,43,119,,,0,,0,5,114,9335,0,0,0,0,1,1,,,,0,,,,,,,,,,-,9562,0,0,,,,,,,,,,,,,,,,,,,,,,
asset_proxy,BACKEND,0,0,0,3,5000,162,337432,1765043,0,0,,0,0,0,0,DOWN,0,0,0,,1,213485,213485,,1,12,0,,162,,1,0,,6,,,,0,82,0,80,0,0,,,,162,0,0,0,0,0,0,213544,,,0,2,27,220,,,,,,,,,,,,,,http,roundrobin,,,,,,,0,43,119,0,0,,,0,5,114,9335,0,,,,,0,0,0,0,,,,,,,,,,,-,9562,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40061,40180,9556632,1039258,0,0,
```

[DNS를 백엔드로](https://gitlab.com/gitlab-cookbooks/gitlab-haproxy/-/blob/aa118861af117894acc26a6eab2bfd3b4597b564/attributes/default.rb#L311) 사용하는 대상은
`asset_proxy`이며, 이전 IP가 더 이상 응답하지 않으면 그 IP가 상태 파일에 그대로 남아 백엔드를 `DOWN`으로 표시하게 됩니다.

이 문제는 <https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/12421>에서 수정될 예정입니다.

## 런북

```shell
steve@haproxy-main-06-lb-gprd.c.gitlab-production.internal:~$ sudo systemctl stop haproxy # will take a while.
steve@haproxy-main-06-lb-gprd.c.gitlab-production.internal:~$ sudo rm -f /etc/haproxy/state/global
steve@haproxy-main-06-lb-gprd.c.gitlab-production.internal:~$ sudo systemctl start haproxy
steve@haproxy-main-06-lb-gprd.c.gitlab-production.internal:~$ echo show stat | sudo socat stdio /run/haproxy/admin.sock | grep 'asset_proxy'
asset_proxy,asset-bucket,0,0,0,0,,0,0,0,,0,,0,0,0,0,UP,1,1,0,0,0,5,0,,1,12,1,,0,,2,0,,0,L7OK,200,19,0,0,0,0,0,0,,,,0,0,0,,,,,-1,,,0,0,0,0,,,,Layer7 check passed,,2,3,4,,,,108.177.12.207:443,,http,,,,,,,,0,0,0,,,0,,0,0,0,0,0,0,0,0,1,1,,,,0,,,,,,,,,,-,2,0,0,,,,,,,,,,,,,,,,,,,,,,
asset_proxy,BACKEND,0,0,0,0,5000,0,0,0,0,0,,0,0,0,0,UP,1,1,0,,0,5,0,,1,12,0,,0,,1,0,,0,,,,0,0,0,0,0,0,,,,0,0,0,0,0,0,0,-1,,,0,0,0,0,,,,,,,,,,,,,,http,roundrobin,,,,,,,0,0,0,0,0,,,0,0,0,0,0,,,,,1,0,0,0,,,,,,,,,,,-,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,1636,142,0,0,
```
