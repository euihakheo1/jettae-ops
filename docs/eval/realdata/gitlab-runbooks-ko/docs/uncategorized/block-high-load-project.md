# 높은 부하를 유발하는 프로젝트 차단하기

> **이 런북을 사용하는 경우:** 어뷰징이 특정 GitLab 프로젝트에서 발생하며 애플리케이션 계층(Rails admin / rails console)에서 완화해야 할 때.
> 엣지에서 IP, 경로 또는 User-Agent로 차단하려면 대신 [Cloudflare 트래픽 차단 및 관리 런북](../cloudflare/managing-traffic.md#when-to-block-and-how-to-block)을 참고하세요 — Cloudflare는 **HTTP/S 트래픽 차단에 우선적으로 선택할 첫 번째 옵션**입니다.

## 단계

1. `sudo gitlab-rails console` 명령을 실행하여 rails console을 시작하세요.

1. 해당 프로젝트를 비공개로 설정하세요. 예를 들어 `my-namespace/my-group/my-project`의 경우 다음과 같습니다:

    ```ruby
    project = Project.find_by_full_path('my-namespace/my-group/my-project')
    project.visibility_level = Gitlab::VisibilityLevel::PRIVATE
    project.save
    ```

1. 어떤 이유로 이 방법이 실패하면, 관리자
계정으로 프로젝트에 이동하여 이를 비활성화하세요. 예를 들어 위 예시의 URL은 다음과 같습니다:

    <https://gitlab.com/my-namespace/my-group/my-project/-/settings/edit>

    `Permissions` 아래에서 `Expand`를 클릭하고, `Project visibility`를 `Public`에서 `Private`으로 설정하세요.

1. 확실히 하기 위해 `Repository` 버튼도 켬에서 끔으로 전환하는 것이 좋습니다.

1. 프로젝트 소유자가 이 설정을 변경하지 못하도록, 프로젝트의 소유자를
차단하는 것도 좋습니다. 예를 들면:

    ```ruby
    project.owner.block!
    ```

1. 소유자를 차단해도 소유자가 이미 시작한 임포트는 중지되지 **않으며**, 해당 임포트는 계속 부하를 발생시킵니다.
[실행 중인 임포트 취소](../importers/cancel-running-imports.md)를 따라 임포트를 취소하세요.
