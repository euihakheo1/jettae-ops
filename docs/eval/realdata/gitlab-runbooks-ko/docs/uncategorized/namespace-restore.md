## 요약

이 런북은 데이터베이스 백업과 디스크 스냅샷에서 네임스페이스를 복원하는 방법에 대한 몇 가지 힌트와 세부 사항을 제공합니다.

> 이 방법은 삭제된 프로젝트를 복원할 때 권장되는 방법이 아닙니다.  그 경우에는 [삭제된 프로젝트 복원](deleted-project-restore.md) 런북을 참조하세요

이 런북은 그 런북에 설명된 절차에 따라 [PITR 데이터베이스 복원 인스턴스](deleted-project-restore.md#restore-full-database-backup-and-perform-pitr)를 이미 만들었다고 가정하므로, 이 런북의 내용을 실행하기 전에 그 런북을 먼저 읽어 보는 것이 아마 좋을 것입니다.

## 데이터베이스 인스턴스를 읽기/쓰기 가능하게 만들기

기본적으로 PITR 데이터베이스 복원 인스턴스는 읽기 전용입니다.  읽기/쓰기 가능하게 만들려면:

```shell
gitlab-ctl stop postgresql
rm /var/opt/gitlab/postgresql/data/standby.signal
gitlab-ctl start postgresql
```

## 스냅샷 복원하기

zone 및 type 옵션을 원하는 대상에 맞게 변경합니다

```shell
gcloud compute --project=gitlab-restore disks create file-09-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/6720213477708092934 --zone us-west2-a --type pd-ssd
gcloud compute --project=gitlab-restore disks create file-10-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/6065067043346273798 --zone us-west2-a --type pd-ssd
gcloud compute --project=gitlab-restore disks create file-11-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/4636480772311963142 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-12-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/3340256640197023238 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-13-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/2349193301717243398 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-14-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/3635909187987658246 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-15-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/4238534030654789126 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-16-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/4498711041933661702 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-17-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/642195138748438022 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-18-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/2250439440802862598 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-19-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/5664272673947342342 --zone us-west2-a
gcloud compute --project=gitlab-restore disks create file-20-stor-gprd-data-08-15-2020 --source-snapshot https://www.googleapis.com/compute/v1/projects/gitlab-production/global/snapshots/1884121680127022598 --zone us-west2-a
```

## 모든 샤드를 한 번에 마운트하기

먼저 위에서 생성한 모든 샤드를 GCP 콘솔에서 연결합니다

`/etc/fstab`의 끝에 다음을 추가합니다

```
UUID=98a22450-f02e-4ee7-ab76-97ff490bc90a   /mnt/file-09    ext4    defaults    0 0
UUID=ba8c6f55-c8d9-4e3c-a955-4c49d5e0efb6   /mnt/file-10    ext4    defaults    0 0
UUID=01b2c8c8-5d7f-48a3-b143-1178387007bc   /mnt/file-11    ext4    defaults    0 0
UUID=ea60a66c-b279-45cf-8896-02e92b9404e1   /mnt/file-12    ext4    defaults    0 0
UUID=99edc798-1ee8-43d9-abd1-1a4b849d0b0e   /mnt/file-13    ext4    defaults    0 0
UUID=3ccc3991-cda8-45b1-9dc4-b504692d196e   /mnt/file-14    ext4    defaults    0 0
UUID=377695e8-3b6b-4d07-ad05-358bb930b6f0   /mnt/file-15    ext4    defaults    0 0
UUID=be7101d0-7223-4805-8c0c-d3b656e1f9ea   /mnt/file-16    ext4    defaults    0 0
UUID=33fbf110-2416-4d45-b791-2ebc6fdd5f6c   /mnt/file-17    ext4    defaults    0 0
UUID=2c63d85d-bf20-4859-82b7-d8a942da719f   /mnt/file-18    ext4    defaults    0 0
UUID=9ebfb115-ba06-481e-a961-b183461e38df   /mnt/file-19    ext4    defaults    0 0
UUID=9c65caa8-f3ef-4bfc-89ee-3a7364f51a29   /mnt/file-20    ext4    defaults    0 0
```

그런 다음 모든 샤드를 마운트합니다:

```shell
for i in {09..20} ; do mkdir /mnt/file-$i ; mount /mnt/file-$i; done
```

## 개별 샤드 마운트하기

연결된 디스크를 표시하고 마운트합니다:

```shell
$ lsblk | grep 15.6T
sdc       8:32   0 15.6T  0 disk /mnt/file-16

$ mkdir /mnt/file-16
$ mount /dev/sdc /mnt/file-16
```

## 권한 수정하기

gitaly 노드와 복원 노드는 `passwd` 파일이 서로 다르고 그에 따라 UID도 다르기 때문에, 이를 일치시켜야 합니다. `chown -R git /mnt/file-16/git-data
`는 샤드마다 시간이 매우 오래 걸립니다(백업에서 디스크 전체를 복원하는 것보다 훨씬 오래 걸립니다).  복원 노드를 다시 구축하게 된다면, 파일을 새 UID로 설정하는 대신 `git` UID를 `500`으로 설정하는 것이 좋을 것입니다.

기존 git 사용자와 그룹의 UID와 GIT을 찾습니다:

```shell
grep git /etc/passwd
grep git /etc/group
```

ID가 다르다면 아래의 998을 해당 ID로 바꿉니다:

```shell
usermod -u 500 git
groupmod -g 500 git
find /var/opt/gitlab -group 998 -exec chgrp -h git {} \;
find /var/opt/gitlab -xdev -user 998 -exec chown -h git {} \;
find / -xdev -group 998 -exec chown -h git {} \;
find / -xdev -group 998 -exec chgrp -h git {} \;
```

## Gitaly 구성하기

`gitlab.rb`에 다음 섹션을 추가합니다:

```ruby
git_data_dirs({
        "default" => { "path" => "/var/opt/gitlab/git-data" },
        "nfs-file09" => { "path"=> "/mnt/file-09/git-data" },
        "nfs-file10" => { "path"=> "/mnt/file-10/git-data" },
        "nfs-file11" => { "path"=> "/mnt/file-11/git-data" },
        "nfs-file12" => { "path"=> "/mnt/file-12/git-data" },
        "nfs-file13" => { "path"=> "/mnt/file-13/git-data" },
        "nfs-file14" => { "path"=> "/mnt/file-14/git-data" },
        "nfs-file15" => { "path"=> "/mnt/file-15/git-data" },
        "nfs-file16" => { "path"=> "/mnt/file-16/git-data" },
        "nfs-file17" => { "path"=> "/mnt/file-17/git-data" },
        "nfs-file18" => { "path"=> "/mnt/file-18/git-data" },
        "nfs-file19" => { "path"=> "/mnt/file-19/git-data" },
        "nfs-file20" => { "path"=> "/mnt/file-20/git-data" }
})
```

그런 다음 `gitlab-ctl reconfigure`

### 서비스 시작하기

reconfigure가 성공하면 서비스를 개별적으로 시작해야 합니다.

앞 섹션의 단계를 따라 postgres가 실행 중이고 읽기/쓰기 모드인지 확인합니다.

```shell
gitlab-ctl start gitaly
gitlab-ctl start gitlab-workhorse
gitlab-ctl start redis
gitlab-ctl start puma
gitlab-ctl start nginx
```

## 관리자 토큰 생성하기

연결 인증에 사용할 수 있는 기존 관리자 토큰이 없다면, 다음과 같이 새 토큰을 생성합니다.  모두 대문자로 된 부분을 사용하려는 값으로 바꿉니다.

```ruby
user = User.find_by_username('YOUR_ADMIN_USER')
token_digest = Gitlab::CryptoHelper.sha256 'PLAIN_TEXT_YOU_MAKE_UP_FOR_THE_TOKEN'
PersonalAccessToken.create!(name: 'Full Access', scopes: [:api], user: user, token_digest: token_digest)
```

`PLAIN_TEXT_YOU_MAKE_UP_FOR_THE_TOKEN`은 `congregate configure` 명령에 입력할 값입니다

## congregate 도구 설치 및 구성하기

[congregate 도구](https://gitlab.com/gitlab-org/professional-services-automation/tools/migration/congregate)는 Professional Services 팀이 관리하는 마이그레이션 도구입니다.

설치 및 구성 방법은 [여기에 설명되어 있습니다](https://gitlab.com/gitlab-org/professional-services-automation/tools/migration/congregate/-/blob/master/docs/full_setup.md#configuring).

`congregate init`으로 도구를 초기화한 후, `data/congregate.conf` 파일에 구성 설정을 입력합니다. 모든 필드는 [템플릿에 설명되어 있습니다](https://gitlab.com/gitlab-org/professional-services-automation/tools/migration/congregate/-/blob/master/congregate.conf.template).
개인 액세스 토큰은 소스 인스턴스와 대상 인스턴스에서 생성해야 합니다. 그런 다음 토큰 값을 난독화하기 위해 Base64로 인코딩하여 config에 입력합니다. 난독화는 `congregate obfuscate`를 사용하여 할 수 있습니다. 구성 예시:

```ini
[SOURCE]
src_parent_group_path = deletedNamespace
src_parent_group_id = deletedNamespaceID
```

이후 토큰과 IP가 올바르게 구성되어 있다면 `congregate list` 실행이 성공해야 합니다.  성공하지 않으면 `/opt/congregate/data/logs`에 있는 [Congregate 로그](https://gitlab.com/gitlab-org/professional-services-automation/tools/migration/congregate/-/blob/af88620cfa0b4530b9fc71ec519beae1978b7864/runbooks/airgapped-migration-usage.md#checking-logs)를 확인하거나, 소스 서버의 로그를 보며 원인을 확인합니다 (`sudo gitlab-ctl tail gitlab-rails`)

## 테스트 인스턴스에서 네임스페이스를 삭제할 수 없나요?

복원이 제대로 동작하는지 검증하기 위해 테스트 인스턴스를 만들었나요?  테스트 인스턴스에 파일이 없어서, 다른 인스턴스에서 마이그레이션하기 전에 네임스페이스를 제거하지 못하고 있나요?  네임스페이스를 삭제하는 대신 이름을 변경하세요.

```ruby
irb(main):001:0> g = Group.find_by_full_path("deletedgroup")
=> #<Group id:1857667 @ deletedgroup>
irb(main):003:0> g.destroy
Traceback (most recent call last):
       16: from app/models/concerns/storage/legacy_namespace.rb:102:in `block in rm_dir'
       15: from lib/gitlab/gitaly_client/namespace_service.rb:12:in `allow'
       14: from lib/gitlab/temporarily_allow.rb:9:in `temporarily_allow'
       13: from lib/gitlab/gitaly_client/namespace_service.rb:12:in `block in allow'
       12: from app/models/concerns/storage/legacy_namespace.rb:103:in `block (2 levels) in rm_dir'
       11: from lib/gitlab/shell.rb:170:in `mv_namespace'
       10: from lib/gitlab/gitaly_client/namespace_service.rb:40:in `rename'
        9: from lib/gitlab/gitaly_client/namespace_service.rb:46:in `gitaly_client_call'
        8: from lib/gitlab/gitaly_client.rb:169:in `call'
        7: from lib/gitlab/gitaly_client/call.rb:17:in `call'
        6: from lib/gitlab/gitaly_client/call.rb:55:in `recording_request'
        5: from lib/gitlab/gitaly_client/call.rb:18:in `block in call'
        4: from lib/gitlab/gitaly_client.rb:176:in `execute'
        3: from lib/gitlab/gitaly_client.rb:221:in `request_kwargs'
        2: from lib/gitlab/gitaly_client.rb:211:in `authorization_token'
        1: from lib/gitlab/gitaly_client.rb:261:in `token'
RuntimeError (storage not found: "nfs-file11")
irb(main):012:0> g.path
=> "deletedgroup"
irb(main):013:0> g.path = "renamedgroup"
=> "renamedgroup"
irb(main):017:0> g.save
=> true
irb(main):018:0> g = Group.find_by_full_path("deletedgroup")
=> nil
irb(main):019:0> g = Group.find_by_full_path("renamedgroup")
=> #<Group id:1857667 @ renamedgroup>
irb(main):020:0> exit
```

## 몇 가지 유용한 명령어

### 그룹 네임스페이스 살펴보기

rails 콘솔에서 (`sudo gitlab-rails console`):

```ruby
g = Group.find_by_full_path("interesting-group")
=> #<Group id:1857667 @interesting-group>
irb(main):2:0> g.all_projects.count
=> 158
```

### 리포지토리 파일의 샤드와 해시된 위치 표시하기

```ruby
irb(main):4:0> Group.find_by_full_path("interesting-group").all_projects.map.with_index do |project, index|
irb(main):5:1*   puts "#{index+1} - #{project.full_path}"
irb(main):6:1>   puts project.repository_storage
irb(main):7:1>   puts project.disk_path
irb(main):8:1> end; nil
```
