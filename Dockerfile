# syntax=docker/dockerfile:1.7
# 제때옵스(JETTAE Ops) — API + 웹 화면 단일 이미지
#   docker build -t jettae-ops .
#   docker run -p 8765:8765 -v jettae-data:/data jettae-ops
#   공개 베타 배포(Render): render.yaml · docs/deploy.md

# ---------------------------------------------------------------- web build
FROM node:24-alpine AS web
WORKDIR /web
# npm ci: 저장소에 커밋된 package-lock.json 그대로만 설치합니다(package.json과 어긋나면 빌드 실패).
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------- runtime
# 베이스 이미지는 3.12 계열의 최신 보안 패치를 받도록 태그로 둡니다. 파이썬 패키지는 constraints.txt로 고정합니다.
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JETTAE_WEB_DIST=/app/web/dist \
    JETTAE_DATABASE_URL=sqlite:////data/jettae.db \
    JETTAE_LOG_JSON=true
WORKDIR /app

COPY pyproject.toml README.md alembic.ini constraints.txt ./
COPY src ./src
# -c constraints.txt: 직접·간접 의존성을 모두 테스트한 버전(==)으로 고정 — 자동 배포 때마다 새 버전이 섞이지 않게.
# 생성형 AI SDK(anthropic·openai)는 넣지 않습니다: 공개 사이트는 JETTAE_LLM_MODE=off이고, 코드는 SDK 없이도
# 동작합니다(LLM을 켤 이미지가 필요하면 ".[anthropic,postgres]"처럼 추가 기능을 더하고 constraints.txt도 다시 만드세요).
RUN pip install -c constraints.txt ".[postgres]"

COPY examples ./examples
COPY --from=web /web/dist ./web/dist

RUN useradd --create-home --uid 10001 jettae \
 && mkdir -p /data && chown jettae:jettae /data
USER jettae
VOLUME ["/data"]
EXPOSE 8765

# 포트는 $PORT(Render 등 호스팅이 넣어 줌, 없으면 8765). 시작 시 데모 진단을 한 번 돌리므로 여유를 둔다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT:-8765}/healthz', timeout=4).status == 200 else 1)"

# 1) 스키마를 Alembic 마이그레이션으로 올리고(JETTAE_DATABASE_URL, 이미 최신이면 아무 일도 안 함)
# 2) 별빛상점 데모 워크스페이스를 같은 DB에 만들거나 다시 불러온 뒤(재시작해도 이름으로 찾아 재사용)
# 3) $PORT에서 서버를 띄운다(exec: 컨테이너 종료 신호가 서버에 바로 전달됨).
# 다른 컴퓨터에 열 때는 JETTAE_AUTH_MODE=token(공개 웹사이트는 public)과 JETTAE_API_TOKENS를 설정하세요
# (기본값 dev의 헤더 역할은 로컬·데모 전용). 프록시 뒤에 두면 접속 주소별 한도가 읽을 방문자 주소
# (JETTAE_CLIENT_IP_SOURCE, 기본값 peer)도 맞추고 /api/v1/admin/request-info로 확인하세요(docs/deploy.md 4-2).
CMD ["sh", "-c", "jettae db upgrade && exec jettae demo --host 0.0.0.0 --port \"${PORT:-8765}\""]
