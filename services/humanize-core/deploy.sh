#!/usr/bin/env bash
# Production deploy for Lightsail Core.
#
# Run on the server as the ubuntu user:
#   /opt/dadeum/dadeum-backend/services/humanize-core/deploy.sh [--force]
#
# Fast-forwards the checked-out branch, builds the new image while the old
# container keeps serving, swaps containers, and checks /health. A failed build
# leaves the running service untouched; a failed health check restores the
# previous image and commit.
set -euo pipefail

SERVICE=humanize-core
HEALTH_URL=http://localhost/health
HEALTH_TIMEOUT_SECONDS=90
ROLLBACK_TAG=rollback

log() { printf '\n[deploy] %s\n' "$*"; }
die() { printf '\n[deploy] 중단: %s\n' "$*" >&2; exit 1; }

wait_healthy() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if curl -fsS -m 5 "$HEALTH_URL" 2>/dev/null | grep -q '"status": *"ok"'; then
      return 0
    fi
    sleep 3
  done
  return 1
}

# Names keys this pull added to .env.example that .env does not set.
warn_new_env_keys() {
  local key missing=()
  for key in $(git diff "$1" HEAD -- .env.example | sed -nE 's/^\+#? ?([A-Z][A-Z0-9_]*)=.*/\1/p' | sort -u); do
    grep -qE "^$key=" .env || missing+=("$key")
  done
  if ((${#missing[@]})); then
    log "새 설정 항목이 생겼는데 .env에 없습니다. 필요하면 추가하세요: ${missing[*]}"
  fi
}

main() {
  local force=false
  case "${1:-}" in
    "") ;;
    --force) force=true ;;
    *) die "모르는 옵션: $1 (사용법: deploy.sh [--force])" ;;
  esac

  cd "$(dirname "$(readlink -f "$0")")"

  exec 9>"/tmp/$SERVICE-deploy.lock"
  flock -n 9 || die "다른 배포가 이미 진행 중입니다."

  [[ -f .env ]] || die ".env 파일이 없습니다."

  local branch
  branch=$(git symbolic-ref --short -q HEAD) ||
    die "브랜치가 아니라 특정 커밋에 머물러 있습니다. git checkout <브랜치> 후 다시 실행하세요."
  # Server-only edits stay in place. Git itself refuses the fast-forward when
  # an incoming commit touches one of them, so they are never overwritten.
  if ! git diff --quiet HEAD; then
    log "서버에서 직접 고친 파일이 있습니다. 그대로 두고 진행합니다:"
    git status --short --untracked-files=no
  fi

  local prev_commit new_commit
  prev_commit=$(git rev-parse HEAD)

  log "$branch 브랜치의 최신 코드를 받습니다."
  git fetch -q origin "$branch"
  if [[ $(git rev-parse "origin/$branch") == "$prev_commit" ]] && ! $force; then
    log "이미 최신입니다: $(git log --oneline -1). 같은 버전을 다시 빌드하려면 --force 를 붙이세요."
    return 0
  fi
  git merge -q --ff-only "origin/$branch" ||
    die "코드를 받지 못했습니다. 받을 변경이 서버에서 고친 파일과 겹치거나, GitHub과 서버의 기록이 갈라졌습니다. 서비스는 그대로입니다."
  new_commit=$(git rev-parse HEAD)

  if [[ $new_commit != "$prev_commit" ]]; then
    log "이번에 들어가는 변경:"
    git log --oneline "$prev_commit..$new_commit"
    warn_new_env_keys "$prev_commit"
  fi

  # Keep the image the running container uses so a bad release can be undone
  # without rebuilding. Retagging drops the tag from the release before it.
  local container image=""
  container=$(docker compose ps -q "$SERVICE")
  if [[ -n $container ]]; then
    image=$(docker inspect -f '{{.Config.Image}}' "$container")
    image=${image%:latest}
    docker image tag "$(docker inspect -f '{{.Image}}' "$container")" "$image:$ROLLBACK_TAG"
  fi

  log "새 버전을 빌드합니다. 빌드하는 동안 기존 서비스는 그대로 돌아갑니다."
  if ! docker compose build "$SERVICE"; then
    git reset -q --keep "$prev_commit"
    die "빌드에 실패했습니다. 서비스는 기존 버전 그대로이고, 코드도 $(git log --oneline -1) 로 되돌렸습니다."
  fi

  log "새 버전으로 교체합니다."
  docker compose up -d --no-build
  if ! git diff --quiet "$prev_commit" HEAD -- Caddyfile; then
    # Caddyfile is a single-file bind mount. Git replaces the file, so the
    # running container keeps reading the old one until it is recreated.
    docker compose up -d --no-build --force-recreate caddy
  fi

  log "정상 동작을 확인합니다 (최대 ${HEALTH_TIMEOUT_SECONDS}초)."
  if wait_healthy; then
    docker image prune -f >/dev/null
    log "배포 완료: $(git log --oneline -1)"
    return 0
  fi

  log "정상 동작 확인에 실패했습니다. 최근 로그:"
  docker compose logs --tail=40 "$SERVICE" || true
  [[ -n $image ]] || die "되돌릴 이전 버전이 없습니다. 위 로그를 확인하세요."

  log "이전 버전으로 되돌립니다."
  docker image tag "$image:$ROLLBACK_TAG" "$image:latest"
  # The service must come back even if the checkout cannot be reset.
  git reset -q --keep "$prev_commit" || log "코드는 되돌리지 못했습니다. 서비스만 이전 이미지로 되돌립니다."
  docker compose up -d --no-build --force-recreate
  if wait_healthy; then
    if [[ $new_commit == "$prev_commit" ]]; then
      die "새로 빌드한 이미지에 문제가 있어서 직전에 돌던 이미지로 되돌렸습니다. 서비스는 정상입니다."
    fi
    die "새 버전에 문제가 있어서 이전 버전 $(git log --oneline -1) 로 되돌렸습니다. 서비스는 정상입니다."
  fi
  die "되돌린 뒤에도 정상 동작 확인에 실패했습니다. 바로 확인이 필요합니다: docker compose ps, docker compose logs $SERVICE"
}

# Calling main from one line lets bash finish reading this file before the
# pull can rewrite it.
main "$@"; exit
