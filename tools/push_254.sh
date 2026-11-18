#!/bin/bash
# 第254轮: 网络不稳时的稳妥推送 —— fetch(多次) -> rebase(自动解冲突) -> push(多次)
cd /workspace/repo || exit 1

FETCHED=0
for i in $(seq 1 10); do
  echo "== fetch $i $(date +%H:%M:%S)"
  if timeout 170 git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
       -c http.postBuffer=524288000 fetch --depth=1 origin main 2>&1 | tail -1; then
    if git rev-parse origin/main >/dev/null 2>&1; then
      echo "origin/main = $(git rev-parse --short origin/main)"
      FETCHED=1
      break
    fi
  fi
  sleep 12
done

if [ "$FETCHED" = "0" ]; then echo FETCH_FAILED; exit 2; fi

# shallow 拉取后本地可能变成浅克隆, 先确保能 rebase
if ! git -c http.version=HTTP/1.1 rebase origin/main >/tmp/rb254.log 2>&1; then
  echo "REBASE_NEEDS_FIX"
  for f in $(git diff --name-only --diff-filter=U); do
    echo "  conflict: $f -> take theirs"
    git checkout --theirs -- "$f" 2>/dev/null && git add "$f"
  done
  GIT_EDITOR=true git rebase --continue >>/tmp/rb254.log 2>&1 || { echo REBASE_FAILED; tail -5 /tmp/rb254.log; exit 3; }
fi
echo "rebase ok, HEAD = $(git rev-parse --short HEAD)"

for i in $(seq 1 12); do
  echo "== push $i $(date +%H:%M:%S)"
  out=$(timeout 150 git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=45 \
        -c http.postBuffer=524288000 push origin main 2>&1)
  echo "$out" | tail -1
  if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main 2>/dev/null)" ]; then
    echo PUSHED_OK
    exit 0
  fi
  if echo "$out" | grep -q "rejected\|non-fast-forward"; then
    echo "  need fetch again"
    timeout 170 git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --depth=1 origin main >/dev/null 2>&1
    git -c http.version=HTTP/1.1 rebase origin/main >/dev/null 2>&1 || {
      for f in $(git diff --name-only --diff-filter=U); do git checkout --theirs -- "$f" 2>/dev/null && git add "$f"; done
      GIT_EDITOR=true git rebase --continue >/dev/null 2>&1
    }
  fi
  sleep 15
done
echo PUSH_STILL_FAILED
exit 4
