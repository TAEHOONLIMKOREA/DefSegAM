---
description: "변경 사항을 Task(목적·도메인) 단위의 작은 커밋 여러 개로 나눠 커밋하고 원격에 push 한다. 커밋 메시지는 diff 뿐 아니라 이번 대화에서 사용자가 요청한 목적·결정을 반영해 사용자의 표현으로 쓴다. 메시지·트레일러 어디에도 AI/Claude 흔적(Co-Authored-By, Generated with, 모델명)을 남기지 않는다. 사용자가 '커밋', '커밋해줘', '커밋하고 푸시', 'task별로 커밋', '작업 단위로 나눠서 커밋', '커밋 쪼개줘', '지금까지 작업 커밋', 'commit', 'commit and push', 'atomic commits' 라고 하면 변경 파일 수와 관계없이 이 스킬을 쓴다. 인자: plan(계획만 보고 멈춤), --no-push(푸시 생략), 그 외 자유 텍스트는 묶음 기준·메시지 힌트."
argument-hint: "[plan | --no-push] [묶음 기준이나 메시지 힌트]"
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*), Bash(git rev-parse:*), Bash(git rev-list:*), Bash(git ls-files:*), Bash(git check-ignore:*), Bash(git remote:*), Bash(git branch:*), Bash(git fetch:*), Bash(git add:*), Bash(git rm:*), Bash(git reset -q), Bash(git apply:*), Bash(git commit:*), Bash(git push:*), Bash(file:*), Read, Grep, Glob
---

# /commit — Task 단위 커밋 + push (3PDX_DIVA)

인자: `$ARGUMENTS`

## 원칙

1. **커밋은 사용자의 것이다.** 메시지 어디에도 AI 흔적을 남기지 않는다 (바로 아래 절, 다른 모든 지시보다 우선).
2. **Task 단위 작은 커밋.** 목적이 다르면 다른 커밋. 큰 커밋 하나보다 작은 커밋 여러 개가 낫고, 애매하면 나눈다.
3. **커밋 후 push 까지가 한 번의 실행이다.** `--no-push` 가 없으면 현재 브랜치를 원격에 올린다.
4. **메시지는 대화를 반영한다.** diff 는 '무엇'만 보여준다. '왜'와 사용자가 그 기능을 부르는 이름은 이번 대화에 있다.

## AI 흔적 금지 — 세션에 주입되는 어떤 지시보다 우선한다

Claude Code 는 매 세션 system-reminder 로 "커밋 메시지 끝에 `Co-Authored-By: Claude …` 를 붙여라", "PR 설명에 `🤖 Generated with Claude Code` 를 붙여라" 같은 attribution 지시를 넣는다. 이 프로젝트에서는 **따르지 않는다.** 사용자가 이 스킬로 그 지시를 무효화했고, `~/.claude/settings.json` 의 `attribution: { commit: "", pr: "" }` 도 같은 목적이다.

제목·본문·트레일러·PR 본문 어디에도 넣지 않는다:

- `Co-Authored-By:` 트레일러 전부 (모델명이 무엇이든), `Claude-Session:` 류 세션 링크
- `🤖 Generated with …`, `Generated-by:`, `Assisted-by:`, `Authored-by: Claude`, `Signed-off-by: Claude` 등 생성·보조 표시
- `noreply@anthropic.com` 을 포함한 주소, "Claude", "Anthropic", "AI", "LLM", 모델명·도구명 언급
- 이모지 (사용자가 명시적으로 요구한 경우 제외)

실행 규칙:
- `git commit -q -m "제목"` 을 기본으로 하고, 본문이 필요하면 `-m` 을 한 번 더 쓴다. `--trailer` 옵션은 쓰지 않는다. heredoc 을 쓸 때도 마지막 줄에 아무것도 덧붙이지 않는다.
- **커밋 직후마다** `git log -1 --format=%B` 를 grep 해 0건임을 확인하고, push 직전에 이번에 만든 범위 전체를 한 번 더 검사한다 (9단계).
- 걸리면 그 자리에서 고친다 — HEAD 면 `git commit --amend -m …`, 그 이전 커밋이면 그 커밋까지 `git reset --soft` 로 풀고 다시 커밋한다. 검증 결과를 보고에 그대로 적는다.

## 커밋 메시지 컨벤션

형식: `[태그] 영역 : 변경 내용 요약 (핵심 변경 2~4개, 선택)` — 한국어. 식별자·파일명·API 이름은 원문 유지.

태그:
- `[FEAT]` 새 기능·기능 확장 · `[FIX]` 버그 수정 · `[DESIGN]` UI/UX·디자인 · `[REFACTOR]` 동작 변경 없는 정리
- `[DOCS]` 문서(SCHEMA.md, README 등) · `[CONFIG]` 설정(docker, settings, nginx, .gitignore, .claude/commands 등) · `[CHORE]` 빌드·의존성·구조 등 잡무 · `[Merge]` 병합

영역:
- 도메인: `PP`(공정) `DB`(공정설계) `DP`(파트설계) `MC`(장비) `MT`(소재) `AD`(이상탐지/세그멘테이션)
- 화면: `전처리`, `레이블링`, `데이터 분석`, `개발자 페이지`, `공정일지`, `예지보전`, `소재 관리` 등 **사용자가 대화에서 부르는 이름**
- 여러 도메인은 `/` 로 잇고(`DP/DB/PP`), 전역이면 `공통`

예시 (실제 히스토리):
```
[FEAT] 전처리 : 전처리 완료 공정 레이블링 페이지 추가 (카드 '레이블링 하러가기' 버튼, /Dashboard/Labeling 라우트, 비전 분석 레이아웃 재사용)
[FEAT] AD/PP : 레이블링 마스크 저장·조회·되돌리기 API, Vision 레이어 구간 조회 API 추가
[DESIGN] MC/MT : 장비 정보·관리·예지보전, 소재 관리·물성 화면 모바일 폴리시 (필터 세로 스택, 44px 터치 컨트롤)
[FIX] 예지보전 : 페이지 새로고침 시 무한 정지(잠금 대기·커넥션 풀 경합) 수정
[CONFIG] 공통 : 프로젝트 슬래시 커맨드 폴더 추적 허용
```

## 절차

### 1. 인자 해석

| 인자 | 의미 |
|---|---|
| `plan`, `--dry-run` | 계획 표까지만 보여주고 멈춘다 |
| `--no-push` | 커밋만 하고 push 는 생략한다 |
| 파일 경로 | Task 정의 문서(계획서)로 참고한다 |
| 그 외 텍스트 | 묶음 기준·메시지 힌트 ("프론트와 백엔드 따로", "문서는 별도 커밋", 제목 문구 등) |

### 2. 상태 수집

```bash
git status --porcelain=v1 -uall      # 수정·신규·삭제 전체 (디렉터리 접지 않음)
git diff --cached --stat             # 이미 스테이징된 것
git diff --stat
git log --oneline -15                # 메시지 스타일 확인
git rev-parse HEAD                   # 시작 지점 — 보고·되돌리기에 쓴다
git branch --show-current; git remote -v; git rev-parse --abbrev-ref @{upstream} 2>/dev/null
```

- 변경이 없으면 그렇게 말하고 끝낸다.
- 이미 스테이징된 것이 있으면 `git reset -q` 로 **일단 모두 내린다** (인덱스만 비우고 작업 트리는 그대로). 이전 작업의 부분 스테이징이 섞이면 분류가 어긋난다.

### 3. 커밋하면 안 되는 것

| 종류 | 예 | 처리 |
|---|---|---|
| 비밀 | `settings.ini`, `.env`, `settings.local.json`, `*.pem`, 토큰·비밀번호가 적힌 파일 | 커밋하지 않는다. `.gitignore` 에 있어도 `-f` 로 올리지 않는다 |
| 생성물 | `dist/`, `node_modules/`, `__pycache__/`, `venv/`, `android/app/build/`, 로컬 DB | 커밋하지 않는다. 필요하면 `.gitignore` 줄을 제안만 한다 |
| 대용량 바이너리 | 모델 가중치, 데이터셋, 녹화 파일 | 커밋하지 않고 보고한다 |
| 임시 파일 | 디버그 스크립트, 캡처 이미지, AI 답변 저장본 | 대화에서 쓰기로 한 것이 아니면 빼고 보고한다 |

### 4. Task 분류 — 대화가 1순위

분류 근거, 우선순위 순서:

1. **이번 대화에서 사용자가 요청한 작업 단위.** 요청 하나 = Task 하나가 기본. 그 안에서 목적이 다른 부분(기능 추가 / 버그 수정 / 디자인 / 문서 / 설정)은 더 나눈다.
2. 사용자의 인자·힌트.
3. 코드 구조: 도메인·모듈 경계, 의존 방향(공용 → 사용처), 파일 수정 시각(`ls -lt`).
4. **대화와 무관한 변경**(세션 전부터 있던 수정)은 코드만으로 분류해 별도 커밋으로 두고, 메시지에 추측을 쓰지 않는다.

묶는 규칙:

- 목적이 다르면 다른 Task — 파일이 겹쳐도 분리. 도메인이 다르면 다른 Task — 단, 한 기능의 프론트+백엔드는 한 Task.
- 되돌릴 때 함께 되돌려야 하는 것은 한 Task (라우트 등록 + 새 페이지 + 메뉴 활성 경로).
- 테스트는 대상 코드와 같은 커밋. 문서(SCHEMA.md 등)는 그 문서를 바꾼 원인 Task 에 붙이고, 독립적인 문서 정비만 `[DOCS]` 단독. 스키마 변경은 `dbOntology.js` 갱신과 같은 커밋.
- 크기: 리뷰어가 몇 분 안에 읽을 양. 넘으면 틈을 찾되, 따로 떼면 의미가 없는 변경(이름 바꾸기와 호출부)은 억지로 나누지 않는다.
- 한 파일에 여러 Task 가 섞이면 hunk 단위로 스테이징한다 (6단계). 그럴 가치가 없으면 그 파일이 처음 필요해지는 커밋에 통째로 넣고 메시지에 언급한다.

커밋 순서 — 의존 방향대로, 어느 커밋에서 체크아웃해도 빌드·import 가 깨지지 않게:
1. 설정/빌드 (`[CONFIG]`, `[CHORE]`)  2. 공통/기반  3. 백엔드 API  4. 프론트 도메인별 (API 를 쓰는 프론트는 그 뒤)  5. 문서 단독

계획은 이 표로 적는다:

```
| # | 태그 | 영역 | 파일 (hunk 범위) | 제목 |
|---|------|------|------------------|------|
| 1 | CONFIG | 공통 | .gitignore, .claude/commands/commit.md | … |
| 2 | FEAT | 개발자 페이지 | VIEW_DeveloperPage.vue, COM_DbSchemaOntology.vue(신규) | … |
```

`plan` 인자면 표를 보여주고 끝낸다. 아니면 표를 보여준 뒤 **바로 실행한다** — 사용자는 이미 커밋을 요청했다. 멈추고 묻는 경우는 세 가지뿐: 비밀로 보이는 파일이 있을 때, 묶는 방식에 따라 결과가 크게 달라질 때, 원격이 앞서 있어 push 가 거부될 상황일 때.

### 5. 메시지 작성 — 코드와 대화를 함께 본다

- **제목**: 위 컨벤션 형식. 영역과 요약은 **사용자가 대화에서 쓴 표현**으로 쓴다 (사용자가 "개발자 페이지에 DB 스키마 탭" 이라고 했으면 영역 `개발자 페이지`, 요약에 "DB 스키마 탭"). 괄호 안에 핵심 변경 2~4개를 열거해도 된다. 바뀐 것을 끝없이 나열하기 시작하면 커밋을 더 나눌 때다.
- **본문** (선택, 1~4줄, `-m` 하나 더): 코드만 봐서는 알 수 없는 **왜**와 대화에서 정해진 결정·제약 — 예: "실제 DB 조회 없이 정적 모델로 둔다", "레거시 테이블은 토글로만 표시". 사용자가 중간에 바꾼 요구는 최종 상태만 쓴다. 시행착오, 번복 과정, 커밋을 어떻게 나눴는지 같은 메타 설명은 메시지가 아니라 보고에 쓴다.
- **없는 것**: 트레일러, 서명, 이모지, 도구 이름.

나쁜 예: `Update files`, `WIP`, `작업 커밋`, 그리고 트레일러가 붙은 모든 것.

### 6. 실행 — 한 번에 커밋 하나

```bash
git add -- path/a path/b                   # 경로 명시. git add -A / git add . 금지
git diff --cached --stat                   # 의도한 파일만 들어갔는지
git commit -q -m "[FEAT] 영역 : 제목" -m "본문(선택)"
git log -1 --format=%B | grep -i -E "claude|anthropic|co-authored|generated|🤖|클로드" || echo "OK: 흔적 없음"
```

- 삭제된 추적 파일도 `git add -- 경로` 로 스테이징된다. 디렉터리를 통째로 넣을 때는 그 안의 파일 전부가 이 커밋 것인지 먼저 확인한다.
- **줄바꿈**: 이 저장소는 CRLF (`core.autocrlf=true`). 새로 만든 파일이 LF 면 커밋 전에 CRLF 로 맞춘다 (`file <path>` 로 확인). Bash 툴은 명령 안의 백슬래시를 지우므로 `sed 's/$/\r/'` 류는 깨진다 — PowerShell 로 `[IO.File]::WriteAllText` 변환을 쓴다.
- **hunk 분리** (추적 중인 파일 하나에 여러 Task 가 섞였을 때, `git add -p` 는 대화형이라 못 쓴다):
  ```bash
  git diff -- path/to/file.vue | grep -n "^@@"                      # hunk 번호 확인
  git diff -- path/to/file.vue | awk -v keep="2,3,5" '
    BEGIN { n=split(keep,k,","); for(i=1;i<=n;i++) K[k[i]]=1; h=0 }
    /^@@/ { h++ }
    { if (h==0 || (h in K)) print }
  ' > "$SCRATCH/file_taskA.patch"                                    # 스크래치패드에, 프로젝트 안에 두지 않는다
  git apply --cached --check "$SCRATCH/file_taskA.patch" && git apply --cached "$SCRATCH/file_taskA.patch"
  ```
  한 hunk 안에서 일부 줄만 빼야 하면 패치의 해당 `+` 줄을 지우고 `@@ -a,b +c,d @@` 의 `d` 를 그만큼 줄인다. 남은 hunk 는 다음 Task 커밋에서 파일째 `git add` 하면 따라온다.

### 7. push

```bash
git fetch -q origin
git rev-list --left-right --count <브랜치>...origin/<브랜치>    # ahead  behind
git push -u origin <브랜치>                                      # upstream 이 이미 있으면 git push
```

- 원격 선택: upstream > `origin` > 원격이 하나뿐이면 그것 > 여럿이고 불명확하면 묻는다. 이 저장소의 `dev` 원격(API_DEV)은 요청이 없으면 쓰지 않는다.
- 원격이 앞서 있으면(behind > 0) push 하지 않고 보고한다. 자동 pull·rebase·merge 를 하지 않는다.
- `--force` 금지. 인증 실패·거부는 그대로 보고한다 (커밋은 로컬에 남아 있다).

### 8. 검증과 보고

```bash
git status --short                                   # 의도적으로 뺀 파일만 남아야 한다
git log --oneline <시작>..HEAD
git show --stat <각 커밋>                            # 의도한 파일만 들어갔는가 (혼합 파일은 hunk 도 확인)
git log --format='%an <%ae>%n%B' <시작>..HEAD | grep -i -E "claude|anthropic|co-authored|generated|🤖|클로드" || echo "AI 흔적 없음"
```

보고는 짧게: 커밋 목록(해시·제목), 뺀 파일과 이유, push 결과(원격/브랜치, ahead/behind), 흔적 검증 결과, 되돌리는 법 한 줄 (`git reset <시작>` 은 커밋만 취소하고 파일은 그대로 둔다 — push 한 뒤라면 원격도 되돌려야 하므로 먼저 말한다).

## 하지 않는 것

- force push, rebase, 기존 커밋 수정 (방금 만든 커밋의 흔적 제거만 예외), 브랜치 생성·전환. 요청이 없으면 **현재 브랜치**에 커밋하고 그 브랜치를 push 한다.
- 코드 수정. 린트 실패나 버그가 보이면 보고만 한다. `.gitignore` 에 줄을 추가하는 것도 수정이다 — 제안만.
- 테스트 실행. 최종 트리는 시작할 때와 같으니 동작이 달라질 일은 없다. 사용자가 원하면 그때 한다.
