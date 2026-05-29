# Fast Rewrite Rules

Rewrite 단계에서 사용하는 fast-plus 룰북. 원문 전체를 한 번 자연스럽게 고치고, Audit/review 단계는 문체를 다시 평가하지 않고 잘못 바뀐 의미·수치·날짜·인용·고유명사만 복원한다.

원칙: 먼저 old fast mode보다 한 단계 더 적극적으로 고친다. 원문을 그대로 복사하는 것은 윤문 실패다. 보존 대상 값은 그대로 두고, 나머지 업무 문장은 흐름·어순·리듬·명확성·번역투를 실제로 개선한다.

## Protect

Do not edit: 고유명사, 제품명, 모델명, 기관명, 사람 이름, 숫자, 날짜, 시간, 단위, URL, 이메일, 코드, 법률·규정 조문, 수식, 직접 인용, 사용자가 지정한 protected_terms.

Do not flag: 의미 변화가 없는 어순·조사·접속어·문장 길이 조정, 스타일을 더 좋게 만들 수 있다는 일반 의견, 보존 대상 주변의 안전한 문장 개선.

## A. Translationese

| ID | Sev | Signal | Fix |
|---|---|---|---|
| A-1 | S1 | "~에 대해/대해서" | 목적격 조사나 직접 서술로 바꾼다 |
| A-2 | S1 | "~을/를 통해/통하여" 남발 | "~로", "~해서", 행위 동사로 푼다 |
| A-3 | S1 | "~에 있어/있어서" | "~에서", "~을 볼 때"로 줄인다 |
| A-4 | S2 | "~라는 점에서" 반복 | "~서", "~라는 이유로" 등으로 분산 |
| A-5 | S2 | "~와 관련하여/관련된" | "~에", "~의", 직접 목적어로 줄인다 |
| A-6 | S2 | "~에 기반하여/~을 바탕으로" 반복 | "~로", "~을 보고", 직접 판단으로 바꾼다 |
| A-7 | S1 | "가지고 있다", have/make/take/give 직역 | 형용사·동사형으로 환원한다 |
| A-8 | S1 | 이중 피동 "~되어진다/~지게 된다" | 능동 또는 단일 피동으로 줄인다 |
| A-9 | S2 | "~에 의해" 피동 | 행위자를 주어로 복귀한다 |
| A-10 | S2 | "~할 수 있다" 남발 | 단언 가능한 곳은 현재형·확정형으로 바꾼다 |
| A-11 | S2 | "~을 위해" 목적절 반복 | "~려고", "~하도록", 직접 동사로 바꾼다 |
| A-15 | S2 | 추상 주어 + 사역·인지·발화 동사 | 사람·팀·회사·행위 중심 문장으로 다시 쓴다 |
| A-16 | S1 | 그/그녀/그것/그들 직역 반복 | 대부분 생략하고 필요 지점만 명사구로 바꾼다 |
| A-18 | S2 | 긴 좌향 수식·관계절 직역 | 문장 분리 또는 후치 동격절로 푼다 |
| A-19 | S2 | "-에서의/-으로부터의" 등 복합 조사 | 절·구로 풀어쓴다 |

A-2 example: "데이터 분석을 통해 인사이트를 얻는다" -> "데이터를 분석해 인사이트를 얻는다". 실제 경로·매체 의미인 "API를 통해 호출한다"는 유지한다.

## B. English Terms And Quotations

| ID | Sev | Signal | Fix |
|---|---|---|---|
| B-1 | S2 | 모든 용어에 괄호 영어 병기 | 첫 등장 1회만 병기하고 이후 한국어 중심 |
| B-2 | S2 | 한국어로 충분한 영어 단어를 그대로 사용 | 자연스러운 한국어 업무어로 옮긴다 |
| B-3 | S2 | 영어 문장 인용을 과하게 삽입 | 필요하면 한국어로 풀고 출처만 남긴다 |

## C. Structural AI Patterns

| ID | Sev | Signal | Fix |
|---|---|---|---|
| C-1 | S1 | "첫째/둘째/셋째"가 글 전체를 지배 | 일부를 산문으로 녹이고 항목 길이를 조정 |
| C-5 | S1 | 이모지 머리표·강조 | 업무 글에서는 삭제한다 |
| C-7 | S2 | "먼저/반면/결국" 3단 공식 | 접속사를 줄이고 문단 흐름으로 연결 |
| C-9 | S2 | "1) 2) 3)" 숫자 괄호 인덱싱 | 산문이나 단순 줄바꿈으로 바꾼다 |
| C-10 | S1 | 콜론 부제 헤딩 반복 | 짧은 헤딩 또는 평서형 제목으로 정리 |
| C-11 | S1 | -고/-며/-지만 뒤 불필요한 쉼표 | 쉼표를 제거한다 |

## D. Signature Phrases

| ID | Sev | Signal | Fix |
|---|---|---|---|
| D-1 | S1 | "결론적으로/따라서/이를 통해/요약하면" 반복 | 1~2건만 남기고 삭제·분산 |
| D-2 | S1 | "시사하는 바가 크다/주목할 만하다" | 삭제하거나 구체 결론으로 |
| D-3 | S1 | "본질적으로/핵심적으로" | 삭제 |
| D-4 | S1 | 파격적·압도적·획기적 같은 hype | 구체 사실 중심으로 환원 |
| D-6 | S1 | "~할 때다/~해야 한다" 공식 결말 | 평서로 닫거나 삭제 |

## E/F/G/H/I. Rhythm And Clarity

| ID | Sev | Signal | Fix |
|---|---|---|---|
| E-1 | S2 | 문장 길이가 지나치게 균일 | 단문과 장문을 섞어 리듬 조정 |
| E-2 | S2 | 동일 종결어미 반복, "~고 있다" 반복 | 시제·종결을 자연스럽게 분산 |
| F-4 | S2 | 한자어 명사화 -성/-적/-화 누적 | 동사·형용사 표현으로 환원 |
| F-5 | S2 | "~적 N" 추상 체인 | 명사+명사 또는 풀어쓰기 |
| G-1 | S2 | "~것이다/~할 것이다" 미래 단정 남발 | 현재형·확정형으로 |
| G-2 | S2 | "~로 보인다/~인 듯하다" 추정 남발 | 단언 가능한 곳은 단언 |
| H-1 | S1 | 또한·따라서·즉·나아가 등 문두 접속사 남발 | 접속사를 줄이고 문장 자체로 흐름 형성 |
| I-1 | S1 | "~인 것이다/~한 것이다" 결말 | 평서형으로 |
| I-4 | S2 | 권고형 결말 "~해야 한다/합니다" 반복 | 평서·단언으로 분산 |

## Rewrite/Audit Contract

- Rewrite returns the complete revised passage, not findings, excerpts, or a continuation.
- Rewrite must make visible improvements when the source has S1/S2 signals, repeated wording, translationese, or stiff business prose.
- Rewrite should apply the rulebook slightly more assertively than the old fast mode because preservation audit runs afterward.
- Prefer S1/S2 fixes over exhaustive low-value S3 polishing.
- A high change-rate target is not a goal. The goal is better naturalness with preserved meaning.
- Audit flags only harmful preservation problems: changed facts, numbers, dates, names, quotes, protected terms, order, polarity, causality, omitted content, or added claims.
- Review applies only audit corrections. It must not do a second style rewrite.

## Rewrite Quality Grade

- A: S1 잔존 0건, S2 잔존 2건 이하, 변경률 15~35%, self-check 모두 통과.
- B: S1 잔존 0건, S2 잔존 4건 이하, self-check 대부분 통과.
- C: S1 잔존 1~2건 또는 self-check 일부 미통과. warning에 남긴다.
- D: 의미 보존 실패, 직접 인용·수치·고유명사 훼손, 출력 잘림, 또는 원문과 실질적으로 같은 no-op 결과.
