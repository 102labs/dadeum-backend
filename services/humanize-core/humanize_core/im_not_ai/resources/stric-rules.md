# Stric Rules

윤문 대상 사례 리스트업 전용 compact rulebook. 이 파일은 부자연스러운 패턴, 대표 신호, 수정 방향을 찾는 데만 쓴다.

## Severity

- S1: 한 번만 나와도 AI 티가 강한 패턴. 보존 규칙과 충돌하지 않으면 우선 수정.
- S2: 반복되거나 한 문단에 몰리면 AI 티가 강한 패턴. 밀도와 문맥을 보고 수정.
- S3: 단독으로는 약한 신호. 다른 문제와 겹칠 때만 가볍게 조정.

## Protect

고유명사, 제품명, 모델명, 기관명, 숫자, 날짜, 단위, URL, 코드, 법률·규정 조문, 수식, 직접 인용은 탐지·윤문 대상이 아니라 보존 대상이다. 업계 표준 약어인 AI, API, GPU, LLM, SDK 등은 기본 보존한다.

## A. Translationese

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| A-1 | S1 | "~에 대해/대해서" 남발 | 목적격 조사나 직접 서술로 바꾼다 | 자연스러운 주제 제시는 유지 |
| A-2 | S1 | "~을/를 통해/통하여" 남발 | "~로", "~해서", 행위 동사로 푼다 | 실제 경로·매체 의미는 유지 |
| A-3 | S1 | "~에 있어/있어서" | "~에서", "~을 볼 때"로 줄인다 | 법률·공식 문구는 보존 |
| A-4 | S2 | "~라는 점에서" 반복 | "~서", "~라는 이유로" 등으로 분산 | 핵심 근거 연결은 유지 |
| A-5 | S2 | "~와 관련하여/관련된" | "~에", "~의", 직접 목적어로 줄인다 | 전문 용어 일부는 유지 |
| A-6 | S2 | "~에 기반하여/~을 바탕으로" 반복 | "~로", "~을 보고", 직접 판단으로 바꾼다 | 실제 근거 출처는 보존 |
| A-7 | S1 | "가지고 있다", have/make/take/give 직역 | 형용사·동사형으로 환원한다 | 소유 의미가 실제면 유지 |
| A-8 | S1 | 이중 피동 "~되어진다/~지게 된다" | 능동 또는 단일 피동으로 줄인다 | 책임 주체가 불명확해야 할 때만 피동 유지 |
| A-9 | S2 | "~에 의해" 피동 | 행위자를 주어로 복귀한다 | 법률·학술 인용의 고정 표현은 보존 |
| A-10 | S2 | "~할 수 있다" 가능형 반복 | 단언 가능한 곳은 현재형·확정형으로 바꾼다 | 실제 가능성·불확실성은 유지 |
| A-11 | S2 | "~을 위해" 목적절 반복 | "~려고", "~하도록", 직접 동사로 바꾼다 | 목적 강조가 필요한 곳은 유지 |
| A-12 | S2 | "만들어지다/이루어지다" 피동 | "만들다", "합의했다"처럼 주체·행위로 푼다 | 사건 자체가 중요하면 단일 피동 유지 |
| A-13 | S2 | 조사 빠진 명사 나열 | 조사와 서술어를 복원한다 | 제목·표 항목은 유지 |
| A-14 | S2 | "그리고"로 평문을 계속 연결 | "-고", "-며", 문장 통합·분리로 조정한다 | 의도적 병렬은 유지 |
| A-15 | S2 | 무생물·추상 주어 + 사역·인지·발화 동사 | 사람·팀·회사·행위 중심, 원인절, 또는 "...에 따르면" 구문으로 다시 쓴다 | 원문 주체 의미는 보존 |
| A-16 | S1 | 그/그녀/그것/그들 직역 반복 | 50~70%는 생략하고 필요 지점만 호칭·명사구로 바꾼다 | 지시 대상 혼동이 있으면 명사구로 명시 |
| A-17 | Hold | 무정물·추상명사 "-들" 복수 표지 | 현재는 active finding으로 쓰지 않는다. PE/NMT 점검 때는 분포성 없으면 삭제 후보 | 향후 평가용 ID로만 유지 |
| A-18 | S2 | 긴 좌향 수식·관계절 직역 | 문장 분리 또는 후치 동격절로 푼다 | 전문 명칭 내부 수식은 보존 |
| A-19 | S2 | "-에서의/-에로의/-으로의/-에의/-으로부터의/-로부터의" | 절·구로 풀어쓴다 | 단순 "~의"만으로는 탐지하지 않음 |

### A Operational Notes

- A-2 example: "데이터 분석을 통해 인사이트를 얻는다" -> "데이터를 분석해 인사이트를 얻는다". Do not flag literal channels like "API를 통해 호출한다" unless repeated mechanically.
- A-7 example: "강한 경쟁력을 가지고 있다" -> "경쟁력이 강하다". Do not change real possession, ownership, or inventory statements.
- A-7 also covers light verb constructions: "회의를 가지다" -> "회의를 했다", "결정을 만들다" -> "결정했다", "한번 봄을 가지다" -> "한번 보다".
- A-15 has three v2.0 recipes: causative "X made Y..." -> "X 때문에/덕분에/로 인해 Y...", cognitive/reporting verbs "suggest/show/indicate/reveal" -> "...에 따르면 ...이다" or "...으로 ...이 드러났다", and "This book has 300 pages" -> "이 책은 300쪽이다".
- A-12 example: "합의가 이루어졌다" -> "합의했다" when the actor is clear. Keep passive when the actor is intentionally unknown or irrelevant.
- A-13 finding needs a readable noun pile, not every compact title. Flag "AI 기술 발전 속도 가속화" in prose; keep table headers and product labels.
- A-14 becomes important when "그리고" repeatedly links flat sentences. A single natural "그리고" is not a finding.
- A-16 should trigger when a paragraph has three or more English-style pronouns or awkward possessives. Korean usually omits "그/그녀/그의" when reference is obvious. Mr./Ms./Dr. literal handling also belongs here: use Korean titles such as "선생님", "박사님", "과장님", or omit when natural.
- A-17 remains hold because user corpora produced no positive cases, but PE5 keeps the repair: remove inanimate/abstract "-들" unless it marks real distributivity; use "여러", "다양한", "갖가지", "저마다", "각자" only when needed.
- A-18 should trigger when the head noun arrives too late because modifiers stack. Prefer splitting after the head noun or using "그 X는..." only when reference remains clear.
- A-19 excludes ordinary "~의". Flag only compound postpositions such as "-에서의" and "-으로부터의". Repeated "의 의 의" is not an A-19 hit by itself, but it is a strong rewrite cue to unpack the phrase.

## B. English Terms And Quotations

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| B-1 | S2 | 모든 용어에 괄호 영어 병기 | 첫 등장 1회만 병기하고 이후 한국어 중심 | 표준 약어·제품명은 유지 |
| B-2 | S2 | 한국어로 충분한 영어 단어를 그대로 사용 | 자연스러운 한국어 업무어로 옮긴다 | API, SDK 등 표준 용어는 유지 |
| B-3 | S2 | 영어 문장 인용을 과하게 삽입 | 필요하면 한국어로 풀고 출처만 남긴다 | 원문 표현 자체가 논점이면 유지 |
| B-4 | S3 | "~라고 알려진/~로 일컬어지는" 직역 | 짧은 명칭 또는 괄호 병기로 줄인다 | 정의가 필요한 첫 등장에는 허용 |

### B Operational Notes

- B-1 is about repeated parenthetical glosses, not one useful first-use expansion. Keep standard abbreviations and product names untouched.
- B-2 should avoid forced translation of industry terms. Flag "framework를 leverage" style wording; keep API, SDK, OAuth, Transformer, and named models.
- B-2 common replacements: framework -> 체계/틀/구조, leverage -> 활용하다/기대다, seamless -> 매끄러운/끊김 없는, robust -> 튼튼한/견고한, scalable -> 확장성 있는, insight -> 통찰/눈/시사점, impact -> 영향/파장, holistic -> 전체적/총체적. "pipeline" can stay when it is an actual technical pipeline.
- B-3 is a finding when English quotations interrupt Korean flow without analytic need. Do not translate direct quotes when exact wording is protected or legally relevant.
- B-4 is low priority. Include it only when it stacks with other translationese or appears several times.

## C. Structural AI Patterns

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| C-1 | S1 | "첫째/둘째/셋째"가 글 전체를 지배 | 일부를 산문으로 녹이고 항목 길이를 조정 | 실제 절차·순서는 유지 |
| C-2 | S2 | 불릿이 3개 이상 연속되고 설명이 얕음 | 나열 의미가 약하면 문단으로 합친다 | 체크리스트·요건 목록은 유지 |
| C-3 | S2 | 반복적 일반 헤딩 | 구체 헤딩으로 바꾸거나 삭제한다 | 사용자가 준 구조는 보존 |
| C-4 | S2 | 모든 문단 첫 문장이 요약문 | 일부 문단은 사례·결과·맥락으로 시작 | 논리적 안내가 필요한 문단은 유지 |
| C-5 | S1 | 이모지 머리표·강조 | 업무 글에서는 삭제한다 | 사용자가 의도한 캐주얼 톤이면 최소 유지 |
| C-6 | S2 | 헤딩 아래 "이 섹션에서는..." 안내문 | 삭제하고 본문으로 바로 들어간다 | 긴 문서의 필수 안내는 유지 |
| C-7 | S2 | "먼저/반면/결국" 3단 공식 | 접속사를 줄이고 문단 흐름으로 연결 | 실제 대비·결론 표지는 유지 |
| C-8 | S2 | "A인가, B인가" 대칭 대구 반복 | 한 번만 살리고 나머지는 평서로 바꾼다 | 핵심 문제제기 1회는 유지 |
| C-9 | S2 | "1) 2) 3)" 숫자 괄호 인덱싱 | 산문이나 단순 줄바꿈으로 바꾼다 | 계약·절차 번호는 유지 |
| C-10 | S2 | 콜론 부제 헤딩 "X: Y", "X: A에서 B로" 반복 | 짧은 헤딩 또는 평서형 제목으로 정리 | 고유 제목 형식은 보존 |
| C-11 | S1 | -고/-며/-지만/-면서/-아서/-어서/-자/-는데 뒤 쉼표 | 불필요한 쉼표를 제거한다 | 긴 삽입구 경계는 유지 |
| C-12 | S2 | 쉼표 포함 문장이 과도하게 많음 | 일부를 마침표, 연결어미, 삭제로 분산 | 의미 경계가 필요한 쉼표는 유지 |

### C Operational Notes

- C-1 is S1 only when the enumeration controls the passage and makes every paragraph predictable. Keep real procedures, requirements, and ordered steps.
- C-2 should not flatten useful checklists. Flag when bullets are shallow, repetitive, and could read better as a paragraph. Fast rules also treat column/report bullet blocks as a J-style visual issue, but the actual case is covered here.
- C-3/C-10 are repeated-pattern findings. One clear heading or one colon title is usually not enough. C-10 includes headings like "서론: 제조업의 미래" and "본론 1: 빛과 그림자".
- C-4 is document-level. Flag when most paragraphs start with summary claims and the prose feels like an outline.
- C-6 targets boilerplate guide sentences such as "이 섹션에서는..." under every heading. Keep one navigational sentence in long instructions if needed.
- C-9 excludes legal, contractual, or procedural numbering. For prose, prefer a sentence sequence or plain line breaks.
- C-11 is strong because Korean rarely needs a comma immediately after connective endings. Keep only when it separates a long inserted clause.
- C-12 is distributional. Treat it as a finding when many sentences have commas and the rhythm feels segmented, not when a single long sentence needs punctuation.

## D. Signature Phrases

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| D-1 | S1 | 결론적으로/따라서/이를 통해/요약하면/정리하면 반복 | 1~2개만 남기고 문맥 종결로 바꾼다 | 실제 결론 표지는 유지 |
| D-2 | S1 | "시사하는 바가 크다/주목할 만하다" 같은 의의 과장 | 삭제하거나 구체 결론으로 바꾼다 | 평가 근거가 있으면 구체화 |
| D-3 | S1 | "크게 세 가지", "다음과 같은 특징", "다음과 같이 요약" 열거 도입 또는 "본질적으로/핵심적으로" 공허한 강조 | 대부분 삭제하거나 바로 본문으로 들어간다 | 논리 초점 표시 1회는 허용 |
| D-4 | S1 | 혁신적/전례 없는/파격적/압도적/폭발적/획기적/치명적 등 hype | 구체 사실·수치·효과로 낮춘다 | 원문이 광고 문구면 강도만 낮춤 |
| D-5 | S2 | 기술·시대·시장·전략이 묻는다/요구한다/증명한다/질문을 던진다 | 사람·조직·상황 주어로 바꾼다 | 은유가 원문 핵심이면 유지 |
| D-6 | S2 | "~할 때입니다/~시점입니다/~할 순간입니다/~해야 할 때다/지금이야말로" 결말 공식 | 평서형 결론으로 닫는다 | 명확한 행동 촉구가 필요하면 약화 |
| D-7 | S2 | "X에서 Y로/X을 넘어 Y로" 변환 공식 반복 | 한 번만 남기고 일반 서술로 바꾼다 | 핵심 슬로건 1회는 유지 |

### D Operational Notes

- D findings should not invent stronger claims. Replace empty emphasis with the original concrete point, or delete it if it adds no information.
- D-1 includes "~라고 할 수 있다", "~라고 볼 수 있다", "~라 하겠다", "~라 할 것이다", "~에 다름 아니다", and the KatFish lexicon "결론적으로/따라서/이를 통해/그러므로". It becomes S1 when several pivot phrases appear as paragraph openers or closers. A single "따라서" can remain if it carries logic.
- D-2 includes "매우 중요하다", "반드시 기억해야 한다", "간과할 수 없다", "무시할 수 없다", "~의 지평을 연다", "~에 방점을 찍는다", "그 의미가 적지 않다", "의미심장하다". Delete or replace with concrete grounds.
- D-3 taxonomy source is list-introduction boilerplate; fast quick-rules also uses D-3 for empty focus adverbs. Keep both as deletion-first cues.
- D-4 includes "막강한", "대대적", "강력한", "치열한", "뜨거운", "~의 가능성을 열어준다", "~의 새로운 장을 열다", "~시대가 도래했다". Lower hype, not the user's actual evaluation. If the original has evidence, keep the claim and make wording more specific.
- D-5/D-6 preserve the business conclusion while removing formulaic drama. For D-7, convert slogans like "'지식 전달자'에서 '학습 조력자'로" into direct sentences: "교사는 지식만 전달하지 않는다. 학생 곁에서 학습을 돕는다."

## E. Rhythm And Sentence Shape

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| E-1 | S2 | 문장 길이가 지나치게 균일 | 단문·중문·장문을 섞는다 | 짧은 안내문은 과하게 늘리지 않음 |
| E-2 | S2 | 동일 종결어미·진행형 반복 | 종결과 시제를 자연스럽게 다양화한다 | register는 유지 |
| E-3 | S2 | 모든 문단이 3~4문장 공식 | 문단 길이를 의미 단위로 조정한다 | 사용자가 준 문단 경계는 존중 |
| E-4 | S2 | 단문 일변도 | 일부 문장을 연결어미·조건절로 묶는다 | 강조 단문은 유지 |
| E-5 | S2 | 쉼표 분절이 길고 무겁다 | 문장 분리 또는 절 재배치로 낮춘다 | 명확한 병렬 구조는 유지 |
| E-6 | S2 | 쉼표 주변 구조가 과도하게 복잡하다 | 구문을 단순화하고 핵심 술어를 앞세운다 | 전문 개념 병렬은 유지 |
| E-7 | S2 | 해라/하게/하오/해요/합쇼체 register 혼용 | 하나의 register로 통일한다 | 원문 화자 전환은 보존 |

### E Operational Notes

- E-1/E-3 are passage-level rhythm findings. Do not flag a short 2~3 sentence input just because lengths are similar.
- E-2 should preserve the requested register. Vary endings inside the same register rather than switching from formal to casual. A "~다" ending streak of four or more sentences is a PE9 trigger. Check whether repetitive "~고 있다" can return to a simple tense: "읽고 있다" -> "읽는다", "생각해 오고 있다" -> "생각해 봤다" when context allows.
- E-4 is not "short sentence bad". Flag only when every sentence has the same clipped shape and no flow.
- E-5/E-6 should improve readability by splitting or reordering. E-5 catches long comma-separated clauses; E-6 catches indiscriminate comma placement after nouns, adverbs, verbs, modifiers, quotations, and connectors. Do not add rhetorical connectors just to vary rhythm.
- E-7 is a preservation issue as much as a style issue. Apply it mainly to dialogue, spoken text, interviews, and quoted speech where speaker-listener relation matters. Formal reports usually just need one formal register maintained.

## F. Over-Modification And Redundancy

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| F-1 | S2 | 매우/상당히/굉장히/정말/진짜로/대단히/극히 등 정도부사 중독 | 약한 부사는 삭제하고 필요한 곳만 남긴다 | 실제 강도 정보는 유지 |
| F-2 | S2 | 중요하고 핵심적인/새롭고 혁신적인/지속적이고 꾸준한 등 동의어 이중 수식 | 하나만 남기거나 구체화한다 | 의미가 다른 병렬 수식은 유지 |
| F-3 | S2 | 기능+역할, 역할과 기능, 의미와 가치 복합구 반복 | 실제 행위와 대상으로 풀어 쓴다 | 제품 기능명은 보존 |
| F-4 | S2 | "~적 측면/~적 관점", -성/-적/-화, -tion/-ment/-ness/-ity 명사화 누적 | 동사·형용사·구체 명사로 해체한다 | 전문 용어는 유지 |
| F-5 | S2 | "~적 N" 추상어 체인 | 명사+명사 또는 풀어쓰기로 바꾼다 | 고정 학술 용어는 보존 |

### F Operational Notes

- F-1 should not erase measured intensity. Delete weak boosters when they only inflate tone; keep "매우 낮은 지연" if it is a meaningful technical claim.
- F-2 example: "중요하고 핵심적인 과제" -> "핵심 과제". Keep both modifiers only when they add distinct information.
- F-3 example: "사용자 경험 개선 역할을 수행한다" -> "사용자 경험을 개선한다". Preserve feature names and product labels.
- F-4/F-5 are common sources of stiff prose. Prefer verbs and concrete nouns, but keep fixed terms like "전략적 제휴" when changing them would alter meaning. When hanja nominalizers "-성/-적/-화" exceed 12 uses in a document, strengthen the S2 finding. For English nominalizations, "the implementation of the policy" can become "정책 시행" or "정책을 시행하기". F-5 examples include "에이전트적 자율성", "기술적 안정성", "경제적 자립", "시스템적 접근".

## G. Hedging

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| G-1 | S2 | "~것이다/~할 것이다", "~할 수 있을 것으로 보인다", "~로 보인다/~인 듯하다/~라고 여겨진다/~인 것으로 판단된다" 반복 | 단언 가능한 곳은 현재형·확정형으로 쓴다 | 실제 전망·불확실성은 유지 |
| G-2 | S2 | "~할 가능성이 있을 수 있다/~로 보여질 수 있다" 이중·삼중 완곡 | 완곡 표현을 하나만 남기거나 단언한다 | 실제 불확실성은 유지 |
| G-3 | S2 | 양쪽 모두/두 가지 모두/장점도 있지만/신중하게/균형 등 안전 어휘 반복 | 기준과 결론을 더 분명히 쓴다 | 균형 자체가 논점이면 유지 |

### G Operational Notes

- G-1 covers both taxonomy's 추측·관측형 endings and fast rules' future-default endings. Keep real forecasts; only reduce formulas that hide a present conclusion.
- G-3 is strongest in policy/report prose. Replace vague balance language with a criterion, example, or conditional split: "X일 때는 A, Y일 때는 B".

## H. Connectors

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| H-1 | S2 | 또한/따라서/즉/나아가/아울러/게다가/더욱이 등 문두 접속사 과다 | 70% 안팎을 줄이고 문장 흐름으로 연결 | 논리 전환이 필요한 곳은 유지 |
| H-2 | S2 | 하지만/그러나 혼용 남발 | 하나로 통일하거나 문장 구조로 대비를 만든다 | 의미 대비는 유지 |
| H-3 | S2 | "이는/이 점에서/이 관점에서/이 말은" 지시·메타 진입 반복 | 선행 내용을 문장 안에 녹인다 | 지시어가 명확성에 필요하면 유지 |
| H-4 | S2 | "즉" 재정의 남발 | 1회 정도만 남기거나 곧/말하자면/다시 말해/바꿔 말하면으로 변주한다 | 핵심 정의는 유지 |

### H Operational Notes

- H-1 is about density. Do not flag one necessary paragraph transition; flag a pattern of repeated paragraph-start connectors.
- H-2 should make contrast clearer, not mechanically standardize every "하지만" or "그러나".
- H-3 often combines with A/D patterns. Replace vague "이는" with the actual subject when clarity improves.
- H-4 can remain when it introduces a precise definition. Repeated "즉" after every sentence is the problem.

## I. Bound Nouns

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| I-1 | S1 | "~인 것이다/~한 것이다" 결말 | 평서형으로 줄인다 | 강조 결론 1회는 허용 |
| I-2 | S2 | 점/바/수/데 반복, "X은 ~라는 점에 있다" | 일부를 구체 명사나 동사로 바꾼다 | 자연스러운 의존명사는 유지 |
| I-3 | S2 | "~라는 것", "~다는 뜻이다/의미다" 결산 반복 | 직접 명사·절 또는 평서형으로 바꾼다 | 정의 자체는 유지 |
| I-4 | S2 | "~할 필요가 있다", "~해야 한다/합니다", "마련해야/지원해야/구축해야" 권고형 결말 반복 | 주체와 행동을 밝히거나 평서·조건·구체 동사로 변주한다 | 완곡한 권고가 필요하면 유지 |
| I-5 | S2 | "~이/가 필요하다" 반복 | 필요한 주체와 행위를 밝힌다 | 실제 필요 조건은 보존 |
| I-6 | S2 | "~능력" 추상명사 연쇄 | 역량의 대상과 행동을 구체화한다 | 공식 역량명은 보존 |

### I Operational Notes

- I-1 is strong because it often creates formulaic endings. Keep one emphatic ending only if it carries the user's intended stress.
- I-2/I-3 should be handled with sentence-level rewrites, not word-for-word substitutions. "핵심은 ...다는 점에 있다" -> "핵심은 ...다"; "병목이 될 수 있다는 뜻이다" -> "병목은 ...이다" when the claim level allows.
- I-4 repeated policy endings exceed threshold around five uses. Do not mechanically convert every "필요하다" into "해야 한다"; that can create another I-4 finding or alter obligation strength.
- I-4/I-5 example: "개선이 필요하다" -> "팀은 이 부분을 개선해야 한다" when the actor is clear and obligation is intended. If the actor is intentionally omitted, keep a softer form.
- I-6 is a finding when several abstract ability nouns stack. Keep official competency names and product capability labels.

## J. Visual Decoration

| ID | Sev | Signal | Fix | Keep |
|---|---|---|---|---|
| J-1 | S2 | 문장마다 핵심어에 마크다운 볼드 강조 | 대부분 평문으로 돌린다 | 사용자가 준 필수 강조는 유지 |
| J-2 | S2 | 개념어·강조어 따옴표 5회+ | 핵심 1~2개만 남긴다 | 직접 인용은 수정하지 않음 |
| J-3 | S3 | 대시 장식 남용 | 쉼표·괄호·문장 분리로 바꾼다 | 의미상 삽입구는 유지 |
| J-4 | S3 | "(이는 ~을 의미한다)" 같은 괄호 부연 과다 | 본문에 녹이거나 삭제한다 | 약어 첫 등장은 유지 |

### J Operational Notes

- J findings are low priority unless decoration repeatedly substitutes for clear prose.
- J-1/J-2 must distinguish emphasis marks from protected direct quotations. Never rewrite the quoted content itself. Quote emphasis above about five instances in one document is a strong J-2 signal.
- J-3 should remove decorative dash rhythm, not all dashes. Keep ranges, minus signs, and meaningful appositives.
- J-4 should reduce parenthetical clutter after first-use expansions. Keep abbreviations and clarifications that prevent ambiguity.

## K. PE Case Crosswalk

These are the 15 English-Korean post-editing cases from the playbook, mapped back to the rule IDs above.

| PE | Trigger | Fix | IDs |
|---|---|---|---|
| PE1 | 무생물 주어 + 사역·인지 동사 | 원인절, "...에 따르면", 인간 주어로 환원 | A-15, D-5 |
| PE2 | "~에 의해" by-passive | 능동태 또는 단순 조사로 복귀 | A-9 |
| PE3 | "~되어지다/~여지다/잊혀지다/보여지다" | 단순 피동 또는 능동 | A-8 |
| PE4 | 그/그녀/그것/그들 한 단락 3회 이상 | 생략 50% 이상 + 호칭·명사구 | A-16 |
| PE5 | 무정물·추상명사 "-들" | Hold: active finding은 아니지만 PE 점검에서는 대부분 삭제 | A-17 hold |
| PE6 | 명사 앞 3어절 이상 관형구 | 문장 분리 또는 후치 동격절 | A-18 |
| PE7 | have/make/take/give + N 직역 | 동사 환원 또는 이중주어 | A-7 |
| PE8 | "-에서의/-에로의/-으로의/-에의" | 절·구로 풀기, 단순 "~의" 제외 | A-19 |
| PE9 | "~다" 4문장 이상 연속 | 같은 register 안에서 종결 다양화 | E-2 |
| PE10 | "~고 있다" 남발 | 단순 시제 환원 가능성 검토 | E-2 |
| PE11 | -tion/-ment/-ness/-ity 명사 직역 | 동사·형용사·구체 명사로 해체 | F-4 |
| PE12 | "~로부터/~에 관하여/~을 통하여" 전치사구 직역 | 문맥 자연 표현으로 대체 | A-2, A-5 |
| PE13 | 영어 단순 현재·과거 단조 매핑 | 한국어 서사 시제·서법 다양화 | E-2 |
| PE14 | 대화체 화자-청자 관계 누락 | 해라/하게/하오/해요/합쇼체 일관 적용 | E-7 |
| PE15 | Mr./Ms./Dr. 또는 he/she 호칭 직역 | 한국어 호칭 또는 생략 | A-16 |
