# 서버 에이전트 지시문 — 논문 figure 생성 (F1, F4)

`scripts/egoown_figures.py`로 AAAI-27 논문 figure 2종을 생성한다.
개념도(F2 taxonomy, F3 pipeline)는 대상 아님 — 로컬(Albert)에서 제작.

## 0. 사전 조건

```bash
python scripts/egoown_report.py --outputs ./outputs   # egoown_report.csv 갱신
ls outputs/                                            # 모델 디렉토리명 확인
```

- matplotlib 필요 (`pip install matplotlib`).
- **seed-0 런만 사용** — `optseed*` workdir는 스크립트가 자동 제외.
- report에 prompt_version 혼입 경고가 뜨면 중단하고 Albert에게 보고.

## 1. F4 — confusion 2-panel + label-collapse (§5)

대표 모델 3개: **Qwen2.5-VL-32B / EgoThinker / GPT-5.4-mini**.
`<model_dir>`는 `ls outputs/`의 실제 디렉토리명으로 치환하고, `:` 뒤에 논문 표기명을 준다.

```bash
python scripts/egoown_figures.py f4 \
    --outputs ./outputs --report egoown_report.csv --dataset EgoOwn \
    --models "<32B_dir>:Qwen2.5-VL-32B" "<egothinker_dir>:EgoThinker" "<gpt_dir>:GPT-5.4-mini" \
    --out figures/fig4_confusion_collapse
```

- `--dataset EgoOwn` = sparse 3프레임 모드(메인). 파일 매칭은 `*_EgoOwn_acc.csv` 정확 접미사라 `_Single`/`_Blind`와 섞이지 않음.
- panel (b) GT prior 기본값 = n=3,227 분포(.358/.379/.216/.046). 벤치마크가 바뀌지 않았으면 그대로 둘 것.

## 2. F1 — teaser 실패 막대 (§1)

```bash
python scripts/egoown_figures.py f1 \
    --report egoown_report.csv --dataset EgoOwn --human 0.834 \
    --rename "<지저분한 모델명>=<논문 표기명>" ... \
    --out figures/fig1_teaser
```

- human 0.834 = blind audit 전체 수치(Albert 확정).
- probe 행(clip/siglip/egovlp)은 자동으로 연한 색 + 별도 범례 처리됨.

## 3. 검증 (필수)

- panel (a) 대각합/행 분포가 `egoown_main_table.md`의 acc:LABEL과 일치하는지 스팟체크.
- panel (b)에서 §5.3 서사 재현 확인: 3B→SHARED ~75%, Omni-3B→MINE ~57%, EgoThinker→OTHERS(PERSON_k) ~53%.
- F1 최고 VLM(32B sparse) ≈ .542, probe ≈ .80 확인.

## 4. 산출물 전달

`figures/*.pdf`(+ `.png` preview)를 **이 레포에 커밋**하고 브랜치/커밋 해시를 보고.
Slack Connect API 전송 불가 — 파일 전달은 git 경유가 유일하게 동작.
LaTeX에서는 `Figures/fig1_teaser.pdf` 등으로 참조 예정이므로 파일명 변경 금지.
