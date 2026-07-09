# VLMEvalKit-EGO — Egocentric Implicit Ownership 평가 하네스

open-compass [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 포크에 **EgoOwn 벤치마크**(egocentric implicit ownership, 4-class MCQ)를 얹은 평가 코드입니다. AAAI-27 논문의 전 실험 세팅(§5)을 단일 인터페이스로 실행합니다.

원본 VLMEvalKit 문서는 [docs/README_upstream.md](docs/README_upstream.md) 참조.

---

## 1. 설치

```bash
git clone <this-repo> && cd eval-code
pip install -e .
# 프레임 다운로드에 HF gated 접근 필요:
export HF_TOKEN=<Albertmade/ego-implicit-ownership-multiperson 접근 토큰>
# API 모델 평가 시:
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GOOGLE_API_KEY=...
```

## 2. 실험 세팅 = 데이터셋 이름

입력 모드는 데이터셋 이름 접미사로 선택합니다. **코드 수정 없이 이름만 바꾸면 됩니다.**

| 데이터셋 이름 | 입력 | 논문 세팅 |
|---|---|---|
| `EgoOwn_Single` | 타겟 프레임 t 1장 | §5 setting (a) single-frame |
| `EgoOwn` | sparse 3프레임 (t-2, t-1, t) | §5 setting (b) sparse-frame |
| `EgoOwn_Clip` | short clip (dense frames 또는 mp4) | clip 확장 세팅 |
| `EgoOwn_Blind` | 이미지 없음, narration 텍스트만 | §5.4 image-blind |
| `EgoOwn_EgoLife`, `EgoOwn_EgoLife_Single`, ... | egolife parquet에 동일 모드 | |
| `EgoOwn_NarrA` | 구 narration-only parquet (호환용) | |

```bash
# 단일 실행 예
python run.py --model GPT4o --data EgoOwn_Single --reuse
python run.py --model Qwen2.5-VL-7B-Instruct --data EgoOwn EgoOwn_Single EgoOwn_Blind --reuse
```

## 3. 전체 sweep + 통합 리포트

```bash
# 스모크 테스트 (50개, 2세팅)
MODELS="GPT4o" DATASETS="EgoOwn_Single EgoOwn" EGOOWN_LIMIT=50 ./scripts/run_egoown_sweep.sh

# 본 실험
MODELS="GPT4o Claude-Sonnet Gemini-Pro Qwen2.5-VL-7B-Instruct InternVL3-8B" \
  ./scripts/run_egoown_sweep.sh

# §5.4 선택지 순서 permutation (seed별 재실행)
MODELS="GPT4o" PERMUTE_SEEDS="0 1 2" ./scripts/run_egoown_sweep.sh
```

sweep이 끝나면 자동으로 리포트가 생성됩니다 (수동 실행: `python scripts/egoown_report.py --outputs ./outputs`):

- `outputs/egoown_report.csv` — (모델 × 세팅 × seed)별 전체 지표 long-form
- `outputs/egoown_main_table.md` — 논문 메인 표 형태 (per-label acc / macro-F1 / per-taxonomy / abstention)

## 4. 측정 지표

`evaluate()`가 `*_score.json`에 기록하는 것들 (논문 §3 평가 프로토콜과 1:1):

- `acc:{MINE|PERSON_k|SHARED|AMBIGUOUS}`, `f1:{...}`, `macro_f1`, `overall_acc`
- `acc:taxonomy=T1..T4` — per-taxonomy 진단 (핵심 신호)
- `abstain_precision` / `abstain_recall` / `over_abstention_rate` — AMBIGUOUS를 abstention으로 간주
- `pred_frac:{label}` — 예측 분포 (label collapse / prior 진단용)
- `parsed_rate`, confusion matrix(`*_acc.csv`), per-item 스코어(`*_scored.xlsx`)

## 5. 재현성 규칙

모든 `*_score.json`에 **manifest**가 박힙니다: `prompt_version`(현재 `v2-2026-07-04`), `opt_seed`, `ref_field`, `mode`, `eval_code_rev`(git), `timestamp_utc`. 리포트 스크립트는 서로 다른 prompt_version/ref_field가 섞이면 ⚠️ 경고를 출력합니다 — **섞인 결과는 논문에 쓰지 마세요.**

프롬프트 설계 불변 조건 (수정 시 `PROMPT_VERSION` 반드시 bump):

- 라벨링 가이드 v2 인코딩 — ownership ≠ possession + 경계규칙 5종. `label-pipeline`의 `vlm_crosscheck.py` 프롬프트와 동기 유지할 것.
- **진단 메타데이터(taxonomy, source_dataset)는 절대 프롬프트에 노출 금지.** narration은 Blind 모드에서만 (ablation 시 `EGOOWN_INCLUDE_NARRATION=1` 명시).
- 선택지 순서는 (clip_id, `EGOOWN_OPT_SEED`)로 아이템별 결정적 셔플. seed를 바꾸면 §5.4 permutation test.

## 6. 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `EGOOWN_REF_FIELD` | `vlm_label` | 정답 필드. **human 재검수 완료 후 `human_label`로 전환** |
| `EGOOWN_OPT_SEED` | `0` | 선택지 셔플 seed (permutation test용) |
| `EGOOWN_LIMIT` | `0`(전체) | 스모크 테스트용 행 제한 |
| `EGOOWN_LOCAL_ROOT` | — | `data/*.parquet` + `frames/` 로컬 미러 (HF 다운로드 생략) |
| `EGOOWN_DL_WORKERS` | `16` | 프레임 병렬 다운로드 수 |
| `EGOOWN_INCLUDE_NARRATION` | — | `1`이면 시각 모드에도 narration 주입 (ablation 전용) |
| `EGOOWN_VIDEOS_ROOT` | — | `{video_id}.mp4` 디렉토리 — **Clip 모드 필수** |
| `EGOOWN_CLIP_NFRAMES` | `8` | Clip 모드 dense frame 수 |
| `EGOOWN_CLIP_AS_VIDEO` | — | `1`이면 mp4를 직접 전달 (video-native 모델용) |

## 7. Clip 모드 주의사항

Clip 모드는 원본 영상에서 ffmpeg로 구간을 추출하므로 **`EGOOWN_VIDEOS_ROOT` + parquet의 타이밍 컬럼(`source_video_start_sec`, `frame_times_sec`)이 필요**합니다. 현 parquet revision에 타이밍 컬럼이 없으면 명확한 에러와 함께 중단됩니다 → 데이터셋 export를 재생성하거나 Sparse/Single 모드를 사용하세요. 추출 결과는 `LMUDataRoot()/egoown_clips`에 캐시됩니다.

## 8. 모델 추가

VLMEvalKit 표준 방식 그대로: `vlmeval/config.py`의 supported_VLM에 모델을 등록하면 EgoOwn 전 세팅에서 자동으로 사용 가능합니다. ego 모델(EgoGPT 등) serving 관련은 Albert에게 문의.

---

문의: Hanwool Albert Lee (hanwool@aim-intelligence.com) · 벤치마크 데이터: [HF `Albertmade/ego-implicit-ownership-multiperson`](https://huggingface.co/datasets/Albertmade/ego-implicit-ownership-multiperson) (gated)
