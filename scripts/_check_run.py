"""run.py 실행 후 결과 건전성 검사 — 사일런트 실패(eval_error / 추론 100% 실패 /
score.json 없음)를 탐지해 한 줄 요약을 stdout으로 낸다. 문제 없으면 아무것도 출력 안 함.

usage: _check_run.py <work_dir> <model_key> <dataset...>
드라이버가 run_model 후 호출 → 출력이 비면 DONE, 있으면 ISSUE로 기록.
"""
import glob
import json
import os
import sys

work, model = sys.argv[1], sys.argv[2]
datasets = sys.argv[3:]
problems = []
for ds in datasets:
    # score.json (심볼릭/최상위) 우선, 없으면 T-타임스탬프 하위 최신
    cands = [f for f in glob.glob(os.path.join(work, model, f"{model}_{ds}_score.json"))]
    if not cands:
        cands = sorted(glob.glob(os.path.join(work, model, "T*", f"{model}_{ds}_score.json")))
    if not cands:
        problems.append(f"{ds}=NO_SCORE")
        continue
    try:
        d = json.load(open(cands[-1]))
    except Exception:
        problems.append(f"{ds}=UNREADABLE")
        continue
    acc = d.get("overall_acc")
    parsed = d.get("parsed_rate")
    # eval-code score.json엔 overall_acc/parsed_rate가 있음. run summary의
    # eval_error는 score.json이 안 생기는 형태라 위 NO_SCORE로 잡힘.
    if acc is None:
        problems.append(f"{ds}=NO_ACC")
    elif parsed is not None and parsed == 0.0:
        problems.append(f"{ds}=PARSE0")  # 전량 추론/파싱 실패
if problems:
    print("; ".join(problems))
