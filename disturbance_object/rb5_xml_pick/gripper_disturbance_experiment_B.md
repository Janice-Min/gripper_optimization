# 외란 실험 B (rb5_xml_pick 파이프라인) — 짧은 충격 후 파지 유지

기존 `on_rb5` 실험 B(`gripper_disturbance_experiment_B.md`)를 새 파이프라인에 맞게 다시 만든 것.
새 파이프라인: `make_finger_designs.py`(301개, th_open_d = 0) → `generate_mjcf.py` → `evaluate_grasp_workspace.py`(2지 (r, y) workspace) → `run_rb5_workspace.py`(RB5 pick, 4442 cell 중 2545 PASS를 `grasp_cells/`에 저장).

## 코드
- `rb5_xml_pick/disturb_impulse_rb5.py`: 시뮬레이션 + 판정 + 표. `--rescore`, `--tag`, 이어하기, 크기·방향 추가(append), `--view`
- `rb5_xml_pick/plot_impulse_rb5.py`: levels 분포, 방향 × 크기 heatmap(env/tip), 설계 산점도(성공 cell vs cond@k, Pareto), 설계 변수 vs 강건성, (r, y) 지도, 시간 응답
- 자세한 사용법: rb5_xml_pick `README_KO.md`의 "실험 B" 절

## 기본 설정
- 대상: grasp_cells의 PASS cell. `state.npz`(mjSTATE_INTEGRATION) 복원 → 외력 없이 0.5 s 안정화 → S0
- 시작 접촉 형태: 안정화 마지막 0.1 s 링크 접촉력 평균(4링크 모두 > 0.1 N = env, 양쪽 = tip)
- 펄스: 물체 질량중심 `xfrc_applied`, 50 ms 사각, k = 0.5 / 1 / 2 / 3 mg, 손 좌표 6방향(±x jaw, +y 빼내기 / −y palm 쪽, ±z 원기둥 축 = 위/아래)
- 조건: 저장 상태 그대로 질량 0.3 kg(mg = 2.94 N), 텐던 10 N, 중력 켬, 팔 마지막 명령 유지. 기존 on_rb5 B(0.1 kg, 20 N)와 절대값 비교 불가
- 판정: 펄스 후 10 s 동안 slip 5 mm / 회전 5° / 한쪽 접촉 < 0.1 N이 20 ms / 손 밖 물체와 접촉 0.05 N / 발산 중 하나면 실패. 실패 구간 pulse / transient(0.5 s) / delayed. 외력 0 대조 실패 cell은 zero_fail로 제외
- 지표: level, level_min, pass@k, cond@k, J_impulse@k = env_pass + 0.2·tip_pass, impact_area = Σ level_min

## 확인 (2026-11-15, MuJoCo 3.3.7, RB5 mesh 없이)
- design0001 r60 y15 (env): 복원 후 링크 접촉력이 summary.json과 일치(1.198 / 1.725 / 1.199 / 1.722 N). 접촉 없는 물체에 1 mg × 50 ms → Δv = g·T
- 결과 level_min = 1 mg: ±x, +y, −z는 2 mg에서 slip, −y·+z는 3 mg도 버팀. 외력 0 대조 10 s 변위 0.94 mm, 회전 1.1°
- 같은 cell을 `--tension 20`으로 하면 시작 형태가 tip으로 바뀌고 level_min = 1(약한 방향 +y)
- 속도: cell(25회) 약 10 s → 2545 cell 약 7 core-hour, trace 약 0.2 MB/cell

## 남은 결정
- tip cell이 0.5 mg에서 대부분 실패하면 `--mags 0.25` 추가
- 펄스 길이 20 ms(`--tag p20`), halfsine, residual slip / tilt 판정은 기존 B 문서의 선택지와 같음