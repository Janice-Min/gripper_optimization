# 외란 실험 C — 팔 흔들기 (예정)

| 실험 | 방법 | 목적 |
|---|---|---|
| C. 팔 흔들기 | 들어 올린 뒤 손을 ±y·±z로 가속·감속 (0.5 g, 1 g) | 실제 운반 상황 |

아직 코드가 없습니다. 만들 때 아래처럼 A·B와 같은 구조를 따릅니다.

- 시작 상태: 파지 성공 cell의 저장 상태 `grasp_cells/<label>/` (`grasp/disturb_collect.py`).
  A·B와 같은 상태에서 출발하므로 세 실험을 cell 단위로 비교할 수 있습니다.
- 대상 cell: A의 graspable 설계와 성공 cell (`A/results/graspable_designs.csv`, `A/results/<label>/force_cells.csv`).
- 결과: `C/results/<label>/` (경로 함수 `layout.c_root(run)`).
- 스크립트 첫머리:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # common/, grasp/, A/, B/, C/ 가 import 경로에 들어감
from disturb_collect import label_dir   # 파지 성공 cell 상태
from disturb_force import force_dir      # A 결과
```
