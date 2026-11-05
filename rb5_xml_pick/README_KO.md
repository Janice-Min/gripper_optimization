# XML 그리퍼를 RB5에 장착하여 물체 집고 들어 올리기

`design001.xml` 계열의 독립형 그리퍼 XML을 읽어 RB5-850e 손목에 부착합니다.
기본 실행 시 MuJoCo viewer가 열리고, 접근 → 텐던 파지 → 150 mm 상승 → 2초 유지 동작을 수행합니다.
RB5 mesh, 로봇/환경 XML, 예제 XML을 모두 포함하므로 기존 `run_task.py`, `gripper_core.py`, `rb5_eval.py`가 필요 없습니다.

## 실행 — Windows PowerShell

**ZIP 전체를 압축 해제**하고 `run_rb5_xml.py`가 있는 폴더에서 실행합니다. `assets`와 `sources` 폴더를 지우거나 Python 파일만 따로 옮기면 안 됩니다.

```powershell
python -m pip install -r requirements.txt
python run_rb5_xml.py --xml design001.xml
```

다른 XML도 동일하게 넣습니다. 공백이 있는 경로는 따옴표로 감쌉니다.

```powershell
python run_rb5_xml.py --xml "C:\Users\User\Desktop\BEAR lab\design002.xml"
```

가상환경을 새로 만들려면:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_rb5_xml.py --xml design001.xml
```

Python 3.12.14 / MuJoCo 3.17.0으로 물리 시뮬레이션, 영상 렌더링, MuJoCo viewer 실행을 검증했습니다.
macOS에서는 GUI 실행 시 `python` 대신 MuJoCo가 제공하는 `mjpython`을 사용합니다.

## 바꿀 변수는 모두 코드 맨 아래

`run_rb5_xml.py`의 `if __name__ == '__main__':` 아래에 `parser = argparse.ArgumentParser(...)`와 모든 `parser.add_argument(...)`가 있습니다.
원하는 옵션의 `default=`만 바꾸거나 실행 명령에 옵션을 추가하세요. 나머지 함수는 수정할 필요가 없습니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--xml` | 같은 폴더의 design001.xml | 입력 XML |
| `--tension` | 20 | **실제 텐던 장력**, N; 다중 텐던이면 각각 동일한 장력 |
| `--lift` | 0.15 | 상승 거리, m |
| `--radius` | None | 원기둥 반지름, m; None이면 XML 값 |
| `--object-height` | None | 원기둥 전체 높이, m; None이면 XML 값 |
| `--mass` | None | 물체 질량, kg; None이면 XML 값 |
| `--grasp-xy` | None | palm 좌표계의 물체 중심 x/y, m; 입력하면 자동 위치 조정 해제 |
| `--auto-grasp` | True | 초기 겹침을 피하도록 파지 위치를 손가락 진행 방향으로 자동 이동 |
| `--object-xy` | 0.60 0 | 세계 좌표계에서 물체를 놓을 위치, m |
| `--support-height` | 0.08 | 짧은 원기둥의 파지를 위한 받침대 높이, m |
| `--mount-length` | 0.10 | 손목 flange부터 palm까지 어댑터 길이, m |
| `--approach-distance` | 0.12 | 물체 뒤쪽에서 접근하는 거리, m |
| `--ramp-time` | 1.5 | 장력 증가 시간, s |
| `--lift-time` | 3.0 | 상승 시간, s |
| `--hold-time` | 2.0 | 상승 후 유지 시간, s |
| `--speed` | 1.0 | viewer 재생 속도; 물리 시간 간격은 바뀌지 않음 |
| `--camera-distance` | 1.45 | 카메라 거리; 작을수록 확대 |
| `--headless` | False | 지정할 때만 viewer 생략 |
| `--exit-after-task` | False | 지정하면 동작 후 창을 자동으로 닫음 |
| `--output` | results | 통합 XML 및 결과를 저장할 폴더 |

예시:

```powershell
python run_rb5_xml.py --xml design002.xml --tension 30 --lift 0.12
python run_rb5_xml.py --xml design001.xml --radius 0.045 --mass 0.2
python run_rb5_xml.py --xml design001.xml --grasp-xy -0.09 0
python run_rb5_xml.py --xml design001.xml --camera-distance 1.0
```

## XML에서 자동으로 가져오는 내용

- palm 아래의 손가락 형상, 길이, 관절 위치와 각도 제한.
- 관절 stiffness / damping / springref / armature.
- tendon의 site / wrapping geom / sidesite / gear.
- `open` keyframe의 관절각. keyframe이 없으면 관절 springref를 사용합니다.
- cylinder의 반지름, 높이, 질량, 마찰.
- 기존 손과 물체 사이의 상대 위치. 필요하면 아래 자동 조정이 적용됩니다.

손가락 link 이름이나 관절 이름은 코드에 고정되어 있지 않습니다. 두 finger branch의 collidable geom을 찾아 접촉력을 집계합니다.
서로 다른 XML과 RB5 사이의 이름 충돌을 막기 위해 통합 장면에서만 입력 XML 이름 앞에 `hand__`를 붙입니다.
원본 XML 파일은 수정하지 않습니다.

## design001 형식의 호환 범위

다음 구조를 유지하며 치수, 각도, 강성, 물체 크기 등을 바꾼 파일을 대상으로 합니다.

1. `palm`이라는 고정 body와 그 아래의 두 articulated finger branch.
2. 각 finger의 joint는 이름이 있는 hinge joint. 접촉 geom도 이름이 있어야 합니다.
3. `object_body`에 직접 속한 cylinder geom 한 개와 freejoint 한 개.
4. cylinder 축은 palm z와 평행.
5. actuator는 이름과 0이 아닌 gear를 가진 tendon motor. 여러 개도 동일 장력으로 구동합니다.
6. object weld 등의 equality constraint는 사용하지 않습니다.
7. default / contact / tendon이 참조하는 대상은 가져오는 손/물체 내부에 존재해야 합니다.

body 이름만 바뀐 경우 다음처럼 지정합니다.

```powershell
python run_rb5_xml.py --xml another.xml --palm-body my_palm --object-body my_object
```

design001에서 손가락이 뻗는 방향은 palm **-x**이므로 `--forward-angle 180`입니다.
같은 hierarchy를 다른 방향으로 작성했다면 이 값을 변경합니다. +x는 0, +y는 90입니다.
기존 `R_prox` / `obj` 등 다른 hierarchy의 XML을 모든 경우 자동 변환하는 범용 MJCF 로더는 아닙니다.

**형식이 같다는 것과 파지가 성공한다는 것은 다릅니다.** 닫힐 수 없는 설계, 너무 무거운 물체, 부족한 장력/마찰, 로봇 작업반경 밖 목표는 물리적으로 실패할 수 있습니다.
그 경우 성공을 꾸미지 않고 `FAIL`과 원인을 기록합니다. 실행 전 모델 구조, 열린 손의 겹침, 접근 경로, IK와 팔 충돌을 확인합니다.

## 기본 design001에 적용되는 보정

- 원본 r=55 mm / h=38 mm / m=0.3 kg를 그대로 사용합니다.
- 원본 위치 x=-75 mm에서 손과 물체가 약 4.1 mm 겹칩니다. 기본 자동 탐색으로 상대 위치가 **x=-90 mm, y=0**으로 이동합니다.
- 형상/길이/물체 크기/관절 stiffness/마찰을 자동으로 변경하지 않습니다.
- 원기둥이 낮아 손목이 책상에 닿을 수 있으므로 높이 **80 mm**의 받침대 위에 둡니다. 받침대와 물체는 실제 접촉합니다.
- 접근 중 손가락이 관성 때문에 닫히는 현상을 줄이기 위해 임시 관절 PD opening assistance를 사용합니다. 장력을 주기 시작하면 0.3초 동안 부드럽게 해제하고, 이후 파지·상승 중에는 **완전히 0**입니다. `--open-kp`, `--open-kd`, `--open-torque-limit`로 조정할 수 있습니다.
- 장력은 `ctrl = -T / gear`로 변환합니다. gear=-0.5, T=20 N이면 **ctrl=40**입니다. 원본 ctrlrange=0..20도 지정한 장력을 전달할 수 있도록 통합 XML에서 확장됩니다. 이것은 실제 하드웨어 모터 한계 검증을 뜻하지 않습니다.
- 손목은 position actuator로 움직이고 RB5 링크의 기존 gravity compensation을 유지합니다. 물체에는 freejoint와 중력이 적용되며, 물체를 손에 붙이는 weld / 물체 위치 덮어쓰기 / 위쪽 외력은 사용하지 않습니다.

원본 상대 위치 그대로 검사하려면 `--no-auto-grasp`를 사용합니다. 기본 design001은 초기 겹침 때문에 해당 설정에서 FAIL이 정상입니다.

## 결과와 viewer 동작

약 14.1초 동작 후 다음 파일이 `results`에 생성됩니다.

- `rb5_attached_scene.xml`: 실행할 때 생성한 통합 장면. 원본 mesh 경로가 들어 있으므로 다른 폴더/PC로 옮기면 스크립트를 다시 실행해 재생성하세요.
- `summary.json`: 성공/실패 이유, 실제 상승량, 각 링크 접촉력, 자동 파지 위치, 설정값.
- `trace.csv`: 시간에 따른 물체 위치, 파지점 위치, 링크 normal force와 받침대 접촉력.

기본 성공 조건은 마지막 1초 동안 **목표 상승량의 80% 이상**, **양쪽 finger 접촉**, **책상/받침대 접촉 해제**를 모두 유지하는 것입니다.
4개 링크 전부 접촉해야만 성공인 조건은 아닙니다. `four_link_contact`는 별도로 기록합니다.

종료 시 `PASS` 또는 `FAIL`이 출력됩니다. 정상 성공은 종료코드 0, 실패는 1입니다.
viewer는 최종 상태를 **정지해서 보여주며**, 창을 닫으면 종료됩니다. 실패할 때도 창을 닫기 전 상태를 확인할 수 있습니다.
다시 실행하면 처음부터 시작합니다. 출력 폴더가 같으면 이전 결과 파일을 덮어쓰므로 여러 실험을 보존하려면 `--output`을 다르게 지정하세요.

## 검증 및 선택 영상 녹화

```powershell
python run_rb5_xml.py --xml design001.xml --headless
python -m pip install -r requirements-video.txt
python run_rb5_xml.py --xml design001.xml --video pick_lift.mp4
```

Linux에서 디스플레이 없이 영상만 저장할 때는 EGL 드라이버가 필요합니다.

```bash
MUJOCO_GL=egl python run_rb5_xml.py --headless --video pick_lift.mp4
```

기본 XML 검증: **147.398 mm 상승**, 최종 받침대 접촉력 **0 N**, 양쪽 distal 접촉 유지, MuJoCo warning **0**.
이번 결과는 tip contact이며 four-link contact는 아닙니다.
추가 입력/실패 검사 결과는 `validation/verification.json`에 있습니다.

## 모델 출처와 라이선스

RB5 및 실험실 모델은 앞서 사용한 `rb5_tendon_pick_lift` 패키지의 모델을 포함했습니다.
RB5 kinematics/inertias/mesh는 Rainbow Robotics `rbpodo_ros2`의 `rb5_850e` 기반입니다.

- https://github.com/RainbowRobotics/rbpodo_ros2
- RB5 라이선스: `assets/rb5_850e/LICENSE`
- 환경 소스 라이선스: `sources/Class_IR_LICENSE`

이 패키지는 MuJoCo 시뮬레이션용이며 실제 RB5 통신/구동 코드는 포함하지 않습니다.
