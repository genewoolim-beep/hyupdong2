# legacy — 지금 플로우에서 쓰지 않는 것

**여기 있는 것은 정상 동작 플로우가 한 번도 부르지 않는다.** 지우지 않고 남긴
이유는 하나다 — 이 저장소가 음성 pick & place 로 시작했고, 그 시절 코드가
지금 쓰는 것들의 출처이기 때문이다(좌표 변환, 그리퍼 제어, 검출 호출 방식).
무언가가 왜 이렇게 돼 있는지 따라가야 할 때 여기를 본다.

지금 쓰는 입력은 **수어와 손동작뿐이다. 음성 인식은 쓰지 않는다.**

| 폴더 | 무엇 | 왜 안 쓰는가 |
|---|---|---|
| `pick_and_place_voice/` | 음성 → STT → LLM → 검출 → 파지 (원본 데모) | 입력이 음성이다 |
| `pick_and_place_text/` | 위와 같은데 입력만 텍스트 | 지금은 `block_sort.py pick` 이 그 자리다 |
| `voice_processing/` | 웨이크워드·STT·마이크 | 음성 입력용. `sign_processing` 이 같은 `/get_keyword` 를 수어로 대체했다 |
| `robot_control/` | 음성 데모의 로봇 노드 (`person_tracker` 포함) | 로봇 동작은 `block_sort/` 로 옮겨졌다 |
| `rokey/` | 껍데기 패키지 (`__init__.py` 하나) | 내용이 없다 |
| `README-voice_pickandplace.md` | 옛 워크스페이스 문서 | 위 프로그램들의 설명 |

## 되살리려면 알아야 할 것

- **워크스페이스 밖으로 나왔다.** `sign_pickandplace/src/` 가 아니라 여기 있으므로
  `colcon build` 가 이것들을 빌드하지 않는다. 되살리려면 `src/` 로 옮겨야 한다.
- **핸드아이 행렬은 여기 없다.** `robot_control/resource/T_gripper2camera.npy` 는
  저장소 최상위 `calib/` 로 옮겼다 — `block_sort` 와 `hand_gesture_control`(파지 AR)이
  함께 읽는 값이라 한 프로그램 안에 두면 안 되기 때문이다. 옛 코드는 자기 패키지
  share 폴더에서 찾으므로 되살릴 때 경로를 고쳐야 한다.
- **LLM 키 자리도 옮겼다.** `voice_processing/resource/.env` → 저장소 최상위 `.env`.
  키는 음성이든 수어든 공용이고, 지금은 `block_sort/sign_command.py` 가 쓴다
  (환경변수 `OPENAI_API_KEY` 도 그대로 먹는다).
