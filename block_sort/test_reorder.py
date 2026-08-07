#!/usr/bin/env python3
"""복제 계획에서, 뒤 순번 색이 앞 순번 색을 막고 있을 때 순서를 바꾸는 것을
로봇 없이 검증한다.

    python3 test_reorder.py

예: 인간구역1=파랑, 인간구역2=보라 인데 보라가 파랑을 막고 있다면, 파랑 먼저
집으려다 relocate_blocker 가 보라를 **임시 자리로 잠깐 치웠다가** 나중에
보라 차례가 왔을 때 제 목적지로 또 옮기게 된다 — 두 번 움직이는 낭비다.
보라가 어차피 이 계획에서 옮겨야 할 색이면, 그 자리에서 곧장 제 목적지로
보내는 편이 한 번으로 끝난다. reorder_for_conflicts 는 이 순서 조정만 한다 —
실제로 옮기는 것은 block_sort.py 의 pick_cached/relocate_blocker 몫이다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_geom import reorder_for_conflicts, FINGER_T  # noqa: E402

# ── 1. 안 막혀 있으면 순서를 그대로 둔다 ──
plan = [("h1", 1, "파랑", 0.0), ("h2", 2, "보라", 0.0)]
positions = {"파랑": [(400.0, 0.0)], "보라": [(800.0, 0.0)]}   # 서로 멀다
assert reorder_for_conflicts(plan, positions) == plan
print("1 안 막혀 있으면 순서 그대로")

# ── 2. 뒤 순번(보라)이 앞 순번(파랑)을 막고 있으면 보라를 먼저 ──
#     파랑(400,0) 을 두 축에서 막는다: x축은 노랑(440,0, plan 밖 — 무관한 색도
#     막을 수 있다), y축은 보라(400,42, plan 안 — 이쪽이 "더 나쁜" 축으로
#     골라진다: 틈 7mm > 5mm 라 y축이 최선이라 그쪽 블로커가 지목된다).
plan = [("h1", 1, "파랑", 0.0), ("h2", 2, "보라", 0.0)]
positions = {
    "파랑": [(400.0, 0.0)],
    "보라": [(400.0, 42.0)],     # 파랑의 y축을 막음 (틈 7mm)
    "노랑": [(440.0, 0.0)],      # 파랑의 x축을 막음 (틈 5mm) — plan 밖 색
}
out = reorder_for_conflicts(plan, positions)
assert out == [("h2", 2, "보라", 0.0), ("h1", 1, "파랑", 0.0)], out
print("2 보라가 파랑을 막고 있으면 보라를 먼저 처리하도록 순서를 바꿈")

# ── 3. 막는 색이 plan에 없으면(어차피 relocate_blocker가 처리) 순서 그대로 ──
plan = [("h1", 1, "파랑", 0.0)]
positions = {
    "파랑": [(400.0, 0.0)],
    "노랑": [(400.0, 42.0)],     # plan에 없는 색
    "초록": [(440.0, 0.0)],      # plan에 없는 색
}
assert reorder_for_conflicts(plan, positions) == plan
print("3 막는 색이 plan에 없으면 순서 조정 대상이 아님 (relocate_blocker의 몫)")

# ── 4. 이미 순서가 맞으면(막는 색이 앞에 있으면) 그대로 ──
plan = [("h2", 2, "보라", 0.0), ("h1", 1, "파랑", 0.0)]   # 보라가 이미 먼저
positions = {
    "파랑": [(400.0, 0.0)],
    "보라": [(400.0, 42.0)],
    "노랑": [(440.0, 0.0)],
}
assert reorder_for_conflicts(plan, positions) == plan
print("4 막는 색이 이미 먼저 와 있으면 그대로")

# ── 5. 순환 의존(서로 막고 있음)이면 풀 수 없다 — 원래 순서를 유지한다 ──
#     A(400,0) 은 B(400,42, y축, 틈7)·C(440,0, x축, 틈5, plan 밖)에 막혀
#     최선축(y)의 블로커가 B. B(400,42) 는 A(y축, 틈7)·D(440,42, x축, 틈5,
#     plan 밖)에 막혀 최선축(y)의 블로커가 A. 서로가 서로의 블로커다.
plan = [("h1", 1, "A", 0.0), ("h2", 2, "B", 0.0)]
positions = {
    "A": [(400.0, 0.0)],
    "B": [(400.0, 42.0)],
    "C": [(440.0, 0.0)],
    "D": [(440.0, 42.0)],
}
assert reorder_for_conflicts(plan, positions) == plan
print("5 순환 의존이면 무리하게 순서를 만들지 않고 원래 순서 유지")

# ── 6. 같은 색이 plan에 두 번 있어도 항목 수·상대순서가 보존된다 ──
plan = [("h1", 1, "파랑", 0.0), ("h2", 2, "보라", 0.0), ("h3", 3, "파랑", 0.0)]
positions = {
    "파랑": [(400.0, 0.0), (900.0, 0.0)],
    "보라": [(400.0, 42.0)],
    "노랑": [(440.0, 0.0)],
}
out = reorder_for_conflicts(plan, positions)
assert len(out) == 3 and set(out) == set(plan), out
assert out[0] == ("h2", 2, "보라", 0.0), out           # 보라가 맨 앞으로
print(f"6 같은 색이 여러 항목이어도 개수·내용 보존 ({out})")

print("\n전부 통과 — 실제로 옮기는 동작(집기·놓기)은 block_sort.py의 "
      "copy_human()/pick_cached() 몫, 로봇에서만 확인 가능")
