#!/usr/bin/env python3
"""**지금 바로 집히는 것을 먼저** — 복제의 두 번 통과 규칙을 로봇 없이 검증한다.

    python3 test_order.py

계획 순서대로 밀어붙이면 앞 순번이 막혔을 때 그것부터 풀려고 판을 흔든다.
그래서 choose_next() 가 '지금 바로 집을 수 있는 것' 을 먼저 고르는데, 그 판단은
순회(patrol) 좌표로 한 것이라 **막상 가 보면 다를 수 있다.**

    실측 2026-08-11 (로그 python3_414742): 파란색(594,88)을 '집힌다' 로 보고
    갔는데 현장에서 틈이 34mm 였다(그때 문턱 38). 그 길로 치우기 연쇄가 돌아
    주황색 실패 → 직각 축의 빨간색 → 그 빨간색을 막는 보라색까지 3단으로
    번졌고, 보라색은 하필 원래 목표(파란색) 쪽으로 끌려갔다.

그래서 copy_human 은 두 번 통과한다:
    1차  '집힌다' 고 본 것 → **치우기 없이** 집어만 본다. 아니면 계획에 되돌리고
         `_blocked_once` 에 표시한 뒤 다음 후보로 넘어간다.
    2차  남은 것이 전부 그렇게 되면(ready=False) 그때 치우기를 허용한다.

여기서 보는 것은 그 순환이 ① 다음 후보로 넘어가는지 ② 무한히 돌지 않는지
③ 하나 빠지면 다시 열어 보는지다. block_sort.py 는 import 하는 순간 DSR 에
붙으므로 choose_next() 를 소스에서 그대로 떼어내 가짜 판 위에 얹는다.
"""
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_geom import FINGER_T, best_axis, tool_yaw  # noqa: E402

SRC = os.path.join(HERE, "block_sort.py")
tree = ast.parse(open(SRC, encoding="utf-8").read())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "BlockSort")
fn = next(n for n in cls.body
          if isinstance(n, ast.FunctionDef) and n.name == "choose_next")

ns = {"np": np, "best_axis": best_axis, "tool_yaw": tool_yaw,
      "FINGER_GAP_MIN": FINGER_T, "SELF_R": 30.0}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
             "<block_sort>", "exec"), ns)


def det(x, y, ang=0.0):
    """_detect_all 형식의 최소 검출 하나. pose 의 자세각은 수직 + 요만 쓴다."""
    return {"pose": [x, y, -13.0, 0.0, 179.9, ang], "angle": ang}


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)

    warn = error = info


class Fake:
    choose_next = ns["choose_next"]

    def __init__(self, pending, stock, really_blocked=()):
        self.copy_pending = list(pending)
        self.stock = stock
        self.really_blocked = set(really_blocked)   # 가 보면 막혀 있는 색
        self._blocked_once = set()
        self.log = _Log()
        self.order = []          # 실제로 집으러 간 순서 (색, 치우기허용)

    def get_logger(self):
        return self.log

    def in_reach(self, xy):
        return xy[0] <= 640.0

    def graspable_first(self, dets, stock, why=""):
        return list(dets)

    def run(self, rounds=20):
        """copy_human 의 while 루프를 그대로 흉내낸다."""
        for _ in range(rounds):
            if not self.copy_pending:
                return True
            item, ready = self.choose_next(self.stock)
            if item is None:
                return True
            _hz, dst, color, _a = item
            self.copy_pending.remove(item)
            self.order.append((color, not ready))       # (색, 치우기 허용했나)
            blocked = color in self.really_blocked
            if blocked and ready:
                # 1차 통과 — 치우지 않고 되돌린다
                self.copy_pending.append(item)
                self._blocked_once.add((dst, color))
                continue
            if blocked and not ready:
                # 치우기를 허용해 풀었다고 본다 (실기의 relocate 성공)
                self.really_blocked.discard(color)
            self.stock.pop(color, None)                 # 옮겨졌다
            self._blocked_once = set()                  # 판이 헐거워졌다
        raise AssertionError(f"루프가 안 끝났다: {self.order}")


# ── 실기 배치 (2026-08-11 로그 python3_414742, 노란색을 끝낸 직후) ──
#     계획: 빨간색→3번, 파란색→1번   /   판에 있는 것들
PLAN = [(3, 3, "빨간색", 0.0), (1, 1, "파란색", 0.0)]
BOARD = {
    "빨간색": [det(534, 90, 88.0)],
    "파란색": [det(594, 88, 88.0)],
    "주황색": [det(529, 50, 46.0)],
    "보라색": [det(529, 140, 89.0)],
}

# 그 배치에서 빨간색(534,90)은 주황색(529,50)이 40mm 옆이라 **순회 좌표로도**
# 막혀 있다. 실기 로그에서 "빨간색(544,85) 도 갇혀 있습니다(여유 14mm)" 로
# 나온 그 상태다 — 아래 5번이 이 배치를 그대로 본다.

# 같은 계획에 **정말 뻥 뚫린** 항목을 하나 더 둔 판. 1~4번은 이 판을 쓴다.
PLAN3 = [(1, 1, "파란색", 0.0), (4, 4, "초록색", 0.0)]
BOARD3 = {
    "파란색": [det(594, 88, 88.0)],
    "초록색": [det(450, -200, 0.0)],      # 사방이 비었다
    "주황색": [det(569, 9, 46.0)],        # 파란색을 막는 것
}

# ── 1. 가 보니 막혀 있으면 **치우지 않고 다음 후보부터** ──
f = Fake(PLAN3, dict(BOARD3), really_blocked={"파란색"})
assert f.run() is True
assert f.order[0] == ("파란색", False), f"1차는 치우기 없이 가야 한다 — {f.order}"
assert ("초록색", False) in f.order, f.order
i_green = [i for i, (c, _r) in enumerate(f.order) if c == "초록색"][0]
i_blue_retry = [i for i, (c, _r) in enumerate(f.order) if c == "파란색"][-1]
assert i_green < i_blue_retry, f"초록색이 파란색 재시도보다 먼저여야 한다 — {f.order}"
print(f"1 막힌 것은 되돌리고 다음 후보부터 — {[c for c, _ in f.order]}")

# ── 2. 무한히 돌지 않는다 (전부 막히면 그때 치우기 허용) ──
f = Fake(PLAN3, dict(BOARD3), really_blocked={"파란색", "초록색"})
assert f.run() is True
relo = [c for c, r in f.order if r]
assert relo, f"전부 막히면 치우기를 허용해야 한다 — {f.order}"
print(f"2 전부 막히면 그때 치우기 허용 — 치우며 집은 것 {relo}")

# ── 3. 하나 빠지면 막혔던 것을 **다시 본다** ──
#     판이 헐거워져 저절로 열리는 일이 잦다. _blocked_once 를 비우는 이유다.
f = Fake(PLAN3, dict(BOARD3), really_blocked={"파란색"})
f.run()
assert f._blocked_once == set(), "성공 뒤에는 표시를 비워야 한다"
print("3 하나 옮기고 나면 막혔던 표시를 지우고 다시 본다")

# ── 4. 막힌 것이 없으면 예전과 같다 (치우기를 아예 안 켠다) ──
f = Fake(PLAN3, dict(BOARD3))
f.run()
assert all(not r for _c, r in f.order), f"치우기를 켜면 안 된다 — {f.order}"
assert len(f.order) == 2, f.order
print(f"4 막힌 것이 없으면 치우기를 안 켠다 — {[c for c, _ in f.order]}")

# ── 5. 실기 그 배치 — 둘 다 막혀 있으면 치우기는 **1차를 다 해본 뒤**에 ──
#     파란색은 현장에서만 막혀 있고(순회 좌표로는 집힌다고 봤다), 빨간색은
#     순회 좌표로도 막혀 있다. 그래도 파란색을 **치우기 없이 먼저** 해보고,
#     그게 안 될 때 비로소 치우기가 나서야 한다 — 예전에는 그 첫 시도가
#     그대로 3단 연쇄가 됐다.
f = Fake(PLAN, dict(BOARD), really_blocked={"파란색"})
assert f.run() is True
assert f.order[0] == ("파란색", False), f"첫 시도는 치우기 없이 — {f.order}"
print(f"5 실기 배치 — 첫 시도는 치우기 없이 끝낸다 — {f.order}")

print("\n전부 통과 — 실기에서는 '가 보니 막혀 있습니다 — 치우지 않고 다음 것부터' "
      "가 뜨고, 그 뒤 다른 색이 먼저 옮겨지는지 로그로 본다")
