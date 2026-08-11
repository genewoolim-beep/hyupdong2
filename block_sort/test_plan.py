#!/usr/bin/env python3
"""한 명령에 여러 짝이 올 때의 **계획 관리**를 로봇 없이 검증한다.

    python3 test_plan.py

'노랑 1번 파랑 2번 보라 3번' 처럼 짝이 여럿이면, 그것은 그 자체로 계획이다.
노란색을 집는데 파란색이 막고 있고 그 파란색이 **어차피 2번에 갈 색**이라면,
옆으로 60mm 끌어 두었다가 차례가 와서 다시 집는 것은 두 번 일하는 것이다 —
곧장 2번에 놓으면 한 번에 끝난다(relocate_to_zone).

    실측 2026-08-11 (로그 python3_250427): 위 명령에서 노란색을 집는데
    파란색(540,-39)이 막았다. (580,-84)로 끌어만 뒀고, 정작 그 차례에는
    "파란색 → 2번 실패" 로 끝났다.

복제(copy_human)에는 이 장치가 있었는데 색·구역 명령에는 없었다. 여기서 보는
것은 run_steps() 가 그 계획을 제대로 걸고, 빼고, 지우는지다.

block_sort.py 는 import 하는 순간 DSR 에 붙어 로봇 없이는 한 줄도 못 돌린다.
그래서 이 파일은 **소스에서 그 두 메서드만 떼어내** 가짜 run_one 위에 얹는다 —
사본을 만드는 게 아니라 실제 코드를 그대로 읽어 시험한다.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "block_sort.py")

WANT = {"run_steps", "pending_target"}
tree = ast.parse(open(SRC, encoding="utf-8").read())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "BlockSort")
methods = [n for n in cls.body
           if isinstance(n, ast.FunctionDef) and n.name in WANT]
assert len(methods) == len(WANT), f"못 찾은 메서드: {WANT - {m.name for m in methods}}"

ns = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             "<block_sort>", "exec"), ns)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)

    warn = error = info


class Fake:
    """run_one 이 불릴 때 '그 색을 집는데 어떤 색이 막았다' 고 가정한다.

    막는 색이 계획에 있으면 실기의 relocate_to_zone 이 하는 일을 그대로 흉내낸다
    (제 구역에 놓고, 계획에서 빼고, 놓은 것을 기록한다). 없으면 끌기다.
    """

    run_steps = ns["run_steps"]
    pending_target = ns["pending_target"]

    def __init__(self, blocker_of=None):
        self.log, self.calls = _Log(), []
        self.blocker_of = blocker_of or {}

    def get_logger(self):
        return self.log

    def copy_human(self, mirror=False):
        self.calls.append(("copy", mirror))
        return True

    def run_one(self, src, zone):
        self.calls.append((src, zone))
        b = self.blocker_of.get(src)
        if b is None:
            return True
        item = self.pending_target(b)
        if item is None:
            self.calls.append(("끌기", b))       # 예전 길: 임시 자리로 치운다
            return True
        self.calls.append(("→구역", b, item[1]))  # relocate_to_zone
        self.copy_pending.remove(item)
        self.copy_placed[item[1]] = b
        return True


STEPS = [("노란색", 1), ("파란색", 2), ("보라색", 3)]

# ── 1. 로그의 그 상황 — 막는 색이 계획에 있으면 곧장 제 구역으로 ──
f = Fake(blocker_of={"노란색": "파란색"})
assert f.run_steps(STEPS, str) is True
assert ("→구역", "파란색", 2) in f.calls, f.calls
assert ("끌기", "파란색") not in f.calls, "끌어 두면 안 된다"
assert ("파란색", 2) not in f.calls, "이미 끝난 짝을 다시 집으면 안 된다"
assert any("이미 끝냈" in l for l in f.log.lines), f.log.lines
print(f"1 막는 색이 계획에 있으면 곧장 제 구역으로 — {f.calls}")

# ── 2. 계획에 없는 색이 막으면 예전 그대로(임시 자리로 치우기) ──
f = Fake(blocker_of={"노란색": "주황색"})
f.run_steps(STEPS, str)
assert ("끌기", "주황색") in f.calls, f.calls
print("2 계획에 없는 색은 예전대로 임시 자리로 치운다")

# ── 3. 지금 집으러 간 색은 계획에서 미리 빠져 있다 ──
#     안 빼면 그 블록을 집는 도중에 도는 relocate 가 '아직 안 한 짝' 으로 보고
#     자기 자신을 또 보내려 한다 (copy_human 이 같은 이유로 먼저 뺀다).
f = Fake(blocker_of={"노란색": "노란색"})
f.run_steps(STEPS, str)
assert ("끌기", "노란색") in f.calls, f.calls
print("3 지금 집으러 간 색은 계획에서 빠져 있다 (자기를 또 보내지 않는다)")

# ── 4. 명령이 끝나면 계획이 남지 않는다 ──
#     남겨 두면 다음 명령이 지난 계획을 보고 엉뚱한 구역으로 보낸다.
assert f.copy_pending is None and f.copy_placed == {}
print("4 명령이 끝나면 계획을 지운다")

# ── 5. 구역→구역 짝(숫자)은 계획에 안 들어간다 ──
#     pending_target 은 색으로 찾고, 그 블록은 이미 제 구역에 있다.
f = Fake()
f.run_steps([(3, 1), ("파란색", 2)], str)
assert f.copy_pending is None          # 끝나고 지워졌다
f2 = Fake(blocker_of={"파란색": "파란색"})
f2.run_steps([(3, 1), ("파란색", 2)], str)
assert (3, 1) in f2.calls and ("파란색", 2) in f2.calls, f2.calls
print("5 구역→구역 짝은 계획에 넣지 않는다")

# ── 6. '똑같이'(copy) 가 섞여도 뒤 짝이 계획을 잃지 않는다 ──
#     copy_human 은 제 계획을 걸었다가 finally 에서 self.copy_* 를 None 으로
#     되돌린다. run_steps 가 매 짝마다 다시 걸지 않으면 그 뒤 짝들은 계획을
#     잃고, None 을 순회하다 터진다.
class FakeCopy(Fake):
    def copy_human(self, mirror=False):
        self.calls.append(("copy", mirror))
        self.copy_pending = self.copy_occ = self.copy_stock = None
        self.copy_placed = {}
        return True


f = FakeCopy(blocker_of={"보라색": "파란색"})
assert f.run_steps([("copy", 0), ("보라색", 3), ("파란색", 2)], str) is True
assert ("→구역", "파란색", 2) in f.calls, f.calls
print("6 '똑같이' 뒤에 오는 짝도 계획을 그대로 쓴다")

print("\n전부 통과 — 실기에서는 막는 블록을 곧장 제 구역에 놓는지, "
      "그 짝을 다시 집지 않는지 로그로 본다")
