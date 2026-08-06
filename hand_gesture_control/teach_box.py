#!/usr/bin/env python3
"""손동작 조종의 작업영역 상자를 교시로 잡는다.

로봇을 손으로 옮겨 두 모서리를 찍으면, 그 둘을 감싸는 직육면체를
robot_teleop.py 가 읽는 환경변수 형태로 뽑아준다.

  python3 teach_box.py            # 대화식으로 두 점 찍기
  python3 teach_box.py now        # 지금 위치 한 번만 보기

로봇을 **수동 모드**로 두고 팔을 손으로 옮긴 뒤 Enter 를 치면 된다.
읽기만 하므로 로봇을 움직이지 않는다.
"""
import os
import sys

ROBOT_ID, ROBOT_MODEL = "dsr01", "m0609"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teleop_box.env")


def connect():
    import rclpy
    import DR_init
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("teach_box", namespace=ROBOT_ID)
    # 모듈 수준 대입이어야 한다. 함수 안에서 self. 로 쓰면 이름 맹글링에 걸린다.
    setattr(DR_init, "__dsr__node", node)
    from DSR_ROBOT2 import get_current_posx
    return node, get_current_posx


def read(get_posx, label):
    """지금 위치를 읽는다. get_current_posx 는 타임아웃이 없어 스레드로 감싼다."""
    import threading
    box = {}

    def _r():
        try:
            box["p"] = get_posx()[0]
        except Exception as e:
            box["e"] = e

    th = threading.Thread(target=_r, daemon=True)
    th.start()
    th.join(timeout=5.0)
    if "p" not in box:
        print(f"  ! {label} 위치를 5초 안에 못 읽었습니다 "
              "— 다른 프로그램이 DSR 을 잡고 있는지 확인하세요")
        return None
    p = box["p"]
    print(f"  {label}: x={p[0]:.1f}  y={p[1]:.1f}  z={p[2]:.1f}")
    return p


def main():
    import rclpy
    node, get_posx = connect()
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "now":
            read(get_posx, "지금 위치")
            return

        print("\n작업영역 상자를 교시합니다. 로봇을 수동 모드로 두고 손으로 옮기세요.\n")
        print("① 상자의 **한쪽 모서리**로 팔을 옮기고 Enter")
        input()
        a = read(get_posx, "모서리 1")
        if a is None:
            return

        print("\n② 상자의 **대각선 반대쪽 모서리**로 옮기고 Enter")
        input()
        b = read(get_posx, "모서리 2")
        if b is None:
            return

        lo = [min(a[i], b[i]) for i in range(3)]
        hi = [max(a[i], b[i]) for i in range(3)]
        size = [hi[i] - lo[i] for i in range(3)]

        print("\n" + "=" * 52)
        print("작업영역 상자")
        for i, ax in enumerate("XYZ"):
            print(f"  {ax}  {lo[i]:8.1f} ~ {hi[i]:8.1f}   (폭 {size[i]:.0f}mm)")
        print("=" * 52)

        # 경계는 **밖으로 나가는 방향의 속도를 0 으로** 만드는 방식이다. 정지에도
        # 거리가 필요하므로 실제로는 안쪽에서 미리 끊고(BRAKE 15mm, MARGIN 15mm),
        # 그래도 조금 미끄러진다. 벽·판에 가까운 쪽은 여유를 두고 잡는 게 좋다.
        print("\n주의: 경계 30mm 안쪽에서 그 방향 속도가 0 이 되고, 정지까지 조금 더 갑니다.")
        print("      벽·판에 가까운 쪽은 그만큼 안쪽으로 잡으세요.\n")

        env = (f"TELEOP_X_MIN={lo[0]:.1f} TELEOP_X_MAX={hi[0]:.1f} "
               f"TELEOP_Y_MIN={lo[1]:.1f} TELEOP_Y_MAX={hi[1]:.1f} "
               f"TELEOP_Z_MIN={lo[2]:.1f} TELEOP_Z_MAX={hi[2]:.1f}")
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(env + "\n")
        print(f"저장됨: {OUT}\n")
        # robot_teleop.py 가 이 파일을 직접 읽는다. 환경변수로 다시 적어줄 필요가
        # 없다 — 두 곳에 적으면 갈라지고, 갈라지면 팔이 판에 부딪히는 쪽이 된다.
        print("조종(제어모드)은 다음 실행부터 이 값을 그대로 씁니다.")
        print("제어모드는 작업모드 안에서 돕니다 — 수어로 '모드변경' 을 하세요:\n")
        print("  cd .. && ./run_all.sh\n")
        print(f"환경변수로 한 번만 다르게 쓰려면:\n\n  {env}\n")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
