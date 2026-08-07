#!/usr/bin/env python3
"""파지 지점 AR 을 로봇·카메라 없이 검증한다.

    python3 test_grasp_overlay.py        # 프레임 그림도 한 장 저장한다

실제 핸드아이 행렬과 RealSense 1280x720 대표 내부파라미터로 계산한다.
여기서 확인하는 것은 **판정이 뒤집히지 않는가** 다 — 초록이 잘못 켜지면
사람이 내려서 블록 모서리를 물거나 옆을 친다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grasp_overlay as go   # noqa: E402

# D435i 컬러 1280x720 대표값. 실제로는 camera_info 에서 받는다.
INTR = {"fx": 915.0, "fy": 915.0, "ppx": 640.0, "ppy": 360.0}
W, H = 1280, 720

v = go.GraspView(intr=INTR)
v.set_opening(55.0)          # 반폭 55mm = 110mm 벌림

# ── 1. TCP 가 화면 안에 찍힌다 (AR 이 성립하는 최소 조건) ──
uv = v.project(v.to_cam([0, 0, 0]))
assert uv is not None and 0 < uv[0] < W and 0 < uv[1] < H, uv
print(f"1 TCP 투영 ({uv[0]:.0f}, {uv[1]:.0f})  화면 안 — 팔 자세와 무관하게 고정")

# ── 2. 좌표 왕복이 맞는다 ──
for p in ([0, 0, 0], [10, -20, 120], [-30, 5, 300]):
    back = v.to_tcp(v.to_cam(p))
    assert np.allclose(back, p, atol=1e-6), (p, back)
print("2 TCP↔카메라 좌표 왕복 일치")

# ── 3. 내려갈수록 파지 구역이 화면에서 옮겨간다 (시차) ──
seen = []
for z in (0, 100, 200, 300):
    q = v.corridor_quad(z)
    assert q is not None
    seen.append((z, np.mean([p[0] for p in q]), np.mean([p[1] for p in q])))
assert seen[0][2] > seen[-1][2] + 50, seen      # 아래로 갈수록 화면 위쪽으로
print("3 하강 거리별 구역 중심 — " +
      "  ".join(f"{z}mm:({x:.0f},{y:.0f})" for z, x, y in seen))

# ── 4. 손가락 사이에 있으면 가능, 벗어나면 불가 ──
mid = v.to_cam([0, 0, 150])                     # 정중앙 150mm 아래
ok, why = v.graspable(mid)
assert ok, why
lim = v.half_open - go.GRASP_MARGIN - go.BLOCK_MM / 2
for x, want in ((lim - 2, True), (lim + 2, False)):
    got, why = v.graspable(v.to_cam([x, 0, 150]))
    assert got is want, (x, got, why)
print(f"4 좌우 한계 ±{lim:.0f}mm — 안쪽 가능 / 밖 불가  "
      f"(벌림 {v.half_open*2:.0f} - 여유 {go.GRASP_MARGIN*2:.0f} - 블록 {go.BLOCK_MM:.0f})")

# ── 5. 너무 멀거나 위에 있으면 불가 ──
for z, want in ((-20, False), (5, True), (go.REACH - 10, True), (go.REACH + 10, False)):
    got, why = v.graspable(v.to_cam([0, 0, z]))
    assert got is want, (z, got, why)
print(f"5 하강 거리 0~{go.REACH:.0f}mm 만 가능 — 위(-20mm)·먼 곳 불가")

# ── 6. 벌림이 좁아지면 판정도 좁아진다 (TF 값이 그대로 반영되는가) ──
#     여유는 **한쪽당** GRASP_MARGIN 이므로, 들어가려면
#     벌림 >= 블록 + 여유*2 = 35 + 12 = 47mm 여야 한다.
need = go.BLOCK_MM + 2 * go.GRASP_MARGIN
v.set_opening(20.0)                             # 벌림 40mm — 47 보다 좁다
assert not v.graspable(v.to_cam([0, 0, 150]))[0], "40mm 벌림에 35mm 블록은 안 들어간다"
v.set_opening(25.0)                             # 벌림 50mm — 여유 있게 들어간다
assert v.graspable(v.to_cam([0, 0, 150]))[0]
v.set_opening(55.0)
assert v.graspable(v.to_cam([0, 0, 150]))[0]
print(f"6 벌림 {need:.0f}mm 이상이어야 가능 — 40mm 불가 / 50·110mm 가능")

# ── 7. 빨강 ↔ 초록이 실제로 바뀐다 ──
import cv2   # noqa: E402

frame = np.full((H, W, 3), (32, 28, 26), np.uint8)
hit, name = go.draw(frame, v, [])
assert hit is False and name is None
red_only = frame.copy()

frame2 = np.full((H, W, 3), (32, 28, 26), np.uint8)
hit, name = go.draw(frame2, v, [(v.to_cam([4, -3, 160]), "빨간색", "깊이")])
assert hit is True and name.startswith("빨간색")
assert not np.array_equal(red_only, frame2)
print("7 블록 없으면 빨강 / 들어오면 초록 + 파지 가능")

out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "grasp_ar.png")
cv2.imwrite(out, np.hstack([red_only[380:720, 520:1080], frame2[380:720, 520:1080]]))
print(f"\n전부 통과 — 그림: {out}  (왼쪽 평소 / 오른쪽 파지 가능)")


# ── 8. 합성 영상으로 검출 → 판정까지 끝까지 ──
#     여기까지 통과하면 카메라만 붙이면 되는 상태다. 색 판정은 검출 노드의
#     그 코드를 그대로 부르므로(load_detector) HSV 범위를 고쳐도 따라온다.
#     워크스페이스를 안 소싱하면 건너뛴다 — 그때도 구역은 그려진다.
det, why = go.load_detector()
if det is None:
    print(f"8 건너뜀 — 색 검출을 못 불러옴 ({why.splitlines()[0][:60]})")
else:
    def synth(z_tcp, x_tcp=0.0, hsv=(0, 200, 200)):
        """TCP 기준 (x, 0, z) 자리에 블록이 있는 화면과 깊이 영상을 만든다."""
        p_cam = v.to_cam([x_tcp, 0.0, z_tcp])
        u, w_ = v.project(p_cam)
        d = float(p_cam[2])
        side = int(go.BLOCK_MM * INTR["fx"] / d)          # 35mm 가 몇 픽셀
        bgr = cv2.cvtColor(np.uint8([[list(hsv)]]), cv2.COLOR_HSV2BGR)[0][0]
        img = np.full((H, W, 3), (30, 30, 30), np.uint8)
        dep = np.full((H, W), 1200, np.uint16)            # 배경은 먼 바닥
        x0, y0 = int(u - side / 2), int(w_ - side / 2)
        img[y0:y0 + side, x0:x0 + side] = bgr
        dep[y0:y0 + side, x0:x0 + side] = int(d)
        return img, dep

    v.set_opening(55.0)
    # (a) 구역 안 — 초록
    img, dep = synth(160.0)
    blocks = go.find_blocks(v, img, dep, det)
    assert blocks, "합성 블록을 아예 못 찾았다"
    hit, name = go.draw(img, v, blocks, depth=dep)
    assert hit and name.startswith("빨간색"), (hit, name, blocks)
    got_z = v.to_tcp(blocks[0][0])
    assert abs(got_z[2] - 160.0) < 8.0, got_z            # 깊이로 되돌린 높이가 맞나
    print(f"8a 합성 블록 검출 {len(blocks)}개 → 초록  "
          f"복원 높이 {got_z[2]:.0f}mm (넣은 값 160mm)")

    # (b) 옆으로 비켜 있으면 — 빨강. 화면에서는 겹쳐 보여도 손가락 사이가 아니다.
    img2, dep2 = synth(160.0, x_tcp=60.0)
    blocks2 = go.find_blocks(v, img2, dep2, det)
    assert blocks2, "비켜 놓은 블록도 검출은 돼야 한다"
    hit2, _ = go.draw(img2, v, blocks2, depth=dep2)
    assert not hit2, v.to_tcp(blocks2[0][0])
    print(f"8b 좌우 60mm 비킨 블록 → 빨강 (좌우 "
          f"{v.to_tcp(blocks2[0][0])[0]:+.0f}mm, 한계 ±{lim:.0f}mm)")

    out2 = os.path.join(os.environ.get("TMPDIR", "/tmp"), "grasp_ar_synth.png")
    cv2.imwrite(out2, np.hstack([img2[380:720, 520:1180], img[380:720, 520:1180]]))
    print(f"   그림: {out2}  (왼쪽 비킴=빨강 / 오른쪽 정렬=초록)")

    # (c) 깊이가 0 이어도(가까이서 센서가 죽는 구간) 크기로 거리를 메운다
    #     실측 증상: 그리퍼가 파지 높이로 내려가면 추적이 끊기고 구역이 사라졌다.
    img3, dep3 = synth(120.0)
    dep3[:] = 0                                   # 깊이 전멸
    blocks3 = go.find_blocks(v, img3, dep3, det)
    assert blocks3, "깊이가 없으면 크기로라도 찾아야 한다"
    p3 = v.to_tcp(blocks3[0][0])
    assert blocks3[0][2] == "크기", blocks3[0]
    assert abs(p3[2] - 120.0) < 20.0, p3           # 크기 추정 오차
    hit3, name3 = go.draw(img3, v, blocks3, depth=dep3)
    assert hit3, p3
    print(f"8c 깊이 0 → 크기로 추정  높이 {p3[2]:.0f}mm (넣은 값 120mm) → 초록 유지")

    # (d) 깊이 영상이 아예 없어도 같다
    blocks4 = go.find_blocks(v, img3, None, det)
    assert blocks4 and blocks4[0][2] == "크기"
    print("8d 깊이 토픽이 없어도 구역·판정 유지")

    # ── 9. 착지점 = 그대로 내려가면 닿는 표면 (블록을 따라다니는 게 아니다) ──
    #     블록 위에서는 블록 윗면, 빈 판 위에서는 판에 찍혀야 한다.
    img5, dep5 = synth(150.0)
    got = go.landing_z(v, dep5)
    assert got is not None and abs(got - 150.0) < 6.0, got
    print(f"9a 블록 위 착지점 {got:.0f}mm (블록 윗면 150mm)")

    #     블록을 지우면(판만 남으면) 판 거리로 찍힌다.
    dep6 = np.full((H, W), 0, np.uint16)
    plate_z = 300.0                                   # TCP 아래 300mm 에 판
    for yy in range(0, H, 4):                         # 판 전체를 채운다
        for xx in range(0, W, 4):
            p = v.unproject(xx, yy, 1.0)              # 방향만 쓴다
            # 그 광선이 z=plate_z 평면과 만나는 깊이
            t = v.to_cam([0, 0, plate_z])[2] / p[2] if p[2] else 0
            dep6[yy:yy+4, xx:xx+4] = int(1.0 * t)
    got2 = go.landing_z(v, dep6)
    assert got2 is not None and abs(got2 - plate_z) < 12.0, got2
    print(f"9b 빈 판 착지점 {got2:.0f}mm (판 300mm) — 블록이 없어도 찍힌다")

    #     깊이가 없으면 None → 부르는 쪽이 크기 추정으로 넘어간다
    assert go.landing_z(v, None) is None
    assert go.landing_z(v, np.zeros((H, W), np.uint16)) is None
    print("9c 깊이 없음/전멸 → None (크기 추정으로 대체됨)")
