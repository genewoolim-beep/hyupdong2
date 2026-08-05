########## HsvProbe ##########
# 카메라 프레임 한 장을 받아서, 배경(나무판/체스보드)을 뺀 색깔 픽셀들의 실제 hue 분포를
# 로그로 찍어준다. 어떤 색이 실제로 몇 번 hue에 몰려있는지 눈으로 보고 정확한 HSV 범위를
# 추측이 아니라 데이터로 정하기 위한 1회성 진단 도구다.
import json
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from object_detection.color_model import CONFIG_PATH

INPUT_TOPIC = '/camera/camera/color/image_raw'
S_MIN, V_MIN = 30, 40  # 배경(나무/회색/흰색)을 뺄 정도로만 느슨하게 잡는다

SMOOTH_WIDTH = 3       # 잔잔한 요철을 없애 구간이 잘게 쪼개지는 걸 막는다
BAND_FLOOR_RATIO = 0.08  # 가장 높은 봉우리의 이 비율 이상인 구간만 색으로 본다

# 붙어 있는 두 색을 갈라내는 기준. 빨강(≈5)과 주황(≈13)처럼 hue 가 이웃한 색은
# 사이가 BAND_FLOOR_RATIO 아래로 안 내려가서 한 덩어리로 잡힌다.
PEAK_MIN_RATIO = 0.30      # 그 덩어리 최고점의 이 비율보다 낮은 봉우리는 잡음으로 본다
VALLEY_DEPTH_RATIO = 0.75  # 골짜기가 낮은 쪽 봉우리의 이 비율보다 깊게 파여야 다른 색으로 인정
MIN_PEAK_GAP = 4           # 봉우리끼리 최소 이만큼 떨어져 있어야 별개 색으로 본다


def _smooth(hist):
    return np.convolve(hist, np.ones(SMOOTH_WIDTH) / SMOOTH_WIDTH, mode='same')


def find_peaks(seg):
    """구간 안에서 색 하나로 볼 만한 봉우리들의 위치."""
    floor = seg.max() * PEAK_MIN_RATIO
    return [i for i in range(1, seg.size - 1)
            if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1] and seg[i] >= floor]


def split_at_valleys(smooth, lo, hi):
    """한 덩어리 안에 봉우리가 둘 이상이면 그 사이 골짜기에서 잘라 나눈다.

    빨강과 주황은 hue 가 바로 이어져 있어 둘 사이가 기준선 위에 머문다.
    그래서 기준선만 보고 자르면 둘이 한 구간으로 뭉쳐 나오고, 어디서
    끊어야 할지 알 수 없어 결국 감으로 경계를 정하게 된다 — 그 경계가
    실제 블록과 어긋나면 두 색이 서로의 화소를 물어 간다.
    봉우리 사이의 가장 낮은 지점이 곧 두 색의 실제 경계다.
    """
    seg = smooth[lo:hi + 1]
    if seg.size < MIN_PEAK_GAP * 2:
        return [(lo, hi)]

    peaks = find_peaks(seg)
    if len(peaks) < 2:
        return [(lo, hi)]

    cuts = []
    for left, right in zip(peaks, peaks[1:]):
        if right - left < MIN_PEAK_GAP:
            continue
        valley = left + int(np.argmin(seg[left:right + 1]))
        # 평평한 어깨에 생긴 잔물결이 아니라 진짜 골짜기인지 확인한다
        if seg[valley] <= min(seg[left], seg[right]) * VALLEY_DEPTH_RATIO:
            cuts.append(valley)

    if not cuts:
        return [(lo, hi)]

    parts, start = [], lo
    for cut in cuts:
        parts.append((start, lo + cut))
        start = lo + cut + 1
    parts.append((start, hi))
    return parts


def find_color_bands(hist):
    """픽셀이 몰린 hue 구간을 찾아낸다. 구간 하나가 대체로 색 하나에 해당한다.

    봉우리 꼭짓점만 보면 어디서 끊어야 할지 알 수 없으므로,
    기준선 위로 이어지는 덩어리를 먼저 구간으로 삼고,
    그 안에 봉우리가 여럿이면 골짜기에서 다시 나눈다.
    """
    smooth = _smooth(hist)
    floor = smooth.max() * BAND_FLOOR_RATIO

    bands = []
    start = None
    for h in range(180):
        if smooth[h] >= floor and start is None:
            start = h
        elif smooth[h] < floor and start is not None:
            bands.append((start, h - 1))
            start = None
    if start is not None:
        bands.append((start, 179))

    split = []
    for lo, hi in bands:
        split.extend(split_at_valleys(smooth, lo, hi))
    return split


def load_configured_hues():
    """지금 검출에 쓰이는 색 구간을 한글 이름 기준으로 읽어 온다.

    측정한 구간이 실제로 어느 색으로 잡히는지 바로 대조하기 위한 것이라,
    검출이 읽는 파일(color_ranges.json)을 그대로 읽는다.
    """
    with open(CONFIG_PATH, encoding='utf-8') as f:
        spec = json.load(f)
    return {name: [(r['h'][0], r['h'][1]) for r in entry['ranges']]
            for name, entry in spec['colors'].items()}


def colors_claiming(lo, hi, configured):
    """hue 구간 [lo, hi] 를 물고 있는 색 이름들."""
    return [name for name, bands in configured.items()
            if any(b_lo <= hi and lo <= b_hi for b_lo, b_hi in bands)]


class HsvProbe(Node):
    def __init__(self):
        super().__init__('hsv_probe_node')
        self.bridge = CvBridge()
        self.done = False
        self.create_subscription(Image, INPUT_TOPIC, self.on_image, 10)
        self.get_logger().info(f"hsv_probe_node started, waiting for one frame from {INPUT_TOPIC}")

    def on_image(self, msg):
        if self.done:
            return
        self.done = True

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        log, warn = self.get_logger().info, self.get_logger().warn
        analyze(frame, log, warn)

        log("측정 끝. 노드를 종료합니다.")
        rclpy.shutdown()


def analyze(frame, log, warn):
    """BGR 이미지 한 장의 hue 분포를 재서 보고한다.

    카메라 토픽이든 사진 파일이든 여기로 들어오면 같은 결과가 나오게,
    측정 자체는 ROS 와 떼어 둔다.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = (s > S_MIN) & (v > V_MIN)
    hues = h[mask]

    if hues.size == 0:
        log("배경 제외하니 남는 픽셀이 없음 (S/V 기준을 더 낮춰야 할 수 있음)")
        return

    hist = np.bincount(hues, minlength=180)
    report_bands(hist, hsv, mask, log, warn)
    report_top_hues(hist, log)


def report_bands(hist, hsv, mask, log, warn):
    """색 구간과 그 구간의 채도를 함께 보여준다. 그대로 색 정의에 옮겨 쓸 수 있다.

    측정한 구간이 지금 설정에서 어느 색으로 잡히는지도 같이 보여준다.
    한 구간(= 한 물체)을 두 색이 나눠 물고 있으면 그게 바로
    "빨강인데 주황으로도 잡힌다" 의 원인이므로 경고로 짚어 준다.
    """
    bands = find_color_bands(hist)
    log("=== 색 구간 (color_ranges.json 의 h/s 에 넣을 값) ===")
    if not bands:
        log("  뚜렷한 구간이 없음")
        return

    configured = load_configured_hues()
    smooth = _smooth(hist)
    conflicts = []
    unsplittable = []

    h_ch, s_ch = hsv[:, :, 0], hsv[:, :, 1]
    for lo, hi in bands:
        # 봉우리가 여럿인데도 한 구간으로 남았다면 골이 얕아 못 나눈 것이다.
        # 이건 두 색의 hue 가 실제로 거의 같다는 뜻이라, 값을 어떻게 고쳐도
        # 색만으로는 안 갈라진다 — 조명이나 물체를 바꿔야 한다.
        if len(find_peaks(smooth[lo:hi + 1])) > 1:
            unsplittable.append((lo, hi))
        in_band = mask & (h_ch >= lo) & (h_ch <= hi)
        sat = s_ch[in_band]
        pixels = int(hist[lo:hi + 1].sum())
        peak = int(np.argmax(hist[lo:hi + 1])) + lo
        owners = colors_claiming(lo, hi, configured)
        if len(owners) > 1:
            conflicts.append((lo, hi, peak, owners))
        owned = '/'.join(owners) if owners else '지금 아무 색도 안 잡음'
        log(
            f'  "h": [{lo}, {hi}]  "s": [{max(int(np.percentile(sat, 5)) - 10, 0)}, 255]'
            f'   (픽셀 {pixels}개, 봉우리 hue={peak}, 채도 {sat.min()}~{sat.max()})'
            f'  → 현재 설정: {owned}'
        )

    if len(bands) >= 2 and bands[0][0] <= 2 and bands[-1][1] >= 177:
        log("  ※ 첫 구간과 마지막 구간은 빨강 하나가 hue 0 을 넘어가며 갈라진 것일 수 있음")

    for lo, hi, peak, owners in conflicts:
        warn(
            f'  ⚠ hue {lo}~{hi} (봉우리 {peak}) 한 덩어리를 {", ".join(owners)} 가 나눠 물고 있다. '
            f'물체 하나가 두 색으로 쪼개진다는 뜻이므로, 이 구간이 통째로 한 색에만 들어가게 고칠 것.'
        )

    for lo, hi in unsplittable:
        warn(
            f'  ⚠ hue {lo}~{hi} 안에 봉우리가 여럿인데 사이 골이 얕아 나누지 못했다. '
            f'두 물체의 색이 실제로 거의 같다는 뜻이라 hue 값만 고쳐서는 안 갈라진다 — '
            f'조명을 바꾸거나(그림자/반사 줄이기) 채도·명도까지 같이 봐야 한다.'
        )


def report_top_hues(hist, log):
    log("=== Hue 분포 (배경 제외, 픽셀 많은 순) ===")
    top = np.argsort(hist)[::-1]
    shown = 0
    for hue in top:
        if hist[hue] < 150:
            break
        log(f"  hue={int(hue):3d}  pixels={int(hist[hue])}")
        shown += 1
        if shown >= 20:
            break


def run_on_file(path):
    """카메라 없이 사진 한 장으로 측정한다.

    카메라를 쓸 수 없을 때(로봇을 다른 사람이 쓰는 중이거나 자리에 없을 때)
    찍어 둔 이미지로 값을 맞춰 보기 위한 경로다. ROS 를 띄우지 않는다.

    다만 HSV 값은 카메라마다 다르다. 휴대폰으로 찍은 사진은 "두 색이
    갈라지긴 하는가" 를 보는 데까지만 쓰고, 설정에 넣을 값은 실제
    D435i 로 찍은 이미지에서 뽑아야 한다.
    """
    frame = cv2.imread(path)
    if frame is None:
        print(f"이미지를 못 읽었다: {path}")
        return 1
    print(f"파일 측정: {path}  ({frame.shape[1]}x{frame.shape[0]})")
    analyze(frame, print, print)
    return 0


def main(args=None):
    # ros2 run 은 --ros-args 를 붙여 넘기므로, 우리 인자만 골라 본다
    argv = sys.argv[1:] if args is None else args
    if '--image' in argv:
        idx = argv.index('--image')
        if idx + 1 >= len(argv):
            print("--image 다음에 파일 경로가 필요하다")
            return 1
        return run_on_file(argv[idx + 1])

    rclpy.init(args=args)
    node = HsvProbe()
    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == '__main__':
    sys.exit(main() or 0)
