#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로봇 탑재 카메라 MJPEG 릴레이 노드 (포트 5002)

[구조]
    rpicam-vid --codec mjpeg → stdout
        └ 리더 스레드: JPEG 경계(FFD8~FFD9)로 잘라 최신 프레임만 보관
              └ Flask /video_feed → 브라우저

  파이썬이 인코딩을 하지 않는다. rpicam-vid 가 이미 JPEG 으로 뱉으므로
  바이트를 잘라 넘기기만 한다. Pi 5 에서 MediaPipe 가 이미 CPU 를 쓰고
  있으므로 여기서 추가 인코딩 부하를 만들지 않는 것이 중요하다.

[모드 연동]
  /mirobot/control_mode ('arm' / 'base') 를 구독해서 카메라를 바꾼다.
  두 카메라를 동시에 켜지 않고 프로세스를 죽였다 다시 띄운다.
  동시 구동은 CPU·대역폭을 두 배로 쓰는데, 어차피 화면에는 한 번에
  하나만 보이므로 낭비다.

[호밍 안전 — 명시적 검수]
  이 노드는 표시 전용이다. 관절/그리퍼 목표값을 만들지도, 큐잉하지도,
  지연 전송하지도 않는다. 발행하는 토픽이 하나도 없고 구독하는 것은
  모드 문자열뿐이다. 따라서 _hard_reset_control_state() 에 등록할
  상태가 없으며, 호밍 직후 낡은 값이 재생될 경로 자체가 존재하지 않는다.
  latest_jpeg 는 낡은 이미지가 잠시 남을 수 있으나 이는 화면 표시일 뿐
  로봇 동작에 영향이 없다.
"""

import os
import time
import signal
import threading
import re
import subprocess

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from flask import Flask, Response

# ── 설정 ──────────────────────────────────────────────────────────────────
PORT = 5002

# 모드 → 카메라 번호. 실기에서 반대면 이 두 값만 바꾼다.
#
# [주의] 여기 적힌 카메라가 실제로 없으면 rpicam-vid 가 즉시 죽고, 리더가
# 빈 읽기를 받아 '종료됨 → 재기동' 루프에 빠진다. 주행 시야가 가장 필요한
# 순간에 스트림이 끊기므로, 기동 시 실제 장착된 카메라를 확인해서 없는
# 번호는 있는 것으로 대체한다(_resolve_cam_map).
CAM_FOR_MODE = {'arm': 0, 'base': 1}
DEFAULT_MODE = 'arm'

WIDTH, HEIGHT, FPS = 640, 480, 15
JPEG_QUALITY = 70          # rpicam-vid 의 MJPEG 품질(1~100)

# 모드가 빠르게 왕복해도 카메라를 계속 재시작하지 않도록 하는 디바운스(초).
# ai_node 의 MODE_COOLDOWN_SEC 가 2.0 이라 실제로는 거의 걸리지 않지만,
# 토픽이 1Hz 로 반복 발행되므로 방어용으로 둔다.
SWITCH_DEBOUNCE_SEC = 1.5

RESTART_DELAY_SEC = 2.0    # rpicam-vid 가 죽었을 때 재기동까지 대기
CERT_DIR = os.path.expanduser('~/webgui_certs')

SOI = b'\xff\xd8'          # JPEG 시작
EOI = b'\xff\xd9'          # JPEG 끝


class RobotCamNode(Node):

    def __init__(self):
        super().__init__('robot_cam_node')

        self.latest_jpeg = None
        self.jpeg_lock = threading.Lock()

        self.cam_for_mode = self._resolve_cam_map()
        self.mode = DEFAULT_MODE
        self.desired_cam = self.cam_for_mode[DEFAULT_MODE]
        self.active_cam = None
        self.switch_request_t = 0.0
        self.proc = None
        self.running = True
        self.frame_count = 0
        self.last_stat_t = time.time()

        self.create_subscription(
            String, '/mirobot/control_mode', self._mode_callback, 10)

        threading.Thread(target=self._capture_loop, daemon=True).start()
        self._start_flask()

        self.get_logger().info(
            f"로봇 카메라 노드 시작 — 포트 {PORT}, "
            f"{WIDTH}x{HEIGHT}@{FPS}fps, arm=cam{self.cam_for_mode['arm']} "
            f"base=cam{self.cam_for_mode['base']}")

    # ── 모드 구독 ─────────────────────────────────────────────────────────
    def _mode_callback(self, msg):
        mode = msg.data.strip().lower()
        if mode not in self.cam_for_mode:
            return
        if mode == self.mode:
            return
        self.mode = mode
        self.desired_cam = self.cam_for_mode[mode]
        self.switch_request_t = time.time()
        self.get_logger().info(
            f"[모드] {mode.upper()} — cam{self.desired_cam} 로 전환 예정 "
            f"({SWITCH_DEBOUNCE_SEC:.1f}초 후)")

    # ── rpicam-vid 관리 ───────────────────────────────────────────────────
    @staticmethod
    def _available_cams():
        """실제로 장착된 카메라 번호 목록. 확인 불가하면 빈 리스트.

        rpicam-hello --list-cameras 의 출력에서 '0 : imx219 ...' 형태의
        선두 숫자를 뽑는다. 실패하면 판단을 포기하고 빈 리스트를 돌려주어,
        호출부가 기존 매핑을 그대로 쓰도록 한다(잘못 추측해 멀쩡한 설정을
        덮어쓰는 것보다 낫다).
        """
        try:
            out = subprocess.run(['rpicam-hello', '--list-cameras'],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return []
        cams = []
        for line in out.splitlines():
            m = re.match(r'\s*(\d+)\s*:', line)
            if m:
                cams.append(int(m.group(1)))
        return sorted(set(cams))

    def _resolve_cam_map(self):
        """CAM_FOR_MODE 에서 존재하지 않는 카메라를 있는 것으로 대체한다.

        카메라가 하나뿐인 구성에서 base 모드로 넘어가면 없는 cam1 을 띄우려다
        스트림이 통째로 끊긴다. 그럴 바엔 같은 카메라를 계속 보여주는 편이 낫다.
        나중에 두 번째 카메라를 달면 이 함수가 알아서 원래 매핑을 쓴다.
        """
        avail = self._available_cams()
        if not avail:
            self.get_logger().warn(
                "카메라 목록을 확인하지 못했습니다 — 설정값을 그대로 씁니다.")
            return dict(CAM_FOR_MODE)
        fallback = avail[0]
        resolved = {}
        for mode, cam in CAM_FOR_MODE.items():
            if cam in avail:
                resolved[mode] = cam
            else:
                resolved[mode] = fallback
                self.get_logger().warn(
                    f"{mode.upper()} 모드에 지정된 cam{cam} 이 없습니다 "
                    f"(장착: {avail}) — cam{fallback} 으로 대체합니다. "
                    f"두 번째 카메라를 달면 자동으로 원래 설정을 씁니다.")
        return resolved

    def _spawn(self, cam):
        cmd = [
            'rpicam-vid',
            '--nopreview',
            '--camera', str(cam),
            '--codec', 'mjpeg',
            '--quality', str(JPEG_QUALITY),
            '--width', str(WIDTH),
            '--height', str(HEIGHT),
            '--framerate', str(FPS),
            '--timeout', '0',        # 무한
            '--output', '-',         # stdout
            '--flush',               # 프레임마다 즉시 내보냄 (지연 감소)
        ]
        self.get_logger().info(f"rpicam-vid 기동 — cam{cam}")
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=open('/tmp/rpicam_vid.log', 'ab', buffering=0),
            bufsize=0, preexec_fn=os.setsid)

    def _kill(self):
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=3.0)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        self.proc = None

    # ── 캡처 루프 ─────────────────────────────────────────────────────────
    def _capture_loop(self):
        buf = bytearray()
        while self.running:
            # 전환 요청 처리 (디바운스 경과 후에만)
            if (self.desired_cam != self.active_cam and
                    time.time() - self.switch_request_t >= SWITCH_DEBOUNCE_SEC):
                self._kill()
                buf.clear()
                # 마지막 프레임은 지우지 않는다. 전환하는 1초 남짓 동안
                # 검은 화면 대신 정지 화면이 보여야 조작자가 "끊겼다"고
                # 오해하지 않는다.
                try:
                    self.proc = self._spawn(self.desired_cam)
                    self.active_cam = self.desired_cam
                except Exception as e:
                    self.get_logger().error(f"rpicam-vid 기동 실패: {e}")
                    time.sleep(RESTART_DELAY_SEC)
                    continue

            if self.proc is None:
                try:
                    self.proc = self._spawn(self.desired_cam)
                    self.active_cam = self.desired_cam
                except Exception as e:
                    self.get_logger().error(f"rpicam-vid 기동 실패: {e}")
                    time.sleep(RESTART_DELAY_SEC)
                    continue

            chunk = self.proc.stdout.read(4096)
            if not chunk:
                self.get_logger().warn(
                    f"rpicam-vid(cam{self.active_cam}) 종료됨 — "
                    f"{RESTART_DELAY_SEC:.0f}초 후 재기동")
                self._kill()
                self.active_cam = None
                buf.clear()
                time.sleep(RESTART_DELAY_SEC)
                continue

            buf.extend(chunk)

            # 완성된 JPEG 을 잘라낸다. 여러 장이 한 번에 들어올 수 있으므로
            # 루프를 돌되, 마지막 한 장만 보관한다(최신 프레임 우선).
            while True:
                start = buf.find(SOI)
                if start < 0:
                    buf.clear()
                    break
                end = buf.find(EOI, start + 2)
                if end < 0:
                    # 아직 다 안 들어옴. 앞쪽 쓰레기만 버리고 대기.
                    if start > 0:
                        del buf[:start]
                    break
                frame = bytes(buf[start:end + 2])
                del buf[:end + 2]
                with self.jpeg_lock:
                    self.latest_jpeg = frame
                self.frame_count += 1

            # 30초마다 수신 상태 요약
            now = time.time()
            if now - self.last_stat_t >= 30.0:
                fps = self.frame_count / (now - self.last_stat_t)
                self.get_logger().info(
                    f"[카메라] cam{self.active_cam} 수신 {fps:.1f}fps")
                self.frame_count = 0
                self.last_stat_t = now

    # ── Flask ─────────────────────────────────────────────────────────────
    def _start_flask(self):
        app = Flask(__name__)

        @app.after_request
        def cors(resp):
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return resp

        @app.route('/video_feed')
        def video_feed():
            def gen():
                last = None
                while self.running:
                    with self.jpeg_lock:
                        jpg = self.latest_jpeg
                    # 같은 프레임을 반복 전송하지 않는다(대역폭 절약).
                    if jpg is not None and jpg is not last:
                        last = jpg
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / (FPS * 2))
            return Response(
                gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/cam_status')
        def cam_status():
            with self.jpeg_lock:
                has = self.latest_jpeg is not None
            return ({'mode': self.mode,
                     'active_cam': self.active_cam,
                     'desired_cam': self.desired_cam,
                     'streaming': has}, 200)

        def run():
            cert = os.path.join(CERT_DIR, 'cert.pem')
            key = os.path.join(CERT_DIR, 'key.pem')
            if os.path.exists(cert) and os.path.exists(key):
                self.get_logger().info(f"HTTPS 로 기동합니다 (포트 {PORT}).")
                app.run(host='0.0.0.0', port=PORT, threaded=True,
                        use_reloader=False, ssl_context=(cert, key))
            else:
                self.get_logger().warn(
                    f"인증서가 없어 HTTP 로 기동합니다 (포트 {PORT}). "
                    f"HTTPS 페이지에서는 혼합 콘텐츠로 차단됩니다.")
                app.run(host='0.0.0.0', port=PORT, threaded=True,
                        use_reloader=False)

        threading.Thread(target=run, daemon=True).start()

    def destroy_node(self):
        self.running = False
        self._kill()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
