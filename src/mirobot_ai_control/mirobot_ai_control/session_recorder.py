# ══════════════════════════════════════════════════════════════════════════
#  session_recorder.py — GRU 학습 데이터 세션 녹화 모듈
#
#  역할:
#    MediaPipe에 들어가기 "직전"의 480×480 프레임(좌우반전 포함, 오버레이 제외)을
#    JPEG 시퀀스로 저장한다. 이 프레임이 곧 학습 입력(x)의 원천이고,
#    같은 프레임에 NLF를 돌려 타겟(y)을 만들기 때문에 저장 지점이 중요하다.
#
#  설계 원칙:
#    - 라이브 파이프라인에 부하를 주지 않기 위해 저장은 별도 스레드 + 큐로 처리.
#      process_frame() 쪽에서는 큐에 넣기만 하고 즉시 반환한다.
#    - 큐에는 "이미지와 시각 정보만" 들어간다. 조인트 목표값 등 제어 상태는
#      절대 섞지 않는다 (호밍 안전 규약: 제어 경로와 완전 분리).
#    - 타임스탬프는 프레임마다 텍스트 파일에 즉시 기록(크래시 대비)하고,
#      정상 종료 시 .npy로도 변환 저장한다.
#    - NLF_RECORD_DIR 환경변수 아래에 "시작 시각 이름의 하위 폴더"를 자동 생성.
#      → 같은 변수로 여러 번 실행해도 절대 겹쳐쓰기/섞임이 발생하지 않는다.
#
#  사용법 (ai_node_new.py 쪽):
#    from session_recorder import SessionRecorder
#    self.recorder = SessionRecorder.from_env(self.get_logger())   # __init__에서
#    self.recorder.submit(frame)                                   # process_frame에서
#    self.recorder.close()                                         # destroy_node에서
# ══════════════════════════════════════════════════════════════════════════

import os
import json
import time
import queue
import threading
from datetime import datetime

import cv2
import numpy as np

JPEG_QUALITY = 95          # 재압축 손실 최소화 (학습 데이터 품질 우선)
QUEUE_MAX = 400            # 약 20초 분량 버퍼. 디스크가 못 따라가면 여기서 드랍 카운트됨
FLUSH_EVERY = 20           # 타임스탬프 텍스트 파일 flush 주기 (프레임 수)


class SessionRecorder:
    """MediaPipe 입력 프레임을 비동기로 저장하는 녹화기.

    enabled=False 로 만들어지면 모든 메서드가 아무 일도 하지 않는다
    (환경변수 미설정 시 라이브 파이프라인에 영향 0).
    """

    # ── 생성 ──────────────────────────────────────────────────────────────
    def __init__(self, session_dir, logger=None):
        self.enabled = session_dir is not None
        self.logger = logger
        if not self.enabled:
            return

        self.session_dir = session_dir
        self.frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=False)  # 새 폴더여야만 함 (겹침 원천 차단)

        self.q = queue.Queue(maxsize=QUEUE_MAX)
        self.frame_idx = 0          # 제출된 프레임 수 (파일명 인덱스)
        self.saved_count = 0        # 실제 저장 완료 수
        self.dropped_count = 0      # 큐 가득참으로 버린 수
        self.dup_count = 0          # 같은 프레임이 반복 저장된 수 (업로드 밀림)
        self._last_src_t = -1.0
        self._dup_run = 0           # 현재 연속 중복 길이
        self._closing = False

        # 타임스탬프 즉시 기록용 텍스트 파일 (크래시가 나도 여기까지는 남음)
        self._ts_file = open(os.path.join(session_dir, "timestamps.txt"), "w")
        self._ts_file.write("# frame_idx, t_mono, t_wall, src_t\n")

        # 시작 메타데이터 (종료 시 통계를 덧붙여 다시 저장)
        self.meta = {
            "session_dir": session_dir,
            "started_wall": datetime.now().isoformat(timespec="seconds"),
            "started_mono": time.monotonic(),
            "jpeg_quality": JPEG_QUALITY,
            "frame_size": [480, 480],
            "note": "frames = MediaPipe 입력 직전(좌우반전 후, 오버레이 전) 프레임",
        }
        self._write_meta()

        # ── 블록 마커 (브라우저에서 눌러도 "서버 시계"로 기록됨) ──────────
        # 브라우저 시계와 녹화 타임스탬프는 서로 다른 시계라 그대로 쓰면
        # 폰으로 촬영할 때 오차가 생긴다. 그래서 마커 요청이 도착한 순간의
        # time.monotonic()을 서버가 직접 찍는다 → 프레임과 같은 시계.
        self.markers = []
        self._marker_lock = threading.Lock()
        self._markers_path = os.path.join(session_dir, "markers.json")

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        self._log(f"[녹화] 시작 → {session_dir}")

    # ── 마커 기록 ─────────────────────────────────────────────────────────
    def add_marker(self, event, kind="block", note=None):
        """블록/단계 경계를 현재 서버 시계로 기록한다.

        kind: "block"(블록 경계) | "step"(블록 내 세부 단계) | "end"(촬영 종료)
        기록할 때마다 즉시 파일로 저장하므로 중간에 꺼져도 남는다.
        """
        if not self.enabled:
            return None
        m = {
            "event": event,
            "kind": kind,
            "t_mono": time.monotonic(),
            "t_wall": time.time(),
            "wall_str": datetime.now().strftime("%H:%M:%S"),
        }
        if note:
            m["note"] = note
        with self._marker_lock:
            self.markers.append(m)
            with open(self._markers_path, "w") as f:
                json.dump({"markers": self.markers}, f, indent=2, ensure_ascii=False)
        if kind != "step":     # 세부 단계는 로그가 너무 잦아 블록/종료만 출력
            self._log(f"[마커] {event}")
        return m

    def undo_marker(self):
        """직전 마커 취소 (잘못 눌렀을 때)."""
        if not self.enabled:
            return None
        with self._marker_lock:
            if not self.markers:
                return None
            m = self.markers.pop()
            with open(self._markers_path, "w") as f:
                json.dump({"markers": self.markers}, f, indent=2, ensure_ascii=False)
        self._log(f"[마커] 취소됨: {m['event']}")
        return m

    def has_end_marker(self):
        with self._marker_lock:
            return any(m.get("kind") == "end" or m.get("event") == "END"
                       for m in self.markers)

    @classmethod
    def from_env(cls, logger=None):
        """NLF_RECORD_DIR 환경변수를 읽어 녹화기를 만든다.

        미설정이면 비활성 녹화기를 반환한다 (평상시 실행에 영향 없음).
        설정돼 있으면 그 아래에 시작 시각 이름의 하위 폴더를 만들어 사용한다.
        예) NLF_RECORD_DIR=~/dataset/session1  이고 14시 23분 01초에 시작하면
            → ~/dataset/session1/20260808_142301/  에 저장
        """
        base = os.environ.get("NLF_RECORD_DIR")
        if not base:
            return cls(None, logger)
        base = os.path.expanduser(base)
        sub = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(os.path.join(base, sub), logger)

    # ── 프레임 제출 (라이브 파이프라인 쪽에서 호출, 논블로킹) ─────────────
    def submit(self, frame_bgr, src_t=-1.0):
        """MediaPipe 입력 직전 프레임을 저장 큐에 넣는다. 즉시 반환.

        src_t: 원격(폰/노트북 업로드) 프레임의 도착 시각. 로컬 웹캠이면 -1.
               업로드가 20Hz보다 느려 같은 프레임이 중복 처리된 구간을
               후처리에서 식별하는 데 쓴다 (src_t가 같으면 중복 프레임).
        """
        if not self.enabled or self._closing:
            return
        t_mono = time.monotonic()
        t_wall = time.time()

        # ── 멈춤 감지 ─────────────────────────────────────────────────────
        # 업로드가 밀리면 서버의 remote_frame이 갱신되지 않고, 같은 프레임이
        # 반복 저장된다(화면이 멈춘 것처럼 보임). src_t가 직전과 같으면
        # 새 프레임이 아니라는 뜻이므로 그 횟수를 센다.
        if src_t > 0 and src_t == self._last_src_t:
            self.dup_count += 1
            self._dup_run += 1
            # 1초 이상(20프레임) 연속으로 같은 프레임이면 즉시 알린다.
            if self._dup_run == 20 or (self._dup_run > 20 and self._dup_run % 100 == 0):
                self._log(f"[녹화] ⚠ 프레임이 {self._dup_run/20:.0f}초째 갱신되지 않습니다 "
                          f"— 업로드가 밀리는 중 (누적 중복 {self.dup_count})", warn=True)
        else:
            if self._dup_run >= 20:
                self._log(f"[녹화] 프레임 수신 정상화 (멈춤 {self._dup_run/20:.1f}초)")
            self._dup_run = 0
            self._last_src_t = src_t

        idx = self.frame_idx
        self.frame_idx += 1
        try:
            # copy(): 호출 측이 이후 프레임에 오버레이를 그리므로 원본을 격리
            self.q.put_nowait((idx, frame_bgr.copy(), t_mono, t_wall, src_t))
        except queue.Full:
            self.dropped_count += 1
            if self.dropped_count % 50 == 1:
                self._log(f"[녹화] 저장 큐 가득참 — 누적 드랍 {self.dropped_count}", warn=True)

    # ── 저장 스레드 ───────────────────────────────────────────────────────
    def _writer_loop(self):
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        while True:
            item = self.q.get()
            if item is None:          # 종료 신호
                break
            idx, frame, t_mono, t_wall, src_t = item
            path = os.path.join(self.frames_dir, f"frame_{idx:06d}.jpg")
            ok = cv2.imwrite(path, frame, params)
            if ok:
                self.saved_count += 1
                self._ts_file.write(f"{idx},{t_mono:.6f},{t_wall:.6f},{src_t:.6f}\n")
                if self.saved_count % FLUSH_EVERY == 0:
                    self._ts_file.flush()
            else:
                self._log(f"[녹화] 프레임 저장 실패: {path}", warn=True)

    # ── 종료 (큐 드레인 → npy 변환 → 메타 확정) ─────────────────────────
    def close(self):
        if not self.enabled or self._closing:
            return
        self._closing = True
        pending = self.q.qsize()
        if pending > 0:
            self._log(f"[녹화] 종료 중 — 남은 {pending}프레임 저장을 마무리합니다...")
        self.q.put(None)
        self._thread.join(timeout=60.0)
        self._ts_file.flush()
        self._ts_file.close()

        # 텍스트 → npy 변환 (후속 파이프라인은 npy를 사용)
        ts_path = os.path.join(self.session_dir, "timestamps.txt")
        try:
            data = np.loadtxt(ts_path, delimiter=",", comments="#", ndmin=2)
            if data.size > 0:
                order = np.argsort(data[:, 0])   # 저장 순서가 섞였어도 인덱스순 정렬
                data = data[order]
                np.save(os.path.join(self.session_dir, "timestamps.npy"),
                        data.astype(np.float64))
        except Exception as e:
            self._log(f"[녹화] timestamps.npy 변환 실패(텍스트는 남아있음): {e}", warn=True)

        # 최종 통계를 메타에 기록
        self.meta.update({
            "ended_wall": datetime.now().isoformat(timespec="seconds"),
            "submitted": self.frame_idx,
            "saved": self.saved_count,
            "dropped_queue_full": self.dropped_count,
            "duplicate_frames": self.dup_count,
        })
        self._write_meta()
        dup_pct = self.dup_count / max(self.frame_idx, 1) * 100
        self._log(f"[녹화] 종료 — 저장 {self.saved_count} / 제출 {self.frame_idx} "
                  f"(드랍 {self.dropped_count}, 중복 {self.dup_count} = {dup_pct:.1f}%) "
                  f"→ {self.session_dir}")
        if dup_pct > 10:
            self._log(f"[녹화] ⚠ 중복 프레임 비율이 높습니다({dup_pct:.1f}%). "
                      f"업로드가 자주 밀렸다는 뜻이며, 그만큼 실제 동작이 기록되지 "
                      f"않았습니다. 재촬영을 권장합니다.", warn=True)

    # ── 내부 유틸 ─────────────────────────────────────────────────────────
    def _write_meta(self):
        with open(os.path.join(self.session_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    def _log(self, msg, warn=False):
        if self.logger is not None:
            (self.logger.warning if warn else self.logger.info)(msg)
        else:
            print(msg)
