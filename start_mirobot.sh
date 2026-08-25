#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  Mirobot 통합 실행 스크립트
#
#  터미널 하나만 열어서 이 스크립트를 실행하면 아래를 순서대로 다 띄웁니다.
#     1) rosbridge      : 웹 GUI ↔ ROS2 통신 다리 (포트 9090)
#     2) robot_node     : Mirobot 시리얼 드라이버 (실행 시 자동 호밍)
#        └ 호밍이 끝날 때까지 기다린 뒤에 다음 단계로 넘어감 (안전)
#     3) ai_node_new    : MediaPipe 제스처 인식 (+ 내부에 카메라 서버 5000/5001 포함)
#     4) https_server   : 웹 GUI(index.html) 서빙 (포트 8443)
#
#  종료: 이 터미널에서 Ctrl+C 한 번 → 위에서 띄운 모든 프로세스가 같이 종료됩니다.
#
#  사용법:
#     chmod +x start_mirobot.sh     (최초 1회만)
#     ./start_mirobot.sh            (전체 실행)
#     ./start_mirobot.sh --no-ai    (AI 제스처 없이, 로봇+주행만)
#     ./start_mirobot.sh --no-arm   (팔 없이, 웹GUI+주행만)
# ══════════════════════════════════════════════════════════════════════════

# ┌────────────────────────────────────────────────────────────────────────┐
# │ [설정] 이 부분만 본인 환경에 맞는지 확인하세요.                        │
# │ 특히 PKG / EXEC 이름은 setup.py의 entry_points와 같아야 합니다.        │
# │ (확인법:  ros2 pkg executables mirobot_ai_control)                     │
# └────────────────────────────────────────────────────────────────────────┘
ROS_DISTRO_NAME="jazzy"                       # ROS2 배포판 (humble이면 humble로)
WS_DIR="$HOME/mirobot"                        # colcon 워크스페이스 경로
PKG="mirobot_ai_control"                      # 패키지 이름
EXEC_ROBOT="robot_node"                       # 팔 드라이버 실행파일 이름
EXEC_AI="ai_node_new"                         # AI 노드 실행파일 이름
WEBGUI_DIR="$HOME/mirobot_gui"            # index.html + roslib.min.js + https_server.py 폴더
CERT_DIR="$HOME/webgui_certs"                 # cert.pem / key.pem 폴더
SERIAL_PORT="/dev/ttyUSB0"                    # Mirobot 시리얼 포트

# 메카넘 주행 노드를 같이 띄우려면 아래 두 줄을 본인 것에 맞게 채우고
# RUN_MECANUM=true 로 바꾸세요. (모르면 false로 두고 따로 실행해도 됩니다)
RUN_MECANUM=false
PKG_MECANUM="mirobot_ai_control"
EXEC_MECANUM="mecanum_driver_node"

# ══════════════════════════════════════════════════════════════════════════
# 여기서부터는 수정할 필요 없습니다.
# ══════════════════════════════════════════════════════════════════════════
set -o pipefail

RUN_AI=true
RUN_ARM=true
for arg in "$@"; do
  case "$arg" in
    --no-ai)  RUN_AI=false ;;
    --no-arm) RUN_ARM=false ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  esac
done

LOG_DIR="/tmp/mirobot_logs"
mkdir -p "$LOG_DIR"
PIDS=()

C_G="\033[1;32m"; C_Y="\033[1;33m"; C_R="\033[1;31m"; C_B="\033[1;36m"; C_0="\033[0m"
say()  { echo -e "${C_B}[런처]${C_0} $1"; }
ok()   { echo -e "${C_G}  ✓${C_0} $1"; }
warn() { echo -e "${C_Y}  !${C_0} $1"; }
die()  { echo -e "${C_R}  ✗ $1${C_0}"; exit 1; }

# ── Ctrl+C 시 자식 프로세스 전부 정리 ────────────────────────────────────
cleanup() {
  echo ""
  say "종료 중... 실행된 프로세스를 모두 정리합니다."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      # 프로세스 그룹째 종료 (ros2 run은 자식을 또 만들기 때문)
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    fi
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && { kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; }
  done
  ok "모두 종료되었습니다."
  exit 0
}
trap cleanup INT TERM

# 백그라운드 실행 헬퍼 (프로세스 그룹으로 띄워서 통째로 종료 가능하게 함)
launch() {  # launch <표시이름> <로그파일명> <명령...>
  local name="$1"; local logf="$LOG_DIR/$2"; shift 2
  setsid "$@" > "$logf" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  sleep 0.5
  if kill -0 "$pid" 2>/dev/null; then
    ok "$name 실행됨  (로그: $logf)"
  else
    warn "$name 이 바로 종료됐습니다. 로그를 확인하세요: $logf"
    tail -n 15 "$logf"
  fi
}

# ══════════════════════════════════════════════════════════════════════════
say "사전 점검"
# ══════════════════════════════════════════════════════════════════════════
[ -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ] \
  || die "ROS2 '$ROS_DISTRO_NAME' 을 찾을 수 없습니다. 스크립트 상단 ROS_DISTRO_NAME 을 확인하세요."
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
ok "ROS2 $ROS_DISTRO_NAME"

[ -f "$WS_DIR/install/setup.bash" ] \
  || die "워크스페이스가 빌드되어 있지 않습니다: $WS_DIR
       먼저 실행하세요:  cd $WS_DIR && colcon build --packages-select $PKG"
source "$WS_DIR/install/setup.bash"
ok "워크스페이스: $WS_DIR"

[ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ] \
  || die "인증서가 없습니다: $CERT_DIR/cert.pem, key.pem
       setup_https.sh 또는 generate_cert.sh 를 먼저 실행하세요."
ok "인증서: $CERT_DIR"

[ -f "$WEBGUI_DIR/index.html" ] \
  || die "index.html 이 없습니다: $WEBGUI_DIR"
[ -f "$WEBGUI_DIR/roslib.min.js" ] \
  || die "roslib.min.js 가 없습니다: $WEBGUI_DIR
       (index.html 과 같은 폴더에 반드시 있어야 웹 GUI가 동작합니다)"
[ -f "$WEBGUI_DIR/https_server.py" ] \
  || die "https_server.py 가 없습니다: $WEBGUI_DIR"
ok "웹 GUI 파일: $WEBGUI_DIR"

if $RUN_ARM; then
  if [ -e "$SERIAL_PORT" ]; then
    ok "시리얼 포트: $SERIAL_PORT"
  else
    warn "시리얼 포트 $SERIAL_PORT 가 안 보입니다. 팔이 연결·전원 켜져 있는지 확인하세요."
    warn "(그래도 계속 진행합니다. 팔 없이 테스트하려면 --no-arm 옵션을 쓰세요)"
  fi
fi

# 포트가 이미 쓰이고 있으면 이전 실행이 안 죽은 것 → 미리 알려줌
for port in 9090 8443 5000 5001; do
  if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$port "; then
    warn "포트 $port 를 이미 누가 쓰고 있습니다. 이전 실행이 남아있을 수 있어요."
    warn "  정리하려면:  pkill -f rosbridge; pkill -f $EXEC_AI; pkill -f https_server"
  fi
done

# 내 IP (웹 GUI 주소 안내용)
MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$MY_IP" ] && MY_IP="<이 컴퓨터의 IP>"

echo ""
# ══════════════════════════════════════════════════════════════════════════
say "실행 시작"
# ══════════════════════════════════════════════════════════════════════════

# ── 1) rosbridge (웹 GUI ↔ ROS2) ─────────────────────────────────────────
# 웹 페이지가 https 로 열리므로 브라우저가 wss(보안 웹소켓)를 요구함 → ssl:=true
launch "rosbridge (9090)" "rosbridge.log" \
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
    ssl:=true \
    certfile:="$CERT_DIR/cert.pem" \
    keyfile:="$CERT_DIR/key.pem"

# ── 2) robot_node (팔 드라이버) + 호밍 완료 대기 ─────────────────────────
if $RUN_ARM; then
  launch "robot_node (팔 드라이버)" "robot_node.log" \
    ros2 run "$PKG" "$EXEC_ROBOT"

  echo ""
  say "호밍(원점 복귀)을 기다리는 중입니다. 팔에서 손을 떼고 주변을 비워주세요."
  HOMED=false
  for i in $(seq 1 60); do   # 최대 60초 대기
    if grep -q "호밍 및 가동 초기화가 완전히 완료" "$LOG_DIR/robot_node.log" 2>/dev/null; then
      HOMED=true; break
    fi
    # robot_node 가 죽었으면 더 기다릴 필요 없음
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
      warn "robot_node 가 종료되었습니다. 로그를 확인하세요: $LOG_DIR/robot_node.log"
      tail -n 20 "$LOG_DIR/robot_node.log"; break
    fi
    printf "."
    sleep 1
  done
  echo ""
  if $HOMED; then
    ok "호밍 완료 — 이제 다음 노드를 실행합니다."
  else
    warn "호밍 완료 신호를 못 받았습니다. 그래도 진행하지만 팔 동작을 주의해서 확인하세요."
  fi
fi

# ── 3) ai_node_new (제스처 인식 + 카메라 서버 5000/5001) ─────────────────
if $RUN_AI; then
  launch "ai_node_new (제스처 + 카메라)" "ai_node.log" \
    ros2 run "$PKG" "$EXEC_AI"
fi

# ── 4) 메카넘 주행 노드 (선택) ───────────────────────────────────────────
if $RUN_MECANUM; then
  launch "메카넘 주행 노드" "mecanum.log" \
    ros2 run "$PKG_MECANUM" "$EXEC_MECANUM"
fi

# ── 5) 웹 GUI 서버 ───────────────────────────────────────────────────────
cd "$WEBGUI_DIR" || die "웹 GUI 폴더로 이동 실패: $WEBGUI_DIR"
launch "웹 GUI 서버 (8443)" "https_server.log" \
  python3 "$WEBGUI_DIR/https_server.py"

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${C_G}════════════════════════════════════════════════════════${C_0}"
echo -e "${C_G}  준비 완료${C_0}"
echo -e "${C_G}════════════════════════════════════════════════════════${C_0}"
echo ""
echo -e "  폰/노트북 브라우저에서 아래 주소를 여세요:"
echo -e "      ${C_B}https://$MY_IP:8443/index.html${C_0}"
echo ""
echo -e "  ${C_Y}처음 접속할 때 주의${C_0} (자체 서명 인증서라 한 번만 해주면 됩니다)"
echo -e "    1) 위 주소에서 '고급 → 계속 진행' 으로 경고를 승인"
echo -e "    2) ${C_B}https://$MY_IP:9090${C_0} 도 한 번 열어서 승인 (rosbridge)"
echo -e "    3) ${C_B}https://$MY_IP:5001${C_0} 도 한 번 열어서 승인 (카메라 업로드)"
echo ""
echo -e "  GUI 안에서 IP 칸에 ${C_B}$MY_IP${C_0} 를 넣고 '연결'을 누르세요."
echo ""
echo -e "  로그 보기 (다른 터미널에서):  tail -f $LOG_DIR/ai_node.log"
echo -e "  ${C_Y}종료: 이 터미널에서 Ctrl+C${C_0}"
echo ""

# 로그를 실시간으로 흘려보며 대기 (Ctrl+C 로 전체 종료)
tail -f "$LOG_DIR"/*.log &
PIDS+=($!)
wait
