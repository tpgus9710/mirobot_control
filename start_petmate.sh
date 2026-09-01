#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  start_petmate.sh — 라즈베리파이 전체 시스템 실행 (SSH 환경 기준)
#
#  띄우는 것
#    1) rosbridge      (9090, WSS)  — 웹 GUI ↔ ROS2
#    2) mecanum_driver              — 메카넘 베이스 (별도 워크스페이스)
#    3) robot_node                  — Mirobot 팔 드라이버  [--no-arm 으로 생략]
#    4) ai_node_new    (5000/5001)  — MediaPipe + 제스처 제어  [--no-ai 로 생략]
#    5) https_server   (8443)       — 웹 GUI
#
#  워크스페이스를 합치지 않는 이유
#    ROS2 오버레이 방식이 표준이고, 한쪽만 고쳐도 전체를 다시 빌드하지 않아도
#    된다. 파이에서 colcon 빌드는 느리므로 이 차이가 크다.
#    source 순서: /opt/ros → ros2_ws(언더레이) → mirobot(오버레이)
#
#  사용법
#      chmod +x start_petmate.sh        (최초 1회)
#      ./start_petmate.sh               (전체)
#      ./start_petmate.sh --no-arm      (팔 없이 — 베이스 제어 테스트용)
#      ./start_petmate.sh --no-ai       (슬라이더만)
#      ./start_petmate.sh --no-base     (메카넘 없이)
#
#  SSH 로 쓸 때는 tmux 안에서 실행할 것:
#      tmux new -s robot
#      ./start_petmate.sh --no-arm
#      (빠져나오기 Ctrl+B → D / 재접속 tmux attach -t robot)
# ══════════════════════════════════════════════════════════════════════════

ROS_DISTRO_NAME="jazzy"
MIROBOT_WS="$HOME/mirobot"
BASE_WS="$HOME/ros2_ws"
WEBGUI_DIR="$HOME/mirobot_web_gui"
CERT_DIR="$HOME/webgui_certs"
export ROS_DOMAIN_ID=19

MIROBOT_PKG="mirobot_ai_control"
BASE_PKG="mecanum_driver_pid"
BASE_EXEC="mecanum_driver_node"      # = mecanum_driver_node_final:main

set -o pipefail

USE_ARM=true; USE_AI=true; USE_BASE=true
for a in "$@"; do
  case "$a" in
    --no-arm)  USE_ARM=false ;;
    --no-ai)   USE_AI=false ;;
    --no-base) USE_BASE=false ;;
    *) echo "알 수 없는 옵션: $a"; exit 1 ;;
  esac
done

LOG_DIR="/tmp/petmate_logs"; mkdir -p "$LOG_DIR"
PIDS=()

C_G="\033[1;32m"; C_Y="\033[1;33m"; C_R="\033[1;31m"; C_B="\033[1;36m"; C_0="\033[0m"
say()  { echo -e "${C_B}[런처]${C_0} $1"; }
ok()   { echo -e "${C_G}  ✓${C_0} $1"; }
warn() { echo -e "${C_Y}  !${C_0} $1"; }
die()  { echo -e "${C_R}  ✗ $1${C_0}"; exit 1; }

CLEANED=false
cleanup() {
  $CLEANED && return
  CLEANED=true
  echo ""; say "종료 중 — 프로세스를 정리합니다."
  # robot_node 는 SIGINT 로 보내야 시리얼을 정상적으로 닫는다.
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && { kill -INT -"$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null; }
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && { kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; }
  done
  echo ""; ok "정리 완료. 로그: $LOG_DIR"
  exit 0
}
trap cleanup INT TERM

launch() {   # launch <이름> <로그파일> <명령...>
  local name="$1"; local logf="$LOG_DIR/$2"; shift 2
  setsid "$@" > "$logf" 2>&1 &
  local pid=$!
  PIDS+=("$pid"); sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    ok "$name  (로그: $logf)"
  else
    warn "$name 이 바로 종료됨 — 로그 확인"
    tail -n 15 "$logf"
  fi
}

# ══════════════════════════════════════════════════════════════════════════
say "사전 점검"
# ══════════════════════════════════════════════════════════════════════════
[ -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ] || die "ROS2 '$ROS_DISTRO_NAME' 없음"
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
ok "ROS2 $ROS_DISTRO_NAME  (DOMAIN_ID=$ROS_DOMAIN_ID)"

if $USE_BASE; then
  [ -f "$BASE_WS/install/setup.bash" ] || die "메카넘 워크스페이스 빌드 필요:
       cd $BASE_WS && colcon build --packages-select $BASE_PKG"
  source "$BASE_WS/install/setup.bash"
  ok "베이스 워크스페이스: $BASE_WS"
fi

[ -f "$MIROBOT_WS/install/setup.bash" ] || die "미로봇 워크스페이스 빌드 필요:
       cd $MIROBOT_WS && colcon build --packages-select $MIROBOT_PKG"
source "$MIROBOT_WS/install/setup.bash"
ok "미로봇 워크스페이스: $MIROBOT_WS"

[ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ] \
  || die "인증서 없음: $CERT_DIR  ($WEBGUI_DIR/setup_https.sh 실행)"
ok "인증서: $CERT_DIR"

[ -f "$WEBGUI_DIR/index.html" ]     || die "index.html 없음: $WEBGUI_DIR"
[ -f "$WEBGUI_DIR/https_server.py" ] || die "https_server.py 없음: $WEBGUI_DIR"
ok "웹 GUI: $WEBGUI_DIR"

if $USE_ARM; then
  SERIAL=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)
  if [ -z "$SERIAL" ]; then
    warn "시리얼 장치가 없습니다. 팔 없이 테스트하려면 --no-arm 을 쓰세요."
    warn "  CH340(1a86:7523) 인식 실패라면:  sudo apt remove brltty"
  else
    ok "시리얼: $SERIAL"
  fi
fi

# 이전 실행이 포트를 쥐고 있으면 새 프로세스가 조용히 죽는다.
for port in 9090 8443 5000 5001; do
  if ss -ltn 2>/dev/null | grep -q ":$port "; then
    warn "포트 $port 사용 중 — 이전 실행이 남아있습니다."
    warn "  정리:  pkill -f 'ai_node_new|rosbridge|https_server|$BASE_EXEC'"
  fi
done

MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$MY_IP" ] && MY_IP="<파이 IP>"

echo ""
# ══════════════════════════════════════════════════════════════════════════
say "실행 시작"
# ══════════════════════════════════════════════════════════════════════════

launch "rosbridge (9090, WSS)" "rosbridge.log" \
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
    ssl:=true certfile:="$CERT_DIR/cert.pem" keyfile:="$CERT_DIR/key.pem"

$USE_BASE && launch "메카넘 드라이버" "mecanum.log" \
  ros2 run "$BASE_PKG" "$BASE_EXEC"

$USE_ARM && launch "robot_node (Mirobot 팔)" "robot_node.log" \
  ros2 run "$MIROBOT_PKG" robot_node

# ai_node 는 팔 호밍이 끝난 뒤 붙는 편이 안전하다.
# 호밍 중에 제스처 목표값이 들어가면 낡은 값 재생 문제가 생길 수 있다.
$USE_ARM && $USE_AI && { say "호밍 대기 (5초)"; sleep 5; }

$USE_AI && launch "ai_node_new (5000/5001)" "ai_node.log" \
  ros2 run "$MIROBOT_PKG" ai_node_new

cd "$WEBGUI_DIR" || die "웹 GUI 폴더 이동 실패"
launch "웹 GUI 서버 (8443)" "https_server.log" python3 "$WEBGUI_DIR/https_server.py"

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${C_G}════════════════════════════════════════════════════════${C_0}"
echo -e "${C_G}  실행 완료${C_0}   팔:$($USE_ARM && echo ON || echo OFF)"\
"  AI:$($USE_AI && echo ON || echo OFF)  베이스:$($USE_BASE && echo ON || echo OFF)"
echo -e "${C_G}════════════════════════════════════════════════════════${C_0}"
echo ""
echo -e "  조작 기기 브라우저에서:  ${C_B}https://$MY_IP:8443/index.html${C_0}"
echo ""
echo -e "  ${C_Y}처음 접속하는 기기라면${C_0} 아래 세 곳을 먼저 열어 인증서를 승인하세요:"
echo -e "    ${C_B}https://$MY_IP:5000/video_feed${C_0}"
echo -e "    ${C_B}https://$MY_IP:5001${C_0}"
echo -e "    ${C_B}https://$MY_IP:9090${C_0}"
echo -e "  IP 입력칸에 ${C_B}$MY_IP${C_0} 를 넣고 연결하세요."
echo ""
echo -e "  토픽 확인 (다른 창):"
echo -e "    ${C_B}source /opt/ros/$ROS_DISTRO_NAME/setup.bash && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID${C_0}"
echo -e "    ${C_B}ros2 topic echo /cmd_vel${C_0}"
echo ""
echo -e "  로그: ${C_B}tail -f $LOG_DIR/ai_node.log${C_0}"
echo -e "  종료: 이 창에서 ${C_B}Ctrl+C${C_0}"
echo ""

# 프로세스가 죽으면 알린다 (조용히 사라지면 원인 파악이 어렵다)
while true; do
  sleep 3
  for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      warn "프로세스 하나가 종료되었습니다 (PID ${PIDS[$i]}). 로그를 확인하세요: $LOG_DIR"
      unset 'PIDS[i]'
    fi
  done
done