#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  start_petmate.sh — 라즈베리파이 전체 시스템 실행 (SSH 환경 기준)
#
#  띄우는 것
#    1) rosbridge      (9090, WSS)  — 웹 GUI ↔ ROS2
#    2) mecanum_driver              — 메카넘 베이스 (별도 워크스페이스)
#    3) robot_node                  — Mirobot 팔 드라이버  [--no-arm 으로 생략]
#    4) robot_cam_node (5002)       — 로봇 탑재 카메라 MJPEG 릴레이
#    5) ai_node_new    (5000/5001)  — MediaPipe + 제스처 제어  [--no-ai 로 생략]
#    6) https_server   (8443)       — 웹 GUI
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

# 이 스크립트가 들어있는 워크스페이스를 쓴다. 경로를 박아두지 않는 이유는
# git worktree 로 사람마다 작업 디렉토리를 나눠 쓰기 때문이다.
#   ~/mirobot/start_petmate.sh       → ~/mirobot 을 씀
#   ~/mirobot-taro/start_petmate.sh  → ~/mirobot-taro 를 씀
# 각 worktree 에 이 스크립트 사본이 있으므로 자기 디렉토리 것을 실행하면
# 알아서 맞는 워크스페이스가 잡힌다. readlink -f 는 ~/start_petmate.sh
# 같은 심볼릭 링크를 실제 위치로 풀기 위한 것이다.
MIROBOT_WS="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
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
ok "미로봇 워크스페이스: $MIROBOT_WS  (브랜치: $(git -C "$MIROBOT_WS" branch --show-current 2>/dev/null || echo '?'))"

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
for port in 9090 8443 5000 5001 5002; do
  if ss -ltn 2>/dev/null | grep -q ":$port "; then
    warn "포트 $port 사용 중 — 이전 실행이 남아있습니다."
    warn "  정리:  pkill -f 'ai_node_new|rosbridge|https_server|$BASE_EXEC'"
  fi
done

# 접속 가능한 주소를 전부 모은다. Tailscale 을 쓰면 wlan0(192.168.x)과
# tailscale0(100.x)이 동시에 존재하는데, 예전처럼 첫 번째 하나만 안내하면
# 다른 망에서 접속할 때 어느 주소를 인증서 승인해야 하는지 알 수 없었다.
ALL_IPS=$(ip -4 -o addr show scope global 2>/dev/null \
          | awk '{split($4,a,"/"); print a[1]}' | sort -u)
MY_IP=$(echo "$ALL_IPS" | grep -v '^100\.' | head -1)
[ -z "$MY_IP" ] && MY_IP=$(echo "$ALL_IPS" | head -1)
[ -z "$MY_IP" ] && MY_IP="<파이 IP>"
TS_IP=$(echo "$ALL_IPS" | grep '^100\.' | head -1)

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

# 로봇 탑재 카메라 릴레이(5002). rpicam-vid 의 MJPEG 을 재인코딩 없이 넘기므로
# 부하가 거의 없다. 팔 없이 테스트할 때도 주행 시야는 필요하므로 --no-arm 과
# 무관하게 띄운다. 카메라가 없으면 이 노드만 재기동을 반복하고 나머지는 정상.
launch "로봇 카메라 (5002)" "robot_cam.log" \
  ros2 run "$MIROBOT_PKG" robot_cam_node

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
# 접속에 쓴 주소와 GUI 의 IP 입력칸 값은 반드시 같아야 한다. 페이지는
# 8443 으로 열고 입력칸에는 다른 주소를 넣으면, WSS(9090/5001)가 인증서
# 예외를 못 받은 origin 으로 가서 조용히 실패한다 — 화면은 나오는데
# 로봇이 안 움직이는 증상이 정확히 이것이다.
for _ip in $ALL_IPS; do
  _label=""
  case "$_ip" in 100.*) _label="  (Tailscale)" ;; esac
  echo -e "  브라우저에서:  ${C_B}https://$_ip:8443/index.html${C_0}$_label"
  echo -e "  ${C_Y}처음 접속하는 기기라면${C_0} 아래 세 곳도 먼저 열어 인증서를 승인:"
  echo -e "    ${C_B}https://$_ip:5000/video_feed${C_0}"
  echo -e "    ${C_B}https://$_ip:5001${C_0}"
  echo -e "    ${C_B}https://$_ip:5002${C_0}   (로봇 카메라)"
  echo -e "    ${C_B}https://$_ip:9090${C_0}"
  echo -e "  그리고 IP 입력칸에 ${C_B}$_ip${C_0} — 페이지를 연 주소와 같은 것을 넣으세요."
  echo ""
done
if [ -n "$TS_IP" ]; then
  echo -e "  ${C_Y}주의${C_0} 인증서는 위 주소들이 모두 들어가야 유효합니다."
  echo -e "        주소가 바뀌었으면:  ${C_B}$WEBGUI_DIR/setup_https.sh${C_0} 재실행 후 전체 재시작"
fi
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