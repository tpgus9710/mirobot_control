--- how to run

~/mirobot #folder location
source install/setup.bash #register at terminal
ros2 run mirobot_ai_control 'filename' #run file


--- ros bridge launch

ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
  ssl:=true \
  certfile:=$HOME/webgui_certs/cert.pem \
  keyfile:=$HOME/webgui_certs/key.pem


--- build

colcon build --packages-select mirobot_ai_control # build only mirobot_ai_folder


--- env source

source install/setup.bash

═══════════════════════════════════════════════════════════════════
 제스처 모드 전환 (ARM ↔ BASE)
═══════════════════════════════════════════════════════════════════

왼팔(제어팔)로 Mirobot 팔을, 오른팔(베이스팔)로 메카넘 주행을 조작한다.
한 손이 양쪽을 겸하면 팔을 옮기다 베이스가 따라 나가므로 손을 나눴다.

--- 규칙

  제어팔만 중립      →  ARM
  베이스팔만 중립    →  BASE
  둘 다 중립         →  유지 (AI ON/OFF 토글용 자세라 비워둠)
  둘 다 비중립       →  유지

'한 손만' 이 절대 조건이다. 위 목표를 2.5초 끊김 없이 유지해야 전환된다.
안 보이는 팔은 '중립이 아님' 으로 본다 (화각 좁은 기기 대응).

--- '중립' 은 쉬는 자세가 아니다

캘리브레이션 1단계에서 잡은 "팔꿈치를 편하게 굽혀 몸 앞쪽에 손을 둔"
자세다. 양팔 모두 이 자세가 기준이며 좌우(lat)만 거울상으로 뒤집는다.

  팔을 내린 쉬는 자세  →  r 5~7  (한참 밖 → 조작 중 오전환 없음)
  몸 앞으로 든 자세    →  r 0.3~0.6 (중립)

전환하려면 그 손을 의도적으로 '몸 앞 중앙' 으로 들어야 한다. 옆으로
벌리면서 들면 lat 이 커져서 r 이 0.9 언저리에 머물고 전환되지 않는다.

--- 중립 기준이 두 벌인 이유 (헷갈리기 쉬움)

  _mode_neutral_ref()   전환 트리거용 — 제어팔 캘리브 중립 (좌우만 미러)
  _drive_neutral_ref()  주행 원점용   — 베이스팔의 실측 '쉬는' 자세

한 값으로 겸하면 반드시 깨진다. 주행 원점에 제어팔 중립을 쓰면, 손을
가만히 내린 정지 상태가 '최대 후진' 으로 읽혀 로봇이 저절로 달린다.
반대로 트리거에 쉬는 자세를 쓰면 그 조건이 상시 성립해 멋대로 전환된다.

베이스팔 중립은 캘리브레이션 1단계에서 반대쪽 팔을 같이 재서 얻는다.
그때 오른팔이 화각 밖이면 측정이 안 되고, 그러면 BASE 진입이 막힌다
(로그: "베이스팔 중립 미측정"). 1단계에서 양팔이 모두 보이게 할 것.

--- 진단: [모드판정] 로그 하나로 끝난다

  tail -f /tmp/petmate_logs/ai_node.log | grep 모드판정

  [모드판정] left r 5.72  right r 0.55(중립) ... 현재 ARM  목표 base 1.2/2.5s
             └ 이 숫자만 보면 된다 ┘                        └ 오르면 정상 ┘

  (중립) 표시가 한쪽에만 있어야 한다. 둘 다거나 둘 다 아니면 목표 '-'.
  유지가 0.5초 넘게 쌓였다 끊기면 "전환 중단 — <이유>" 를 직접 찍는다.
  판정 자체가 막히면 "막힘 — <사유>" 를 찍는다 (호밍/쿨다운/캘리브/미측정/
  양팔 안 보임).

--- 튜닝 (ai_node_new.py 상단 상수)

  증상                                   손댈 값
  ────────────────────────────────────   ─────────────────────────
  양쪽 다 (중립) 이라 목표 '-'           MODE_NEUTRAL_TOL 낮추기
  양쪽 다 (중립) 아님, r 0.6~1.0         MODE_NEUTRAL_TOL 높이기
  목표가 오르다 자꾸 0 으로 리셋         MODE_NEUTRAL_TOL_EXIT 높이기
  유지 시간 조정                         MODE_NEUTRAL_HOLD_SEC
  전환 직후 재전환 금지 시간             MODE_COOLDOWN_SEC

--symlink-install 이므로 상수만 바꿀 때는 재빌드 없이 노드만 재시작하면 된다.

--- 주의: 접속 주소와 인증서

GUI 의 IP 입력칸 값은 페이지를 연 주소와 반드시 같아야 한다. index.html 이
그 값으로 wss://IP:9090 (rosbridge), wss://IP:5001 (프레임 업로드) 를
만들기 때문이다. WSS 는 인증서 경고창을 띄울 수 없어서, 예외를 못 받은
origin 이면 조용히 실패한다 — 화면은 나오는데 로봇이 안 움직이는 증상이
정확히 이것이다.

IP 가 바뀌었으면 setup_https.sh 를 재실행해 인증서를 다시 만들고 전체를
재시작할 것. 이 스크립트는 이 기기의 모든 IPv4(LAN + Tailscale)를 SAN 에
넣는다.
