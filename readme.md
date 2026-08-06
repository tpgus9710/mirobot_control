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