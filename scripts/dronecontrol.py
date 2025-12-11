import socket
import requests
import io
import asyncio
import threading
from queue import Queue
from mavsdk import System
import sys
import os
import tempfile

# Add current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import recognization as inference_engine

# ================= 配置参数 =================
# 1. TCP 音频流配置 (接收端)
# 注意：这里假设有一个外部 TCP Server (如麦克风端) 在给本机发送数据
# 如果本机是 TCP Server，请将 socket 逻辑改为 bind/listen (参考原本的 drone 脚本)
TCP_REMOTE_IP = '192.168.2.62'
TCP_REMOTE_PORT = 9000
EOF_MARKER = b"<END_OF_AUDIO_FILE>" # 结束符
BUFFER_SIZE = 4096

# 2. Whisper Server 配置 (C++ HTTP 接口)
WHISPER_URL = "http://172.17.0.2:8080/inference"

# 3. 无人机配置 (MAVSDK)
DRONE_CONNECTION_STRING = "udp://0.0.0.0:8080" # 仿真器默认端口
# ===========================================

class DroneController:
    """管理无人机连接与控制的类"""
    def __init__(self, command_queue):
        self.drone = System()
        self.queue = command_queue
        self.running = True
    
    # 【新增辅助函数】经纬度偏移计算（核心：米转经纬度）
    def _calculate_geo_offset(self, lat, lon, north_m, east_m):
        """
        基于当前经纬度，计算北/东方向偏移指定米数后的经纬度
        :param lat: 当前纬度（度）
        :param lon: 当前经度（度）
        :param north_m: 北方向偏移（正=北，负=南，米）
        :param east_m: 东方向偏移（正=东，负=西，米）
        :return: 新纬度、新经度
        """
        # 纬度偏移：1米 = 1/(地球周长/360) 度
        lat_offset = (north_m / self.EARTH_RADIUS_M) * 180.0 / math.pi
        # 经度偏移：需乘以当前纬度的余弦（修正极地误差）
        lon_offset = (east_m / (self.EARTH_RADIUS_M * math.cos(math.pi * lat / 180.0))) * 180.0 / math.pi
        return lat + lat_offset, lon + lon_offset

    async def _get_position_once(self):
        """从 telemetry.position() 读取一次当前位置并返回（async generator -> 单次读取）"""
        async for pos in self.drone.telemetry.position():
            return pos
        return None

    # 【新增辅助函数】等待到达目标位置（带阈值校验）
    async def _wait_reach_target(self, target_lat, target_lon, target_alt, threshold=0.5):
        """
        等待无人机到达目标位置（水平+高度阈值默认0.5米）
        """
        while True:
            # 获取当前位置
            current_pos = await self._get_position_once()
            if current_pos is None:
                print("⚠ 获取当前位置失败，无法执行向北移动")
                return
            current_lat = current_pos.latitude_deg
            current_lon = current_pos.longitude_deg
            current_alt = current_pos.absolute_altitude_m

            # 计算水平距离（米）
            lat_diff = (current_lat - target_lat) * math.pi * self.EARTH_RADIUS_M / 180.0
            lon_diff = (current_lon - target_lon) * math.pi * self.EARTH_RADIUS_M * math.cos(math.pi * target_lat / 180.0) / 180.0
            horizontal_dist = math.sqrt(lat_diff**2 + lon_diff**2)
            alt_diff = abs(current_alt - target_alt)
            #vel_data = await self._get_current_velocity()
            #print(f"📊 实时速度 - 东向：{vel_data['east_m_s']:.2f}m/s | 地面合速度：{vel_data['ground_speed_m_s']:.2f}m/s")

            # 满足阈值则退出
            if horizontal_dist < threshold and alt_diff < threshold:
                print(f"✅ 到达目标位置（水平误差：{horizontal_dist:.2f}米，高度误差：{alt_diff:.2f}米）")
                break
            print(f"📌 正在接近目标：水平剩余 {horizontal_dist:.2f}米 | 高度剩余 {alt_diff:.2f}米")
            await asyncio.sleep(0.5)

    async def _get_velocity_ned_once(self):
        """从 telemetry.velocity_ned() 读取一次速度（NED）"""
        async for v in self.drone.telemetry.velocity_ned():
            return v
        return None

    async def _get_current_velocity(self):
        """获取当前速度（NED坐标系 + 地面速度）"""
        velocity = await self._get_velocity_ned_once()
        #ground_speed = await self.drone.telemetry.ground_speed()
        return {
            "north_m_s": velocity.north_m_s,    # 北方向速度（m/s）
            "east_m_s": velocity.east_m_s,      # 东方向速度（核心：往东飞的速度）
            "down_m_s": velocity.down_m_s,      # 下方向速度（m/s）
            #"ground_speed_m_s": ground_speed    # 地面合速度（m/s）
        }
    
    async def start(self):
        """启动无人机连接和指令监听循环"""
        print(f"🚁 正在连接无人机: {DRONE_CONNECTION_STRING}...")
        await self.drone.connect(system_address=DRONE_CONNECTION_STRING)

        # 等待连接成功
        print("Waiting for drone to connect...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("✅ 无人机已连接 (Drone Connected)!")
                break
        #配置无人机速度
        self.set_guided_speed(0.5)        
        # 启动指令处理循环
        await self.process_commands()

    async def process_commands(self):
        """不断检查队列并执行指令"""
        print("🎮 准备接收语音指令...")

        while self.running:
            # 检查队列是否有新指令 (非阻塞检查)
            if not self.queue.empty():
                text_command = self.queue.get()
                await self.execute_action(text_command)

            # 让出控制权，避免死循环卡死 Event Loop
            await asyncio.sleep(0.1)
    # 【新增】配置飞控默认水平速度（GUIDED模式）
    async def set_guided_speed(self, speed_m_s):
        """
        设置GUIDED模式下的水平巡航速度（单位：m/s）
        :param speed_m_s: 目标速度（如2.0表示2米/秒）
        """
        try:
            # PX4飞控参数：MPC_XY_CRUISE（水平巡航速度）
            # ArduPilot飞控参数：WP_SPEED（Waypoint速度）
            # 先读取当前参数
            current_speed = await self.drone.param.get_param_float("MPC_XY_CRUISE")
            print(f"🔧 当前GUIDED模式速度：{current_speed:.2f}m/s，即将修改为 {speed_m_s}m/s")
            
            # 设置新速度
            await self.drone.param.set_param_float("MPC_XY_CRUISE", speed_m_s)
            
            # 验证修改结果
            new_speed = await self.drone.param.get_param_float("MPC_XY_CRUISE")
            print(f"✅ 速度设置完成，当前值：{new_speed:.2f}m/s")
        except Exception as e:
            print(f"❌ 设置速度参数失败：{e}")

    async def execute_action(self, text):
        """解析文本并执行 MAVSDK 动作"""
        print(f"🤖 执行逻辑判断: [{text}]")

        # 统一转换为小写处理
        cmd = text.lower()

        try:
            # === 关键词映射 (同时支持中文和英文) ===

            # 1. 起飞 (Takeoff)
            if "上升" in cmd or "起飞" in cmd or "shangsheng" in cmd or "qifei" in cmd:
                print("🚀 指令确认: 起飞 (Arming & Taking off)")
                await self.drone.action.arm()
                import time
                time.sleep(1)
                await self.drone.action.takeoff()

            # 2. 降落 (Land)
            elif "降落" in cmd or "下降" in cmd or "jiangluo" in cmd or "xiajiang" in cmd or "xiajiao" in cmd or "xiajing" in cmd or "jiangwo" in cmd or "jingwu" in cmd or "jingwo" in cmd:
                print("🛬 指令确认: 降落 (Landing)")
                await self.drone.action.land()

            # 3. 返航 (Return to Launch)
            elif "返航" in cmd or "回家" in cmd or "return" in cmd or "fanhang" in cmd or "huijia" in cmd:
                print("🏠 指令确认: 返航 (RTL)")
                await self.drone.action.return_to_launch()

            # 4. 解锁 (Arm) - 仅解锁不起飞
            elif "解锁" in cmd or "arm" in cmd or "jiesuo" in cmd:
                print("🔓 指令确认: 解锁 (Arming)")
                await self.drone.action.arm()

            # 5. 上锁 (Disarm) - 危险！仅在地面使用
            elif "上锁" in cmd or "锁定" in cmd or "disarm" in cmd or "shangsuo" in cmd or "suoding" in cmd:
                print("🔒 指令确认: 上锁 (Disarming)")
                await self.drone.action.disarm()
# 6. 向北飞2米 (North 2m)
            elif "北" in cmd or "bei" in cmd or "xiangbei" in cmd:
                print("🡹 指令确认: 向北飞2米 (Fly North 2m)")
                # 获取当前位置和高度
                current_pos = await self._get_position_once()
                if current_pos is None:
                    print("⚠ 获取当前位置失败，无法执行向北移动")
                    return
                current_lat = current_pos.latitude_deg
                current_lon = current_pos.longitude_deg
                current_alt = current_pos.absolute_altitude_m
                # 计算目标经纬度（北+2米，东0米）
                target_lat, target_lon = self._calculate_geo_offset(current_lat, current_lon, 2.0, 0.0)
                # 发送移动指令（保持当前高度，偏航角不变）
                await self.drone.action.goto_location(target_lat, target_lon, current_alt, float('nan'))
                # 等待到达目标位置
                await self._wait_reach_target(target_lat, target_lon, current_alt)

            # 7. 向南飞2米 (South 2m)
            elif "南" in cmd or "nan" in cmd or "xiangnan" in cmd:
                print("🡻 指令确认: 向南飞2米 (Fly South 2m)")
                current_pos = await self._get_position_once()
                if current_pos is None:
                    print("⚠ 获取当前位置失败，无法执行向北移动")
                    return
                current_lat = current_pos.latitude_deg
                current_lon = current_pos.longitude_deg
                current_alt = current_pos.absolute_altitude_m
                # 北-2米 = 南+2米
                target_lat, target_lon = self._calculate_geo_offset(current_lat, current_lon, -2.0, 0.0)
                await self.drone.action.goto_location(target_lat, target_lon, current_alt, float('nan'))
                await self._wait_reach_target(target_lat, target_lon, current_alt)

            # 8. 向东飞2米 (East 2m)
            elif "东" in cmd or "dong" in cmd or "xiangdong" in cmd:
                print("🡺 指令确认: 向东飞2米 (Fly East 2m)")
                current_pos = await self._get_position_once()
                if current_pos is None:
                    print("⚠ 获取当前位置失败，无法执行向北移动")
                    return
                current_lat = current_pos.latitude_deg
                current_lon = current_pos.longitude_deg
                current_alt = current_pos.absolute_altitude_m
                # 东+2米
                target_lat, target_lon = self._calculate_geo_offset(current_lat, current_lon, 0.0, 2.0)
                await self.drone.action.goto_location(target_lat, target_lon, current_alt, float('nan'))
                await self._wait_reach_target(target_lat, target_lon, current_alt)

            # 9. 向西飞2米 (West 2m)
            elif "西" in cmd or "xi" in cmd or "xiangxi" in cmd:
                print("🡸 指令确认: 向西飞2米 (Fly West 2m)")
                current_pos = await self.drone.telemetry.position()
                current_pos = await self._get_position_once()
                if current_pos is None:
                    print("⚠ 获取当前位置失败，无法执行向北移动")
                    return
                current_lat = current_pos.latitude_deg
                current_lon = current_pos.longitude_deg
                current_alt = current_pos.absolute_altitude_m
                # 东-2米 = 西+2米
                target_lat, target_lon = self._calculate_geo_offset(current_lat, current_lon, 0.0, -2.0)
                await self.drone.action.goto_location(target_lat, target_lon, current_alt, float('nan'))
                await self._wait_reach_target(target_lat, target_lon, current_alt)

            else:
                print(f"⚠️ 未知指令: {text}")

        except Exception as e:
            print(f"❌ 执行指令出错: {e}")


def transcribe_audio(audio_data):
    """调用 C++ Whisper Server 进行识别"""
    try:
        # 将内存中的 bytes 包装成虚拟文件
        audio_file = io.BytesIO(audio_data)

        files = {
            'file': ('speech.wav', audio_file, 'audio/wav')
        }
        # 显式指定中文，且设置 temperature=0 提高指令准确度
        data = {
            'temperature': '0.0',
            'response_format': 'json',
            'language': 'zh'
        }

        resp = requests.post(WHISPER_URL, files=files, data=data, timeout=1000)

        if resp.status_code == 200:
            result = resp.json()
            # Whisper.cpp 有时返回 {'text': ...} 有时是 segments，通常 'text' 字段最直接
            return result.get('text', '').strip()
        else:
            print(f"❌ Whisper Server Error: {resp.status_code}")
            return None
    except Exception as e:
        print(f"❌ Whisper Request Failed: {e}")
        return None

def tcp_audio_listener(command_queue):
    """
    运行在独立线程中的 TCP 客户端
    负责接收音频 -> 调用 HTTP 识别 -> 放入队列
    """
    # Initialize Inference Engine
    model_path = os.path.join(current_dir, "../model/voice.om")
    acl_resource, model = inference_engine.init_inference(model_path)
    if not acl_resource or not model:
        print("❌ ACL Init Failed in Audio Thread")
        return

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 重连机制
    while True:
        try:
            print(f"🔌 (Audio Thread) 正在连接音频源 {TCP_REMOTE_IP}:{TCP_REMOTE_PORT}...")
            s.connect((TCP_REMOTE_IP, TCP_REMOTE_PORT))
            print("✅ 音频流已连接")

            received_buffer = b""

            while True:
                chunk = s.recv(BUFFER_SIZE)
                if not chunk:
                    raise ConnectionResetError("服务端关闭连接")

                received_buffer += chunk

                # 检查结束符
                if EOF_MARKER in received_buffer:
                    parts = received_buffer.split(EOF_MARKER)

                    # 取出完整的一段音频
                    audio_data = parts[0]

                    if len(audio_data) > 0:
                        print(f"🎤 收到音频 ({len(audio_data)} bytes)，正在识别...")

                        # Save to temp file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(audio_data)
                            tmp_path = tmp.name

                        # 1. 识别 (阻塞调用)
                        # text = transcribe_audio(audio_data)
                        txt, pinyin = inference_engine.process_single_audio(model, tmp_path)
                        
                        # Clean up
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                        # 2. 如果有结果，放入队列传给 Drone 协程
                        if txt:
                            print(f"🗣️  识别结果: Pinyin=[{pinyin}], Text=[{txt}]")
                            command = "".join([i[:-1] for i in pinyin]) + ", " + txt
                            command_queue.put(txt)

                    # 处理粘包，保留剩余部分
                    received_buffer = b"".join(parts[1:])

        except (ConnectionRefusedError, ConnectionResetError) as e:
            print(f"⚠️ 连接断开或失败: {e}，3秒后重连...")
            # 重置 socket
            s.close()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            import time
            time.sleep(3)
        except Exception as e:
            print(f"❌ 音频线程发生严重错误: {e}")
            break
    
    inference_engine.release_inference(acl_resource, model)

async def main():
    # 1. 创建线程安全的队列，用于跨线程通信
    cmd_queue = Queue()

    # 2. 启动音频处理线程 (TCP + Requests 是阻塞的，必须在 Thread 中)
    audio_thread = threading.Thread(
        target=tcp_audio_listener,
        args=(cmd_queue,),
        daemon=True  # 设置为守护线程，主程序退出时它也会退出
    )
    audio_thread.start()

    # 3. 启动无人机控制 (Asyncio 主循环)
    controller = DroneController(cmd_queue)
    await controller.start()

if __name__ == "__main__":
    try:
        # 启动 Async 事件循环
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已停止")