"""
DJI Mavic 2 Pro controller
- WASD + QE 手动控制
- ↑ / ↓ 控制高度
- 简单颜色检测当作“目标识别”（后面可替换为训练模型）
- RangeFinder 记录环境深度（scan_log.csv）
- 每当识别到目标时，用相机 + RangeFinder + GPS + yaw 估算目标坐标
- 去重后输出 targets.csv（目标坐标表）
"""

from controller import Robot, Keyboard
import csv
import cv2
import numpy as np
import math


def clamp(value, low, high):
    return low if value < low else high if value > high else value


robot = Robot()
timestep = int(robot.getBasicTimeStep())

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# ---------- Sensors ----------
gps = robot.getDevice("gps")
gps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

camera = robot.getDevice("camera")
camera.enable(timestep)
cam_width = camera.getWidth()
cam_height = camera.getHeight()
cam_fov = camera.getFov()  # 水平视场角

front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

# camera gimbal (may or may not exist)
try:
    camera_roll_motor = robot.getDevice("camera roll")
    camera_pitch_motor = robot.getDevice("camera pitch")
    roll_min = camera_roll_motor.getMinPosition()
    roll_max = camera_roll_motor.getMaxPosition()
    pitch_min = camera_pitch_motor.getMinPosition()
    pitch_max = camera_pitch_motor.getMaxPosition()
except Exception:
    camera_roll_motor = None
    camera_pitch_motor = None
    roll_min = roll_max = pitch_min = pitch_max = 0.0

# ---------- RangeFinder (optional) ----------
range_finder = None
rf_width = 0
rf_fov = 0.0
try:
    dev = robot.getDevice("range-finder")
    if dev is not None:
        range_finder = dev
except Exception:
    range_finder = None

if range_finder is not None:
    range_finder.enable(timestep)
    rf_width = range_finder.getWidth()
    rf_fov = range_finder.getFov()
    print(f"[INFO] RangeFinder OK width={rf_width} fov={rf_fov:.3f}")
else:
    print("[WARN] RangeFinder not found")

# ---------- Motors ----------
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

for m in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

# ---------- CSV for depth ----------
writer = None
log_file = None
if range_finder is not None:
    log_file = open("scan_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["t", "x", "y", "z", "yaw", "beam_index", "range"])
    print("[INFO] Logging depth -> scan_log.csv")

# ---------- Control params ----------
k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 1.5   # 减小垂直 P，减少上下抖动
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 1.0
auto_track = False  # P toggle

# 暂时用红色作为“目标”的示例；之后可替换成模型输出
LOWER_HSV1 = np.array([0, 120, 70])
UPPER_HSV1 = np.array([10, 255, 255])
LOWER_HSV2 = np.array([170, 120, 70])
UPPER_HSV2 = np.array([180, 255, 255])
MIN_CONTOUR_AREA = 200

# ---------- 目标坐标记录 ----------
targets = []          # 每个元素: {"x":..,"y":..,"z":..,"class":"target"}
TARGET_MIN_SEP = 0.7  # 距离小于这个就认为是同一个目标（去重）

def add_target_if_new(x_obj, y_obj, z_obj, label="target"):
    for t in targets:
        dx = t["x"] - x_obj
        dy = t["y"] - y_obj
        dz = t["z"] - z_obj
        if math.sqrt(dx*dx + dy*dy + dz*dz) < TARGET_MIN_SEP:
            return
    targets.append({"x": x_obj, "y": y_obj, "z": z_obj, "class": label})
    print(f"[TARGET] New {label} at ({x_obj:.2f}, {y_obj:.2f}, {z_obj:.2f})")

# small warm-up
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

try:
    while robot.step(timestep) != -1:
        t = robot.getTime()

        x, y, z = gps.getValues()
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_vel, pitch_vel, yaw_vel = gyro.getValues()

        led_state = int(t) % 2
        front_left_led.set(led_state)
        front_right_led.set(1 - led_state)

        # gimbal stabilization with clamped positions
        if camera_roll_motor is not None:
            cmd_r = clamp(-0.115 * roll_vel, roll_min, roll_max)
            camera_roll_motor.setPosition(cmd_r)
        if camera_pitch_motor is not None:
            cmd_p = clamp(-0.1 * pitch_vel, pitch_min, pitch_max)
            camera_pitch_motor.setPosition(cmd_p)

        # ---------- camera detection ----------
        image = camera.getImage()
        target_found = False
        target_cx = None
        target_cy = None
        target_area = 0.0

        if image is not None:
            img = np.frombuffer(image, np.uint8).reshape((cam_height, cam_width, 4))
            frame = img[:, :, :3]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, LOWER_HSV1, UPPER_HSV1)
            mask2 = cv2.inRange(hsv, LOWER_HSV2, UPPER_HSV2)
            mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area <= MIN_CONTOUR_AREA:
                    continue
                x_box, y_box, w_box, h_box = cv2.boundingRect(c)
                cx_pix = x_box + w_box / 2
                cy_pix = y_box + h_box / 2
                if area > target_area:
                    target_area = area
                    target_cx = cx_pix
                    target_cy = cy_pix
                    target_found = True

        # ---------- keyboard ----------
        roll_disturb = 0.0
        pitch_disturb = 0.0
        yaw_disturb = 0.0

        key = keyboard.getKey()
        while key != -1:
            if key in (ord("P"), ord("p")):
                auto_track = not auto_track
                print("Auto tracking >", auto_track)

            elif key in (ord("W"), ord("w")):
                pitch_disturb = -2.0
            elif key in (ord("S"), ord("s")):
                pitch_disturb = 2.0
            elif key in (ord("D"), ord("d")):
                yaw_disturb = -1.3
            elif key in (ord("A"), ord("a")):
                yaw_disturb = 1.3
            elif key in (ord("E"), ord("e")):
                roll_disturb = -1.0
            elif key in (ord("Q"), ord("q")):
                roll_disturb = 1.0

            # ↑ / ↓ 控制高度（步长 0.02）
            elif key == Keyboard.UP:
                target_altitude += 0.02
                print("Altitude >", target_altitude)
            elif key == Keyboard.DOWN:
                target_altitude -= 0.02
                print("Altitude >", target_altitude)

            key = keyboard.getKey()

        # ---------- auto tracking (yaw only, 简单版本) ----------
        if auto_track and target_found and target_cx is not None:
            offset_x = (target_cx - cam_width / 2) / (cam_width / 2)
            yaw_disturb += -1.0 * offset_x

        # ---------- PID & motors ----------
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_vel + roll_disturb
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + pitch_disturb
        yaw_input = yaw_disturb

        dz = clamp(target_altitude - z + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * dz

        fl = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        fr = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rl = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rr = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        front_left_motor.setVelocity(fl)
        front_right_motor.setVelocity(-fr)
        rear_left_motor.setVelocity(-rl)
        rear_right_motor.setVelocity(rr)

        # ---------- depth logging ----------
        d_front = None
        if range_finder is not None:
            rf_image = range_finder.getRangeImage()
            if writer is not None:
                for k_idx in range(rf_width):
                    r_val = rf_image[k_idx]
                    writer.writerow([t, x, y, z, yaw, k_idx, r_val])
            # 正前方距离（用于估算目标坐标）
            if rf_width > 0:
                d_front = rf_image[rf_width // 2]

        # ---------- 估计目标坐标并记录（用当前帧） ----------
        if target_found and target_cx is not None and d_front is not None and d_front < 1e6:
            # 像素 → 相机横向角度
            # 假设图像宽度 cam_width，对应水平 FOV = cam_fov
            norm_x = (target_cx - cam_width / 2) / (cam_width / 2)  # -1 ~ 1
            angle_offset = norm_x * (cam_fov / 2.0)                 # 相对于机头的偏航角
            bearing = yaw + angle_offset                            # 世界坐标系里的方位角

            # 世界坐标中目标位置（假设地面高度差不大，z 近似相同）
            x_obj = x + d_front * math.cos(bearing)
            y_obj = y + d_front * math.sin(bearing)
            z_obj = z  # 简化

            add_target_if_new(x_obj, y_obj, z_obj, label="target")

except Exception as e:
    print("[ERROR]", e)
finally:
    # 输出目标表
    if targets:
        print("\n=== TARGET LIST ===")
        print("idx\tclass\tx\ty\tz")
        for i, t_item in enumerate(targets):
            print(f"{i}\t{t_item['class']}\t{t_item['x']:.2f}\t{t_item['y']:.2f}\t{t_item['z']:.2f}")
        # 写 CSV
        with open("targets.csv", "w", newline="") as ft:
            tw = csv.writer(ft)
            tw.writerow(["id", "class", "x", "y", "z"])
            for i, t_item in enumerate(targets):
                tw.writerow([i, t_item["class"], t_item["x"], t_item["y"], t_item["z"]])
        print("[INFO] Saved targets.csv")
    else:
        print("[INFO] No targets detected.")

    if log_file is not None:
        log_file.close()
    cv2.destroyAllWindows()
    print("[DONE]")




======



"""
Fully Automatic Mavic 2 Pro Controller (with arena boundary)
- 全自动起飞 / 搜索 / 避障 / 目标识别 / 记录坐标
- 目标：纸箱 (box, 棕色) + 红色灭火器 (fire_extinguisher)
- 避障：RangeFinder 前方距离
- 边界约束：根据 GPS 保持在矩形场地内 (floorSize 20x20)
- 输出: targets.csv (id, label, x, y, z)
"""

from controller import Robot
import math
import csv
import cv2
import numpy as np


def clamp(v, low, high):
    return max(low, min(high, v))


def wrap_pi(angle):
    # 把角度归一化到 [-pi, pi]
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ------------ 设备 ------------
gps = robot.getDevice("gps")
gps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

camera = robot.getDevice("camera")
camera.enable(timestep)
CAM_W = camera.getWidth()
CAM_H = camera.getHeight()
CAM_FOV = camera.getFov()  # 水平视场角

# RangeFinder 用于避障和距离
range_finder = None
RF_WIDTH = 0
try:
    rf = robot.getDevice("range-finder")
    if rf:
        range_finder = rf
        range_finder.enable(timestep)
        RF_WIDTH = range_finder.getWidth()
except Exception:
    range_finder = None

# LED（可选）
try:
    led_left = robot.getDevice("front left led")
    led_right = robot.getDevice("front right led")
except Exception:
    led_left = led_right = None

# 电机
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

for m in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

# ------------ 控制参数 ------------
k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 0.75   # 小一点，垂直更稳
k_roll_p = 50.0
k_pitch_p = 30.0

TARGET_ALTITUDE = 1.2       # 自动目标高度
FORWARD_PITCH = -1.0        # 正常前进速度（负值表示向前）
AVOID_DIST = 1.5            # 避障触发距离
AVOID_TURN_TIME = 1.0       # 避障时持续转向秒数
TARGET_STOP_DIST = 2.0      # 接近目标时停在距离

avoid_timer = 0.0

# ---- 场地边界（RectangleArena floorSize 20 20） ----
# 以 (0,0) 为中心，x,y ∈ [-10,10]，我们留一点安全边界 1.0m
ARENA_HALF_X = 10.0
ARENA_HALF_Y = 10.0
BORDER_MARGIN = 1.0  # 距离边缘小于 1m 就往内侧转
BORDER_YAW_GAIN = 1.2

# ------------ 视觉阈值（需要根据场景微调）------------
# 纸箱：棕色/黄色调（可视情况调 H / S / V）
BOX_LOWER = np.array([10, 60, 60])
BOX_UPPER = np.array([25, 255, 255])
BOX_MIN_AREA = 150
BOX_RATIO_MIN = 0.7   # w/h 接近方形
BOX_RATIO_MAX = 1.5

# 灭火器：红色，高瘦
FIRE_LOWER1 = np.array([0, 120, 70])
FIRE_UPPER1 = np.array([10, 255, 255])
FIRE_LOWER2 = np.array([170, 120, 70])
FIRE_UPPER2 = np.array([180, 255, 255])
FIRE_MIN_AREA = 80
FIRE_MIN_RATIO = 1.5   # h/w 较大

# ------------ 目标记录 ------------
targets = []   # 每个: {"label":..., "x":..., "y":..., "z":...}
MIN_SEP = 0.7  # 重复检测去重半径


def record_target(label, x, y, z):
    for t in targets:
        if math.dist((x, y, z), (t["x"], t["y"], t["z"])) < MIN_SEP:
            return
    targets.append({"label": label, "x": x, "y": y, "z": z})
    print(f"[TARGET] {label} at ({x:.2f}, {y:.2f}, {z:.2f})")


# ------------ 预热 ------------
while robot.step(timestep) != -1 and robot.getTime() < 1.0:
    pass

print("[INFO] Auto flight started.")

try:
    while robot.step(timestep) != -1:
        dt = timestep / 1000.0
        t = robot.getTime()

        x, y, z = gps.getValues()
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_v, pitch_v, yaw_v = gyro.getValues()

        if led_left and led_right:
            s = int(t) % 2
            led_left.set(s)
            led_right.set(1 - s)

        # ---------- RangeFinder ----------
        d_front = None
        if range_finder is not None:
            rf_img = range_finder.getRangeImage()
            if RF_WIDTH > 0:
                d_front = rf_img[RF_WIDTH // 2]

        # ---------- Camera ----------
        image = camera.getImage()
        detected_label = None
        detected_cx = None

        if image is not None:
            img = np.frombuffer(image, np.uint8).reshape((CAM_H, CAM_W, 4))
            frame = img[:, :, :3]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # --- 先找灭火器（红色高瘦） ---
            m1 = cv2.inRange(hsv, FIRE_LOWER1, FIRE_UPPER1)
            m2 = cv2.inRange(hsv, FIRE_LOWER2, FIRE_UPPER2)
            mask_fire = cv2.bitwise_or(m1, m2)
            cnts, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_score = 0
            for c in cnts:
                area = cv2.contourArea(c)
                if area < FIRE_MIN_AREA:
                    continue
                x0, y0, w0, h0 = cv2.boundingRect(c)
                ratio = h0 / max(w0, 1)
                if ratio > FIRE_MIN_RATIO and area * ratio > best_score:
                    best_score = area * ratio
                    detected_label = "fire_extinguisher"
                    detected_cx = x0 + w0 / 2

            # --- 再找纸箱（棕色方块） ---
            if detected_label is None:
                mask_box = cv2.inRange(hsv, BOX_LOWER, BOX_UPPER)
                cnts2, _ = cv2.findContours(mask_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_area = 0
                for c in cnts2:
                    area = cv2.contourArea(c)
                    if area < BOX_MIN_AREA:
                        continue
                    x0, y0, w0, h0 = cv2.boundingRect(c)
                    ratio = w0 / max(h0, 1)
                    if BOX_RATIO_MIN <= ratio <= BOX_RATIO_MAX and area > best_area:
                        best_area = area
                        detected_label = "box"
                        detected_cx = x0 + w0 / 2

        # ---------- 自动控制 ----------
        # 默认前进
        pitch_d = FORWARD_PITCH
        roll_d = 0.0
        yaw_d = 0.0
        target_alt = TARGET_ALTITUDE

        # 避障：前方太近 -> 停止前进 + 左转一段时间
        if d_front is not None and d_front < AVOID_DIST:
            avoid_timer = AVOID_TURN_TIME

        if avoid_timer > 0.0:
            avoid_timer -= dt
            pitch_d = 0.0
            yaw_d = 1.0   # 左转避障
        else:
            # 边界约束：靠近边缘就朝场地中心转向
            near_border = (
                abs(x) > (ARENA_HALF_X - BORDER_MARGIN) or
                abs(y) > (ARENA_HALF_Y - BORDER_MARGIN)
            )
            if near_border:
                # 场地中心在 (0,0)，朝向中心的方位角：
                angle_to_center = math.atan2(-y, -x)
                delta_yaw = wrap_pi(angle_to_center - yaw)
                yaw_d += clamp(BORDER_YAW_GAIN * delta_yaw, -1.5, 1.5)
                # 保持前进，让它往中心飞回
            else:
                # 正常追目标逻辑
                if detected_label is not None and detected_cx is not None and d_front is not None:
                    norm_x = (detected_cx - CAM_W / 2) / (CAM_W / 2)  # -1~1
                    yaw_d += -1.0 * norm_x  # 对准目标

                    # 目标居中且有距离信息 -> 记录坐标
                    if abs(norm_x) < 0.1 and d_front < 30.0:
                        angle = norm_x * (CAM_FOV / 2.0)
                        bearing = yaw + angle
                        x_obj = x + d_front * math.cos(bearing)
                        y_obj = y + d_front * math.sin(bearing)
                        z_obj = z
                        record_target(detected_label, x_obj, y_obj, z_obj)

                        # 距离足够近就停止前进，悬停观察
                        if d_front < TARGET_STOP_DIST:
                            pitch_d = 0.0

        # ---------- PID & 电机 ----------
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_v + roll_d
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_v + pitch_d
        yaw_input = yaw_d

        dz = clamp(target_alt - z + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * dz

        fl = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        fr = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rl = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rr = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        front_left_motor.setVelocity(fl)
        front_right_motor.setVelocity(-fr)
        rear_left_motor.setVelocity(-rl)
        rear_right_motor.setVelocity(rr)

except Exception as e:
    print("[ERROR]", e)

finally:
    # 输出目标列表
    if targets:
        print("\n=== TARGETS DETECTED ===")
        print("id\tlabel\tx\ty\tz")
        for i, t in enumerate(targets):
            print(f"{i}\t{t['label']}\t{t['x']:.2f}\t{t['y']:.2f}\t{t['z']:.2f}")
        # 写 CSV
        with open("targets.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label", "x", "y", "z"])
            for i, t in enumerate(targets):
                w.writerow([i, t["label"], t["x"], t["y"], t["z"]])
        print("[INFO] Saved targets.csv")
    else:
        print("[INFO] No targets detected.")
    print("[DONE]")
