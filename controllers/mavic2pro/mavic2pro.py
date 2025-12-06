
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

# ---------- Control params（完全按官方 C 版） ----------
k_vertical_thrust = 68.5     # 起飞推力
k_vertical_offset = 0.6      # 垂直偏移
k_vertical_p = 3.0           # 垂直 P（立方放大）
k_roll_p = 50.0              # 横滚 P
k_pitch_p = 30.0             # 俯仰 P

target_altitude = 1.0        # 目标高度
auto_track = False           # P 键开关自动跟踪

# 暂时用红色作为“目标”的示例；之后可替换为模型输出
LOWER_HSV1 = np.array([0, 120, 70])
UPPER_HSV1 = np.array([10, 255, 255])
LOWER_HSV2 = np.array([170, 120, 70])
UPPER_HSV2 = np.array([180, 255, 255])
MIN_CONTOUR_AREA = 200


targets = []          
TARGET_MIN_SEP = 0.7 


def add_target_if_new(x_obj, y_obj, z_obj, label="target"):
    for t in targets:
        dx = t["x"] - x_obj
        dy = t["y"] - y_obj
        dz = t["z"] - z_obj
        if math.sqrt(dx * dx + dy * dy + dz * dz) < TARGET_MIN_SEP:
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

        # gimbal stabilization with clamped positions（同 C 版）
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

        # ---------- keyboard（WASD+QE+↑↓） ----------
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

           
            elif key == Keyboard.UP:
                target_altitude += 0.05
                print("Altitude >", target_altitude)
            elif key == Keyboard.DOWN:
                target_altitude -= 0.05
                print("Altitude >", target_altitude)

            key = keyboard.getKey()

       
        if auto_track and target_found and target_cx is not None:
            offset_x = (target_cx - cam_width / 2) / (cam_width / 2)
            yaw_disturb += -1.0 * offset_x

        # roll / pitch / yaw 
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_vel + roll_disturb
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + pitch_disturb
        yaw_input = yaw_disturb

        
        clamped_diff_alt = clamp(target_altitude - z + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * (clamped_diff_alt ** 3)
        
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
            if rf_width > 0:
                d_front = rf_image[rf_width // 2]


        if target_found and target_cx is not None and d_front is not None and d_front < 1e6:
            norm_x = (target_cx - cam_width / 2) / (cam_width / 2)  # -1 ~ 1
            angle_offset = norm_x * (cam_fov / 2.0)                
            bearing = yaw + angle_offset                          

            x_obj = x + d_front * math.cos(bearing)
            y_obj = y + d_front * math.sin(bearing)
            z_obj = z

            add_target_if_new(x_obj, y_obj, z_obj, label="target")

except Exception as e:
    print("[ERROR]", e)
finally:
    
    
    if targets:
        print("\n=== TARGET LIST ===")
        print("idx\tclass\tx\ty\tz")
        for i, t_item in enumerate(targets):
            print(f"{i}\t{t_item['class']}\t{t_item['x']:.2f}\t{t_item['y']:.2f}\t{t_item['z']:.2f}")
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
