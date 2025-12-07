"""
DJI Mavic 2 Pro Webots controller

Main features:
- Manual control using keyboard: WASD + QE, UP/DOWN for altitude.
- Basic color-based target detection using the front camera (can be replaced by a trained model).
- RangeFinder depth logging to scan_log.csv for later 3D reconstruction or analysis.
- When a target is detected, estimate its world coordinates using Camera + RangeFinder + GPS + yaw.
- Deduplicate and save targets into targets.csv.
"""

from controller import Robot, Keyboard
import csv
import cv2
import numpy as np
import math


def clamp(value, low, high):
    """Clamp a value to the interval [low, high]."""
    return low if value < low else high if value > high else value


class PIDController:
    """Generic PID controller used for roll, pitch and vertical axes."""
    def __init__(self, P=0.0, I=0.0, D=0.0):
        self.P = P
        self.I = I
        self.D = D
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        output = self.P * error + self.I * self.integral + self.D * derivative
        self.previous_error = error
        return output


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
cam_fov = camera.getFov()  # horizontal field of view [rad]

front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

# Camera gimbal (may or may not exist in the model)
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

# ---------- RangeFinder (optional depth sensor) ----------
range_finder = None
rf_width = 0
rf_fov = 0.0  # horizontal FOV of the RangeFinder
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
    # Infinite position means velocity control mode
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

# ---------- CSV for depth logging ----------
writer = None
log_file = None
if range_finder is not None:
    log_file = open("scan_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    # Each row: time, drone pose, yaw, beam index in RangeFinder image, range value
    writer.writerow(["t", "x", "y", "z", "yaw", "beam_index", "range"])
    print("[INFO] Logging depth -> scan_log.csv")

# ==========================================================
#                 ATTITUDE / ALTITUDE GAINS (PID)
# ==========================================================

# Base vertical thrust (approximately hover thrust)
k_vertical_thrust = 68.5

# Vertical offset to compensate for model bias
k_vertical_offset = 0.6

# PID controllers (values taken from tuned PID example)
PID = {
    # Roll PID: P only, D via gyro (roll_vel) added later
    "roll": PIDController(P=50.0, I=0.0, D=0.0),
    # Pitch PID: P + small D, plus gyro (pitch_vel) added later
    "pitch": PIDController(P=50.0, I=0.0, D=0.25),
    # Yaw PID can be added if needed; currently yaw is only from disturbance
    "yaw": PIDController(P=5.0, I=0.0, D=0.0),
    # Vertical PID: runs on cubic altitude error
    "vertical": PIDController(P=0.64, I=0.0, D=0.0),
}

# Target altitude in meters (can be changed at runtime using keyboard)
target_altitude = 1.0

# Toggle for simple automatic yaw tracking towards detected target
auto_track = False

# ---------- Simple red-color based target detection thresholds ----------
# These HSV ranges are used only as a placeholder example.
LOWER_HSV1 = np.array([0, 120, 70])
UPPER_HSV1 = np.array([10, 255, 255])
LOWER_HSV2 = np.array([170, 120, 70])
UPPER_HSV2 = np.array([180, 255, 255])
MIN_CONTOUR_AREA = 200  # Ignore very small blobs

# ---------- Target coordinate logging ----------
# Each element: {"x": float, "y": float, "z": float, "class": str}
targets = []
TARGET_MIN_SEP = 0.7  # minimum separation to consider a new target (meters)


def add_target_if_new(x_obj, y_obj, z_obj, label="target"):
    """Store a new detected target if sufficiently far from existing ones."""
    for t in targets:
        dx = t["x"] - x_obj
        dy = t["y"] - y_obj
        dz = t["z"] - z_obj
        if math.sqrt(dx * dx + dy * dy + dz * dz) < TARGET_MIN_SEP:
            # Too close to an existing target; treat as duplicate
            return
    targets.append({"x": x_obj, "y": y_obj, "z": z_obj, "class": label})
    print(f"[TARGET] New {label} at ({x_obj:.2f}, {y_obj:.2f}, {z_obj:.2f})")


# Small warm-up to let sensors stabilize
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

last_time = None  # for PID dt

try:
    while robot.step(timestep) != -1:
        current_time = robot.getTime()
        if last_time is None:
            dt = 0.0
        else:
            dt = current_time - last_time
        last_time = current_time

        # --- Read drone pose and sensor values ---
        x, y, z = gps.getValues()
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_vel, pitch_vel, yaw_vel = gyro.getValues()

        # LED blinking for basic feedback (alternating)
        led_state = int(current_time) % 2
        front_left_led.set(led_state)
        front_right_led.set(1 - led_state)

        # Camera gimbal stabilization using gyro feedback (optional)
        if camera_roll_motor is not None:
            cmd_r = clamp(-0.115 * roll_vel, roll_min, roll_max)
            camera_roll_motor.setPosition(cmd_r)
        if camera_pitch_motor is not None:
            cmd_p = clamp(-0.1 * pitch_vel, pitch_min, pitch_max)
            camera_pitch_motor.setPosition(cmd_p)

        # ---------- Camera-based detection ----------
        image = camera.getImage()
        target_found = False
        target_cx = None
        target_cy = None
        target_area = 0.0

        if image is not None:
            # Convert Webots BGRA image to OpenCV BGR
            img = np.frombuffer(image, np.uint8).reshape((cam_height, cam_width, 4))
            frame = img[:, :, :3]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Threshold for red hue (two ranges in HSV)
            mask1 = cv2.inRange(hsv, LOWER_HSV1, UPPER_HSV1)
            mask2 = cv2.inRange(hsv, LOWER_HSV2, UPPER_HSV2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Find connected components
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area <= MIN_CONTOUR_AREA:
                    continue
                x_box, y_box, w_box, h_box = cv2.boundingRect(c)
                cx_pix = x_box + w_box / 2
                cy_pix = y_box + h_box / 2
                # Keep the largest contour as the main target
                if area > target_area:
                    target_area = area
                    target_cx = cx_pix
                    target_cy = cy_pix
                    target_found = True

        # ---------- Keyboard input (manual control: WASD+QE + arrow keys) ----------
        roll_disturb = 0.0   # extra roll control from keyboard
        pitch_disturb = 0.0  # extra pitch control from keyboard
        yaw_disturb = 0.0    # extra yaw control from keyboard

        key = keyboard.getKey()
        while key != -1:
            if key in (ord("P"), ord("p")):
                # Toggle auto yaw tracking
                auto_track = not auto_track
                print("Auto tracking >", auto_track)

            elif key in (ord("W"), ord("w")):
                pitch_disturb = -2.0   # move forward
            elif key in (ord("S"), ord("s")):
                pitch_disturb = 2.0    # move backward
            elif key in (ord("D"), ord("d")):
                yaw_disturb = -1.3     # yaw right
            elif key in (ord("A"), ord("a")):
                yaw_disturb = 1.3      # yaw left
            elif key in (ord("E"), ord("e")):
                roll_disturb = -1.0    # strafe right
            elif key in (ord("Q"), ord("q")):
                roll_disturb = 1.0     # strafe left

            # Arrow keys UP/DOWN: change target altitude
            elif key == Keyboard.UP:
                target_altitude += 0.05
                print("Altitude target >", target_altitude)
            elif key == Keyboard.DOWN:
                target_altitude -= 0.05
                print("Altitude target >", target_altitude)

            key = keyboard.getKey()

        # ---------- Simple auto yaw tracking (optional) ----------
        if auto_track and target_found and target_cx is not None:
            # Normalized horizontal offset of target in image [-1, 1]
            offset_x = (target_cx - cam_width / 2) / (cam_width / 2)
            # Add yaw disturbance to turn the drone toward the target
            yaw_disturb += -1.0 * offset_x

        # ======================================================
        #        ATTITUDE / ALTITUDE CONTROL (PID-BASED)
        # ======================================================

        # Roll/Pitch/Altitude errors (clamped to avoid extreme values)
        roll_error = clamp(roll, -1.0, 1.0)
        pitch_error = clamp(pitch, -1.0, 1.0)
        altitude_error = clamp(target_altitude - z + k_vertical_offset, -1.0, 1.0)

        # Roll control input = PID(roll) + gyro (extra D) + keyboard disturbance
        roll_input = PID["roll"].compute(roll_error, dt) + roll_vel + roll_disturb

        # Pitch control input = PID(pitch) + gyro (extra D) + keyboard disturbance
        pitch_input = PID["pitch"].compute(pitch_error, dt) + pitch_vel + pitch_disturb

        # Yaw control is only based on disturbances (manual + auto tracking)
        # (PID["yaw"] is defined but not used here, can be enabled later if needed)
        yaw_input = yaw_disturb

        # Vertical control: use cubic altitude error as "nonlinear" input to the PID
        # This matches the tuned behavior from the PID example controller.
        vertical_input = PID["vertical"].compute(math.pow(altitude_error, 3.0), dt)

        # Motor mixing (same layout as official C example)
        fl = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        fr = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rl = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rr = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        front_left_motor.setVelocity(fl)
        front_right_motor.setVelocity(-fr)
        rear_left_motor.setVelocity(-rl)
        rear_right_motor.setVelocity(rr)

        # ---------- Depth logging ----------
        d_front = None
        if range_finder is not None:
            rf_image = range_finder.getRangeImage()
            if writer is not None:
                # Log every beam in the RangeFinder image
                for k_idx in range(rf_width):
                    r_val = rf_image[k_idx]
                    writer.writerow([current_time, x, y, z, yaw, k_idx, r_val])
            # also keep the central beam as "front distance"
            if rf_width > 0:
                d_front = rf_image[rf_width // 2]

        # ---------- Estimate target world coordinates and record ----------
        if target_found and target_cx is not None and d_front is not None and d_front < 1e6:
            # Normalized horizontal coordinate in image [-1, 1]
            norm_x = (target_cx - cam_width / 2) / (cam_width / 2)
            # Convert to angular offset using camera horizontal FOV
            angle_offset = norm_x * (cam_fov / 2.0)
            # Bearing in world frame (yaw + camera horizontal offset)
            bearing = yaw + angle_offset

            # Approximate world coordinates of the detected target
            x_obj = x + d_front * math.cos(bearing)
            y_obj = y + d_front * math.sin(bearing)
            z_obj = z  # simplified: assume target at same height as drone

            add_target_if_new(x_obj, y_obj, z_obj, label="target")

except Exception as e:
    print("[ERROR]", e)
finally:
    # Print and save all detected targets
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
