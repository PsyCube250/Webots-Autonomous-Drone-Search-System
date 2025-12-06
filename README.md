# Webots Autonomous Drone Search System

Fully automatic autonomous drone search and mapping platform built in Webots.  
This system performs real-time object detection, autonomous navigation, coordinate projection, and data export for 3D reconstruction.

---

## 🚀 Features

- Full autonomous quadcopter stabilization (roll, pitch, yaw, altitude)
- Real-time camera detection (HSV-based; replaceable with ML model)
- GPS + RangeFinder based coordinate projection
- Obstacle avoidance response
- Arena boundary containment logic
- Export of detected object coordinates (`targets.csv`)
- RangeFinder scanning output for MATLAB 3D reconstruction (`scan_log.csv`)

---

## 📂 Repository Structure

/controllers
/mavic2pro
controller.py # main autonomous flight logic

/worlds
arena.wbt # Webots simulation environment

/matlab
reconstruction.m # builds 3D scatter / point cloud
depth_reader.m # imports scan_log.csv

/data
scan_log.csv # depth log for MATLAB
targets.csv # detected object coordinates

/docs
algorithm_overview.md
matlab_pipeline.md

README.md
LICENSE

Webots-Autonomous-Drone-Search-System
│
├── README.md
├── LICENSE
│
├── controllers/
│   └── mavic2pro/
│       ├── controller.py             # Main autonomous flight control + detection + logging
│       └── __init__.py
│
├── worlds/
│   └── arena.wbt                     # Webots simulation world (drone + arena + objects)
│
├── matlab/
│   ├── reconstruction.m              # Point cloud / 3D scatter reconstruction
│   └── depth_reader.m                # Reads scan_log.csv and builds map arrays
│
├── data/
│   ├── scan_log.csv                  # RangeFinder full field-of-view log
│   └── targets.csv                   # Exported detected object coordinates
│
└── docs/
    ├── algorithm_overview.md         # Detection + coordinate projection description
    └── matlab_pipeline.md            # MATLAB processing documentation

---

## 📄 Output — `targets.csv`

| id | label | x | y | z |
|----|------|---|---|---|

Each object is logged once by duplicate-radius filtering.

---

## 🧠 System Pipeline

Autonomous stabilization
→ Camera frame → HSV segmentation
→ RangeFinder distance
→ Coordinate projection (GPS + yaw)
→ Export to CSV
→ Optional MATLAB 3D point cloud


---

## 🛠 Requirements

| Software | Version |
|----------|--------|
| Webots | R2023+ |
| Python | 3.8–3.11 |
| OpenCV | 4.x |
| NumPy | Latest |
| MATLAB | Optional |

---

## ▶ How to Run

```bash
Clone repository
Open Webots
Load world
Run controller
```

## MATLAB (optional)

```matlab
run matlab/reconstruction.m
```

🔧 Optional Future Expansions

-Replace HSV with YOLO/TensorRT model

-SLAM path planning

-Multi-category classification

-Full surface reconstruction instead of scatter mapping


---

If you'd like, I can also auto-generate:  
📁 `/docs/algorithm_overview.md` and `/docs/matlab_pipeline.md`

Would you like **very short docs** (bullet style), or **long academic style** (suitable for competition paper)? ✍️
