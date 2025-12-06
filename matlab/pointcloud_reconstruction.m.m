% 读 CSV
data = readtable('scan_log.csv');

% 参数：和 Webots 中 RangeFinder 一致
rf_fov = 0.785398;      % 示例：45° = pi/4，你用 console 打印的真实值填进来
W = max(data.beam_index) + 1;

% 转成 double
x_d   = double(data.x);
y_d   = double(data.y);
z_d   = double(data.z);
yaw   = double(data.yaw);
idx   = double(data.beam_index);
range = double(data.range);

% 各射线相对于机头的角度 alpha_k
alpha = (idx ./ (W-1) - 0.5) * rf_fov;

% 无人机自身坐标系下的点
x_local = range .* cos(alpha);
y_local = range .* sin(alpha);
z_local = zeros(size(range));   % 2D 激光平面

% 旋转到世界坐标系
x_world = x_d + x_local .* cos(yaw) - y_local .* sin(yaw);
y_world = y_d + x_local .* sin(yaw) + y_local .* cos(yaw);
z_world = z_d + z_local;

% 画 3D 点云
figure;
scatter3(x_world, y_world, z_world, 4, z_world, 'filled');
axis equal;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('3D point cloud from Webots RangeFinder');
view(3);
grid on;
