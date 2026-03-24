#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import subprocess
import os
import signal
import time
import threading
import json
import math
import psutil
import numpy as np
from collections import deque
from std_msgs.msg import String, Header, ColorRGBA
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Point, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from actionlib_msgs.msg import GoalStatus, GoalStatusArray, GoalID
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult, MoveBaseActionFeedback
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs

LOG_PREFIX = "[TarkbotBridge]"
CMD_TOPIC = "/switch_map_mode_request"
RESP_TOPIC = "/switch_map_mode_response"

# ================= 配置区域 =================
NAV_LAUNCH_CMD = ["roslaunch", "tarkbot_nav", "nav.launch"]
MAPPING_LAUNCH_CMD = ["roslaunch", "exploration_server", "tarkbot_mapping_exploration.launch"]

# 地图保存配置
BASE_MAP_PATH = os.path.expanduser("~/tarkbot/ros_ws/src/tarkbot_slam/maps")
MAP_SAVE_NAME = os.path.join(BASE_MAP_PATH, "my_map")

MODE_IDLE = -1       # 空闲模式
MODE_MAPPING = 0     # 建图模式
MODE_NAVIGATION = 1  # 导航模式

# ================= 建图子模式 =================
MAPPING_SUB_IDLE = 0           # 建图模式下-空闲（手动建图）
MAPPING_SUB_AUTO_EXPLORING = 1 # 建图模式下-自动探索中

class AutoExplorationNode:
    """自动探索引擎 - 修复版"""
    def __init__(self, parent_node):
        self.parent = parent_node
        self.enabled = False
        self.is_navigating = False
        self.current_goal = None
        self.previous_goals = deque(maxlen=20)
        self.failed_goals = set()
        self.map_data = None
        self.map_info = None
        self.robot_pose = None
        self.nav_status = None
        
        # 探索算法参数
        self.robot_radius = 0.25
        self.obstacle_inflation = 0.40
        self.total_safe_distance = self.robot_radius + self.obstacle_inflation + 0.15
        self.search_radius = 60.0
        self.min_frontier_size = 1
        self.navigation_timeout = 45.0
        self.goal_reached_dist = 0.5
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers 
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.frontier_marker_pub = rospy.Publisher('/exploration/frontiers_markers', MarkerArray, queue_size=10)
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        self.nav_status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.nav_status_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        
        # Timers
        self.exploration_timer = rospy.Timer(rospy.Duration(1.5), self.exploration_cycle)
        self.monitor_timer = rospy.Timer(rospy.Duration(0.5), self.navigation_monitor)
        
        # 调试计数器
        self.cycle_count = 0
        self.goal_sent_count = 0
        self.enable_call_count = 0
        
        # 强制启用标志
        self.force_enable_requested = False
        self.force_enable_time = 0
        
        # 关键修复：停止标志
        self.stop_requested = False
        
        rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Engine Initialized.")

    def enable(self, force=False):
        """启用自动探索"""
        self.enabled = True
        self.stop_requested = False
        self.failed_goals.clear()
        self.previous_goals.clear()
        self.is_navigating = False
        self.current_goal = None
        self.cycle_count = 0
        self.goal_sent_count = 0
        self.enable_call_count += 1
        
        if force:
            rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] >>> FORCE ENABLED! Call #{self.enable_call_count} <<<")
        else:
            rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] >>> ENABLED! Call #{self.enable_call_count} <<<")

    def disable(self, immediate=True):
        """
        禁用自动探索 - 关键修复：立即停止
        immediate: 是否立即停止（用于 mqtt_stop_auto_mapping）
        """
        rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] DISABLE called! immediate={immediate}")
        
        # 关键修复 1：立即设置停止标志
        self.stop_requested = True
        self.enabled = False
        
        # 关键修复 2：立即取消导航目标
        if self.parent.move_base_cancel_pub:
            cancel_msg = GoalID()
            cancel_msg.stamp = rospy.Time.now()
            self.parent.move_base_cancel_pub.publish(cancel_msg)
            rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Navigation CANCELLED")
        
        # 关键修复 3：立即停止机器人运动
        stop_twist = Twist()
        if self.parent.cmd_vel_pub:
            for i in range(10):  # 多发几次确保生效
                self.parent.cmd_vel_pub.publish(stop_twist)
                time.sleep(0.05)
            rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Robot motion STOPPED")
        
        # 关键修复 4：清空状态
        self.is_navigating = False
        self.current_goal = None
        self.failed_goals.clear()
        self.previous_goals.clear()
        
        rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] >>> DISABLED COMPLETE <<<")
        
    def reset_exploration_state(self):
        """重置探索状态"""
        self.disable(immediate=True)
        self.map_data = None 
        self.nav_status = None
        rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] State Reset for Map Reconstruction.")

    def map_callback(self, msg):
        try:
            self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
            self.map_info = msg.info
            if self.enabled and self.cycle_count == 0:
                rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Map received: {msg.info.width}x{msg.info.height}")
        except Exception as e:
            rospy.logerr(f"{LOG_PREFIX} [EXPLORER] Map callback error: {e}")

    def nav_status_callback(self, msg):
        """接收导航状态"""
        self.nav_status = msg
        if msg.status_list:
            last_status = msg.status_list[-1]
            if last_status.status == GoalStatus.SUCCEEDED:
                if self.is_navigating:
                    rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Navigation SUCCEEDED!")
                    self.is_navigating = False
                    self.current_goal = None
            elif last_status.status in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST]:
                if self.is_navigating:
                    rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] Navigation FAILED! Status: {last_status.status}")
                    self.is_navigating = False
                    if self.current_goal:
                        self.failed_goals.add(self.current_goal)
                    self.current_goal = None

    def odom_callback(self, msg):
        """从里程计获取位姿"""
        if self.robot_pose is None:
            self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def get_robot_pose(self):
        """获取机器人位姿"""
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(0.5))
            self.robot_pose = (transform.transform.translation.x, transform.transform.translation.y)
            return True
        except Exception as e:
            if self.robot_pose is not None:
                return True
            if self.cycle_count % 10 == 0:
                rospy.logwarn_throttle(5, f"{LOG_PREFIX} [EXPLORER] TF not available, waiting...")
            return False

    def world_to_map(self, x, y):
        if self.map_info is None: return 0, 0
        mx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return mx, my

    def map_to_world(self, mx, my):
        if self.map_info is None: return 0, 0
        wx = (mx + 0.5) * self.map_info.resolution + self.map_info.origin.position.x
        wy = (my + 0.5) * self.map_info.resolution + self.map_info.origin.position.y
        return wx, wy

    # ================= 核心算法 =================

    def is_frontier_cell(self, r, c):
        h, w = self.map_data.shape
        if not (0 <= r < h and 0 <= c < w):
            return False
        if self.map_data[r, c] != -1:
            return False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if self.map_data[nr, nc] == 0:
                        return True
        return False

    def check_cell_safety_optimized(self, r, c, safety_cells):
        h, w = self.map_data.shape
        if r - safety_cells < 0 or r + safety_cells >= h or c - safety_cells < 0 or c + safety_cells >= w:
            return False
        for dr in range(-safety_cells, safety_cells + 1):
            for dc in range(-safety_cells, safety_cells + 1):
                if dr*dr + dc*dc > safety_cells*safety_cells:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if self.map_data[nr, nc] > 0:
                        return False
        return True

    def detect_safe_frontiers_optimized(self):
        if self.map_data is None or self.robot_pose is None:
            return []

        h, w = self.map_data.shape
        visited = np.zeros_like(self.map_data, dtype=bool)
        frontiers_clusters = []
        
        safety_pixels = int(math.ceil(self.total_safe_distance / self.map_info.resolution))
        safety_pixels = min(safety_pixels, 4) 
        min_size_pixels = self.min_frontier_size
        
        rx, ry = self.robot_pose
        robot_mx, robot_my = self.world_to_map(rx, ry)

        candidates = []
        search_pix = int(self.search_radius / self.map_info.resolution)
        r_min, r_max = max(0, robot_my - search_pix), min(h, robot_my + search_pix)
        c_min, c_max = max(0, robot_mx - search_pix), min(w, robot_mx + search_pix)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if not visited[r, c] and self.is_frontier_cell(r, c):
                    candidates.append((r, c))

        for start_r, start_c in candidates:
            if visited[start_r, start_c]:
                continue
            
            cluster = []
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            
            while queue:
                r, c = queue.popleft()
                cluster.append((r, c))
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc] and self.is_frontier_cell(nr, nc):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            
            if len(cluster) < min_size_pixels:
                continue
                
            cluster_r = int(np.mean([p[0] for p in cluster]))
            cluster_c = int(np.mean([p[1] for p in cluster]))
            
            if not self.check_cell_safety_optimized(cluster_r, cluster_c, safety_pixels):
                continue
                
            cx, cy = self.map_to_world(cluster_r, cluster_c)
            dist_to_robot = math.hypot(cx - rx, cy - ry)
            
            is_near_previous = False
            for px, py in list(self.previous_goals):
                if math.hypot(cx - px, cy - py) < 2.0:
                    is_near_previous = True
                    break
            
            if is_near_previous:
                continue
                
            frontiers_clusters.append((cx, cy, len(cluster)))

        frontiers_clusters.sort(key=lambda x: x[2], reverse=True)
        return [(x, y) for x, y, _ in frontiers_clusters]

    def exploration_cycle(self, event=None):
        """探索主循环 - 关键修复：检查停止标志"""
        self.cycle_count += 1
        
        # 关键修复：优先检查停止标志
        if self.stop_requested:
            if self.cycle_count % 5 == 0:
                rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] Cycle {self.cycle_count}: STOP REQUESTED, skipping exploration")
            return
        
        # 强制启用检查
        if self.force_enable_requested and time.time() - self.force_enable_time > 2.0:
            rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] FORCE ENABLE TRIGGERED at cycle {self.cycle_count}!")
            self.enable(force=True)
            self.force_enable_requested = False
        
        if not self.enabled:
            return
            
        if self.is_navigating:
            if self.cycle_count % 20 == 0:
                rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Cycle {self.cycle_count}: NAVIGATING to {self.current_goal}")
            return
            
        if self.map_data is None:
            if self.cycle_count % 10 == 0:
                rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] Cycle {self.cycle_count}: NO MAP DATA")
            return
            
        if not self.get_robot_pose():
            if self.cycle_count % 10 == 0:
                rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] Cycle {self.cycle_count}: NO ROBOT POSE")
            return

        safe_frontiers = self.detect_safe_frontiers_optimized()
        
        if self.cycle_count % 10 == 0:
            rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Cycle {self.cycle_count}: Found {len(safe_frontiers)} frontiers")
        
        if not safe_frontiers:
            if self.cycle_count % 50 == 0:
                rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] No safe frontiers found!")
            return

        target = None
        for fx, fy in safe_frontiers:
            if (fx, fy) not in self.failed_goals:
                target = (fx, fy)
                break
        
        if target:
            self.send_goal(target)
            self.publish_frontier_markers(safe_frontiers, target)
        else:
            if len(safe_frontiers) > 0 and self.cycle_count % 20 == 0:
                rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] All frontiers in failed_goals, clearing...")
                self.failed_goals.clear()

    def send_goal(self, target_pos):
        """发送导航目标"""
        rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] >>> SENDING GOAL: {target_pos} <<<")
        
        # 降低导航速度
        try:
            rospy.set_param('/move_base/TrajectoryPlannerROS/max_vel_x', 0.15)
            rospy.set_param('/move_base/TrajectoryPlannerROS/max_rotational_vel', 0.25)
            rospy.set_param('/move_base/base_local_planner/max_vel_x', 0.15)
            rospy.set_param('/move_base/base_local_planner/max_rotational_vel', 0.25)
            rospy.set_param('/move_base/DWAPlannerROS/max_vel_x', 0.15)
            rospy.set_param('/move_base/DWAPlannerROS/max_rotational_vel', 0.25)
        except Exception as e:
            rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] Failed to set velocity params: {e}")
        
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = target_pos[0]
        goal_msg.pose.position.y = target_pos[1]
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_msg)
        self.goal_sent_count += 1
        
        rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Goal #{self.goal_sent_count} published (SLOW SPEED)")
        
        self.current_goal = target_pos
        self.is_navigating = True
        self.navigation_start_time = time.time()
        self.previous_goals.append(target_pos)

    def publish_frontier_markers(self, frontiers, current_target):
        markers = MarkerArray()
        marker_id = 0
        for fx, fy in frontiers[:10]:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "frontiers"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = fx
            m.pose.position.y = fy
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.5
            m.lifetime = rospy.Duration(3.0)
            markers.markers.append(m)
            marker_id += 1
            
        if current_target:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "current_goal"
            m.id = 999
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = current_target[0]
            m.pose.position.y = current_target[1]
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = 0.6; m.scale.y = 0.6; m.scale.z = 0.6
            m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 0.8
            m.lifetime = rospy.Duration(3.0)
            markers.markers.append(m)
            
        self.frontier_marker_pub.publish(markers)

    def navigation_monitor(self, event):
        """导航监控 - 关键修复：检查停止标志"""
        # 关键修复：优先检查停止标志
        if self.stop_requested:
            if self.is_navigating:
                rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] Monitor: STOP REQUESTED, cancelling navigation")
                self.is_navigating = False
                self.current_goal = None
                if self.parent.move_base_cancel_pub:
                    self.parent.move_base_cancel_pub.publish(GoalID())
            return
            
        if not self.is_navigating or not self.current_goal:
            return
            
        if time.time() - self.navigation_start_time > self.navigation_timeout:
            rospy.logwarn(f"{LOG_PREFIX} [EXPLORER] TIMEOUT! Goal: {self.current_goal}")
            self.is_navigating = False
            self.failed_goals.add(self.current_goal)
            self.current_goal = None
            if self.parent.move_base_cancel_pub:
                self.parent.move_base_cancel_pub.publish(GoalID())
            return

        if self.get_robot_pose():
            rx, ry = self.robot_pose
            gx, gy = self.current_goal
            dist = math.hypot(rx - gx, ry - gy)
            
            if dist < self.goal_reached_dist:
                rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Goal REACHED! Dist: {dist:.2f}")
                self.is_navigating = False
                self.current_goal = None


class TarkbotNode:
    def __init__(self):
        rospy.init_node('tarkbot_bridge', anonymous=True)
        
        self.current_mode = MODE_IDLE
        self.mapping_sub_mode = MAPPING_SUB_IDLE
        self.lock = threading.Lock()
        self.current_process = None 
        
        self.is_calibrating = False
        self.calibration_publisher = None
        self.calibration_complete = False
        
        # 关键修复：启动状态标志，防止重复启动
        self.is_starting_auto_mapping = False
        self.start_request_count = 0  # 记录启动请求次数
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.move_base_cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.response_pub = rospy.Publisher(RESP_TOPIC, String, queue_size=10)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1, latch=True)
        
        # Subscribers
        rospy.Subscriber(CMD_TOPIC, String, self.business_command_callback)
        
        # 初始化探索引擎
        self.explorer = AutoExplorationNode(self)
        
        if not os.path.exists(BASE_MAP_PATH):
            try:
                os.makedirs(BASE_MAP_PATH)
            except: pass
        
        rospy.loginfo(f"{LOG_PREFIX} System Ready. 正式启动")
        
        # 启动强制启用监控线程
        self.force_enable_thread = threading.Thread(target=self._force_enable_monitor, daemon=True)
        self.force_enable_thread.start()
        
    def _force_enable_monitor(self):
        """强制启用监控线程"""
        rate = rospy.Rate(1.0)
        wait_count = 0
        while not rospy.is_shutdown():
            wait_count += 1
            if self.current_mode == MODE_MAPPING and self.mapping_sub_mode == MAPPING_SUB_AUTO_EXPLORING:
                if not self.explorer.enabled and not self.explorer.stop_requested and wait_count > 10:
                    rospy.logwarn(f"{LOG_PREFIX} [FORCE] Auto-exploring mode active but explorer not enabled! Forcing...")
                    self.explorer.force_enable_requested = True
                    self.explorer.force_enable_time = time.time()
            else:
                wait_count = 0
            rate.sleep()
        
    def _start_mapping_initial_calibration(self):
        """启动建图前的初始位姿校准"""
        if self.is_calibrating:
            rospy.logwarn("校准正在进行中，忽略重复请求")
            return

        self.is_calibrating = True
        self.calibration_complete = False
        
        if self.calibration_publisher is None:
            self.calibration_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
            rospy.sleep(0.5)

        calib_thread = threading.Thread(target=self._mapping_calibration_loop, daemon=True)
        calib_thread.start()

    def _mapping_calibration_loop(self):
        """建图专用校准循环"""
        start_time = time.time()
        duration = 7.0  
        angular_speed = 0.5
        
        twist = Twist()
        twist.angular.z = angular_speed
        
        rospy.loginfo(f"机器人正在旋转建图校准 ({duration}秒)...")
        
        while not rospy.is_shutdown() and self.is_calibrating:
            if time.time() - start_time >= duration:
                break
            self.calibration_publisher.publish(twist)
            time.sleep(0.1)
        
        stop_twist = Twist()
        if self.calibration_publisher:
            for _ in range(5):
                self.calibration_publisher.publish(stop_twist)
                time.sleep(0.1)
                
        self.is_calibrating = False
        self.calibration_complete = True
        rospy.loginfo("[MANUAL_MAP] 建图校准完成！") 
        
    def _execute_start_manual_mapping(self, data):
        """执行手动建图启动任务"""
        rospy.loginfo("=== [开始] 执行手动建图启动任务 ===")
        
        if self.current_mode == MODE_MAPPING:
            msg = "当前已处于建图模式，无需重复启动"
            rospy.logwarn(msg)
            return True, msg

        if self.mapping_sub_mode == MAPPING_SUB_AUTO_EXPLORING:
            rospy.logwarn("检测到之前是自动探索模式，先禁用自动探索...")
            self.explorer.disable(immediate=True)
        
        yaml_path = f"{MAP_SAVE_NAME}.yaml"
        pgm_path = f"{MAP_SAVE_NAME}.pgm"
        
        has_old_map = False
        launch_args = []

        if os.path.exists(yaml_path) and os.path.exists(pgm_path):
            rospy.loginfo(f"检测到现有地图文件：{yaml_path}")
            has_old_map = True
            launch_args = [f"map_file:={yaml_path}"]
        else:
            rospy.loginfo("未检测到现有地图文件，将创建新地图")
            has_old_map = False
            self._clear_local_map()

        rospy.loginfo("正在启动建图节点...")
        final_launch_cmd = MAPPING_LAUNCH_CMD + launch_args
        
        success = self._start_stack(final_launch_cmd, "MAPPING")
        
        if not success:
            msg = "建图节点启动失败"
            rospy.logerr(msg)
            self.current_mode = MODE_IDLE
            self.mapping_sub_mode = MAPPING_SUB_IDLE
            return False, msg

        self.current_mode = MODE_MAPPING
        self.mapping_sub_mode = MAPPING_SUB_IDLE
        rospy.loginfo("建图节点启动成功，状态已切换为 MODE_MAPPING (MANUAL)")

        wait_time = 2.0 if has_old_map else 1.0
        rospy.sleep(wait_time)
        
        try:
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.pose.position.x = 0.0
            pose_msg.pose.pose.position.y = 0.0
            pose_msg.pose.pose.orientation.w = 1.0
            pose_msg.pose.covariance = [0.1] * 36 if has_old_map else [0.5] * 36
            self.initial_pose_pub.publish(pose_msg)
        except Exception as e:
            rospy.logerr(f"发送初始位姿失败：{e}")

        self._start_mapping_initial_calibration()

        if has_old_map:
            msg = "已加载旧地图，正在进行位姿校准（手动建图模式）"
        else:
            msg = "新地图创建中，正在进行初始位姿校准（手动建图模式）"
            
        rospy.loginfo(f"{msg}")
        return True, msg
        
    def _execute_stop_mapping(self, data):
        """执行停止建图并保存任务"""
        rospy.loginfo("=== [开始] 执行停止建图任务 ===")
        
        original_mode = self.current_mode
        original_sub_mode = self.mapping_sub_mode
        
        # 先禁用自动探索
        if self.explorer.enabled or self.explorer.stop_requested:
            rospy.loginfo("禁用自动探索...")
            self.explorer.disable(immediate=True)
        
        if self.is_calibrating:
            rospy.logwarn("⚠️ 检测到正在校准，强制停止...")
            self.is_calibrating = False
            if self.calibration_publisher:
                stop_twist = Twist()
                for _ in range(5):
                    self.calibration_publisher.publish(stop_twist)
                    time.sleep(0.1)

        overall_success = True
        message_parts = []
        map_saved_success = False

        rospy.loginfo("步骤 1: 尝试保存地图...")
        
        def try_save_map():
            try:
                if hasattr(self, '_save_map'):
                    return self._save_map()
                else:
                    map_path = MAP_SAVE_NAME
                    rospy.loginfo(f"[MAP] Saving to: {map_path}...")
                    cmd = ['rosrun', 'map_server', 'map_saver', '-f', map_path]
                    subprocess.run(cmd, check=True, timeout=30)
                    return True
            except Exception as e:
                rospy.logerr(f"[MAP] Error saving map: {e}")
                return False

        if original_mode == MODE_MAPPING:
            map_saved_success = try_save_map()
        
        if not map_saved_success:
            rospy.logwarn("第一次保存失败，尝试停止节点后重试...")
            self._stop_current_stack()
            rospy.sleep(3.0)
            if original_mode == MODE_MAPPING:
                map_saved_success = try_save_map()

        if map_saved_success:
            message_parts.append("地图保存成功")
        else:
            message_parts.append("地图保存失败")
            overall_success = False

        if original_mode == MODE_MAPPING:
            self.current_mode = MODE_IDLE
            self.mapping_sub_mode = MAPPING_SUB_IDLE
            rospy.loginfo("✅ 状态已更新为 MODE_IDLE")
            rospy.loginfo("🚀 建图结束，自动切换到导航模式...")
            self._switch_to_navigation_mode()

        final_message = ",".join(message_parts)
        rospy.loginfo(f"=== [结束] 停止建图任务：{final_message} ===")
        
        return overall_success, final_message

    def _switch_to_navigation_mode(self):
        """切换到导航模式"""
        if self.current_mode == MODE_NAVIGATION:
            return True
            
        if self.explorer.enabled:
            self.explorer.disable(immediate=True)
            
        if self.current_process is not None:
            self._stop_current_stack()
            rospy.sleep(1.5)
            
        if self.current_mode == MODE_IDLE and self.current_process is None:
            if self._start_stack(NAV_LAUNCH_CMD, "NAV"):
                self.current_mode = MODE_NAVIGATION
                self.mapping_sub_mode = MAPPING_SUB_IDLE
                rospy.loginfo("✅ 已成功自动切换到导航模式")
                return True
            else:
                rospy.logerr("❌ 自动切换到导航模式失败")
                return False
        return False
        
    def _is_process_running(self, cmd_list):
        try:
            target_pkg = cmd_list[1]
            target_file = cmd_list[2]
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and len(cmdline) >= 3:
                        if cmdline[1] == target_pkg and cmdline[2] == target_file:
                            return True
                except: continue
            return False
        except: return False

    def _kill_process_group(self, process_obj):
        if process_obj and process_obj.poll() is None:
            try:
                os.killpg(os.getpgid(process_obj.pid), signal.SIGTERM)
                process_obj.wait(timeout=3)
            except: pass
        
    def _clear_local_map(self):
        """清空地图"""
        try:
            files_to_delete = [MAP_SAVE_NAME + ".pgm", MAP_SAVE_NAME + ".yaml"]
            for f_path in files_to_delete:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    rospy.loginfo(f"{LOG_PREFIX} [MAP] Deleted: {f_path}")
            return True
        except Exception as e:
            rospy.logerr(f"{LOG_PREFIX} [MAP] Clear error: {e}")
            return False

    def _save_map(self):
        rospy.loginfo(f"{LOG_PREFIX} [MAP] Saving to: {MAP_SAVE_NAME}...")
        try:
            if not os.path.exists(BASE_MAP_PATH):
                os.makedirs(BASE_MAP_PATH)
            cmd = ["rosrun", "map_server", "map_saver", "-f", MAP_SAVE_NAME]
            result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
            if result.returncode == 0:
                rospy.loginfo(f"{LOG_PREFIX} [MAP] SUCCESS.")
                return True
            return False
        except Exception as e:
            rospy.logerr(f"{LOG_PREFIX} [MAP] Error: {e}")
            return False

    def _start_stack(self, cmd_list, mode_name):
        if self._is_process_running(cmd_list):
            rospy.loginfo(f"{LOG_PREFIX} [{mode_name}] Process already running.")
            return True
        if self.current_process:
            rospy.logwarn(f"{LOG_PREFIX} [{mode_name}] Stopping existing process...")
            self._stop_current_stack()
        
        rospy.loginfo(f"{LOG_PREFIX} [{mode_name}] Starting: {' '.join(cmd_list)}")
        try:
            self.current_process = subprocess.Popen(cmd_list, preexec_fn=os.setsid)
            time.sleep(6) 
            if self.current_process.poll() is None:
                return True
            return False
        except Exception as e:
            rospy.logerr(f"{LOG_PREFIX} [{mode_name}] Failed: {e}")
            return False

    def _stop_current_stack(self):
        if self.current_process:
            self.move_base_cancel_pub.publish(GoalID())
            self._kill_process_group(self.current_process)
            self.current_process = None
            rospy.loginfo(f"{LOG_PREFIX} [CTRL] Stack stopped.")

    def business_command_callback(self, msg):
        try:
            data = json.loads(msg.data.replace('\n', '').replace('\r', ''))
            req_type = data.get('type')
            client_id = data.get('ros_clientid', 'unknown')
            rospy.logwarn(f"{LOG_PREFIX} [!!!] 收到原始消息 type="+req_type)
            
            with self.lock:
                code = 0
                status_msg = "OK"
                final_mode = self.current_mode

                # 1. mqtt_start_auto_mapping: 自动探索建图
                if req_type == 'mqtt_start_auto_mapping':
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_start_auto_mapping")
                    self.start_request_count += 1
                    rospy.loginfo(f"{LOG_PREFIX} Start request count: {self.start_request_count}")
                    
                    # 关键修复 1：检查是否已经在自动探索模式中
                    if self.current_mode == MODE_MAPPING and self.mapping_sub_mode == MAPPING_SUB_AUTO_EXPLORING:
                        if self.explorer.enabled and not self.explorer.stop_requested:
                            status_msg = "Already in Auto Mapping Mode (explorer running)"
                            code = 0
                            final_mode = MODE_MAPPING
                            rospy.loginfo(f"{LOG_PREFIX} [IGNORE] Duplicate start request ignored")
                        else:
                            # 探索器被禁用但模式还在，重新启用
                            rospy.loginfo(f"{LOG_PREFIX} [RE-ENABLE] Explorer was disabled, re-enabling...")
                            self.explorer.stop_requested = False
                            self.explorer.enable(force=True)
                            status_msg = "Auto Mapping Re-enabled"
                            code = 0
                            final_mode = MODE_MAPPING
                    # 关键修复 2：检查是否正在启动过程中
                    elif self.is_starting_auto_mapping:
                        status_msg = "Auto Mapping is starting, please wait..."
                        code = 0
                        final_mode = MODE_MAPPING
                        rospy.logwarn(f"{LOG_PREFIX} [IGNORE] Start request ignored, already starting")
                    else:
                        # 标记正在启动
                        self.is_starting_auto_mapping = True
                        
                        try:
                            # 关键修复 3：切换模式前先禁用之前的探索
                            if self.explorer.enabled or self.explorer.stop_requested:
                                rospy.loginfo("Detected explorer running, disabling first...")
                                self.explorer.disable(immediate=True)
                                rospy.sleep(0.5)  # 等待停止生效
                            
                            self._clear_local_map()
                            
                            success = self._start_stack(MAPPING_LAUNCH_CMD, "MAPPING")
                            
                            if success:
                                self.current_mode = MODE_MAPPING
                                self.mapping_sub_mode = MAPPING_SUB_AUTO_EXPLORING
                                final_mode = MODE_MAPPING
                                
                                pose_msg = PoseWithCovarianceStamped()
                                pose_msg.header.stamp = rospy.Time.now()
                                pose_msg.header.frame_id = "map"
                                pose_msg.pose.pose.orientation.w = 1.0
                                pose_msg.pose.covariance = [0.1]*36 
                                self.initial_pose_pub.publish(pose_msg)
                                
                                rospy.loginfo(f"{LOG_PREFIX} [WAIT] Waiting for Map and TF...")
                                rate = rospy.Rate(1.0)
                                wait_count = 0
                                max_wait = 30
                                map_ready = False
                                tf_ready = False
                                
                                while wait_count < max_wait and not (map_ready and tf_ready):
                                    if self.explorer.map_data is not None:
                                        map_ready = True
                                    try:
                                        self.explorer.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(0.5))
                                        tf_ready = True
                                    except:
                                        tf_ready = False
                                        
                                    if map_ready and tf_ready:
                                        break
                                        
                                    rospy.loginfo_throttle(5, f"{LOG_PREFIX} [WAIT] Map: {map_ready}, TF: {tf_ready} ... ({wait_count}s)")
                                    rate.sleep()
                                    wait_count += 1
                                
                                # 启动校准
                                self._start_mapping_initial_calibration()
                                
                                # 设置强制启用标志
                                rospy.loginfo(f"{LOG_PREFIX} [EXPLORER] Setting FORCE ENABLE flag for AUTO mode...")
                                self.explorer.force_enable_requested = True
                                self.explorer.force_enable_time = time.time()
                                
                                # 启动延迟启用线程
                                enable_thread = threading.Thread(target=self._delayed_enable_explorer, daemon=True)
                                enable_thread.start()
                                
                                status_msg = "Auto Mapping Started (Ready)"
                                code = 0
                            else:
                                code = -1
                                status_msg = "Launch Failed"
                        finally:
                            # 关键修复 4：无论成功失败，都要清除启动标志
                            self.is_starting_auto_mapping = False
                            rospy.loginfo(f"{LOG_PREFIX} Start process completed, is_starting_auto_mapping = False")

                # 2. mqtt_mapping: 手动建图
                elif req_type == 'mqtt_mapping':        
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_mapping (MANUAL MODE)")
                    if self.current_mode == MODE_MAPPING:
                        status_msg = "Already in Mapping Mode"
                        code = 0
                        final_mode = MODE_MAPPING
                    else:
                        success, msg_text = self._execute_start_manual_mapping(data)
                        if success:
                            code = 0
                            status_msg = 'mqtt_mapping success (Manual mode)'
                            final_mode = self.current_mode
                        else:
                            code = -1
                            status_msg = 'mqtt_mapping failed: ' + msg_text
                            final_mode = MODE_IDLE
                            
                # 3. mqtt_stop_mapping: 停止建图并保存
                elif req_type == 'mqtt_stop_mapping': 
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_stop_mapping")  
                    if self.current_mode != MODE_MAPPING:
                        status_msg = "Not in Mapping Mode"
                        code = -1
                    else:
                        success, msg_text = self._execute_stop_mapping(data)
                        if success:
                            code = 0
                            status_msg = 'mqtt_stop_mapping success'
                            final_mode = self.current_mode 
                        else:
                            code = -1
                            status_msg = 'mqtt_stop_mapping failed: ' + msg_text

                # 4. mqtt_stop_auto_mapping: 停止自动探索
                elif req_type == 'mqtt_stop_auto_mapping':
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_stop_auto_mapping")
                    if self.current_mode == MODE_MAPPING:
                        # 关键修复：立即禁用探索器
                        rospy.loginfo(f"{LOG_PREFIX} [STOP] Disabling explorer immediately...")
                        self.explorer.disable(immediate=True)
                        self.mapping_sub_mode = MAPPING_SUB_IDLE
                        status_msg = "Auto Exploration STOPPED immediately. Robot standing by."
                        code = 0
                        final_mode = MODE_MAPPING
                        rospy.logwarn(f"{LOG_PREFIX} [STOP] Explorer disabled, stop_requested={self.explorer.stop_requested}")
                    else:
                        status_msg = "Not in Mapping Mode"
                        code = -1

                # 5. mqtt_ros_mode: 查询当前模式
                elif req_type == 'mqtt_ros_mode':
                     rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_ros_mode")
                     final_mode = self.current_mode
                     status_msg = f"ROS Mode: {final_mode}, Sub-Mode: {self.mapping_sub_mode}, Explorer: enabled={self.explorer.enabled}, stop={self.explorer.stop_requested}"
                     code = 0
                         
                # 6. mqtt_navigation: 进入导航模式
                elif req_type == 'mqtt_navigation':
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_navigation")
                    
                    code = -1 
                    status_msg = "Nav Start Failed"

                    if self.current_mode == MODE_NAVIGATION:
                        status_msg = "Already in Navigation Mode"
                        code = 0
                        final_mode = MODE_NAVIGATION
                    else:
                        if self.explorer.enabled:
                            rospy.loginfo("Disabling explorer before switching to NAV...")
                            self.explorer.disable(immediate=True)
                        
                        if self.current_process is not None:
                            rospy.loginfo(f"{LOG_PREFIX} [NAV] Stopping existing process...")
                            self._stop_current_stack()
                            rospy.sleep(1.5)

                        if self.current_mode == MODE_MAPPING:
                             rospy.warn("Current mode is MAPPING. Stopping mapping to switch to NAV...")
                             self._stop_current_stack()
                             self.current_mode = MODE_IDLE
                             self.mapping_sub_mode = MAPPING_SUB_IDLE
                             rospy.sleep(1.0)

                        if self.current_mode == MODE_IDLE and self.current_process is None:
                            if self._start_stack(NAV_LAUNCH_CMD, "NAV"):
                                self.current_mode = MODE_NAVIGATION
                                self.mapping_sub_mode = MAPPING_SUB_IDLE
                                final_mode = MODE_NAVIGATION
                                status_msg = "Nav Launched Successfully"
                                code = 0
                            else:
                                status_msg = "Nav Launch Command Failed"
                                code = -1
                        else:
                            status_msg = "Error: Mode not Idle or Process busy"
                            code = -1

                # 7. mqtt_stop_navigation
                elif req_type == 'mqtt_stop_navigation':
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_stop_navigation")
                    self._stop_current_stack()
                    self.current_mode = MODE_IDLE
                    self.mapping_sub_mode = MAPPING_SUB_IDLE
                    final_mode = MODE_IDLE
                    status_msg = "Nav Stopped"
                    code = 0
                    
                # 8. mqtt_reconstruct_map
                elif req_type == 'mqtt_reconstruct_map':
                    rospy.logwarn(f"{LOG_PREFIX} [!!!] mqtt_reconstruct_map")
                    if self.explorer.enabled:
                        self.explorer.disable(immediate=True)
                    clear_success = self._clear_local_map()
                    if clear_success:
                        if self.current_mode == MODE_MAPPING:
                            self.explorer.reset_exploration_state()
                            self.mapping_sub_mode = MAPPING_SUB_IDLE
                            status_msg = "Map Reconstructed & Explorer Reset"
                        else:
                            status_msg = "Map Reconstructed (Idle Mode)"
                        code = 0
                    else:
                        status_msg = "Failed to clear map"
                        code = -1
                
                response_data = {
                    "type": req_type, "ros_clientid": client_id,
                    "timestamp": str(int(time.time() * 1000)),
                    "data": {"mode": final_mode, "sub_mode": self.mapping_sub_mode, "code": code, "msg": status_msg}
                }
                rospy.loginfo(f"响应消息 /switch_map_mode_response: {response_data}")
                self.response_pub.publish(String(data=json.dumps(response_data)))
                

        except Exception as e:
            rospy.logerr(f"{LOG_PREFIX} [ERROR] {e}")

    def _delayed_enable_explorer(self):
        """延迟启用探索器"""
        rospy.loginfo(f"{LOG_PREFIX} [THREAD] Waiting for calibration to complete...")
        
        wait_count = 0
        while not self.calibration_complete and wait_count < 20:
            rospy.sleep(1.0)
            wait_count += 1
        
        rospy.sleep(2.0)
        
        if self.mapping_sub_mode == MAPPING_SUB_AUTO_EXPLORING:
            rospy.logwarn(f"{LOG_PREFIX} [THREAD] Calibration complete! Enabling explorer (AUTO mode)...")
            self.explorer.enable(force=True)
            
            rospy.sleep(1.0)
            if self.explorer.enabled:
                rospy.loginfo(f"{LOG_PREFIX} [THREAD] Explorer confirmed ENABLED!")
            else:
                rospy.logerr(f"{LOG_PREFIX} [THREAD] Explorer still NOT enabled!")
        else:
            rospy.loginfo(f"{LOG_PREFIX} [THREAD] Not in AUTO mode (sub_mode={self.mapping_sub_mode}), skipping enable")


if __name__ == '__main__':
    try:
        node = TarkbotNode()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutdown")