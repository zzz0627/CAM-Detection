import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

def detect_interface_regions(img):
    """检测并标记软件界面区域，用于后续过滤"""
    h, w = img.shape[:2]
    interface_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 界面通常分布在边缘区域，标记边界带
    edge_thickness = min(w, h) // 20
    interface_mask[:, :edge_thickness] = 255
    interface_mask[:, w-edge_thickness:] = 255
    interface_mask[:edge_thickness, :] = 255
    interface_mask[h-edge_thickness:, :] = 255
    
    # 基于边缘检测识别工具栏等规则矩形结构
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = w_box * h_box
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        # 界面元素特征：边缘位置 + 规则矩形 + 适中面积
        is_edge_element = (x < edge_thickness or x + w_box > w - edge_thickness or 
                          y < edge_thickness or y + h_box > h - edge_thickness)
        is_regular_shape = 0.2 < aspect_ratio < 5.0
        is_moderate_size = 100 < area < (w * h) // 4
        
        if is_edge_element and is_regular_shape and is_moderate_size:
            cv2.fillPoly(interface_mask, [contour], 255)
    
    return interface_mask

def detect_image_content_roi(img):
    """提取图像主要内容区域（ROI），排除边缘界面干扰"""
    h, w = img.shape[:2]
    
    # 初始化全图ROI并剔除边框
    roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    border_size = min(w, h) // 15
    roi_mask[:border_size, :] = 0
    roi_mask[h-border_size:, :] = 0
    roi_mask[:, :border_size] = 0
    roi_mask[:, w-border_size:] = 0
    
    # Sobel算子检测纹理梯度，定位主要内容区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 阈值分割纹理丰富区域
    _, texture_mask = cv2.threshold(sobel_magnitude, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学闭运算连接碎片化纹理
    kernel = np.ones((15, 15), np.uint8)
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # 提取最大连通域作为主内容区
    contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        content_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(content_mask, [largest_contour], 255)
        roi_mask = cv2.bitwise_and(roi_mask, content_mask)
    
    return roi_mask

def separate_merged_regions(blue_mask, min_separation_distance=5):
    """使用分水岭算法分离粘连的蓝色区域"""
    try:
        from scipy import ndimage
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_maxima
        
        dist_transform = cv2.distanceTransform(blue_mask, cv2.DIST_L2, 5)
        
        # 提取局部极大值作为分水岭种子点
        local_maxima = peak_local_maxima(dist_transform, 
                                       min_distance=min_separation_distance,
                                       threshold_abs=0.3 * dist_transform.max(),
                                       indices=False)
        
        if np.sum(local_maxima) <= 1:
            return blue_mask
        
        markers = ndimage.label(local_maxima)[0]
        labels = watershed(-dist_transform, markers, mask=blue_mask)
        separated_mask = (labels > 0).astype(np.uint8) * 255
        
        return separated_mask
        
    except ImportError:
        return separate_regions_simple(blue_mask, min_separation_distance)

def detect_large_circular_structures(img):
    """基于边缘检测识别焊盘、过孔等大型圆形PCB结构"""
    h, w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 闭运算连接断裂边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circular_contours = []
    circular_centers = []
    
    # 焊盘级结构的合理面积范围：0.5% ~ 10%
    min_structure_area = (w * h) // 200
    max_structure_area = (w * h) // 10
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if not (min_structure_area <= area <= max_structure_area):
            continue
        
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # 半径范围：3.3% ~ 20% 图像尺寸
        min_radius = min(w, h) / 30
        max_radius = min(w, h) / 5
        
        if not (min_radius <= radius <= max_radius):
            continue
        
        circle_area = np.pi * radius * radius
        circularity = area / circle_area if circle_area > 0 else 0
        
        # 圆形度阈值 > 0.3 即可接受
        if circularity > 0.3:
            circular_contours.append(contour)
            circular_centers.append((int(cx), int(cy), radius))
    
    return circular_contours, circular_centers

def is_anomaly_near_structure(anomaly_contour, circular_centers, img_shape):
    """计算异常区域到最近焊盘结构的距离比例，判定是否相邻"""
    if len(circular_centers) == 0:
        return False, float('inf')
    
    M = cv2.moments(anomaly_contour)
    if M["m00"] == 0:
        return False, float('inf')
    
    anomaly_cx = int(M["m10"] / M["m00"])
    anomaly_cy = int(M["m01"] / M["m00"])
    
    min_distance_ratio = float('inf')
    
    for cx, cy, radius in circular_centers:
        distance = np.sqrt((anomaly_cx - cx)**2 + (anomaly_cy - cy)**2)
        distance_ratio = distance / radius if radius > 0 else float('inf')
        min_distance_ratio = min(min_distance_ratio, distance_ratio)
    
    # 距离比例 < 1.2 视为相邻或重叠
    is_near = min_distance_ratio < 1.2
    
    return is_near, min_distance_ratio

def is_anomaly_on_pad(anomaly_contour, pad_mask):
    """计算异常区域与焊盘的重叠比例，判定是否位于焊盘上"""
    h, w = pad_mask.shape[:2]
    
    anomaly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(anomaly_mask, [anomaly_contour], 255)
    
    overlap = cv2.bitwise_and(anomaly_mask, pad_mask)
    overlap_area = np.sum(overlap > 0)
    anomaly_area = np.sum(anomaly_mask > 0)
    overlap_ratio = overlap_area / anomaly_area if anomaly_area > 0 else 0
    
    # 重叠超过50%即判定为焊盘异常
    return overlap_ratio > 0.5, overlap_ratio

def find_nearest_pad_center(anomaly_contour, circular_centers):
    """找到距离异常区域最近的焊盘中心"""
    if not circular_centers:
        return None
    
    M = cv2.moments(anomaly_contour)
    if M["m00"] == 0:
        return None
    
    anomaly_cx = int(M["m10"] / M["m00"])
    anomaly_cy = int(M["m01"] / M["m00"])
    
    min_distance = float('inf')
    nearest_center = None
    
    for cx, cy, radius in circular_centers:
        distance = np.sqrt((anomaly_cx - cx)**2 + (anomaly_cy - cy)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_center = (cx, cy)
    
    return nearest_center

def determine_anomaly_direction(anomaly_contour, pad_center):
    """判断异常相对于焊盘的方向（横向/纵向）"""
    if pad_center is None:
        return 'horizontal'  # 无焊盘时默认横向
    
    M = cv2.moments(anomaly_contour)
    if M["m00"] == 0:
        return 'horizontal'
    
    anomaly_cx = int(M["m10"] / M["m00"])
    anomaly_cy = int(M["m01"] / M["m00"])
    
    pad_cx, pad_cy = pad_center
    dx = anomaly_cx - pad_cx  # 保留符号
    dy = anomaly_cy - pad_cy  # 保留符号
    
    # 使用角度判断更鲁棒
    # atan2 返回 [-π, π]，转换为角度 [-180, 180]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    
    # 角度分区（以焊盘中心为原点）：
    # -45° ~ 45°：右侧 → vertical
    # 45° ~ 135°：下侧 → horizontal  
    # 135° ~ 180° 或 -180° ~ -135°：左侧 → vertical
    # -135° ~ -45°：上侧 → horizontal
    
    abs_angle = abs(angle_deg)
    
    # 左右侧异常（±45°范围外的水平方向）
    if abs_angle < 45 or abs_angle > 135:
        return 'vertical'
    # 上下侧异常（±45°范围内的垂直方向）
    else:
        return 'horizontal'

def get_target_pad_mask(img, circular_centers):
    """
    生成只包含与圆形结构（焊盘）相连的红色区域的掩膜。
    用于剔除背景中无关的红色走线上的异常。
    """
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 宽泛地提取所有红色区域 (焊盘 + 走线)
    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 闭运算填充红色区域内部的小孔
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 保留与 circular_centers (粉色孔) 重叠/相连的红色区域
    target_pad_mask = np.zeros((h, w), dtype=np.uint8)
    
    if not circular_centers:
        return red_mask  # 如果没找到圆，只能返回所有红色作为保底
    
    # 标记所有红色连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    
    valid_labels = set()
    for cx, cy, radius in circular_centers:
        check_x = min(max(int(cx), 0), w-1)
        check_y = min(max(int(cy), 0), h-1)
        
        label_id = labels[check_y, check_x]
        
        # 如果圆心处是空洞，向外搜索最近的红色
        if label_id == 0:
            temp_mask = np.zeros((h, w), np.uint8)
            cv2.circle(temp_mask, (int(cx), int(cy)), int(radius * 1.5), 1, -1)
            intersect = cv2.bitwise_and(temp_mask, (labels > 0).astype(np.uint8))
            if np.sum(intersect) > 0:
                overlap_labels = labels[np.where((temp_mask > 0) & (labels > 0))]
                if len(overlap_labels) > 0:
                    unique, counts = np.unique(overlap_labels, return_counts=True)
                    label_id = unique[np.argmax(counts)]
                    valid_labels.add(label_id)
        else:
            valid_labels.add(label_id)
    
    # 生成最终 Mask
    for label_id in valid_labels:
        target_pad_mask[labels == label_id] = 255
    
    # 稍微膨胀，确保边缘的异常能被包住
    target_pad_mask = cv2.dilate(target_pad_mask, kernel, iterations=2)
    
    return target_pad_mask

def separate_regions_simple(blue_mask, min_separation_distance=5):
    """简化版区域分离，基于形态学操作和连通域分析"""
    dist_transform = cv2.distanceTransform(blue_mask, cv2.DIST_L2, 5)
    
    # Top-hat 提取局部峰值
    kernel = np.ones((min_separation_distance, min_separation_distance), np.uint8)
    local_max = cv2.morphologyEx(dist_transform, cv2.MORPH_TOPHAT, kernel)
    
    threshold = 0.3 * dist_transform.max()
    _, peaks = cv2.threshold(local_max, threshold, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    
    contours_peaks, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_peaks) <= 1:
        return blue_mask
    
    # 腐蚀分离粘连
    erosion_kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(blue_mask, erosion_kernel, iterations=1)
    
    num_labels, labels = cv2.connectedComponents(eroded)
    
    if num_labels <= 2:
        return blue_mask
    
    # 膨胀恢复尺寸
    separated_mask = np.zeros_like(blue_mask)
    for i in range(1, num_labels):
        component = (labels == i).astype(np.uint8) * 255
        component = cv2.dilate(component, erosion_kernel, iterations=1)
        separated_mask = cv2.bitwise_or(separated_mask, component)
    
    return separated_mask

def is_likely_interface_element(contour, img_shape, interface_mask):
    """基于位置、形状和界面区域重叠度判定是否为界面元素"""
    h, w = img_shape[:2]
    
    x, y, w_box, h_box = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(contour_mask, [contour], 255)
    
    overlap = cv2.bitwise_and(contour_mask, interface_mask)
    overlap_ratio = np.sum(overlap > 0) / np.sum(contour_mask > 0) if np.sum(contour_mask > 0) > 0 else 0
    
    aspect_ratio = w_box / h_box if h_box > 0 else 0
    is_at_edge = (x < w//10 or x + w_box > w*9//10 or y < h//10 or y + h_box > h*9//10)
    is_regular = 0.1 < aspect_ratio < 10.0
    has_interface_overlap = overlap_ratio > 0.3
    
    return has_interface_overlap or (is_at_edge and is_regular)

def extract_blue_regions_x_range(image_path, output_dir=None, debug=False):
    """提取蓝色异常区域并输出2.5倍宽度扩展后的X坐标范围"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    img_name = Path(image_path).stem
    h, w = img.shape[:2]
    
    interface_mask = detect_interface_regions(img)
    roi_mask = detect_image_content_roi(img)
    circular_contours, circular_centers = detect_large_circular_structures(img)
    
    # 生成目标焊盘掩膜（约束蓝色异常在焊盘范围内）
    target_pad_mask = get_target_pad_mask(img, circular_centers)
    
    # HSV空间更适合蓝色分割
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 双阈值覆盖标准蓝和亮蓝异常区域
    lower_blue1 = np.array([100, 80, 80])
    upper_blue1 = np.array([130, 255, 255])
    lower_blue2 = np.array([90, 60, 60])
    upper_blue2 = np.array([110, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    blue_mask = cv2.bitwise_or(mask1, mask2)
    
    # 关键约束：将蓝色异常限制在焊盘区域内，切断走线粘连
    blue_mask = cv2.bitwise_and(blue_mask, target_pad_mask)
    
    # 限定在有效ROI内并排除界面
    blue_mask = cv2.bitwise_and(blue_mask, roi_mask)
    blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(interface_mask))
    
    # 根据ROI占比自适应调整形态学内核
    roi_area = np.sum(roi_mask > 0)
    total_area = h * w
    roi_ratio = roi_area / total_area if total_area > 0 else 1.0
    
    base_kernel_size = max(1, int(3 * roi_ratio))
    kernel_small = np.ones((base_kernel_size, base_kernel_size), np.uint8)
    
    close_kernel_size = max(1, int(2 * roi_ratio)) if roi_ratio < 0.5 else 3
    kernel_close = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    
    # 开运算去噪，闭运算轻度连接碎片
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 分水岭分离粘连区域
    try:
        min_separation = max(3, int(5 * roi_ratio)) if roi_ratio < 0.5 else 8
        separated_mask = separate_merged_regions(blue_mask, min_separation_distance=min_separation)
        
        contours_before = len(cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        contours_after = len(cv2.findContours(separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        
        if contours_after > contours_before:
            blue_mask = separated_mask
            if debug:
                print(f"成功分离区域: {contours_before} -> {contours_after}")
        elif debug:
            print("分离未产生新区域，保持原始掩码")
            
    except Exception as e:
        if debug:
            print(f"区域分离失败，使用原始掩码: {e}")
    
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ROI占比越小，面积阈值越低
    base_min_area = max(50, int(100 * roi_ratio))
    min_area = base_min_area
    max_area = max(roi_area // 10, (w * h) // 50)
    
    if debug:
        print(f"ROI占比: {roi_ratio:.3f}")
        print(f"动态面积阈值: {min_area} - {max_area}")
        print(f"内核大小: open={base_kernel_size}, close={close_kernel_size}")
        print(f"找到蓝色轮廓数量: {len(contours)}")
        print(f"找到圆形结构数量: {len(circular_centers)}")
        if circular_centers:
            for i, (cx, cy, r) in enumerate(circular_centers):
                print(f"  结构{i}: 中心({cx}, {cy}), 半径={r:.1f}")
    
    valid_contours = []
    contours_near_structure = []
    candidate_pool = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w_box // 2, y + h_box // 2
        
        # 提取蓝色区域平均饱和度
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour], 255)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=contour_mask)
        saturation_values = masked_hsv[:, :, 1][contour_mask > 0]
        avg_saturation = np.mean(saturation_values) if len(saturation_values) > 0 else 0
        
        # 归一化到图像中心距离
        img_center_x, img_center_y = w // 2, h // 2
        dist_to_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
        max_dist = np.sqrt((w // 2)**2 + (h // 2)**2)
        normalized_center_dist = dist_to_center / max_dist if max_dist > 0 else 1.0
        
        # 计算到最近焊盘结构的距离比
        distance_to_structure = float('inf')
        if len(circular_centers) > 0:
            for struct_cx, struct_cy, radius in circular_centers:
                dist = np.sqrt((cx - struct_cx)**2 + (cy - struct_cy)**2)
                dist_ratio = dist / radius if radius > 0 else float('inf')
                distance_to_structure = min(distance_to_structure, dist_ratio)
        
        # 综合评分：面积30 + 饱和度25 + 中心位置15 + 紧凑度20 + 结构接近度10
        score = 0
        score += min(area / 1000, 30)
        score += min(avg_saturation / 255 * 25, 25)
        score += max(0, (1 - normalized_center_dist) * 15)
        score += min(compactness * 20, 20)
        if distance_to_structure != float('inf'):
            score += max(0, (2 - distance_to_structure) * 10)
        
        candidate_info = {
            'contour': contour,
            'area': area,
            'saturation': avg_saturation,
            'center': (cx, cy),
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'distance_to_structure': distance_to_structure,
            'score': score,
            'filter_reason': []
        }
        
        passed_filters = True
        
        if not (min_area <= area <= max_area):
            candidate_info['filter_reason'].append(f'area_out_of_range({area:.0f})')
            passed_filters = False
        
        if is_likely_interface_element(contour, (h, w), interface_mask):
            candidate_info['filter_reason'].append('interface_element')
            passed_filters = False
        
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            candidate_info['filter_reason'].append(f'aspect_ratio({aspect_ratio:.2f})')
            passed_filters = False
        
        if compactness < 0.1:
            candidate_info['filter_reason'].append(f'low_compactness({compactness:.2f})')
            passed_filters = False
        
        if passed_filters:
            valid_contours.append(contour)
            
            if len(circular_centers) > 0:
                is_near, distance_ratio = is_anomaly_near_structure(contour, circular_centers, (h, w))
                if is_near:
                    contours_near_structure.append((contour, distance_ratio))
                    if debug:
                        print(f"  轮廓 {len(valid_contours)-1} 靠近结构，距离比例: {distance_ratio:.2f}")
        
        candidate_pool.append(candidate_info)
    
    result = {
        'image_name': img_name,
        'anomalies': []
    }
    
    is_fallback = False
    is_default_region = False
    
    # 保底策略：无有效轮廓时启用候选池
    if len(valid_contours) == 0 and len(candidate_pool) > 0:
        is_fallback = True
        candidate_pool.sort(key=lambda x: x['score'], reverse=True)
        
        best_candidate = candidate_pool[0]
        contours_to_process = [best_candidate['contour']]
        
        if debug:
            print(f"  所有轮廓均被过滤，从候选池中选择最佳候选")
            print(f"  最佳候选评分: {best_candidate['score']:.2f}")
            print(f"  面积: {best_candidate['area']:.0f}")
            print(f"  饱和度: {best_candidate['saturation']:.1f}")
            print(f"  紧凑度: {best_candidate['compactness']:.2f}")
            print(f"  过滤原因: {', '.join(best_candidate['filter_reason'])}")
            
            print(f"  候选池排名 (前5名):")
            for i, cand in enumerate(candidate_pool[:5]):
                print(f"    {i+1}. 评分={cand['score']:.2f}, 面积={cand['area']:.0f}, "
                      f"饱和度={cand['saturation']:.1f}, 过滤原因={', '.join(cand['filter_reason']) if cand['filter_reason'] else '通过'}")
    
    elif len(valid_contours) == 0 and len(candidate_pool) == 0:
        # 兜底策略：完全未检测到蓝色时，在ROI中心输出默认区域
        is_fallback = True
        is_default_region = True
        
        if debug:
            print(f"未检测到任何蓝色区域，创建默认检测区域")
        
        roi_points = np.column_stack(np.where(roi_mask > 0))
        if len(roi_points) > 0:
            roi_center_y = int(np.mean(roi_points[:, 0]))
            roi_center_x = int(np.mean(roi_points[:, 1]))
        else:
            roi_center_x, roi_center_y = w // 2, h // 2
        
        default_width = w // 5
        default_height = h // 8
        
        default_x1 = max(0, roi_center_x - default_width // 2)
        default_y1 = max(0, roi_center_y - default_height // 2)
        default_x2 = min(w, roi_center_x + default_width // 2)
        default_y2 = min(h, roi_center_y + default_height // 2)
        
        default_contour = np.array([
            [[default_x1, default_y1]],
            [[default_x2, default_y1]],
            [[default_x2, default_y2]],
            [[default_x1, default_y2]]
        ], dtype=np.int32)
        
        contours_to_process = [default_contour]
        cv2.rectangle(blue_mask, (default_x1, default_y1), (default_x2, default_y2), 255, -1)
        
        if debug:
            print(f"  默认区域位置: 中心({roi_center_x}, {roi_center_y}), 大小: {default_width}x{default_height}")
    
    elif len(contours_near_structure) > 0:
        # 策略1：有焊盘结构时，优先选择相邻异常
        contours_near_structure.sort(key=lambda x: x[1])
        contours_to_process = [c[0] for c in contours_near_structure]
        if debug:
            print(f"策略1: 只保留靠近圆形结构的 {len(contours_to_process)} 个异常区域")
    elif len(valid_contours) > 1:
        # 策略2：多异常时基于结构亲和度选择（优先选择靠近焊盘的）
        contour_info = []
        for contour in valid_contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                area = cv2.contourArea(contour)
                
                # 计算结构亲和度得分
                affinity_score = 0
                distance_ratio = float('inf')
                if len(circular_centers) > 0:
                    is_near, distance_ratio = is_anomaly_near_structure(contour, circular_centers, (h, w))
                    if is_near:
                        affinity_score = 100 / (distance_ratio + 0.1)  # 距离越近分数越高
                
                # 综合评分：结构亲和度(60%) + 面积(40%)
                final_score = affinity_score * 0.6 + (area / 1000) * 0.4
                
                contour_info.append((contour, final_score, distance_ratio))
        
        if contour_info:
            contour_info.sort(key=lambda x: x[1], reverse=True)  # 按得分降序
            contours_to_process = [contour_info[0][0]]
            if debug:
                print(f"策略2: 选择结构亲和度最高的异常区域 (得分={contour_info[0][1]:.2f}, 距离比={contour_info[0][2]:.2f})")
        else:
            contours_to_process = valid_contours
    else:
        contours_to_process = valid_contours
        if debug:
            print(f"处理所有 {len(contours_to_process)} 个有效轮廓")
    
    # 提取异常区域的坐标并按方向自适应扩展
    for i, contour in enumerate(contours_to_process):
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # 计算异常区域重心
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w_box // 2, y + h_box // 2
        
        # --- 步骤1：找到最近焊盘并判断方向 ---
        nearest_pad = find_nearest_pad_center(contour, circular_centers)
        direction = determine_anomaly_direction(contour, nearest_pad)
        
        # --- 步骤2：根据方向自适应生成坐标 ---
        points = contour.reshape(-1, 2)
        min_x = int(np.min(points[:, 0]))
        max_x = int(np.max(points[:, 0]))
        min_y = int(np.min(points[:, 1]))
        max_y = int(np.max(points[:, 1]))
        
        if direction == 'horizontal':
            # === 横向扩展：上下侧异常 ===
            original_width = max_x - min_x
            expanded_width = int(original_width * 2.5)
            expansion = (expanded_width - original_width) // 2
            
            p1_x = max(0, min_x - expansion)
            p1_y = (min_y + max_y) // 2
            p2_x = min(w - 1, max_x + expansion)
            p2_y = (min_y + max_y) // 2
            
        else:  # direction == 'vertical'
            # === 纵向扩展：左右侧异常 ===
            original_height = max_y - min_y
            expanded_height = int(original_height * 2.5)
            expansion = (expanded_height - original_height) // 2
            
            p1_x = (min_x + max_x) // 2
            p1_y = max(0, min_y - expansion)
            p2_x = (min_x + max_x) // 2
            p2_y = min(h - 1, max_y + expansion)
        
        anomaly_info = {
            'id': i,
            'direction': direction,  # 新增：输出方向供设备参考
            'coor1': {'x': p1_x, 'y': p1_y},
            'coor2': {'x': p2_x, 'y': p2_y},
        }
        
        # 保留原有的置信度标记逻辑
        if is_default_region:
            anomaly_info['confidence'] = 'very_low'
            anomaly_info['type'] = 'default_region'
            anomaly_info['warning'] = '未检测到任何蓝色区域，此为默认输出'
        elif is_fallback:
            anomaly_info['confidence'] = 'low'
            anomaly_info['type'] = 'fallback'
        else:
            anomaly_info['confidence'] = 'high'
            anomaly_info['type'] = 'normal'
        
        # 保留debug_info逻辑
        if is_fallback and not is_default_region and len(candidate_pool) > 0:
            best_candidate = candidate_pool[0]
            anomaly_info['debug_info'] = {
                'score': round(best_candidate['score'], 2),
                'area': int(best_candidate['area']),
                'saturation': round(best_candidate['saturation'], 1),
                'compactness': round(best_candidate['compactness'], 2),
                'filter_reasons': best_candidate['filter_reason']
            }
        
        # 新增：记录焊盘参考信息（用于调试）
        if nearest_pad and debug:
            anomaly_info['debug_pad_reference'] = {
                'pad_center': {'x': nearest_pad[0], 'y': nearest_pad[1]},
                'anomaly_center': {'x': cx, 'y': cy}
            }
        
        result['anomalies'].append(anomaly_info)
    
    # 生成调试可视化图像
    if debug and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        debug_img = img.copy()
        
        # 绘制焊盘结构
        if len(circular_centers) > 0:
            for cx, cy, radius in circular_centers:
                cv2.circle(debug_img, (cx, cy), int(radius), (0, 255, 255), 2)
                cv2.circle(debug_img, (cx, cy), 3, (0, 255, 255), -1)
        
        # 按检测类型设置标记颜色
        if is_default_region:
            contour_color = (0, 0, 255)
            box_color = (255, 0, 255)
            label_prefix = 'DEFAULT'
        elif is_fallback:
            contour_color = (0, 165, 255)
            box_color = (0, 140, 255)
            label_prefix = 'FALLBACK'
        else:
            contour_color = (0, 255, 0)
            box_color = (255, 0, 0)
            label_prefix = 'ID'
        
        cv2.drawContours(debug_img, contours_to_process, -1, contour_color, 2)
        
        for i, contour in enumerate(contours_to_process):
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), box_color, 2)
            
            label = f'{label_prefix}:{i}'
            cv2.putText(debug_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if is_default_region:
            cv2.putText(debug_img, 'WARNING: DEFAULT REGION OUTPUT', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif is_fallback:
            cv2.putText(debug_img, 'WARNING: FALLBACK OUTPUT', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        debug_path = Path(output_dir) / f"{img_name}_contours_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        
        mask_path = Path(output_dir) / f"{img_name}_blue_mask.jpg"
        cv2.imwrite(str(mask_path), blue_mask)
        
        pad_mask_path = Path(output_dir) / f"{img_name}_pad_mask.jpg"
        cv2.imwrite(str(pad_mask_path), target_pad_mask)
        
        if debug:
            print(f"  调试图像已保存: {debug_path}")
            print(f"  蓝色掩码已保存: {mask_path}")
            print(f"  焊盘掩码已保存: {pad_mask_path}")
    
    return result

def batch_extract_x_ranges(input_dir, output_dir, debug=False):
    """批量处理图像目录，提取所有图像的蓝色异常区域坐标"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    all_results = []
    fallback_count = 0
    default_region_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        print(f"处理 {i}/{len(image_files)}: {img_file.name}")
        
        result = extract_blue_regions_x_range(img_file, output_path if debug else None, debug)
        
        if result:
            all_results.append(result)
            
            has_default = any(a.get('type') == 'default_region' for a in result['anomalies'])
            has_fallback = any(a.get('type') == 'fallback' for a in result['anomalies'])
            
            if has_default:
                default_region_count += 1
                print(f"  -> 找到 {len(result['anomalies'])} 个蓝色异常区域 [默认区域]")
            elif has_fallback:
                fallback_count += 1
                print(f"  -> 找到 {len(result['anomalies'])} 个蓝色异常区域 [保底输出]")
            else:
                print(f"  -> 找到 {len(result['anomalies'])} 个蓝色异常区域")
            
            single_result_file = output_path / f"{result['image_name']}_anomalies.json"
            with open(single_result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            print(f"  -> 处理失败")
    
    summary_file = output_path / "blue_anomalies_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n处理完成!")
    print(f"汇总结果保存到: {summary_file}")
    
    total_anomalies = sum(len(r['anomalies']) for r in all_results)
    normal_count = len(all_results) - fallback_count - default_region_count
    
    print(f"\n{'='*60}")
    print(f"统计信息:")
    print(f"{'='*60}")
    print(f"总共处理图像: {len(all_results)}")
    print(f"总共检测到异常区域: {total_anomalies}")
    print(f"\n检测质量分布:")
    print(f"正常检测: {normal_count}/{len(all_results)} ({normal_count/len(all_results)*100:.1f}%)")
    
    if fallback_count > 0:
        print(f"  保底输出: {fallback_count}/{len(all_results)} ({fallback_count/len(all_results)*100:.1f}%)")
        print(f"  (所有轮廓被过滤，从候选池选择最佳)")
    
    if default_region_count > 0:
        print(f"  默认区域: {default_region_count}/{len(all_results)} ({default_region_count/len(all_results)*100:.1f}%)")
        print(f"  完全未检测到蓝色，使用默认位置)")
    
    print(f"{'='*60}")
    
    if fallback_count > 0 or default_region_count > 0:
        print(f"\n 建议检查低质量输出图像：调整HSV范围、面积阈值或查看调试图像")
    
    if debug:
        print(f"\n调试图像保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='检测图像中的蓝色异常区域并输出坐标范围')
    parser.add_argument('input_dir', nargs='?', help='输入图像目录路径（批量处理时必需）')
    parser.add_argument('-o', '--output', default='./anomaly_results', 
                       help='输出目录路径 (默认: ./anomaly_results)')
    parser.add_argument('--debug', action='store_true', 
                       help='保存调试可视化图像')
    parser.add_argument('--single', help='处理单张图像（提供图像路径）')
    
    args = parser.parse_args()
    
    if args.single:
        result = extract_blue_regions_x_range(args.single, args.output, args.debug)
        if result:
            output_file = Path(args.output) / f"{result['image_name']}_anomalies.json"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"单张图像处理完成，结果保存到: {output_file}")
            print(f"找到 {len(result['anomalies'])} 个蓝色异常区域")
            
            for anomaly in result['anomalies']:
                if anomaly.get('type') == 'default_region':
                    confidence_marker = " [默认区域 - 未检测到任何蓝色]"
                elif anomaly.get('type') == 'fallback':
                    confidence_marker = " [保底输出]"
                else:
                    confidence_marker = ""
                    
                print(f"  区域 {anomaly['id']}: 方向={anomaly.get('direction', 'N/A')}, "
                      f"坐标范围 coor1({anomaly['coor1']['x']}, {anomaly['coor1']['y']}) -> "
                      f"coor2({anomaly['coor2']['x']}, {anomaly['coor2']['y']}){confidence_marker}")
                
                if 'warning' in anomaly:
                    print(f"    {anomaly['warning']}")
                
                if 'debug_info' in anomaly:
                    debug = anomaly['debug_info']
                    print(f"    评分: {debug['score']}, 面积: {debug['area']}, 饱和度: {debug['saturation']}, 紧凑度: {debug['compactness']}")
                    if debug['filter_reasons']:
                        print(f"    被过滤原因: {', '.join(debug['filter_reasons'])}")
        else:
            print("图像处理失败")
    else:
        if not args.input_dir:
            parser.error("批量处理模式需要提供 input_dir 参数")
        batch_extract_x_ranges(args.input_dir, args.output, args.debug)

if __name__ == "__main__":
    main() 