import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


DEBUG_MODE = False


@dataclass(frozen=True)
class DetectorConfig:
    """集中管理视觉检测参数，避免逻辑中散落硬编码。"""

    interface_edge_ratio: float = 0.05
    interface_canny_low: int = 50
    interface_canny_high: int = 150
    interface_min_area_pixels: int = 100
    interface_max_area_ratio: float = 0.25
    interface_aspect_min: float = 0.2
    interface_aspect_max: float = 5.0

    roi_border_ratio: float = 1.0 / 15.0
    roi_texture_threshold: int = 30
    roi_kernel_size: int = 15
    roi_dilate_iterations: int = 2

    structure_canny_low: int = 50
    structure_canny_high: int = 150
    structure_close_kernel_size: int = 3
    structure_close_iterations: int = 2
    structure_min_area_ratio: float = 0.005
    structure_max_area_ratio: float = 0.10
    structure_min_radius_ratio: float = 1.0 / 30.0
    structure_max_radius_ratio: float = 0.20
    structure_min_circularity: float = 0.30

    red_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = field(
        default_factory=lambda: (
            ((0, 43, 46), (10, 255, 255)),
            ((156, 43, 46), (180, 255, 255)),
        )
    )
    cyan_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = field(
        default_factory=lambda: (
            ((78, 50, 50), (98, 255, 255)),
            ((90, 40, 40), (110, 255, 255)),
        )
    )

    pad_close_kernel_size: int = 5
    pad_component_probe_radius_scale: float = 1.6
    pad_soft_dilate_kernel_size: int = 25
    pad_soft_dilate_iterations: int = 1

    cyan_open_kernel_size: int = 3
    cyan_close_kernel_size: int = 5
    cyan_open_iterations: int = 1
    cyan_close_iterations: int = 1
    separation_min_distance: int = 5

    contour_min_area_pixels: int = 40
    contour_min_area_ratio: float = 0.00005
    contour_max_area_ratio: float = 0.15
    contour_aspect_min: float = 0.08
    contour_aspect_max: float = 12.0
    min_reconstructed_radius_pixels: float = 3.0

    pad_affinity_radius_ratio: float = 0.90
    pad_affinity_min_pixels: float = 8.0
    structure_near_ratio: float = 1.2

    direction_anchor_gradient_kernel_size: int = 3
    direction_min_reference_distance_pixels: float = 6.0
    direction_min_anisotropy_pixels: float = 3.0

    operation_box_expand_ratio: float = 2.5
    operation_box_padding_pixels: int = 4
    minimum_box_side_pixels: int = 4

    fallback_width_ratio: float = 0.20
    fallback_height_ratio: float = 0.12
    fallback_min_side_pixels: int = 24

    debug_reconstructed_circle_color: tuple[int, int, int] = (0, 255, 255)
    debug_operation_box_color: tuple[int, int, int] = (0, 0, 255)
    debug_selected_contour_color: tuple[int, int, int] = (0, 255, 0)
    debug_fallback_color: tuple[int, int, int] = (0, 165, 255)


CONFIG = DetectorConfig()


def build_kernel(size: int) -> np.ndarray:
    kernel_size = max(1, int(size))
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def build_hsv_mask(hsv_img: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_img, lower_arr, upper_arr))
    return mask


def detect_interface_regions(img: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """检测并标记软件界面区域，用于后续过滤。"""
    h, w = img.shape[:2]
    interface_mask = np.zeros((h, w), dtype=np.uint8)

    edge_thickness = max(1, int(min(w, h) * config.interface_edge_ratio))
    interface_mask[:, :edge_thickness] = 255
    interface_mask[:, w - edge_thickness :] = 255
    interface_mask[:edge_thickness, :] = 255
    interface_mask[h - edge_thickness :, :] = 255

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, config.interface_canny_low, config.interface_canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = w_box * h_box
        aspect_ratio = w_box / h_box if h_box > 0 else 0.0

        is_edge_element = (
            x < edge_thickness
            or x + w_box > w - edge_thickness
            or y < edge_thickness
            or y + h_box > h - edge_thickness
        )
        is_regular_shape = config.interface_aspect_min < aspect_ratio < config.interface_aspect_max
        is_moderate_size = config.interface_min_area_pixels < area < int(w * h * config.interface_max_area_ratio)

        if is_edge_element and is_regular_shape and is_moderate_size:
            cv2.fillPoly(interface_mask, [contour], 255)

    return interface_mask


def detect_image_content_roi(img: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """提取图像主要内容区域（ROI），排除边缘界面干扰。"""
    h, w = img.shape[:2]
    roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    border_size = max(1, int(min(w, h) * config.roi_border_ratio))

    roi_mask[:border_size, :] = 0
    roi_mask[h - border_size :, :] = 0
    roi_mask[:, :border_size] = 0
    roi_mask[:, w - border_size :] = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, texture_mask = cv2.threshold(
        sobel_magnitude, config.roi_texture_threshold, 255, cv2.THRESH_BINARY
    )

    kernel = build_kernel(config.roi_kernel_size)
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
    texture_mask = cv2.dilate(texture_mask, kernel, iterations=config.roi_dilate_iterations)

    contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        content_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(content_mask, [largest_contour], 255)
        roi_mask = cv2.bitwise_and(roi_mask, content_mask)

    return roi_mask


def separate_regions_simple(blue_mask: np.ndarray, min_separation_distance: int) -> np.ndarray:
    """简化版区域分离，基于形态学操作和连通域分析。"""
    if np.count_nonzero(blue_mask) == 0:
        return blue_mask

    dist_transform = cv2.distanceTransform(blue_mask, cv2.DIST_L2, 5)
    kernel = build_kernel(min_separation_distance)
    local_max = cv2.morphologyEx(dist_transform, cv2.MORPH_TOPHAT, kernel)

    threshold = float(0.3 * dist_transform.max())
    _, peaks = cv2.threshold(local_max, threshold, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)

    contours_peaks, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_peaks) <= 1:
        return blue_mask

    erosion_kernel = build_kernel(2)
    eroded = cv2.erode(blue_mask, erosion_kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(eroded)

    if num_labels <= 2:
        return blue_mask

    separated_mask = np.zeros_like(blue_mask)
    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8) * 255
        component = cv2.dilate(component, erosion_kernel, iterations=1)
        separated_mask = cv2.bitwise_or(separated_mask, component)

    return separated_mask


def detect_large_circular_structures(img: np.ndarray, config: DetectorConfig):
    """基于边缘检测识别焊盘、过孔等大型圆形PCB结构。"""
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, config.structure_canny_low, config.structure_canny_high)
    close_kernel = build_kernel(config.structure_close_kernel_size)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, close_kernel, iterations=config.structure_close_iterations
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circular_contours = []
    circular_centers = []
    min_structure_area = w * h * config.structure_min_area_ratio
    max_structure_area = w * h * config.structure_max_area_ratio
    min_radius = min(w, h) * config.structure_min_radius_ratio
    max_radius = min(w, h) * config.structure_max_radius_ratio

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_structure_area <= area <= max_structure_area):
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (min_radius <= radius <= max_radius):
            continue

        circle_area = np.pi * radius * radius
        circularity = area / circle_area if circle_area > 0 else 0.0
        if circularity >= config.structure_min_circularity:
            circular_contours.append(contour)
            circular_centers.append((int(round(cx)), int(round(cy)), float(radius)))

    return circular_contours, circular_centers


def get_target_pad_mask(img: np.ndarray, circular_centers, config: DetectorConfig) -> np.ndarray:
    """
    生成只包含与圆形结构相连的红色铜区掩膜。
    该掩膜仅作为“空间亲和度参考”，不再与青色掩膜做硬相交。
    """
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = build_hsv_mask(img_hsv, config.red_hsv_ranges)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, build_kernel(config.pad_close_kernel_size))

    if not circular_centers:
        return red_mask

    target_pad_mask = np.zeros((h, w), dtype=np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    if num_labels <= 1:
        return red_mask

    valid_labels = set()
    for cx, cy, radius in circular_centers:
        check_x = int(np.clip(cx, 0, w - 1))
        check_y = int(np.clip(cy, 0, h - 1))
        label_id = labels[check_y, check_x]

        if label_id == 0:
            probe_mask = np.zeros((h, w), dtype=np.uint8)
            probe_radius = max(1, int(round(radius * config.pad_component_probe_radius_scale)))
            cv2.circle(probe_mask, (check_x, check_y), probe_radius, 255, -1)
            overlap_labels = labels[(probe_mask > 0) & (labels > 0)]
            if overlap_labels.size > 0:
                unique_labels, counts = np.unique(overlap_labels, return_counts=True)
                label_id = int(unique_labels[np.argmax(counts)])

        if label_id > 0:
            valid_labels.add(label_id)

    for label_id in valid_labels:
        target_pad_mask[labels == label_id] = 255

    return target_pad_mask


def build_soft_pad_mask(target_pad_mask: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """
    对焊盘掩膜做超大膨胀，主动弥合被青色 DRC 标记遮挡形成的“黑洞”。
    这里不做硬裁剪，只把它作为后续空间距离约束参考。
    """
    if np.count_nonzero(target_pad_mask) == 0:
        return target_pad_mask.copy()
    dilate_kernel = build_kernel(config.pad_soft_dilate_kernel_size)
    return cv2.dilate(target_pad_mask, dilate_kernel, iterations=config.pad_soft_dilate_iterations)


def build_direction_anchor_mask(target_pad_mask: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """
    从红铜掩膜提取边界锚点，方向判断优先参考“最近铜边界”而不是焊盘圆心。
    这样即使圆形结构检测失败，也能利用真实铜面边缘提供稳定方向基准。
    """
    if np.count_nonzero(target_pad_mask) == 0:
        return target_pad_mask.copy()

    gradient_kernel = build_kernel(config.direction_anchor_gradient_kernel_size)
    anchor_mask = cv2.morphologyEx(target_pad_mask, cv2.MORPH_GRADIENT, gradient_kernel)
    if np.count_nonzero(anchor_mask) == 0:
        return target_pad_mask.copy()
    return anchor_mask


def is_likely_interface_element(contour: np.ndarray, img_shape, interface_mask: np.ndarray) -> bool:
    """基于位置、形状和界面区域重叠度判定是否为界面元素。"""
    h, w = img_shape[:2]

    x, y, w_box, h_box = cv2.boundingRect(contour)
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(contour_mask, [contour], 255)

    contour_pixels = np.count_nonzero(contour_mask)
    overlap = cv2.bitwise_and(contour_mask, interface_mask)
    overlap_ratio = np.count_nonzero(overlap) / contour_pixels if contour_pixels > 0 else 0.0

    aspect_ratio = w_box / h_box if h_box > 0 else 0.0
    is_at_edge = x < w // 10 or x + w_box > w * 9 // 10 or y < h // 10 or y + h_box > h * 9 // 10
    is_regular = 0.1 < aspect_ratio < 10.0

    return overlap_ratio > 0.3 or (is_at_edge and is_regular)


def find_nearest_pad_center(point, circular_centers):
    """找到距离目标点最近的焊盘中心。"""
    if not circular_centers:
        return None

    px, py = point
    best_center = None
    best_distance = float("inf")
    for cx, cy, _ in circular_centers:
        distance = float(np.hypot(px - cx, py - cy))
        if distance < best_distance:
            best_distance = distance
            best_center = (cx, cy)
    return best_center


def find_nearest_mask_point(point, mask_points: np.ndarray):
    """在预提取的掩膜像素集合中查找最近锚点。"""
    if mask_points.size == 0:
        return None, float("inf")

    point_xy = np.array([point[0], point[1]], dtype=np.float32)
    points_xy = mask_points[:, ::-1].astype(np.float32)
    deltas = points_xy - point_xy
    distances_sq = np.sum(deltas * deltas, axis=1)
    nearest_index = int(np.argmin(distances_sq))
    nearest_point = points_xy[nearest_index]
    return (int(nearest_point[0]), int(nearest_point[1])), float(np.sqrt(distances_sq[nearest_index]))


def determine_anomaly_direction(
    candidate: dict,
    direction_anchor_point=None,
    pad_center=None,
    config: DetectorConfig = CONFIG,
):
    """
    优先使用最近铜边界锚点，避免圆形结构缺失时全部回退到固定方向。次优使用焊盘圆心。若外部参考无效，则退化到几何形状推断。
    """
    anomaly_cx, anomaly_cy = candidate["circle_center"]

    if direction_anchor_point is not None:
        ref_cx, ref_cy = direction_anchor_point
        dx = anomaly_cx - ref_cx
        dy = anomaly_cy - ref_cy
        reference_distance = float(np.hypot(dx, dy))

        if reference_distance >= config.direction_min_reference_distance_pixels:
            if abs(dx) >= abs(dy):
                return "vertical", "pad_boundary", direction_anchor_point
            return "horizontal", "pad_boundary", direction_anchor_point

    if pad_center is not None:
        pad_cx, pad_cy = pad_center
        dx = anomaly_cx - pad_cx
        dy = anomaly_cy - pad_cy
        reference_distance = float(np.hypot(dx, dy))

        if reference_distance >= config.direction_min_reference_distance_pixels:
            if abs(dx) >= abs(dy):
                return "vertical", "pad_center", pad_center
            return "horizontal", "pad_center", pad_center

    min_x, max_x, min_y, max_y = candidate["contour_bounds"]
    contour_width = max_x - min_x
    contour_height = max_y - min_y
    width_height_delta = contour_width - contour_height

    if abs(width_height_delta) >= config.direction_min_anisotropy_pixels:
        if width_height_delta >= 0:
            return "horizontal", "contour_aspect", None
        return "vertical", "contour_aspect", None

    # 当轮廓接近方形时，观察重构圆心到边界的有效跨度，选择占用更大的轴作为主扩展轴。
    left_span = anomaly_cx - min_x
    right_span = max_x - anomaly_cx
    top_span = anomaly_cy - min_y
    bottom_span = max_y - anomaly_cy
    horizontal_span = max(left_span, right_span)
    vertical_span = max(top_span, bottom_span)

    if horizontal_span >= vertical_span:
        return "horizontal", "reconstructed_extent", None
    return "vertical", "reconstructed_extent", None


def contour_bounds(contour: np.ndarray):
    points = contour.reshape(-1, 2)
    min_x = int(np.min(points[:, 0]))
    max_x = int(np.max(points[:, 0]))
    min_y = int(np.min(points[:, 1]))
    max_y = int(np.max(points[:, 1]))
    return min_x, max_x, min_y, max_y


def clamp_rectangle(x1, y1, x2, y2, img_shape, config: DetectorConfig):
    h, w = img_shape[:2]

    min_side = max(1, config.minimum_box_side_pixels)
    x1 = int(np.clip(np.floor(x1), 0, w - 1))
    x2 = int(np.clip(np.ceil(x2), 0, w - 1))
    y1 = int(np.clip(np.floor(y1), 0, h - 1))
    y2 = int(np.clip(np.ceil(y2), 0, h - 1))

    if x2 - x1 < min_side:
        center_x = (x1 + x2) / 2.0
        half = min_side / 2.0
        x1 = int(np.clip(np.floor(center_x - half), 0, w - 1))
        x2 = int(np.clip(np.ceil(center_x + half), 0, w - 1))
    if y2 - y1 < min_side:
        center_y = (y1 + y2) / 2.0
        half = min_side / 2.0
        y1 = int(np.clip(np.floor(center_y - half), 0, h - 1))
        y2 = int(np.clip(np.ceil(center_y + half), 0, h - 1))

    if x2 <= x1:
        x2 = min(w - 1, x1 + min_side)
        x1 = max(0, x2 - min_side)
    if y2 <= y1:
        y2 = min(h - 1, y1 + min_side)
        y1 = max(0, y2 - min_side)

    return x1, y1, x2, y2


def build_operation_rectangle(candidate: dict, direction: str, img_shape, config: DetectorConfig):
    """
    使用“重构圆 + 正交轴边界继承机制”生成最终操作框。
    主轴按最小外接圆直径做 2.5 倍外推，副轴继承原始残缺轮廓边界并追加 padding，
    从而既修复被黑色间距切断的主尺寸失真，也避免坐标塌成 1D 线段。
    """
    cx, cy = candidate["circle_center"]
    radius = candidate["circle_radius"]
    min_x, max_x, min_y, max_y = candidate["contour_bounds"]

    native_diameter = max(2.0 * radius, float(config.minimum_box_side_pixels))
    expanded_diameter = native_diameter * config.operation_box_expand_ratio
    half_major_axis = expanded_diameter / 2.0
    padding = config.operation_box_padding_pixels

    if direction == "horizontal":
        x1 = cx - half_major_axis
        x2 = cx + half_major_axis
        y1 = min_y - padding
        y2 = max_y + padding
    else:
        x1 = min_x - padding
        x2 = max_x + padding
        y1 = cy - half_major_axis
        y2 = cy + half_major_axis

    return clamp_rectangle(x1, y1, x2, y2, img_shape, config)


def build_default_candidate(roi_mask: np.ndarray, img_shape, config: DetectorConfig) -> dict:
    """
    完全未检测到候选目标时，返回具备明确 2D 面积的保底矩形，防止后续设备 API 因畸形坐标崩溃。
    """
    h, w = img_shape[:2]
    roi_points = np.column_stack(np.where(roi_mask > 0))
    if roi_points.size > 0:
        center_y = int(np.mean(roi_points[:, 0]))
        center_x = int(np.mean(roi_points[:, 1]))
    else:
        center_x = w // 2
        center_y = h // 2

    width = max(config.fallback_min_side_pixels, int(w * config.fallback_width_ratio))
    height = max(config.fallback_min_side_pixels, int(h * config.fallback_height_ratio))
    x1, y1, x2, y2 = clamp_rectangle(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
        img_shape,
        config,
    )

    fallback_contour = np.array(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
        dtype=np.int32,
    )
    fallback_radius = float(max((x2 - x1) / 2.0, (y2 - y1) / 2.0))

    return {
        "contour": fallback_contour,
        "area": float((x2 - x1) * (y2 - y1)),
        "circle_center": (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
        "circle_radius": fallback_radius,
        "contour_bounds": (x1, x2, y1, y2),
        "pad_distance": float("inf"),
        "distance_to_structure": float("inf"),
        "aspect_ratio": (x2 - x1) / max(1, (y2 - y1)),
        "avg_saturation": 0.0,
        "filter_reason": ["default_region"],
        "pad_affinity_passed": False,
        "near_structure": False,
        "score": 0.0,
    }


def build_candidate_rank(candidate: dict, inf_fallback: float) -> tuple:
    pad_distance = candidate["pad_distance"]
    structure_distance = candidate["distance_to_structure"]
    return (
        pad_distance if np.isfinite(pad_distance) else inf_fallback,
        structure_distance if np.isfinite(structure_distance) else inf_fallback,
        -candidate["circle_radius"],
        -candidate["area"],
    )


def save_debug_mask(output_dir: Path, img_name: str, suffix: str, mask: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{img_name}_{suffix}.jpg"), mask)


def build_exception_fallback_result(
    image_path, output_dir, debug: bool, config: DetectorConfig, error: Exception
):
    """当主流程异常时，返回具备 2D 面积的应急结果，避免自动化链路中断。"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"异常兜底失败，无法重新读取图像: {image_path}")
        return None

    img_name = Path(image_path).stem
    full_roi_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    candidate = build_default_candidate(full_roi_mask, img.shape, config)
    x1, x2, y1, y2 = candidate["contour_bounds"]

    result = {
        "image_name": img_name,
        "anomalies": [
            {
                "id": 0,
                "direction": "horizontal",
                "coor1": {"x": x1, "y": y1},
                "coor2": {"x": x2, "y": y2},
                "reconstructed_circle": {
                    "center": {"x": candidate["circle_center"][0], "y": candidate["circle_center"][1]},
                    "radius": round(candidate["circle_radius"], 2),
                },
                "confidence": "very_low",
                "type": "fallback",
                "warning": f"检测流程异常，已输出应急矩形: {error}",
                "debug_info": {"filter_reasons": ["pipeline_exception"], "exception": str(error)},
            }
        ],
    }

    if debug and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exception_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(exception_mask, [candidate["contour"]], 255)
        save_debug_mask(output_path, img_name, "pad_mask_filled", exception_mask)
        save_debug_mask(output_path, img_name, "blue_mask_filtered", exception_mask)

        debug_img = img.copy()
        cv2.circle(
            debug_img,
            candidate["circle_center"],
            max(1, int(round(candidate["circle_radius"]))),
            config.debug_reconstructed_circle_color,
            2,
        )
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), config.debug_fallback_color, 2)
        cv2.putText(
            debug_img,
            "EXCEPTION_FALLBACK",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            config.debug_fallback_color,
            2,
        )
        cv2.imwrite(str(output_path / f"{img_name}_contours_debug.jpg"), debug_img)

    return result


def extract_blue_regions_x_range(image_path, output_dir=None, debug=False, config: DetectorConfig = CONFIG):
    try:
        return _extract_blue_regions_x_range_impl(image_path, output_dir, debug, config)
    except Exception as error:
        print(f"处理图像时发生异常，启用应急兜底: {error}")
        return build_exception_fallback_result(
            image_path=image_path,
            output_dir=output_dir,
            debug=bool(debug or DEBUG_MODE),
            config=config,
            error=error,
        )


def _extract_blue_regions_x_range_impl(image_path, output_dir=None, debug=False, config: DetectorConfig = CONFIG):
    """提取 Genesis DRC 青色异常区域，并输出稳定的 2D 操作框坐标。"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    img_name = Path(image_path).stem
    h, w = img.shape[:2]
    debug_enabled = bool(debug or DEBUG_MODE)

    interface_mask = detect_interface_regions(img, config)
    roi_mask = detect_image_content_roi(img, config)
    _, circular_centers = detect_large_circular_structures(img, config)

    target_pad_mask = get_target_pad_mask(img, circular_centers, config)
    soft_pad_mask = build_soft_pad_mask(target_pad_mask, config)
    direction_anchor_mask = build_direction_anchor_mask(target_pad_mask, config)
    direction_anchor_points = np.column_stack(np.where(direction_anchor_mask > 0))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cyan_mask = build_hsv_mask(hsv, config.cyan_hsv_ranges)
    cyan_mask = cv2.bitwise_and(cyan_mask, roi_mask)
    cyan_mask = cv2.bitwise_and(cyan_mask, cv2.bitwise_not(interface_mask))

    cyan_mask = cv2.morphologyEx(
        cyan_mask,
        cv2.MORPH_OPEN,
        build_kernel(config.cyan_open_kernel_size),
        iterations=config.cyan_open_iterations,
    )
    cyan_mask = cv2.morphologyEx(
        cyan_mask,
        cv2.MORPH_CLOSE,
        build_kernel(config.cyan_close_kernel_size),
        iterations=config.cyan_close_iterations,
    )
    cyan_mask = separate_regions_simple(cyan_mask, config.separation_min_distance)

    if np.count_nonzero(soft_pad_mask) > 0:
        pad_distance_map = cv2.distanceTransform(cv2.bitwise_not(soft_pad_mask), cv2.DIST_L2, 5)
    else:
        pad_distance_map = np.full((h, w), np.inf, dtype=np.float32)

    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = max(1, int(np.count_nonzero(roi_mask)))
    min_area = max(config.contour_min_area_pixels, int(roi_area * config.contour_min_area_ratio))
    max_area = max(min_area + 1, int(roi_area * config.contour_max_area_ratio))

    if debug_enabled:
        print(f"ROI像素面积: {roi_area}")
        print(f"候选面积阈值: {min_area} - {max_area}")
        print(f"找到青色轮廓数量: {len(contours)}")
        print(f"找到圆形焊盘数量: {len(circular_centers)}")

    valid_candidates = []
    near_structure_candidates = []
    candidate_pool = []
    affinity_filtered_blue_mask = np.zeros_like(cyan_mask)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = w_box / h_box if h_box > 0 else 0.0

        if area <= 0:
            continue

        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour], 255)
        saturation_values = hsv[:, :, 1][contour_mask > 0]
        avg_saturation = float(np.mean(saturation_values)) if saturation_values.size > 0 else 0.0

        # 此处使用最小外接圆重构被间距切割的残缺 DRC 标记，以获取真实物理圆心。
        # 后续距离计算、方向判定和主轴扩展全部以该理想圆为准，避免残缺像素块造成中心偏移。
        (circle_cx, circle_cy), circle_radius = cv2.minEnclosingCircle(contour)
        circle_radius = max(float(circle_radius), config.min_reconstructed_radius_pixels)
        center_x = int(np.clip(round(circle_cx), 0, w - 1))
        center_y = int(np.clip(round(circle_cy), 0, h - 1))

        pad_distance = float(pad_distance_map[center_y, center_x]) if np.isfinite(pad_distance_map[center_y, center_x]) else float("inf")
        pad_distance_limit = max(config.pad_affinity_min_pixels, circle_radius * config.pad_affinity_radius_ratio)
        pad_affinity_passed = pad_distance <= pad_distance_limit

        if pad_affinity_passed:
            cv2.fillPoly(affinity_filtered_blue_mask, [contour], 255)

        nearest_pad = find_nearest_pad_center((center_x, center_y), circular_centers)
        if nearest_pad is None:
            distance_to_structure = float("inf")
            near_structure = False
        else:
            structure_distance_pixels = float(np.hypot(center_x - nearest_pad[0], center_y - nearest_pad[1]))
            nearest_radius = next(
                radius
                for cx, cy, radius in circular_centers
                if cx == nearest_pad[0] and cy == nearest_pad[1]
            )
            distance_to_structure = (
                structure_distance_pixels / nearest_radius if nearest_radius > 0 else float("inf")
            )
            near_structure = distance_to_structure <= config.structure_near_ratio

        candidate = {
            "contour": contour,
            "area": area,
            "circle_center": (center_x, center_y),
            "circle_radius": circle_radius,
            "contour_bounds": contour_bounds(contour),
            "pad_distance": pad_distance,
            "distance_to_structure": distance_to_structure,
            "aspect_ratio": aspect_ratio,
            "avg_saturation": avg_saturation,
            "filter_reason": [],
            "pad_affinity_passed": pad_affinity_passed,
            "near_structure": near_structure,
            "score": area,
        }

        if not (min_area <= area <= max_area):
            candidate["filter_reason"].append(f"area_out_of_range({area:.0f})")
        if not (config.contour_aspect_min <= aspect_ratio <= config.contour_aspect_max):
            candidate["filter_reason"].append(f"aspect_ratio({aspect_ratio:.2f})")
        if is_likely_interface_element(contour, img.shape, interface_mask):
            candidate["filter_reason"].append("interface_element")
        if not pad_affinity_passed:
            candidate["filter_reason"].append(
                f"pad_affinity(distance={pad_distance:.2f}, limit={pad_distance_limit:.2f})"
            )

        passed_filters = len(candidate["filter_reason"]) == 0
        if passed_filters:
            valid_candidates.append(candidate)
            if near_structure:
                near_structure_candidates.append(candidate)

        candidate_pool.append(candidate)

        if debug_enabled:
            print(
                "  候选: "
                f"center=({center_x}, {center_y}), "
                f"r={circle_radius:.1f}, "
                f"area={area:.0f}, "
                f"pad_dist={pad_distance:.2f}, "
                f"struct_ratio={distance_to_structure:.2f}"
            )

    result = {"image_name": img_name, "anomalies": []}
    is_fallback = False
    is_default_region = False

    if not valid_candidates and candidate_pool:
        is_fallback = True
        fallback_sort_inf = float(max(h, w) * config.operation_box_expand_ratio)
        candidate_pool.sort(key=lambda item: build_candidate_rank(item, fallback_sort_inf))
        selected_candidates = [candidate_pool[0]]

        if debug_enabled:
            print("所有候选均未完全通过过滤，启用 fallback 最优候选。")
            print(f"  过滤原因: {', '.join(selected_candidates[0]['filter_reason'])}")
    elif not valid_candidates:
        is_fallback = True
        is_default_region = True
        selected_candidates = [build_default_candidate(roi_mask, img.shape, config)]
        cv2.fillPoly(cyan_mask, [selected_candidates[0]["contour"]], 255)
        cv2.fillPoly(affinity_filtered_blue_mask, [selected_candidates[0]["contour"]], 255)

        if debug_enabled:
            print("未检测到任何青色候选，输出具备 2D 面积的默认矩形。")
    elif near_structure_candidates:
        fallback_sort_inf = float(max(h, w) * config.operation_box_expand_ratio)
        near_structure_candidates.sort(key=lambda item: build_candidate_rank(item, fallback_sort_inf))
        selected_candidates = near_structure_candidates
        if debug_enabled:
            print(f"优先输出靠近焊盘结构的 {len(selected_candidates)} 个候选。")
    elif len(valid_candidates) > 1:
        fallback_sort_inf = float(max(h, w) * config.operation_box_expand_ratio)
        valid_candidates.sort(key=lambda item: build_candidate_rank(item, fallback_sort_inf))
        selected_candidates = [valid_candidates[0]]
        if debug_enabled:
            print("存在多个有效候选，输出空间亲和度最高的单一目标。")
    else:
        selected_candidates = valid_candidates
        if debug_enabled:
            print("输出单个有效候选。")

    for index, candidate in enumerate(selected_candidates):
        nearest_anchor_point, nearest_anchor_distance = find_nearest_mask_point(
            candidate["circle_center"], direction_anchor_points
        )
        nearest_pad = find_nearest_pad_center(candidate["circle_center"], circular_centers)
        direction, direction_source, effective_reference_point = determine_anomaly_direction(
            candidate=candidate,
            direction_anchor_point=nearest_anchor_point,
            pad_center=nearest_pad,
            config=config,
        )
        x1, y1, x2, y2 = build_operation_rectangle(candidate, direction, img.shape, config)

        anomaly_info = {
            "id": index,
            "direction": direction,
            "coor1": {"x": x1, "y": y1},
            "coor2": {"x": x2, "y": y2},
            "reconstructed_circle": {
                "center": {"x": candidate["circle_center"][0], "y": candidate["circle_center"][1]},
                "radius": round(candidate["circle_radius"], 2),
            },
        }

        if is_default_region:
            anomaly_info["confidence"] = "very_low"
            anomaly_info["type"] = "default_region"
            anomaly_info["warning"] = "未检测到任何青色区域，此为具备 2D 面积的默认输出"
        elif is_fallback:
            anomaly_info["confidence"] = "low"
            anomaly_info["type"] = "fallback"
        else:
            anomaly_info["confidence"] = "high"
            anomaly_info["type"] = "normal"

        if is_fallback and not is_default_region:
            anomaly_info["debug_info"] = {
                "area": int(candidate["area"]),
                "pad_distance": round(candidate["pad_distance"], 2)
                if np.isfinite(candidate["pad_distance"])
                else "inf",
                "structure_distance_ratio": round(candidate["distance_to_structure"], 2)
                if np.isfinite(candidate["distance_to_structure"])
                else "inf",
                "filter_reasons": candidate["filter_reason"],
            }

        if debug_enabled:
            debug_reference = {
                "direction_source": direction_source,
                "anomaly_center": {
                    "x": candidate["circle_center"][0],
                    "y": candidate["circle_center"][1],
                },
                "anchor_distance": round(nearest_anchor_distance, 2)
                if np.isfinite(nearest_anchor_distance)
                else "inf",
                "pad_boundary_point": (
                    {"x": nearest_anchor_point[0], "y": nearest_anchor_point[1]}
                    if nearest_anchor_point
                    else None
                ),
                "pad_center": (
                    {"x": nearest_pad[0], "y": nearest_pad[1]}
                    if nearest_pad
                    else None
                ),
                "effective_reference_point": (
                    {"x": effective_reference_point[0], "y": effective_reference_point[1]}
                    if effective_reference_point
                    else None
                ),
            }
            anomaly_info["debug_pad_reference"] = debug_reference

        result["anomalies"].append(anomaly_info)

    if debug_enabled and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_debug_mask(output_path, img_name, "pad_mask_filled", soft_pad_mask)
        save_debug_mask(output_path, img_name, "blue_mask_filtered", affinity_filtered_blue_mask)
        save_debug_mask(output_path, img_name, "direction_anchor_mask", direction_anchor_mask)

        debug_img = img.copy()

        if circular_centers:
            for pad_x, pad_y, radius in circular_centers:
                cv2.circle(debug_img, (pad_x, pad_y), int(round(radius)), (255, 255, 0), 2)
                cv2.circle(debug_img, (pad_x, pad_y), 3, (255, 255, 0), -1)

        for index, candidate in enumerate(selected_candidates):
            draw_color = (
                config.debug_fallback_color
                if is_fallback
                else config.debug_selected_contour_color
            )
            cv2.drawContours(debug_img, [candidate["contour"]], -1, draw_color, 2)

            circle_center = candidate["circle_center"]
            circle_radius = max(1, int(round(candidate["circle_radius"])))
            cv2.circle(
                debug_img,
                circle_center,
                circle_radius,
                config.debug_reconstructed_circle_color,
                2,
            )
            cv2.circle(debug_img, circle_center, 3, config.debug_reconstructed_circle_color, -1)

            anomaly = result["anomalies"][index]
            top_left = (anomaly["coor1"]["x"], anomaly["coor1"]["y"])
            bottom_right = (anomaly["coor2"]["x"], anomaly["coor2"]["y"])
            cv2.rectangle(debug_img, top_left, bottom_right, config.debug_operation_box_color, 2)

            label_prefix = "DEFAULT" if is_default_region else ("FALLBACK" if is_fallback else "ID")
            cv2.putText(
                debug_img,
                f"{label_prefix}:{index}",
                (top_left[0], max(20, top_left[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                config.debug_operation_box_color,
                2,
            )

        cv2.imwrite(str(output_path / f"{img_name}_contours_debug.jpg"), debug_img)

        if debug_enabled:
            print(f"调试图像已保存到: {output_path}")

    return result


def batch_extract_x_ranges(input_dir, output_dir, debug=False, config: DetectorConfig = CONFIG):
    """批量处理图像目录，提取所有图像的青色异常区域坐标。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    all_results = []
    fallback_count = 0
    default_region_count = 0

    for index, img_file in enumerate(sorted(image_files), 1):
        print(f"处理 {index}/{len(image_files)}: {img_file.name}")
        result = extract_blue_regions_x_range(img_file, output_path if debug else None, debug, config)

        if result is None:
            print("  -> 处理失败")
            continue

        all_results.append(result)

        has_default = any(anomaly.get("type") == "default_region" for anomaly in result["anomalies"])
        has_fallback = any(anomaly.get("type") == "fallback" for anomaly in result["anomalies"])

        if has_default:
            default_region_count += 1
            print(f"  -> 找到 {len(result['anomalies'])} 个异常区域 [默认区域]")
        elif has_fallback:
            fallback_count += 1
            print(f"  -> 找到 {len(result['anomalies'])} 个异常区域 [保底输出]")
        else:
            print(f"  -> 找到 {len(result['anomalies'])} 个异常区域")

        single_result_file = output_path / f"{result['image_name']}_anomalies.json"
        with open(single_result_file, "w", encoding="utf-8") as file_obj:
            json.dump(result, file_obj, indent=2, ensure_ascii=False)

    summary_file = output_path / "blue_anomalies_summary.json"
    with open(summary_file, "w", encoding="utf-8") as file_obj:
        json.dump(all_results, file_obj, indent=2, ensure_ascii=False)

    print("\n处理完成!")
    print(f"汇总结果保存到: {summary_file}")

    total_anomalies = sum(len(item["anomalies"]) for item in all_results)
    normal_count = len(all_results) - fallback_count - default_region_count

    print(f"\n{'=' * 60}")
    print("统计信息:")
    print(f"{'=' * 60}")
    print(f"总共处理图像: {len(all_results)}")
    print(f"总共检测到异常区域: {total_anomalies}")
    print("\n检测质量分布:")

    if all_results:
        total_images = len(all_results)
        print(f"正常检测: {normal_count}/{total_images} ({normal_count / total_images * 100:.1f}%)")
        if fallback_count > 0:
            print(f"保底输出: {fallback_count}/{total_images} ({fallback_count / total_images * 100:.1f}%)")
        if default_region_count > 0:
            print(
                f"默认区域: {default_region_count}/{total_images} "
                f"({default_region_count / total_images * 100:.1f}%)"
            )
    else:
        print("没有成功处理的图像。")

    print(f"{'=' * 60}")
    if debug:
        print(f"\n调试图像保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="检测图像中的 Genesis DRC 青色异常区域并输出稳定操作框")
    parser.add_argument("input_dir", nargs="?", help="输入图像目录路径（批量处理时必需）")
    parser.add_argument(
        "-o",
        "--output",
        default="./anomaly_results",
        help="输出目录路径 (默认: ./anomaly_results)",
    )
    parser.add_argument("--debug", action="store_true", help="保存调试可视化图像")
    parser.add_argument("--single", help="处理单张图像（提供图像路径）")

    args = parser.parse_args()

    if args.single:
        result = extract_blue_regions_x_range(args.single, args.output, args.debug, CONFIG)
        if result is None:
            print("图像处理失败")
            return

        output_file = Path(args.output) / f"{result['image_name']}_anomalies.json"
        Path(args.output).mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file_obj:
            json.dump(result, file_obj, indent=2, ensure_ascii=False)

        print(f"单张图像处理完成，结果保存到: {output_file}")
        print(f"找到 {len(result['anomalies'])} 个异常区域")

        for anomaly in result["anomalies"]:
            if anomaly.get("type") == "default_region":
                confidence_marker = " [默认区域 - 未检测到任何青色]"
            elif anomaly.get("type") == "fallback":
                confidence_marker = " [保底输出]"
            else:
                confidence_marker = ""

            print(
                f"  区域 {anomaly['id']}: "
                f"方向={anomaly.get('direction', 'N/A')}, "
                f"coor1({anomaly['coor1']['x']}, {anomaly['coor1']['y']}) -> "
                f"coor2({anomaly['coor2']['x']}, {anomaly['coor2']['y']})"
                f"{confidence_marker}"
            )

            if "warning" in anomaly:
                print(f"    {anomaly['warning']}")
            if "debug_info" in anomaly:
                debug_info = anomaly["debug_info"]
                print(
                    f"    面积: {debug_info['area']}, "
                    f"pad距离: {debug_info['pad_distance']}, "
                    f"结构距离比: {debug_info['structure_distance_ratio']}"
                )
                if debug_info["filter_reasons"]:
                    print(f"    被过滤原因: {', '.join(debug_info['filter_reasons'])}")
    else:
        if not args.input_dir:
            parser.error("批量处理模式需要提供 input_dir 参数")
        batch_extract_x_ranges(args.input_dir, args.output, args.debug, CONFIG)


if __name__ == "__main__":
    main()
