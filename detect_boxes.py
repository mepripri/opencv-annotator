# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
import sys
import os

def detect_columns(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vertical_projection = np.sum(binary, axis=0)
    vertical_projection_norm = vertical_projection / np.max(vertical_projection) if np.max(vertical_projection) > 0 else vertical_projection

    window_size = max(10, w // 100)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(vertical_projection_norm, kernel, mode='same')

    local_minima = [i for i in range(1, len(smoothed)-1) if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]]

    margin = w // 20
    filtered_separators = []
    for x in local_minima:
        if x < margin or x > w - margin:
            continue
        left_region = smoothed[max(0, x-window_size):x]
        right_region = smoothed[x:min(len(smoothed), x+window_size)]
        if len(left_region) > 0 and len(right_region) > 0:
            left_max = np.max(left_region)
            right_max = np.max(right_region)
            if smoothed[x] < 0.7*left_max and smoothed[x] < 0.7*right_max:
                filtered_separators.append(x)

    min_distance = w // 10
    column_separators = []
    for x in filtered_separators:
        if not column_separators or x - column_separators[-1] >= min_distance:
            column_separators.append(x)

    boundaries = [0] + column_separators + [w]
    columns = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    return columns

def get_text_bounding_box(region_img):
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if len(region_img.shape) == 3 else region_img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) == 0:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, x + w, y + h)

def detect_text_lines_in_column(column_img, min_text_density=0.02):
    h, w = column_img.shape[:2]
    gray = cv2.cvtColor(column_img, cv2.COLOR_BGR2GRAY) if len(column_img.shape) == 3 else column_img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_projection = np.sum(binary, axis=1)
    h_window = max(3, h // 150)
    h_smoothed = np.convolve(horizontal_projection, np.ones(h_window)/h_window, mode='same')
    h_norm = h_smoothed / np.max(h_smoothed) if np.max(h_smoothed) > 0 else h_smoothed

    threshold = 0.15
    text_regions = h_norm > threshold
    transitions = np.diff(text_regions.astype(int))
    line_starts = np.where(transitions == 1)[0]
    line_ends = np.where(transitions == -1)[0]
    if text_regions[0]: line_starts = np.concatenate([[0], line_starts])
    if text_regions[-1]: line_ends = np.concatenate([line_ends, [h-1]])

    min_line_height = h // 150
    max_line_height = h // 10
    lines = []

    for start, end in zip(line_starts, line_ends[:len(line_starts)]):
        line_height = end - start
        if line_height < min_line_height or line_height > max_line_height:
            continue
        y_start = max(0, start-2)
        y_end = min(h, end+2)
        line_region = column_img[y_start:y_end, :]
        bbox = get_text_bounding_box(line_region)
        if bbox is None: continue
        x_min, _, x_max, _ = bbox
        line_binary = cv2.threshold(cv2.cvtColor(line_region, cv2.COLOR_BGR2GRAY) if len(line_region.shape) == 3 else line_region,
                                    0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        density = np.sum(line_binary>0)/line_binary.size if line_binary.size>0 else 0
        if density < min_text_density: continue
        lines.append((y_start, y_end, x_min, x_max))
    return lines

def auto_detect_boxes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}", file=sys.stderr)
        return []
    columns = detect_columns(img)
    results = []
    for x_start, x_end in columns:
        column_img = img[:, x_start:x_end]
        lines = detect_text_lines_in_column(column_img)
        for y_start, y_end, x_min, x_max in lines:
            results.append({
                "text": "",
                "x": int(x_start + x_min),
                "y": int(y_start),
                "w": int(x_max - x_min),
                "h": int(y_end - y_start)
            })
    return results

def save_json(image_path, output_json_path='output.json'):
    boxes = auto_detect_boxes(image_path)
    with open(output_json_path, 'w') as f:
        json.dump(boxes, f, indent=2)

if __name__ == "__main__":
    IMAGE_FILE = "input.jpg"
    OUTPUT_FILE = "output.json"
    save_json(IMAGE_FILE, OUTPUT_FILE)
