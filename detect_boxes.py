import cv2
import json
import numpy as np
import sys
import os

def remove_outlier_lines(lines, tolerance_ratio=1.5):
    if len(lines) <= 2:
        return lines
    spacings = [lines[i+1][0] - lines[i][0] for i in range(len(lines)-1)]
    avg_spacing = np.median(spacings)
    valid_indices = set()
    for i, spacing in enumerate(spacings):
        if 0.5 * avg_spacing <= spacing <= tolerance_ratio * avg_spacing:
            valid_indices.add(i)
            valid_indices.add(i+1)
    return [lines[i] for i in range(len(lines)) if i in valid_indices]

def detect_columns(binary, min_column_width=50, space_threshold=0.1):
    vertical_proj = np.sum(binary == 255, axis=0)
    threshold = np.max(vertical_proj) * space_threshold
    spaces = vertical_proj < threshold

    columns = []
    in_col = False
    for i, is_space in enumerate(spaces):
        if not is_space and not in_col:
            start = i
            in_col = True
        elif is_space and in_col:
            end = i
            if end - start >= min_column_width:
                columns.append((start, end))
            in_col = False
    if in_col:
        end = len(spaces)
        if end - start >= min_column_width:
            columns.append((start, end))

    if len(columns) > 1:
        merged = []
        cur_start, cur_end = columns[0]
        for s, e in columns[1:]:
            if s - cur_end < 3:
                cur_end = e
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        columns = merged

    total_width = sum([e - s for s, e in columns])
    if len(columns) == 0 or total_width > 0.85 * binary.shape[1]:
        columns = [(0, binary.shape[1])]

    return columns

def detect_lines_in_column(binary, col_range, min_space_height=5, min_white_pixels=5):
    x_start, x_end = col_range
    col_crop = binary[:, x_start:x_end]

    horizontal_proj = np.sum(col_crop == 255, axis=1)
    threshold = np.max(horizontal_proj) * 0.15
    spaces = horizontal_proj < threshold

    lines = []
    in_line = False
    for y, is_space in enumerate(spaces):
        if not is_space and not in_line:
            start_y = y
            in_line = True
        elif is_space and in_line:
            end_y = y
            if end_y - start_y > min_space_height and np.sum(col_crop[start_y:end_y, :] == 255) >= min_white_pixels:
                lines.append((start_y, end_y))
            in_line = False

    if in_line:
        end_y = col_crop.shape[0]
        if end_y - start_y > min_space_height and np.sum(col_crop[start_y:end_y, :] == 255) >= min_white_pixels:
            lines.append((start_y, end_y))

    lines = remove_outlier_lines(lines)
    return lines

def detect_text_lines_combined(image_path, visualize=True, expand_y=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read {image_path}", file=sys.stderr)
        return [], None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3),0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 35)

    h, w = binary.shape
    binary[:int(0.05*h), :] = 0
    binary[int(0.95*h):, :] = 0

    columns = detect_columns(binary)
    boxes = []

    for x_start, x_end in columns:
        lines = detect_lines_in_column(binary, (x_start, x_end))
        for i, (start, end) in enumerate(lines):
            line_img = binary[start:end, x_start:x_end]

            kernel = np.ones((1,3), np.uint8)
            line_img_closed = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, kernel)

            col_sum = np.sum(line_img_closed, axis=0)
            sig_cols = np.where(col_sum > 0.05*np.max(col_sum))[0]
            if len(sig_cols) == 0:
                continue
            x1 = x_start + sig_cols[0]
            x2 = x_start + sig_cols[-1]

            pad = 2
            x1 = max(0, x1-pad)
            x2 = min(w-1, x2+pad)

            if expand_y:
                next_start = lines[i+1][0] if i+1 < len(lines) else end
                mid = (next_start - end)//2
                y1 = max(0, start - mid)
                y2 = min(h, end + mid)
            else:
                y1, y2 = start, end

            if i == len(lines) - 1:
                prev_end = lines[i-1][1] if i-1 >= 0 else start
                mid = (start - prev_end)//2
                y1 = max(0, start - mid)
                y2 = min(h, end + mid)

            boxes.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1)
            })
    return boxes

if __name__ == "__main__":
    IMAGE_FILE = "input.jpg"
    boxes = detect_text_lines_combined(IMAGE_FILE, visualize=True)
    print(json.dumps(boxes))
