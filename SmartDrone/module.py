import cv2
import numpy as np
import math


# 이미지 회전 함수 정의
def rotate_and_resize_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 회전 변환 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 이미지 회전
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return rotated_image

def find_perpendicular_line(start_point, end_point, point_x, point_y):
    x1, y1 = start_point
    x2, y2 = end_point
    # print('x1:', x1, ' y1:', y1, ' x2:', x2, ' y2:', y2)
    # 직선의 방정식 계산
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    # print('slope : ', slope, 'intercept : ', intercept)

    # 수직선의 방정식 계산
    if slope == 0:
        intersection_x = point_x
        intersection_y = point_y
    elif slope == float('inf') or slope == float('-inf'):
        intersection_x = x1
        intersection_y = point_y
    else:
        perpendicular_slope = -1 / slope
        perpendicular_intercept = point_y - perpendicular_slope * point_x
        print('perpendicular_slope : ', perpendicular_slope, 'perpendicular_intercept : ', perpendicular_intercept)

        # 두 방정식의 교점 계산
        intersection_x = (perpendicular_intercept - intercept) / (slope - perpendicular_slope)
        intersection_y = slope * intersection_x + intercept

    intersection_point = (intersection_x, intersection_y)
    return intersection_point

# SIFT 특징점 추출 함수
def sitf_matcher(map_keypoints,map_descriptors,frame_keypoints ,frame_descriptors):
    # SIFT 특징점 매칭
    bf = cv2.BFMatcher()
    matches = bf.match(map_descriptors, frame_descriptors)
    # 매칭 결과를 거리 기준으로 오름차순 정렬
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def good_sitf_matcher(map_keypoints,map_descriptors,frame_keypoints ,frame_descriptors):

    # SIFT 특징점 매칭
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(map_descriptors, frame_descriptors, k=2)
    # 좋은 매칭 결과 선별
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # RANSAC 알고리즘으로 이상치 제거
    if len(good_matches) > 10:
        src_pts = np.float32([map_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        good_matches = [m for i, m in enumerate(good_matches) if matchesMask[i]]

        # index_params = dict(algorithm=1, trees=5)
        # search_params = dict(checks=50)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(map_descriptors, frame_descriptors, k=2)

        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(map_descriptors, des2, k=2)

        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # matches = bf.match(map_descriptors, frame_descriptors)

    return good_matches


# SIFT 특징점을 추출합니다.
def extract_sift_features(image, mask=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()  # SIFT 알고리즘 생성
    keypoints, descriptors = sift.detectAndCompute(gray_image, mask)
    return keypoints, descriptors

def calculate_rotation_angle(current_angle: object, target_angle: object) -> object:
    """
    현재 각도와 쳐다봐야 할 각도가 주어졌을 때, 회전해야 하는 각도를 계산하는 함수

    Args:
        current_angle (float): 현재 각도 (단위: 도)
        target_angle (float): 쳐다봐야 할 각도 (단위: 도)

    Returns:
        float: 회전해야 하는 각도 (단위: 도)
    """
    # 각도의 범위를 0도부터 360도까지로 조정
    current_angle = current_angle % 360
    target_angle = target_angle % 360

    # 회전해야 하는 각도 계산
    rotation_angle = target_angle - current_angle

    # 180도를 넘어가는 경우 반대 방향으로 회전하는 각도로 조정
    if rotation_angle > 180:
        rotation_angle -= 360
    elif rotation_angle < -180:
        rotation_angle += 360

    return rotation_angle


def get_angle(start_x, start_y, target_x, target_y):
    """ 시작점과 목표점의 좌표를 이용하여 방위각을 구하는 함수 """
    dx = target_x - start_x
    dy = target_y - start_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    # atan2의 결과는 -pi부터 pi 사이의 값이므로, 각도를 0부터 360도 사이의 값으로 변환
    angle_deg = (angle_deg + 360) % 360
    # 0도가 북쪽이라고 가정하므로, 동쪽일 경우 90도로 변환
    angle_deg = int(angle_deg + 90) % 360

    return angle_deg

