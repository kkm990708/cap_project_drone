
import time
from module import *


def video_find(drone_video, map, start, location_point, select, x, y, end_time):
    # 지도 이미지를 로드합니다.
    map_image = cv2.imread(map)
    map_image_with_locations = map_image.copy()  # 복사본 생성

    # 관심 영역을 설정합니다.
    roi_x = x  # 관심 영역의 시작 x 좌표
    roi_y = y
    roi_width = 600  # 관심 영역의 너비
    roi_height = 600  # 관심 영역의 높이

    # 동영상 파일을 로드합니다.
    video_capture = cv2.VideoCapture(drone_video)
    cv2.namedWindow('video player', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video player', 1215, 862)
    # 초기 위치를 None으로 설정합니다.
    current_location = None
    previous_locations = []  # 이전 위치들을 저장할 리스트 생성

    # 경로 좌표를 출력
    for point in location_point:
        x, y = point
        cv2.circle(map_image_with_locations, (x, y), 10, (0, 0, 255), -1)

    # 직선 그리기
    points = np.array(location_point, dtype=np.int32)
    cv2.polylines(map_image_with_locations, [points], False, (255, 0, 0), 3)

    start_time = time.time()
    while True:


        mask = np.zeros(map_image.shape[:2], dtype=np.uint8)
        mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = 255
        map_keypoints, map_descriptors = extract_sift_features(map_image,mask)
        mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = 0

        # 현재까지 경과한 시간을 계산합니다.
        elapsed_time = time.time() - start_time + start

        if elapsed_time >= end_time:    # end time이 되면 종료
            cv2.waitKey()
            break

        # 동영상에서 프레임을 읽어옵니다.
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (elapsed_time * 1000))
        ret, frame = video_capture.read()

        if current_location:
            roi_x = int(current_location[0] - roi_width / 2)
            roi_y = int(current_location[1] - roi_height / 2)
            roi_x = max(roi_x, 0)  # 좌표가 음수가 되지 않도록
            roi_y = max(roi_y, 0)  # 좌표가 음수가 되지 않도록 보정

        # SIFT 특징점을 추출합니다.
        frame_keypoints, frame_descriptors = extract_sift_features(frame)

        # SIFT 특징점 매칭
        # matcher = cv2.BFMatcher()
        # matches = matcher.knnMatch(map_descriptors, frame_descriptors, k=2)

        # 좋은 매칭 결과 선별
        good_matches = good_sitf_matcher(map_keypoints,map_descriptors,frame_keypoints,frame_descriptors)

        # 상위 매칭 결과를 사용하여 위치를 추정합니다.
        location_points = []
        for match in good_matches[:20]:
            location_points.append(map_keypoints[match.queryIdx].pt)

        # 위치를 계산합니다.
        if location_points:
            location_points = np.array(location_points)
            mean_location = np.mean(location_points, axis=0)
            current_location = (int(mean_location[0]), int(mean_location[1]))

        # print('location_point[select-1] : ', location_point[select - 1], 'location_point[select-1] : ',
        #       location_point[select])

        # 선분의 시작점과 끝점 설정
        start_point = location_point[select-1]
        end_point = location_point[select]

        # 좌표 설정
        point_x = current_location[0]
        point_y = current_location[1]

        # 가장 가까운 좌표, 거리, 방향, 원래 점 계산
        intersection_point = find_perpendicular_line(start_point, end_point, point_x, point_y)
        # print('current_location : ', current_location, '  intersection_point : ', intersection_point, '\n')

        intersection_point = list(intersection_point)
        # 시작점과 끝점을 벗어난 경우 가장 가까운 시작점이나 끝점으로 보정합니다.
        if intersection_point[0] < min(start_point[0], end_point[0]):
            intersection_point[0] = min(start_point[0], end_point[0])
        elif intersection_point[0] > max(start_point[0], end_point[0]):
            intersection_point[0] = max(start_point[0], end_point[0])

        if intersection_point[1] < min(start_point[1], end_point[1]):
            intersection_point[1] = min(start_point[1], end_point[1])
        elif intersection_point[1] > max(start_point[1], end_point[1]):
            intersection_point[1] = max(start_point[1], end_point[1])
        intersection_point = tuple(intersection_point)

        if not math.isnan(intersection_point[0]) and not math.isnan(intersection_point[1]):
            interest_x = int(intersection_point[0])
            intetest_y = int(intersection_point[1])

            # 이동한 좌표 표시
            cv2.circle(map_image_with_locations, [interest_x, intetest_y], 9, (255, 255, 255), -1)

        # 현재 위치를 지도 이미지에 표시합니다.
        # if current_location:
        #     # 이전 위치들과의 선을 그리기 위해 이전 위치들을 저장합니다.
        #     previous_locations.append(current_location)
        #     cv2.circle(map_image_with_locations, current_location, 10, (0, 255, 0), -1)
        #     # cv2.putText(map_image_with_locations, f"({current_location[0]}, {current_location[1]})",
        #     #             current_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #
        # # 이전 위치들을 선으로 연결하여 표시합니다.
        # for i in range(1, len(previous_locations)):
        #     cv2.line(map_image_with_locations, previous_locations[i - 1], previous_locations[i], (0, 255, 0), 2)

        map_image_with_locations_resized = cv2.resize(map_image_with_locations, (1215, 862))
        cv2.imshow('current location', map_image_with_locations)
        cv2.imshow('video player', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imwrite("map_with_locations.jpg", map_image_with_locations)

    cv2.destroyAllWindows()
