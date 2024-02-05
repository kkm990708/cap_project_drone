import cv2

from module import *
dst_x = 0
dst_y = 0
def go(lo,gak,map,time, location_points, select, select2):
    # 마우스 클릭 이벤트 핸들러 함수
    def draw_location(image, p, p2):
        global dst_x,dst_y
        # 두 점의 좌표값 차이 계산 및 출력
        diff_x = p2[0] - p[0]
        diff_y = p2[1] - p[1]
        distans = math.sqrt(diff_x ** 2 + diff_y ** 2) * pixel_length
        # 각도 예시 입력값
        current_angle = gak
        target_angle = get_angle(p[0], p[1], p2[0],p2[1])
        angle_rad = math.radians(current_angle)
        pt2 = (int(p[0] + 50 * math.sin(angle_rad)),
               int(p[1] - 50 * math.cos(angle_rad)))

        # 회전해야 하는 각도 계산
        rotation_angle = calculate_rotation_angle(current_angle, target_angle)

        # 각 값들을 표시
        # cv2.putText(image, f'({p[0]},{p[1]})',
        #             (p[0], p[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 100, 0), 3)
        # cv2.putText(image, f'({rotation_angle})',
        #             (p[0], p[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (50, 255, 0), 3)
        #
        # cv2.putText(image, f'({p2[0]},{p2[1]})',
        #             (p2[0], p2[1] - 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 100, 0), 3)
        # cv2.putText(image, f'({int(distans * 0.8)}M)',
        #             (p2[0], p2[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 255), 3)
        # cv2.putText(image, f'({int(target_angle)}C)',
        #             (p2[0], p2[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 100, 200), 3)

        # 경로 좌표를 출력
        for point in location_points:
            x, y = point
            cv2.circle(image, (x, y), 20, (0, 0, 255), -1)

        # 직선 그리기
        points = np.array(location_points, dtype=np.int32)
        cv2.polylines(image, [points], False, (255, 0, 0), 3)
        cv2.line(image, p, p2, (255, 255, 0), 5)





        cv2.arrowedLine(image, p, pt2, (0, 255, 0), thickness=7, tipLength=0.5)
        return image

    # 이미지 읽어오기

    img1 = cv2.imread(map)
    video_capture = cv2.VideoCapture(lo)
    video_capture.set(cv2.CAP_PROP_POS_MSEC, (time * 1000))
    ret, img2 = video_capture.read()
    global dst_x, dst_y
    # 이미지 축척
    width_m = 625.64  # 가로길이 (m)
    height_m = 352.79  # 세로길이 (m)
    select = select - 1
    # 이미지의 해상도 추출 height 세로 width 가로
    height, width = img1.shape[:2]
    pixel_length = width_m / width

    # 자를 영역 선택 (예: 좌상단 좌표와 우하단 좌표)
    crop_x, crop_y, crop_width, crop_height = 100, 450, 1300, 1200

    crop_rignt_x = min(crop_x + crop_width, width)
    crop_rignt_y = min(crop_y + crop_height, height)


    # 이미지 자르기
    cropMap = img1[crop_y:crop_rignt_y, crop_x:crop_rignt_x]

    cropMap = cv2.rotate(cropMap, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.resize(img2, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    crop_height, crop_width = cropMap.shape[:2]
    # 특징점 검출 및 기술자 계산
    kp1, des1 = extract_sift_features(cropMap)
    kp2, des2 = extract_sift_features(img2)

    matches = good_sitf_matcher(kp1, des1, kp2, des2)

    # 좋은 매칭 결과만 선택
    good_matches = matches[:100]

    # 좋은 매칭 결과 시각화
    result = cv2.drawMatches(cropMap, kp1, img2, kp2, good_matches, cropMap, flags=2)

    # 결과 이미지 전치
    img_matches_transposed = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)


    # 좋은 매칭 결과 중심 좌표 계산
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    M, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    h, w, _ = img2.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dst_x = dst[:, :, 0].mean()
    dst_y = dst[:, :, 1].mean()
    dst_x = int(dst_x)
    dst_y = int(dst_y)
    img3 = img1.copy()

    showCropMap = cropMap.copy()
    showCropMap_rotate = cv2.rotate(showCropMap, cv2.ROTATE_90_CLOCKWISE)
    cropR_height, cropR_width = showCropMap_rotate.shape[:2]

    temp_dst_x = dst_x
    temp_dst_y = dst_y
    temp_point = {dst_x, dst_y}
    print(temp_point)
    dst_x = cropR_width - temp_dst_y + crop_x
    dst_y = temp_dst_x + crop_y
    dst_point = {dst_x, dst_y}
    print(dst_point)

    if select2 == 0:
        # 가운데 좌표값을 지도 이미지에 표시
        cv2.circle(img3, (dst_x, dst_y), 25, (0, 255, 255), -1)

    # 경로 좌표를 출력
    img3 = draw_location(img3, location_points[select], location_points[select+1])

    new_width = round(width * 0.7)
    new_height = round(height * 0.7)
    #  종료
    map_image_with_location = cv2.resize(img3, dsize=(0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

    t_height, t_width = img_matches_transposed.shape[:2]
    img_matches_transposed_resize = cv2.resize(img_matches_transposed, ((int)(t_width * 0.5), (int)(t_height * 0.5) ))

    cv2.circle(showCropMap, (temp_dst_x, temp_dst_y), 25, (0, 255, 255), -1)

    cv2.circle(showCropMap_rotate, (dst_x, dst_y), 25, (0, 255, 255), -1)


    if select2 == 0:
        # cv2.imshow('showCropMap', showCropMap)
        # cv2.imshow('showCropMap_rotate', showCropMap_rotate)
        cv2.imshow('img_matches_transposed_resize', img_matches_transposed_resize)
        cv2.imshow('img_matches_transposed', img_matches_transposed)
    while True:
        cv2.imshow('map with center', map_image_with_location)
        if select2 != 1:
            key2 = cv2.waitKey()
            if key2 == 27:  # ESC 키를 한 번 더 누를 경우
                map_image_with_location2 = img1.copy()
                temp = location_points[select]

                # cv2.putText(map_image_with_location2, f"({temp[0]}, {temp[1]})", temp, cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 255), 2)q
                location_points[select] = [dst_x, dst_y]

                map_image_with_location2 = draw_location(map_image_with_location2, location_points[select],
                                                         location_points[select + 1])


                cv2.circle(map_image_with_location2, (temp[0], temp[1]), 25, (0, 255, 255), -1)

                diff_x = location_points[select + 1][0] - location_points[select][0]
                diff_y = location_points[select + 1][1] - location_points[select][1]
                distans = math.sqrt(diff_x ** 2 + diff_y ** 2) * pixel_length

                print(f"Distance to W_{select + 1} = {distans}m")

                print(f"Coordinate at W_{select} = ({temp[0]},{temp[1]})")
                print(f"Coordinate at W*_{select} = ({location_points[select][0]},{location_points[select][1]})\n\n")


                # map_image_with_location2 = cv2.resize(map_image_with_location2, (new_width, new_height))
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.imshow('map with center2', map_image_with_location2)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    break

        elif cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


