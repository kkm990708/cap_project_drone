from module import *

dst_x = 0
dst_y = 0
def go(image_path, map, sigan, select = 0):
    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            video = cv2.VideoCapture(image_path)
            ret, ima2 = video.read()

            global dst_x, dst_y
            # 두 점의 좌표값 차이 계산 및 출력
            diff_x = x - int(dst_x)
            diff_y = y - int(dst_y)
            # 클릭한 좌표에 원과 텍스트 그리기


            scale_x = width / new_width
            scale_y = height / new_height
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)

            cv2.circle(img1, (new_x, new_y), 10, (0, 0, 255), -1)
            distans = math.sqrt(diff_x ** 2 + diff_y ** 2) * pixel_length * 0.8
            # 좌표 사이에 선 그리기
            cv2.line(img1, (int(dst_x), int(dst_y)), (new_x , new_y), (0, 255, 0), 4)
            cv2.putText(img1, f'({x}, {y})', (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img1, f'({int(distans)}M)', (new_x, new_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 0), 2)

            print('Mouse clicked at: ', new_x, new_y)
            clicked_points.append([new_x, new_y])
            dst_x = new_x
            dst_y = new_y
            # 창 새로고침
            img7 = cv2.resize(img1, dsize=(0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('map with centers', img7)

    # 이미지 읽어오기
    img1 = cv2.imread(map)
    elapsed_time = sigan
    video_capture = cv2.VideoCapture(image_path)
    video_capture.set(cv2.CAP_PROP_POS_MSEC, (elapsed_time * 1000))
    ret, img2 = video_capture.read()
    global dst_x,dst_y
    # 이미지 축척
    width_m = 625.64  # 가로길이 (m)
    height_m = 352.79  # 세로길이 (m)

    # 이미지의 해상도 추출 height 세로 width 가로
    height, width = img1.shape[:2]
    pixel_length = width_m / width


    # SIFT 검출기 생성
    sift = cv2.SIFT_create()

    # 특징점 검출 및 기술자 계산
    kp1, des1 = extract_sift_features(img1)
    kp2, des2 = extract_sift_features(img2)

    matches = []
    if select == 0:
        matches = sitf_matcher(kp1, des1, kp2, des2)
    if select == 1:
        matches = good_sitf_matcher(kp1, des1, kp2, des2)
    # 좋은 매칭 결과만 선택

    good_matches = matches[:50]

    # 좋은 매칭 결과 시각화
    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

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
    clicked_points.append([dst_x, dst_y])

    # 가운데 좌표값을 지도 이미지에 표시
    cv2.circle(img1, (dst_x, dst_y), 15, (0, 0, 255), -1)

    # 좌표값도 함께 출력
    # cv2.putText(img1, f"({int(dst_x)}, {int(dst_y)})", (int(dst_x), int(dst_y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #         (0, 0, 255), 2)

    # 결과 출력
    new_width = 2197
    new_height = 1528
    # result = cv2.resize(result, (1600, 800))
    cv2.imshow('result', result)
    key = cv2.waitKey()
    map_image_with_location = img1.copy()
    map_image_with_location = cv2.resize(map_image_with_location, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
    # map_image_with_location = cv2.resize(map_image_with_location, (new_width, new_height))
    cv2.imshow('map_image_with_location', map_image_with_location)
    # cv2.waitKey()

    # 마우스 이벤트 캡처 함수 등록
    cv2.setMouseCallback('map_image_with_location', mouse_callback, map_image_with_location)

    # 종료
    cv2.waitKey()
    cv2.destroyAllWindows()
    return clicked_points

