import openpyxl as openpyxl
import point_find
import start
import cv2
from module import *
import video_find

workbook = openpyxl.load_workbook('bang.xlsx')
sheet = workbook.active

video_path = "DJI_0051_W.MP4"
map = "Map5.jpg"
# point = [[220, 220], [400, 400], [600, 600], [800, 800]]
point = [[307, 276], [639, 933], [1280, 898], [1147, 267], [307, 276]]

# point = [[192, 136], [411, 623], [766, 589], [717, 164],[192, 136]]

angles = [sheet['W522'].value, sheet['W522'].value, sheet['W865'].value, sheet['W1136'].value]

mapImg = cv2.imread(map)

for index, i in enumerate(point):
    targetAngle = 0

    # 이미지 축척
    width_m = 625.64  # 가로길이 (m)
    height_m = 352.79  # 세로길이 (m)
    # 이미지의 해상도 추출 height 세로 width 가로
    height, width = mapImg.shape[:2]
    pixel_length = width_m / width


    if index != (len(point) - 1):
        targetAngle = get_angle( i[0], i[1], point[index+1][0], point[index+1][1] )
        calculateAngle = calculate_rotation_angle( angles[index],  targetAngle)
        print("Heading Adjustment : " , round(calculateAngle) )

        diff_x = point[index+1][0] - i[0]
        diff_y = point[index+1][1] - i[1]

        distans = math.sqrt(diff_x ** 2 + diff_y ** 2) * pixel_length * 0.8
        if index != (len(point) - 2):
            print(f"Distance to W_{index + 1} = {int(distans)}m")
        else:
            print(f"Distance to W_0 = {int(distans)}m")

    print(f"Coordinate at W_{index} = ({i[0]},{i[1]})\n\n")






# 1.전산관
start.go(video_path, map, 0)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

point_find.go(video_path, sheet['W522'].value, map, 0, point, 1, 1)    # 1이면 처음 0이면 2번째 이상

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

video_find.video_find(video_path, map, 0, point, 1, 100, 100, 19.5)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# 2.도서관

point_find.go(video_path, sheet['W522'].value, map, 19, point, 2, 0)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

video_find.video_find(video_path, map, 19, point,2, 310, 460, 37)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# 3.학술관
point_find.go(video_path, sheet['W865'].value, map, 37 , point, 3,0)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

video_find.video_find(video_path, map , 39, point, 3, 700, 500, 56)

start.go(video_path, map, 56)