
import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle
import math

# Разрешение камеры (стандартное)
HEIGHT = 480.0
WIDTH = 640.0

# коэффициенты для перспективной матрицы
ASPECT = numpy.float(WIDTH / HEIGHT)

# Четыре координаты у каждой вершины т.к. это в однородных координатах
CUBE = numpy.array([[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1],
                    [0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]])

# TODO: Про матрицы лучше почитать здесь: http://opengl-tutorial.blogspot.com/p/3.html
# TODO: Модель матрицы перспективы взята здесь: https://habr.com/ru/post/252771/
n = 0.1 # Расстояние от линзы до матрицы
f = 100.0 # Расстояние от камеры до стола
fovy = numpy.pi / 2 # Угол обзора
PERSPECTIVE_MATRIX = numpy.array([[1 / (ASPECT * (math.tan(fovy / 2))), 0, 0, 0],
                                  [0, 1 / (math.tan(fovy / 2)), 0, 0],
                                  [0, 0, (f + n) / (f - n), 1],
                                  [0, 0, (-2*f*n) / (f - n), 0]])

# TODO: Матрица Вью-Порт взята здесь (в самом низу) https://en.wikibooks.org/wiki/Cg_Programming/Vertex_Transformations
Sx = 0
Sy = 0
MATRIX_VIEW_PORT = numpy.array([[WIDTH / 2, 0, 0, (Sx + WIDTH / 2)],
                                  [0, HEIGHT / 2, 0, (Sy + HEIGHT / 2)],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1]])

# Отрисовка куба
def draw(img, rvec, tvec, corners):

    # Корректировка вращения - т.к. движения в камере противоположны движениям в прсотранстве
    rvec[0][0][0] *= -1
    rvec[0][0][1] *= -1

    tvec[0][0][0] += -1.0   # Корректировка перемещения по оси X
    tvec[0][0][1] += 0.5    # Корректировка перемещения по оси У
    # tvec[0][0][1] += -1.0

    transformationRotate, _ = cv2.Rodrigues(rvec) # Преобразование вектора вращения в матрицу 3х3
    rotateMatrix = numpy.zeros((4, 4))  # Расширение матрицы до размера 4х4
    rotateMatrix[-1][-1] = 1.0

    rotateMatrix[0:3, 0:3] = transformationRotate   # В матрицу поворота добавляем столбец переноса по осям

    rotateMatrix[0][3] = -1 * tvec[0][0][0]
    rotateMatrix[1][3] = -1 * tvec[0][0][1]
    rotateMatrix[2][3] = tvec[0][0][2]  # Матрица перехода

    CUBE_TRANS = numpy.zeros((8, 4), dtype=numpy.float) # Макет матрицы для сохранения преобразованных вершин куба

    for i in range(len(CUBE)): # Вращение и перенос куба
        CUBE_TRANS[i] = rotateMatrix @ CUBE[i]

    for i in range(len(CUBE)): # Перспективная трансформация
        CUBE_TRANS[i] = PERSPECTIVE_MATRIX @ CUBE_TRANS[i]

    for i in range(len(CUBE)): # До этого значения были нормированы, теперь перейдем в масштаб кадра
        CUBE_TRANS[i] = MATRIX_VIEW_PORT @ CUBE_TRANS[i]

    for i in range(len(CUBE)): # Обратное приведение из однородных координат в декартовые
        CUBE_TRANS[i] /= CUBE_TRANS[i][-1]

    CUBE_TRANS = CUBE_TRANS.astype(numpy.int, copy=False)

    CUBE_TOPS = numpy.copy(CUBE_TRANS[0:8, 0:2])    # Копирование первых двух столбцов: х и у координаты вершин куба

    imageTemp = numpy.copy(img)
    # отрисовка нижней грани куба
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[0]), pt2=tuple(CUBE_TOPS[1]), color=(255, 0, 0), thickness=2)
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[1]), pt2=tuple(CUBE_TOPS[2]), color=(255, 0, 0), thickness=2)
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[2]), pt2=tuple(CUBE_TOPS[3]), color=(255, 0, 0), thickness=2)
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[3]), pt2=tuple(CUBE_TOPS[0]), color=(255, 0, 0), thickness=2)

    # отрисовка верхней грани куба
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[4]), pt2=tuple(CUBE_TOPS[5]), color=(0, 255, 0), thickness=2)
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[5]), pt2=tuple(CUBE_TOPS[6]), color=(0, 255, 0), thickness=2)
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[6]), pt2=tuple(CUBE_TOPS[7]), color=(0, 255, 0), thickness=2)
    cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[7]), pt2=tuple(CUBE_TOPS[4]), color=(0, 255, 0), thickness=2)

    # отрисовка боковых граней куба
    for i in range(4):
        cv2.line(imageTemp, pt1=tuple(CUBE_TOPS[i]), pt2=tuple(CUBE_TOPS[i + 4]), color=(0, 255, 255), thickness=2)

    return imageTemp


# TODO: Дальше считайте, что код основной программы
# Проверка наличия файла с параметрами калибровки
if not os.path.exists('calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

# Постоянные параметры, используемые в методе Aruco
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_100)

# Доска с маркерами Аруко с 5х7 маркеров
board = aruco.GridBoard_create(
        markersX=5,
        markersY=7,
        markerLength=0.04,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

# Векторы, которые будем использовать для поворота и переноса
rvecs, tvecs = None, None

cam = cv2.VideoCapture(0) # Захват видеопотока с веб-камеры на ноуте. Если не захватывает, попробуй 0 заменить на -1

while(cam.isOpened()):
    # Захват кадра из видеопотока
    ret, QueryImg = cam.read()

    if ret == True:
        # Перевод в GRAY
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

        # Детектирование Aruco маркеров
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        # Уточнение обнаруженных маркеров
        # Удаление маркеров, не принадлежащих нашей доски и поиск не добавленных ранее
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)

        # Отрисовка контуров обнаруженных маркеров
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        # Если обнаружен маркер, то...
        if ids is not None and len(ids) > 0:
            # Определение его положения
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, ids, cameraMatrix, distCoeffs)

            # Отрисовка куба на кадре
            cube = draw(QueryImg, rvecs, tvecs, corners)
            cv2.imshow('cube', cube)
        # Отображение итогового изображения
        cv2.imshow('QueryImage', QueryImg)
    # Для закрытия видеопотока необходимо нажать 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
