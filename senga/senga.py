import cv2
import numpy as np

# 画像の読み込み
img_src = cv2.imread("./img/python.png")


def contour(img):
    # 24近傍の定義
    neighborhood24 = np.array([[1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1]],
                              np.uint8)
    # グレースケール変換
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("./res/gray_img.png", gray_img)

    # 膨張処理
    dilated_img = cv2.dilate(gray_img, neighborhood24, iterations=1)
    cv2.imwrite("./res/dirated_img.png", dilated_img)

    # 差分の取得
    diff_img = cv2.absdiff(dilated_img, gray_img)
    cv2.imwrite("./res/diff_img.png", diff_img)

    # 白黒反転
    contour_img = 255 - diff_img
    cv2.imwrite("./res/result_img.png", contour_img)

    return contour_img


cv2.namedWindow("res", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
cv2.imshow("res", contour(img_src))
cv2.waitKey(0)
cv2.destroyAllWindows()
