import cv2


def face_detection(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # greyscale == more accurency
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,
                                          scaleFactor=1.1,
                                          minNeighbors=5)
    # scaleFactor => smaller value = higher accurency, with each found face ># scale 5% less
    # minNeighbors => how many neighbors are around the window, most of the time this value is set to 5!!

    print(type(faces))
    print(faces)  # 2dimensional array --> in order of the values: x, y width, height

    # now we need to apply the faces rectangle to the image
    for x, y, width, height in faces:
        image = cv2.rectangle(image,
                            (x, y), (x + width, y + height),
                            (0, 255, 0),
                            3
                            )  # first coordinate pair = top-left-corner, second coordinate pair = bottom-right-corner -> \
    # (0,255,0) = rgb color value for the color green
    return cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))


img1 = cv2.imread('photo.jpg')
img_face_detect_1 = face_detection(img1)
cv2.imshow("Gray", img_face_detect_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = cv2.imread('news.jpg')
img_face_detect_1 = face_detection(img1)
cv2.imshow("Gray", img_face_detect_1)
cv2.waitKey(0)
cv2.destroyAllWindows()