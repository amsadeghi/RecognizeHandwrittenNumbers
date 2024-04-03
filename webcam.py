import cv2
import string
import random


# open camera until Space button hitted to capturing image
def get_image_from_webcam():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    imageName = ""
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            letters = string.ascii_lowercase
            rand = "".join(random.choice(letters) for i in range(10))
            img_name = "webcam/webcam{}.png".format(rand)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            imageName = img_name
            break

    cam.release()
    cv2.destroyAllWindows()
    return imageName

