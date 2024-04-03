import cv2
import matplotlib.pyplot as plt
import numpy as np

# adding extra pading to make it squre
# making image squre is important because we have to resize image to 28*28
def add_padding(img):
    img = np.pad(img, pad_width=50)
    (a, b) = np.shape(img)
    top, down, left, right = 0, 0, 0, 0

    if a > b:
        pad = (a - b) // 2
        left = pad
        right = pad
    else:
        pad = (b - a) // 2
        top = pad
        down = pad

    padding = ((top, down), (left, right))
    img = np.pad(img, padding, mode="constant", constant_values=0)
    return img


# croping white space around the image
def crop_image(img):
    x1, x2, y1, y2 = 0, 0, 0, 0

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 15
    )
    img = cv2.dilate(mask, kernel=(5, 5), iterations=1)

    # print("step 1 in section crop Image => after blure")
    # im2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(im2)
    # plt.show()
    for i in range(len(img)):  # for each row of image
        # print(str(sum(1 for e in img[i] if e == 255)) + ' , '+ str(len(img[i])))
        if sum(1 for e in img[i] if e == 255) != len(img[i]):
            if x1 == 0:
                x1 = i
            x2 = i
    for i in range(len(img.T)):
        if sum(1 for e in img[:, i] if e == 255) != len(img[:, i]):
            if y1 == 0:
                y1 = i
            y2 = i
    return img, x1, x2, y1, y2


# spliting multi digit image to seprate digits
def get_digits(imageName):
    # reading image from file

    originalImage = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    # print("read and convert orginal image to grayscale")
    # originalImage2 = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    # plt.imshow(originalImage2)
    # plt.show()

    # removing white space from around of image
    c_image, cx1, cx2, cy1, cy2 = crop_image(originalImage)
    cropedImage = c_image[cx1:cx2, cy1:cy2]

    # print("show image after crop !!")
    # cropedImage2 = cv2.cvtColor(cropedImage, cv2.COLOR_BGR2RGB)
    # plt.imshow(cropedImage2)
    # plt.show()

    # rectangle cordinats for each digit
    rectangles = []
    x1, x2, y1, y2 = 0, len(cropedImage) - 1, -1, 0
    acceptableImageLen = 2
    col = 0
    images = []
    c = 0
    while c < len(cropedImage.T):
        # finding starting ending column of each image
        for i in range(col, len(cropedImage.T)):
            c = c + 1

            if (
                sum(1 for e in cropedImage[:, i] if e == 0) >= acceptableImageLen
                and y1 == -1
            ):
                y1 = i
            if (
                sum(1 for e in cropedImage[:, i] if e == 255)
                >= len(cropedImage[:, i]) - acceptableImageLen
                and y1 != -1
            ):
                y2 = i
                col = i + 1
                break
        if c == len(cropedImage.T):
            y2 = c - 1
        if y2 - y1 > 1:
            result_orgin = cropedImage[x1:x2, y1:y2]
            r_image, rx1, rx2, ry1, ry2 = crop_image(result_orgin)
            result = r_image[rx1:rx2, ry1:ry2]
            # adding detected digit cordinate

            # print("crop each digit !!")
            # result2 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # plt.imshow(result2)
            # plt.show()

            rectangles.append(
                [x1 + cx1 + rx1, x2 + cx1 + rx2 - x2, y1 + cy1 - ry1, y2 + cy1 - ry1]
            )
            result = np.invert(result)

            # print("after invert each pixel !!")
            # result2 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # plt.imshow(result2)
            # plt.show()

            result = add_padding(result)

            # print("after add_padding each digit !!")
            # result2 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # plt.imshow(result2)
            # plt.show()

            images.append(result)
        y1, y2 = -1, 0
    show_digits_boundry(originalImage, rectangles)
    return images, originalImage, rectangles


##############################################################################
def show_digits_boundry(img, rectangles, predicted=""):
    # Draw rectangles
    if len(predicted):
        for i in range(len(rectangles)):
            x1 = rectangles[i][0]
            x2 = rectangles[i][1]
            y1 = rectangles[i][2]
            y2 = rectangles[i][3]
            # print('('+str(y1)+','+str(x1)+') : ('+str(y2)+','+str(x2)+')')
            cv2.rectangle(img, (y1, x1), (y2, x2), (0, 255, 0), 2)
        #Show predicted number on image
        text = "predicted : " + "".join(map(str, predicted))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        cv2.putText(
            img,
            text,
            (100, 25),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    # Output img with window name as 'image'
    cv2.imshow("image", img)
    # Maintain output window untill user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()

