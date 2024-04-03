import os
from webcam import get_image_from_webcam
from digits import get_digits
from CNN import train, load_trained_net, predict_digits

print("Wellcome To Our Project !")


def file_finder(directory, file_names):
    found_file = False
    for file in file_names:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            found_file = True
        else:
            found_file = False
            break
    return found_file


def proccess_image():
    imageName = ""
    action = input(
        """
****************************** Step 2 ******************************
Do you want use webcam or exists images? (1 => webcam / 2 => images)
********************************************************************
"""
    )
    if action == "1":
        imageName = get_image_from_webcam()
    elif action == "2":
        # imageName = "./webcam/webcamgrwbslhkae.png"
        # imageName = "./webcam/webcamrepqeomfzj.png"
        # imageName = "./webcam/webcamvhctxvqzlm.png"
        imageName = "./webcam/webcamwombyqpwmg.png"

    return get_digits(imageName)


def train_or_evaluate_model(learnd_params):
    if learnd_params:
        user_action = input(
            """
****************************** Step 1 *********************************
You have already trained the model. Would you like to do it again?(y/n)
***********************************************************************
"""
        )
        if user_action == "y":
            train()
            images, originalImage, rectangles = proccess_image()
            load_trained_net(images, originalImage, rectangles)
        elif user_action == "n":
            images, originalImage, rectangles = proccess_image()
            load_trained_net(images, originalImage, rectangles)
    else:
        print("start trainig !!!")
        train()
        images, originalImage, rectangles = proccess_image()
        load_trained_net(images, originalImage, rectangles)


learnd_params = file_finder("./pth", ["m.pth", "o.pth"])
train_or_evaluate_model(learnd_params)


# clear = lambda: os.system("cls")
# clear()
# val_str = "".join(map(str, val))
# print("Predicted value is : " + str("".join(val_str)))
