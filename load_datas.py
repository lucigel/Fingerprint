import os 
import cv2 
import numpy as np 


img_size = 96 

def load_data(path, train=True):
    IDs = []
    genders = []
    hands = []
    fingers = []
    images = []
    loaded_successfully = False

    for img in os.listdir(path):
        imgname, ext = os.path.splitext(img)
        ID_and_info = imgname.split('__')
        if len(ID_and_info) != 2:
            continue

        ID, info = ID_and_info
        ID = int(ID) - 1
        IDs.append(ID)

        remaining_info = info.split("_")
        if len(remaining_info) != 4 and not train:
            continue
        if len(remaining_info) != 5 and train:
            continue
        
        # Extract gender 
        gender = remaining_info[0]

        gender_num = 0 if gender == 'M' else 1
        genders.append(gender_num)


        # Extract hand 
        hand = remaining_info[1]
        hand_num = 0 if hand == 'left' else 1
        hands.append(hand_num)
        

        # Extract finger 
        finger_map = {'little': 0, 'ring':1, 'middle':2, "index": 3, "thumb":4}
        finger = remaining_info[2]
        finger_num = finger_map.get(finger, -1)

        if finger_num == -1:
            continue

        fingers.append(finger_num)

        # load image
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img_array, (img_size, img_size))
        images.append(img_resize)

    loaded_successfully = True

    if loaded_successfully:
        print("Data Loaded from", path)
        
    return np.array(IDs), np.array(genders), np.array(hands), np.array(fingers), np.array(images)
