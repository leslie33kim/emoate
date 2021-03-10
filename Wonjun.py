import os
import torch 
import torchvision.transforms as transforms
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import sys
# from Image_process.Imagepro import*
from network.Transformer import Transformer

[x,y] = [str(input("어떤 사진을 이모티콘으로 만들고 싶나요? 사진파일의 이름을 적어주세요 : ")),str(input("사진에 있는 파일 이름을 적어주세요 : "))]
img_path = "./" + str(y) + "/" +str(x)

pretrained_models = ["Hayao","hayao","Hosoda","hosoda","Paprika","paprika","Shinkai", "shinkai"]
while True:
    model_name = input("어떤 화풍으로 그림을 변환하시겠습니까 (화풍 종류: Hayao, Hosoda, Paprika, Shinkai)? : ")
    if model_name not in pretrained_models:
        print("없는 화풍 입니다. 다시한번 입력해 주세요.")
    elif model_name == "Hayao" or model_name == "hayao":
        model = Transformer()
        model.load_state_dict(torch.load('pretrained_model/Hayao_net_G_float.pth'))
        model.eval()
        print('Hayao Model Loaded')
        break
    elif model_name == "Hosoda" or model_name == "hosoda":
        model = Transformer() 
        model.load_state_dict(torch.load('pretrained_model/Hosoda_net_G_float.pth'))
        model.eval()
        print('Hosoda Model Loaded')
        break
    elif model_name == "Paprika" or model_name == "paprika":
        model = Transformer()
        model.load_state_dict(torch.load('pretrained_model/Paprika_net_G_float.pth'))
        model.eval()
        print('Paprika Model Loaded')
        break
    elif model_name == "Shinkai" or model_name == "shinkai":
        model = Transformer()
        model.load_state_dict(torch.load('pretrained_model/Shinkai_net_G_float.pth'))
        model.eval()
        print('Shinkai Model Loaded')
        break

img_size = 450
img = cv2.imread(img_path)

if img is None:
    print('Image load failed!')
    sys.exit()

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size, 2),
    transforms.ToTensor()
])

img_input = T(img).unsqueeze(0)

img_input = -1 + 2 * img_input

img_output = model(img_input)

img_output = (img_output.squeeze().detach().numpy() + 1.) /2.
img_output = img_output.transpose([1,2,0])

# cv2.namedWindow('image', cv2.WINDOW_NORMAL) #=> 자기가 크기를 조절 할 수 있음  
cv2.namedWindow('Cartoonized Image', cv2.WINDOW_AUTOSIZE) #=> 이건 화면크기에 맞게 자동으로 창크기를 만들어줌 
cv2.namedWindow('Original Image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Cartoonized Image', img_output) 
cv2.imshow('Original Image', img)
cv2.waitKey()
cv2.destroyAllWindows()

options = [0, 1, 2]
answer = "a"
while True:  
    answer = int(input("이모티콘을 만들지 않을경우 : 0 \n이모티콘을 원본사진으로 만들고 싶은 경우 : 1\n\
이모티콘을 그림화한 사진으로 만들고 싶은 경우 : 2 \n를 눌러주세요\n"))
    if answer not in options:
        print("숫자를 잘못 입력하였습니다. 다시 입력하여주십시오.")
    elif answer == 0: #이모티콘 안하는 경우 
        break
    elif answer == 1: #이모티콘 원본사진으로 하는 경우 
        reply1 = input("저장할 파일의 이름을 적어주세요 : ")
        result_image1 = './result1_img/{}.png'.format(reply1)
        cmd_rembg2 = "cat " + img_path + " | python3 ./removebackground.py > " + result_image1
        os.system(cmd_rembg2)  
        img2 = cv2.imread(result_image1)
        cv2.imshow("image", img2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        answer2 = input("배경이 제대로 처리 되었나요? 추가로 배경을 지우고 싶으면 Y를 눌러주시고 괜찮으면 N를 눌러주세요")
        while True: 
            if answer2 == "Y" or answer2 == "y":
                print("처음 왼쪽 마우스키로 드래그하여 배경을 지울 물체를 지정해주세요.")
                print("배경을 지우실려면 오른쪽 마우스키로 드래그하고 enter 키를 눌러주세요")
                print("다 지웠으면 esc키를 눌러 창을 닫아주세요")
                imgseg(result_image1)
                img_path2 = "./test2_img/{}.png".format(reply1)
                save_img2 = "./result1_img/{}.png".format(reply1)
                cmd_rembg3 = "cat "+ img_path2 + " | python3 ./removebackground.py > " + save_img2
                os.system(cmd_rembg3)
                break
            elif answer2 == "N" or answer2 == "n":
                break
        break
    elif answer == 2:
        reply2 = input("저장할 파일의 이름을 적어주세요 : ")
        a = "{}.png".format(reply2)
        save_img = "./test1_img/" + a
        cmd_rembg = "cat "+ img_path + " | python3 ./removebackground.py > " + save_img
        os.system(cmd_rembg)  

        img_output = cv2.convertScaleAbs(img_output, alpha = (255.0)) 
        
        cv2.imwrite('./test_img/{}.png'.format(reply2), img_output)       #그림으로 바뀐 사진 

        dst = cv2.imread(img_path)                                    #원본 사진 
        src1 = cv2.imread("./test1_img/{}.png".format(reply2), cv2.IMREAD_UNCHANGED)  #배경 없앤 사진 
        src = cv2.imread("./test_img/{}.png".format(reply2), cv2.IMREAD_COLOR)        #그림으로 바꾼 사진 
        h, w = dst.shape[:2]    #원본 사진의 shape
        h1, w1 = src1.shape[:2]     #배경 없앤 사진의 shape

        src = cv2.resize(src, dsize=(w, h), interpolation=cv2.INTER_AREA) #원본사진이랑 그림으로 바꾼 사진 크기 맞추기 
        for y in range (h):
            for x in range(w):
                dst[y,x] = 255

        if src is None or src1 is None or dst is None:
            print('Image load failed!')
            sys.exit()

        if [h,w] != [h1, w1]:
            src1 = cv2.rotate(src1, cv2.ROTATE_90_CLOCKWISE)
    
        mask = src1[:, :, -1] #(1440, 666)
        th, mask1 = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask1 = cv2.resize(mask1, dsize=(w,h), interpolation=cv2.INTER_AREA )
        
        i = "./mag_img/{}.png".format(reply2)
        cv2.imwrite(i, mask1)

        j = "./mag/{}.png".format(reply2)
        cv2.copyTo(src, mask1, dst)
        cv2.imwrite(j, dst)

        k = "./result_img/{}.png".format(reply2)
        cmd_rembg1 = "cat " + j + " | python3 ./removebackground.py > " + k
        os.system(cmd_rembg1)
        result_img = cv2.imread(k)
        cv2.imshow("imgae", result_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        os.remove(i)  
        os.remove(j)
        os.remove('./test_img/{}.png'.format(reply2)) 
        os.remove(save_img)
        break 
     
 