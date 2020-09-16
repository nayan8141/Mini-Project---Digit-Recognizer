import tensorflow as tf
import numpy as np
import cv2

new = tf.keras.models.load_model('model.h5')
mnist =  tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

image = x_test[1]
input = cv2.resize(image, (28, 28)).reshape((1, 28, 28))
print(new.predict_classes(input))
print(y_test[1])

mouse=False
ix,iy=-1,-1

def draw_circle(event, x, y, flags, param):
    global mouse,ix,iy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse=True
        ix,iy=x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse == True:
            cv2.circle(img,(x,y),5,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse=False
        cv2.circle(img,(x,y),5,(255,255,255),-1)

img = np.zeros((400,400,3), np.uint8)        
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas',draw_circle)

while True:
    cv2.imshow('Canvas',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    elif key == ord('c'):
        img[0:400,0:400]=0

    elif key == ord('w'):
        out=img[0:400,0:400]=0
        cv2.imwrite('Output.jpg',out)

    elif key == ord('p'):
        image = img[100:500,100:500]
        input = cv2.resize(image, (28, 28)).reshape((1, 28, 28))
        print(new.predict_classes(input))


cv2.destoryAllWindows()




