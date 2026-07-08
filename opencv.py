import cv2
from PIL import Image
import torch
from dataset import test_transformer
from predict import load_model,predict

cap=cv2.VideoCapture(0)
transformer=test_transformer
model=load_model()

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()
    
while True:
    ret,frame=cap.read()
    if not ret:
        print("Error: Could not read frame or video ended.")    
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img=Image.fromarray(rgb_frame)
    tensor_img=transformer(pil_img)
    tensor_img=torch.unsqueeze(tensor_img,dim=0)
    prediction,confidence=predict(model=model,image=tensor_img)
    cv2.putText(
        img=frame,
        text=f"ECO-SORT AI \n Prediction-{prediction} \n confidence-{confidence}",
        org=(30, 50),
        fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        fontScale=1.0,                
        color=(139, 155, 114),            
        thickness=2,                  
        lineType=cv2.LINE_AA
    )
    cv2.imshow("WebCam",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
        
# print(f"predict: {prediction} | confidence: {confidence} ")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        