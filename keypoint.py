import cv2
import numpy as np
import torch


class KeypointDetection(object):


    def __init__(self, model_path):
        self.loaded_model,_,_ = self.load_checkpoint(model_path)
        
    
    def load_checkpoint(self,filepath): ## load the model and it's weight
        checkpoint = torch.load(filepath)
        loaded_model = checkpoint['model']
        loaded_model.load_state_dict(checkpoint['state_dict'])
        L=(checkpoint['training_loss'],checkpoint['val_loss'])    
        return loaded_model,checkpoint['hyper-parameters'],L


    def process_image(self,image):    ## preprocess the image to feed into the model
        #image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
        image = clahe.apply(image)
        image = image/255.0
        image = cv2.resize(image,(224,224))
        image = image.reshape(224,224,1)
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.type(torch.FloatTensor)
        return image


    def procee_keypoints(self,predicted_key_pts,image_shape):   ## un-normalize the output keypoints 
        predicted_key_pts = predicted_key_pts.view( 68, -1)
        predicted_key_pts = predicted_key_pts.data
        predicted_key_pts = predicted_key_pts.numpy()
        predicted_key_pts = predicted_key_pts*50.0+100
        
        original_h,original_w = image_shape[0:2]
        h,w = (224,224)
        predicted_key_pts = predicted_key_pts * [original_w / w, original_h / h]
        
        return predicted_key_pts
    
    
    def put_filter(self, img):
        image_tensor=self.process_image(img)
        predicted_key_pts=self.loaded_model(image_tensor)
        predicted_key_pts=self.procee_keypoints(predicted_key_pts,img.shape)  ## predicted labels
        return predicted_key_pts
