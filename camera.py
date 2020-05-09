import cv2
from keypoint import KeypointDetection
import numpy as np

facec = cv2.CascadeClassifier('./detector_architectures/haarcascade_frontalface_default.xml')  ## load harcascade classifier
model = KeypointDetection('saved_models/keypoints_model_10.pt')   ## load trained model for facial_key_point detection
font = cv2.FONT_HERSHEY_SIMPLEX
sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)  ## load sunglass
moustache = cv2.imread('images/moustache.png', cv2.IMREAD_UNCHANGED)   ## load moustache
#hat = cv2.imread('images/straw_hat.png', cv2.IMREAD_UNCHANGED)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)   ## initialize webcam

    def __del__(self):
        self.video.release()

    # returns camera frames along with the fun_filter placed on face
    def get_frame(self,web=False):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)     ## detect faces present in current frame

        for (x, y, w, h) in faces:
            p_h,p_w = (60,60)     ## padding 
            fc = gray_fr[y-p_h:y+h+p_h, x-p_w:x+w+p_w]
            fc_color = fr[y-p_h:y+h+p_h,x-p_w :x+w+p_w]
            #cv2.rectangle(fr,(x-p_w,y-p_h),(x+w+p_w,y+h+p_h),(0,255,0),3)
            
            key_pts = model.put_filter(fc)  ## predict key_points
            """for p_x,p_y in key_pts:       ## uncomment to visualize key_points on face
                cv2.circle(fc_color,(int(p_x),int(p_y)),2,(255,0,0),-1)"""
           
            ## overlay sunglass over the face
            sx,sy,sh,sw = self.overlay_sunglass(key_pts)
            new_sunglass =  cv2.resize(sunglasses, (sw, sh), interpolation = cv2.INTER_CUBIC)
            ind = np.argwhere(new_sunglass[:,:,3] > 0)
            roi_color = fc_color[sy:sy+sh,sx:sx+sw]
            # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
            for i in range(3):
                roi_color[ind[:,0],ind[:,1],i] = new_sunglass[ind[:,0],ind[:,1],i]
            

            ## overlay moustache over the face
            mx,my,mh,mw = self.overlay_moustache(key_pts)
            new_moustache = cv2.resize(moustache,(mw,mh),interpolation = cv2.INTER_CUBIC)
            ind = np.argwhere(new_moustache[:,:,3] > 0)
            roi_color = fc_color[my:my+mh,mx:mx+mw]
            for i in range(3):
                roi_color[ind[:,0],ind[:,1],i] = new_moustache[ind[:,0],ind[:,1],i]
            
            """hx,hy,hh,hw = self.overlay_hat(key_pts)
            new_hat = cv2.resize(hat,(hw,hh),interpolation = cv2.INTER_CUBIC)
            ind = np.argwhere(new_hat[:,:,3] > 0)
            roi_color = fc_color[hy:hy+hh,hx:hx+hw]
            for i in range(3):
                roi_color[ind[:,0],ind[:,1],i] = new_hat[ind[:,0],ind[:,1],i]"""

        if(web):    
            _, jpeg = cv2.imencode('.jpg', fr)
            return jpeg.tobytes()
        return fr


    def overlay_sunglass(self,key_pts):
        """
            key_pts -> 2D array of keypoints detected by the model, shape-(68,2)
            return the x,y co-ordinate (top-left corner) where the sunglass need to be place 
                    and the height & width of the sunglass image
        """
        sx = int(key_pts[17, 0])
        sy = int(key_pts[17, 1])
        sh = int(abs(key_pts[27,1] - key_pts[34,1]))
        sw = int(abs(key_pts[17,0] - key_pts[26,0]))
        return sx,sy,sh,sw
    

    def overlay_moustache(self,key_pts):
        """
            key_pts-> 2D array of keypoints detected by the model, shape-(68,2)
            return the x,y co-ordinate (top-left corner) where the moustache need to be place 
                    and the height & width of the moustache image
        """
        mx = int(key_pts[2,0])
        my = int(key_pts[2,1])
        mh = int(abs(key_pts[5,1] - key_pts[2,1]))
        mw = int(abs(key_pts[14,0] - key_pts[2,0]))
        return mx,my,mh,mw
    

    """def overlay_hat(self,key_pts):
        hx = int(key_pts[0,0])-80
        hy = int(key_pts[19,1]-80)
        hh = int(abs(key_pts[0,1] - key_pts[19,1]+80))
        hw = int(abs(key_pts[16,0] - key_pts[0,0]))
        return hx,hy,hh,hw"""

if __name__ == '__main__':
    video=VideoCamera()
    #cv2.namedWindow("output",cv2.WINDOW_FREERATIO)
    #cv2.setWindowProperty("output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while(True):
        fr=video.get_frame(web=False)
        cv2.imshow("output",fr)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
