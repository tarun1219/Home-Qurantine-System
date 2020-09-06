import cv2
import datetime
import imutils
import numpy as np
from centroidtrack import CentroidTracker
from itertools import combinations
import math
import urllib.request
url="http://192.168.1.5:8080/shot.jpg"
p_path="MobileNetSSD_deploy.prototxt"
m_path="MobileNetSSD_deploy.caffemodel"
det=cv2.dnn.readNetFromCaffe(prototxt=p_path,caffeModel=m_path)
# det.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# det.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cls=["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
track=CentroidTracker(maxDisappeared=40, maxDistance=50)
def sup(boxes, overlapThresh):
    try:
        if len(boxes)==0:
            return[]
        if boxes.dtype.kind == "i":
            boxes=boxes.astype("float")
        pick=[]
        x1=boxes[:,0]
        y1=boxes[:,1]
        x2=boxes[:,2]
        y2=boxes[:,3]
        area=(x2-x1+1)*(y2-y1+1)
        indexs=np.argsort(y2)
        while len(indexs)>0:
            last=len(indexs)-1
            i=indexs[last]
            pick.append(i)
            xx1=np.maximum(x1[i],x1[indexs[:last]])
            yy1=np.maximum(y1[i],y1[indexs[:last]])
            xx2=np.minimum(x2[i],x2[indexs[:last]])
            yy2=np.minimum(y2[i],y2[indexs[:last]])
            w=np.maximum(0,xx2-xx1+1)
            h=np.maximum(0,yy2-yy1+1)
            overlap=(w*h)/area[indexs[:last]]
            indexs=np.delete(indexs,np.concatenate(([last],np.where(overlap>overlapThresh)[0])))
        return boxes[pick].astype("int")
    except Exception as e:
        print("Problem occured: "+e)
def main():
    #cap=cv2.VideoCapture('testvideo2.mp4')
    fps_st = datetime.datetime.now()
    fps=0
    total_frames = 0
    while True:
        imgResp=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgResp.read()),dtype='uint8')
        frame = cv2.imdecode(imgNp,-1)
        ret=frame
        frame=imutils.resize(frame,width=600)
        total_frames=total_frames+1
        (H,W)=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(frame,0.007843,(W, H),127.5)
        det.setInput(blob)
        person=det.forward()
        rects=[]
        for i in np.arange(0,person.shape[2]):
            con=person[0,0,i,2]
            if con>0.5:
                idx=int(person[0,0,i,1])

                if cls[idx]!="person":
                    continue

                pbox=person[0,0,i,3:7]*np.array([W,H,W,H])
                (sx,sy,ex,ey)=pbox.astype("int")
                rects.append(pbox)

        bound=np.array(rects)
        bound=bound.astype(int)
        rects=sup(bound,0.3)
        cen_dict=dict()
        objects=track.update(rects)
        for (objectId, bbox) in objects.items():
            x1,y1,x2,y2=bbox
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)
            cX=int((x1+x2)/2.0)
            cY=int((y1+y2)/2.0)


            cen_dict[objectId]=(cX,cY,x1,y1,x2,y2)

            # text = "ID: {}".format(objectId)
            # cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        red=[]
        for (id1, p1),(id2, p2) in combinations(cen_dict.items(),2):
            dx,dy=p1[0]-p2[0],p1[1]-p2[1]
            distance=math.sqrt(dx*dx+dy*dy)
            if distance<75.0:
                if id1 not in red:
                    red.append(id1)
                if id2 not in red:
                    red.append(id2)

        for id,box in cen_dict.items():
            if id in red:
                cv2.rectangle(frame,(box[2],box[3]),(box[4],box[5]),(0,0,255),2)
            else:
                cv2.rectangle(frame,(box[2],box[3]),(box[4],box[5]),(0,255,0),2)


        fps_et=datetime.datetime.now()
        time_diff=fps_et-fps_st
        if time_diff.seconds==0:
            fps=0.0
        else:
            fps=(total_frames/time_diff.seconds)
        time=str(datetime.datetime.now())
        time=time[10:19]
        time_text="Time:{0}".format(time)
        fps_text="FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text,(5,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, time_text,(5,325), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("Live Window", frame)
        key=cv2.waitKey(3)
        if key==ord('q'):
            break

    cv2.destroyAllWindows()
main()
