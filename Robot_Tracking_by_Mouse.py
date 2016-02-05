# ===========================================
# Object tracking by manually mouse clicking
# JH@KrappLab
# 05/02/2016
# ===========================================



# =============== python imports ===================
import cv2
import numpy as np
import time
import sys, os

import pandas as pd 
import pprint

# ===================== avi file name input ========
if len(sys.argv)>1:
    cap = cv2.VideoCapture(sys.argv[1])
    filename = os.path.basename(sys.argv[1])
    basename = os.path.splitext(filename)[0]
    print filename
else:
    print "ERROR: please input a filename!"
    sys.exit()

# ==================== Variable initilazation ===========
fRecord = 0
hx,hy = -1,-1
tx,ty = -1,-1
# ==================== Mouse callback function ===========

def mark(event,x,y,flags,param):
    global hx,hy,tx,ty
    global fRecord
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        hx,hy = x,y
        print 'head point: '+str(hx)+','+str(hy)
        fRecord = fRecord | 0x01
    if event == cv2.EVENT_RBUTTONDOWN:
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        tx,ty = x,y
        print 'tail point: '+str(tx)+','+str(ty)
        fRecord = fRecord | 0x02

cv2.namedWindow('raw')
cv2.setMouseCallback('raw',mark)

# ========= Open file for data logging ===========
if not os.path.exists('.\\'+basename + '_trajectory.txt'):
    f = open(basename + '_trajectory.txt', 'a+')
    f.write('frame,head_x,head_y,tail_x,tail_y\n')
    print "Attention: new file establishing..."
else:
    f = open(basename + '_trajectory.txt', 'a+')
    print "Attention: file existing... appending"

# =========== Main loop ============
fPause = 0
framecount = int (cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT ))
while(cap.isOpened()):
    # ========= processing video capture ====
    if fPause == 0:
        ret, img = cap.read()
        hx,hy = -1,-1
        tx,ty = -1,-1
    
        if ret is not 0:
            framenumber = int (cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) 
            print ('Frame: '+str(framenumber)+'/'+str(framecount))
     
            cv2.imshow('raw', img)
            cv2.moveWindow('raw', 0, 0)
                
            if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                print 'End!'
                break
        else:
            break
        fPause = 1

    # ======= processing key press ============
    hitkey = cv2.waitKey(10)
    if hitkey == 0x31:      # Key 1 : go back previous frame
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,framenumber-2)    
        fPause = 0
    if hitkey == 32:        # Key Space : go next frame, data logging
        if fRecord == 0x03:
            f.write((str(framenumber)) + ',' +
                   (str(hx)+','+str(hy)) + ',' + 
                   (str(tx)+','+str(ty)) + '\n')
        fRecord =0
        fPause = 0

    if hitkey == 27:        # Key ESC : quit the program
        break        

# ========= End of program ==========
cap.release()
cv2.destroyAllWindows()
f.close()



# ========================================
# ========= call : data sorting ==========
# ========================================

ans = raw_input('Do you want to sort the data? (y/n): ')
if ans is 'y':
    '''
    if len(sys.argv)>1:
        #cap = cv2.VideoCapture(sys.argv[1])
        filename = os.path.basename(sys.argv[1])
        basename = os.path.splitext(filename)[0]
        print filename
    else:
        print "ERROR: please input a filename!"
        sys.exit()
    '''
    # ===================== Sorting data ========

    df = pd.read_csv('.\\'+basename+'_trajectory.txt')
    #print.pprint(df)
    df = df.drop_duplicates(subset='frame', keep = 'last')
    df= df.sort_values(by = ['frame'], ascending =[True])
    pprint.pprint(df)

    # ===================== Saving data ========

    df.to_csv(basename+'_trajectory.csv', index=False)
    print "Success: Data sorted! "
else:
    print "Warning: Data unsorted... "
    pass
