"""Propose a fresh EfficientTAM box from a saved object's colour reference.

This only initializes a mask; EfficientTAM still performs segmentation/tracking.
Absent or ambiguous candidates require the existing manual selection flow.
"""
import cv2
import numpy as np
from .object_profiles import load_prompt


def propose_box(bgr,profile,role,serial):
    reference,mask,_=load_prompt(profile,role,serial)
    if mask is None:raise ValueError('Saved object profile has no reference mask')
    if bgr.shape!=reference.shape:raise ValueError('Live/reference image dimensions differ')
    binary=(mask>0).astype(np.uint8)
    core=cv2.erode(binary,np.ones((3,3),np.uint8),iterations=1)
    if np.count_nonzero(core)<40:core=binary
    hsv=cv2.cvtColor(reference,cv2.COLOR_BGR2HSV)
    core=core*((hsv[:,:,1]>=60)&(hsv[:,:,2]>=40)).astype(np.uint8)
    if np.count_nonzero(core)<30:raise ValueError('Reference colour is not distinctive enough for automatic selection')
    hist=cv2.calcHist([hsv],[0,1],core,[36,32],[0,180,0,256]);hist=cv2.GaussianBlur(hist,(5,5),0)
    hist_norm=hist/max(float(hist.sum()),1e-12)
    projection_hist=hist/max(float(hist.max()),1e-12)*255
    live=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    probability=cv2.calcBackProject([live],[0,1],projection_hist,[0,180,0,256],1)
    candidate=((probability>=40)&(live[:,:,1]>=60)&(live[:,:,2]>=40)).astype(np.uint8)*255
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    reference_area=int(np.count_nonzero(binary));height,width=bgr.shape[:2];proposals=[]
    for contour in contours:
        area=float(cv2.contourArea(contour));x,y,w,h=cv2.boundingRect(contour)
        if not max(50,.2*reference_area)<=area<=min(6*reference_area,.05*height*width):continue
        if min(w,h)<6 or area/(w*h)<.3:continue
        region=np.zeros((height,width),np.uint8);cv2.drawContours(region,[contour],-1,255,-1)
        current=cv2.calcHist([live],[0,1],region,[36,32],[0,180,0,256]);current=cv2.GaussianBlur(current,(5,5),0);current/=max(float(current.sum()),1e-12)
        similarity=float(np.minimum(hist_norm,current).sum())
        if similarity<.5:continue
        pad=max(2,int(round(.08*max(w,h))));x0=max(0,x-pad);y0=max(0,y-pad)
        box=[x0,y0,min(width,x+w+pad)-x0,min(height,y+h+pad)-y0]
        proposals.append(dict(box_xywh=box,reference_histogram_intersection=similarity,colour_area_pixels=area))
    if len(proposals)!=1:raise ValueError(f'Automatic mask found {len(proposals)} plausible objects; select the target manually')
    return proposals[0]['box_xywh'],dict(method='saved_reference_colour_proposal_then_EfficientTAM',candidate_count=1,**proposals[0])
