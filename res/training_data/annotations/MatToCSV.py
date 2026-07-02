import numpy as np
import scipy.io as sio
import pandas as pd

mat = sio.loadmat('./mpii_human_pose_v1_u12_1.mat', struct_as_record=False)

rel = mat['RELEASE']
print(rel,type(rel),rel.shape)

obj_rel = rel[0,0]
print(obj_rel._fieldnames)

annolist = obj_rel.__dict__['annolist']
img_tra = obj_rel.__dict__['img_train']
act = obj_rel.__dict__['act']
print(annolist.shape,img_tra.shape,act.shape)

n = annolist.shape[1]

data_arr = ['NAME','r ankle_X','r ankle_Y', 'r knee_X','r knee_Y', 'r hip_X','r hip_Y', 'l hip_X','l hip_Y', 'l knee_X','l knee_Y', 'l ankle_X','l ankle_Y','pelvis_X','pelvis_Y','thorax_X','thorax_Y','upper neck_X','upper neck_Y', 'head top_X','head top_Y', 'r wrist_X','r wrist_Y','r elbow_X','r elbow_Y', 'r shoulder_X','r shoulder_Y','l shoulder_X','l shoulder_Y','l elbow_X','l elbow_Y', 'l wrist_X','l wrist_Y','Scale','Activity','Category']
print(data_arr)

data = pd.DataFrame(columns=data_arr)
rows = []

for ix in range(0,annolist.shape[1]):
    if img_tra[0,ix] == 0:
        continue
#     print(ix)
    temp_arr = []
    obj_list = annolist[0,ix]
#     print(obj_list._fieldnames)
    obj_act = act[ix,0]
    
    rect =obj_list.__dict__['annorect']
    img_d = obj_list.__dict__['image']
    if rect.shape[0] ==0:
        continue
        
    obj_rect = rect[0,0]
    obj_img = img_d[0,0]
    
    
    if 'annopoints' not in obj_rect._fieldnames:
        continue
    
    name_d = obj_img.__dict__['name']
    name = name_d[0]
    temp_arr.append(name)
    annopoints = obj_rect.__dict__['annopoints']
    if annopoints.shape[0]==0:
        continue
    obj_points = annopoints[0,0]
    points = obj_points.__dict__['point']
#     if points.shape[1]!=16:
#         continue
#     print(points.shape)
    cnt = 0
    px = 0
    
    for n in range(0,32):
        temp_arr.append(-1)
    
    
    for px in range(0,points.shape[1]):
#         print(points.shape)
#         print('hello')
        po = points[0,px]
        po_id = po.__dict__['id']
        
        po_x = po.__dict__['x']
        po_y = po.__dict__['y']
        ind = 2*po_id[0][0]+1
#         print(ind)
        temp_arr[ind] = po_x[0][0]
        temp_arr[ind+1] = po_y[0][0]
       
        
        
    scale = obj_rect.__dict__['scale']
    temp_arr.append(scale[0][0])
    
    activity = act[ix,0]
    
    a_n = activity.act_name
    c_n = activity.cat_name
    
    if a_n.shape[0]==0:
        temp_arr.append(a_n)
    else:
        temp_arr.append(activity.act_name[0])
    if c_n.shape[0]==0:
        temp_arr.append(c_n)
    else:
        temp_arr.append(activity.cat_name[0])
    
    temp_data_f = pd.DataFrame([temp_arr],columns=data_arr)
    
    data = pd.concat([data,temp_data_f]) 
    rows.append(temp_arr)
    
    if ix%500==0:
        print(ix)
data.to_csv("mpii_annotations.csv", index=False)
