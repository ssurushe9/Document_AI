import base64
import json
import os
import cv2
from google.cloud import storage
from PIL import Image
import requests
import io
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import copy

storage_client = storage.Client()

#Extract the dataframe using AWS Textract from given image and 
#add x_center and y_center to it
def get_ocr_tesstract_df(image):    
    w,h = image.size
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    payload = [{ "content": str(base64.b64encode(byte_im))[2:-1]}]
    
    
    url = "https://y6bedugwx2.execute-api.us-east-1.amazonaws.com/dev"
    headers = {
    'content-type': "application/json",
    'x-api-key': "ASipZo6Xp22djmsxmX72435epWTV7PWg9mMo24Xc"
    }
    
    # url = "https://udj0ogrn19.execute-api.us-east-1.amazonaws.com/test"
    # headers = {
    # 'content-type': "application/json",
    # 'x-api-key': "ASipZo6Xp22djmsxmX72435epWTV7PWg9mMo24Xc"
    # }
    response = requests.request("POST", url, json=payload, headers=headers)
    data_icw = json.dumps(response.json())
    data_icw = json.loads(data_icw)
    
    # print(data_icw)
    dic_aws ={}
    columns = []
    lines = []
    df = pd.DataFrame(columns=['left','top','width','height','text','conf'])
    for item in data_icw[0]["Blocks"]:
        if item["BlockType"] == "WORD":
            bbox_left = item["Geometry"]["BoundingBox"]["Left"]*w
            bbox_right = item["Geometry"]["BoundingBox"]["Top"] *h
            bbox_centre =  item["Geometry"]["BoundingBox"]["Width"]*w
            column_centre = item["Geometry"]["BoundingBox"]["Height"]*h
            text = item["Text"]
            conf = item['Confidence']
            data =pd.DataFrame({'left':bbox_left,'top':bbox_right,'width':bbox_centre,'height':column_centre,'text':text,'conf':conf},index=[0])
            df = df.append(data, ignore_index = True)
            data_req=df.filter(['left','top','width','height','text','conf'])
            data_req.replace("", float("NaN"), inplace=True)
            df = data_req[data_req['text'].notna()]
            df['x_center']=df['left']+df['width']/2
            df['right']=df['left']+df['width']
            # df['x_center']=df['x_center'].astype(int)
            df['y_center']=df['top']+df['height']/2
            df['bottom']=df['top']+df['height']
            # df['y_center']=df['y_center'].astype(int)
            
            df.to_csv('ouput_df.csv')
    
    return df
    
    
#updated
def dataframe_normalisation(df_req,height_template,width_template,height_object,width_object,object_x_min,object_y_min): 
    df=copy.deepcopy(df_req)
    df['top']=df['top']-object_y_min
    df['bottom']=df['bottom']-object_y_min
    df['y_center']=df['y_center']-object_y_min
    df['left']=df['left']-object_x_min
    df['right']=df['right']-object_x_min
    df['x_center']=df['x_center']-object_x_min
    
    df['top']=df['top']*height_template/height_object
    df['bottom']=df['bottom']*height_template/height_object
    df['height']=df['height']*height_template/height_object
    df['left']=df['left']*width_template/width_object
    df['right']=df['right']*width_template/width_object
    df['width']=df['width']*width_template/width_object                   
    df['y_center']=df['y_center']*height_template/height_object          
    df['x_center']=df['x_center']*width_template/width_object 
    
    df.to_csv('normalised_df.csv')
    return df
 
#load json data
def load_json(json_path):
    with open(json_path, 'r') as myfile:
        data=myfile.read()
    dict_obj = json.loads(data)  
    return dict_obj
    

#merge and update two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z
    
    
#resize image based on height or width without losing aspect ratio
def object_resize_with_aspect_ratio(object_height,object_width, width = None, height = None):

    dim = None
    (h, w) = object_height,object_width
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (height,(w * r))
    elif height is None:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = ((h * r),width)
    
    else:
        dim=(height,width)

    # return the resized image
    return dim    
    
# split a dictionary with list to list of dictionaries
def split_list_to_dict(data):
    keys = data.keys()
    vals = zip(*[data[k] for k in keys])
    result = [dict(zip(keys, v)) for v in vals]
    return result


# return the maximum index of matching key from all texts from df
#this is used to get the index of each keyword
def get_matching_key(search_string, window_size,text_list):
    
    fuzzy_ratio=[]
    index_no=0
    
    for each_word in text_list[:-window_size]:
        combined_word=''

        for j in range(0,window_size):
            combined_word=combined_word+' '+text_list[index_no+j]
        
        # fuzzy_ratio=fuzz.ratio(search_string, combined_word)

        fuzzy_ratio.append(fuzz.ratio(search_string, combined_word))

        index_no=index_no+1
        
    if max(fuzzy_ratio)>=50:
        max_index = fuzzy_ratio.index(max(fuzzy_ratio))
    return max_index    

#get the center of keys from list of keys
def get_xy_key(df,list_of_keys):
    #key_x and key_y are the center points of the keys    
    key_x=0
    key_y=0
    key_y_top=0
    key_x_left=0
    
    search_string=' '.join(list_of_keys)
    window_size=len(list_of_keys)
    text_list = df['text'].tolist()
    try:
        max_index=get_matching_key(search_string, window_size,text_list)

        for i in range(0,window_size):
            row = df.iloc[max_index+i]
            key_x=key_x+row['x_center']
            key_y=row['y_center']
            key_y_top=row['top']   
            key_x_left=row['left']

        key_x=(key_x/window_size)
        key_y=(key_y)
        key_y_top=(key_y_top)
    except:
        
        #print('No matching key found for ',search_string)
        pass

    return key_x,key_y,key_y_top,key_x_left   

# get the tokens corresponding to each key
def get_dxy_values(df,dx_min,dx_max,dy_min,dy_max):
    extracted_data=''   
    count=0 
    left,top,right,bottom=0,0,0,0
    confidence=0
    for index,row in df.iterrows():

        if row['y_center']  >= dy_min and row['y_center']<=  dy_max:
            if row['x_center']>= dx_min and row['x_center']<= dx_max:
                extracted_data=extracted_data+' '+row['text']
                confidence=confidence+row['conf']
                      
                if count==0:
                    top=row['top']
                    left=row['left']  
                    right=row['right']
                    bottom=row['bottom']
                else:
                    if row['top']<top:
                        top=row['top']
                    if row['left']<left:
                        left=row['left']
                    if row['right']>right:
                        right=row['right']
                    if row['bottom']>bottom:
                        bottom=row['bottom']
                      
                count=count+1    
    width=right-left
    height=bottom-top
    
    if count>0:
        confidence=confidence/count
        
    value_coordinates=[left,top,width,height]
    return extracted_data,value_coordinates,confidence    
    
#mapping ocr df and single keys with single vales
def mapping_single_key_single_value(df,data_dict,key,list_of_keys,list_of_next_keys,height_template,width_template,key_x, key_y,key_y_top,key_x_left):
    # print(data_dict[key])
    dx_min,dx_max=data_dict[key][1]
    dy_min,dy_max=data_dict[key][2]  
        
    
    if list_of_next_keys:
        if list_of_next_keys[0]=='End of Row':
            
            dy_max=height_template
            
            # pass
            # dy_max=dy_max+abs(height_template-key_y)

        elif list_of_next_keys[0]=='End of Column':
            # pass
            dx_max=width_template
            # dx_max=dx_max+abs(width_template-key_x)
        
        else:        
            next_key_x, next_key_y,next_key_y_top,next_key_x_left= get_xy_key(df,list_of_next_keys) 
            dy_max=next_key_y_top-key_y_top-abs(dy_min)

    token_data,value_coordinates,confidence=get_dxy_values(df,dx_min+key_x,dx_max+key_x,dy_min+key_y,dy_max+key_y)
    return token_data,value_coordinates,confidence
    
    
def filter_empty_dict(key_val, coord,conf):
    kv_new=[]
    coord_new=[]
    conf_new=[]
    for i,j,k in zip(key_val,coord,conf):
        flag=all(x=='' for x in i.values())
        if flag==False:
            kv_new.append(i)
            coord_new.append(j)
            conf_new.append(k)
            
    return kv_new, coord_new,conf_new
    
    
def generate_dymin_dy_max_table(df,dx_min,dx_max,dy_min,stop_key_y_top):
    # print(reference_key)
    # print(data_dict)

    
    dy_top_list=[]
    for index,row in df.iterrows():
        if row['x_center']>= dx_min and row['x_center']<= dx_max:
            if row['y_center']>= dy_min: 
                dy_top_list.append(row['top'])
            # extracted_data=extracted_data+' '+row['text']
    
    dy_top_list.append(stop_key_y_top)
    return dy_top_list
    
#mapping ocr df and single keys with multiple vales
def mapping_table_with_variable_height(df,data_dict,key,list_of_keys,reference_key,stop_key_y_top,key_x, key_y,key_y_top,key_x_left):
    # print(end_row,dy_row)
    token_data_all=[]
    value_coordinates_all=[]
    confidence_all=[]
    dx_min,dx_max=data_dict[key][1]
    dy_min_list=[]
    dy_max_list=[]
    
    list_of_keys=data_dict[reference_key][0]
    dx_min_ref,dx_max_ref=data_dict[reference_key][1]
    
    dy_min_ref,dy_max_ref=data_dict[reference_key][2]
    
    key_x_ref,key_y_ref,_,_=get_xy_key(df,list_of_keys)
    
    
    dy_top_list=generate_dymin_dy_max_table(df,dx_min_ref+key_x_ref,dx_max_ref+key_x_ref,dy_min_ref+key_y_ref,stop_key_y_top)
    
    for k in range(0,len(dy_top_list)-1):
        dy_min_list.append(dy_top_list[k])
    
    for k in range(1,len(dy_top_list)):
        dy_max_list.append(dy_top_list[k])    
    
    for k in range(0,len(dy_max_list)):
        token_data,value_coordinates,confidence=get_dxy_values(df,dx_min+key_x,dx_max+key_x,dy_min_list[k],dy_max_list[k])
        token_data_all.append(token_data)
        value_coordinates_all.append(value_coordinates)
        confidence_all.append(confidence)
    return token_data_all,value_coordinates_all,confidence_all

#mapping ocr df and single keys with multiple vales
def mapping_single_key_multiple_row(df,data_dict,key,list_of_keys,delta,end_row,key_x, key_y,key_y_top,key_x_left):
    # print(end_row,dy_row)
    token_data_all=[]
    value_coordinates_all=[]
    confidence_all=[]
    dx_min,dx_max=data_dict[key][1]
    dy_min,dy_max=data_dict[key][2]
    # key_x, key_y,key_y_top,key_x_left= get_xy_key(df,list_of_keys)
    
    i=0
    next_row=dy_max
    count=0
    while (next_row<=end_row-key_y_top):

        if i==0:
            token_data,value_coordinates,confidence=get_dxy_values(df,dx_min+key_x,dx_max+key_x,dy_min+key_y,dy_max+key_y)
            next_row=next_row+delta
        else:
            dy_min=dy_min+delta
            dy_max=dy_max+delta
            token_data,value_coordinates,confidence=get_dxy_values(df,dx_min+key_x,dx_max+key_x,dy_min+key_y,dy_max+key_y)
            next_row=next_row+delta
        
        i=i+1

        token_data_all.append(token_data)
        value_coordinates_all.append(value_coordinates)
        confidence_all.append(confidence)
    return token_data_all,value_coordinates_all,confidence_all
    
   
#function to get all data of single_key with single_value
def single_key_single_value_extract_data(df,json_data,height_template,width_template,non_keys):
    
    # print(df)
    kv_data={}
    value_coordinates_data={}
    confidence_data={}
    previous_list_of_keys=[]
    key_val_conf_coordinates=[]
    previous_key_x, previous_key_y,previous_key_y_top,previous_key_x_left=0,0,0,0
    for key in json_data:
        list_of_next_keys=[]
        
        if key not in non_keys:
            list_of_keys=json_data[key][0] 
            if previous_list_of_keys==list_of_keys:
                key_x, key_y,key_y_top,key_x_left=previous_key_x, previous_key_y,previous_key_y_top,previous_key_x_left
            else: 
                key_x, key_y,key_y_top,key_x_left= get_xy_key(df,list_of_keys) 
            # print(key)
            kv_data[key]=''            
            
            # print(json_data)
            try:
                if json_data[key][3]: 
                    list_of_next_keys=json_data[key][3]
            except:
                pass
            kv,value_coordinates,confidence=mapping_single_key_single_value(df,json_data,key,list_of_keys,list_of_next_keys,height_template,width_template,key_x, key_y,key_y_top,key_x_left)
            kv_data[key],value_coordinates_data[key+'- Value Coordinates'],confidence_data[key]=kv,value_coordinates,confidence
            previous_list_of_keys=list_of_keys
            previous_key_x,previous_key_y,previous_key_y_top,previous_key_x_left =key_x, key_y,key_y_top,key_x_left

            key_val_conf_coordinates.append({'key':key,'value':kv,'confidence':confidence,'xmin':value_coordinates[0],'ymin':value_coordinates[1],'xmax':value_coordinates[2],'ymax':value_coordinates[3],'regex':None})
 
    return kv_data,value_coordinates_data,confidence_data,key_val_conf_coordinates


#function to get all data of single_key with multiple_values
def single_key_multiple_row_extract_data(df, json_data, height_template,width_template,non_keys):
    
    delta=json_data['Delta']
    remove_end=json_data['Remove End']
    end_row=height_template-remove_end
    
    kv_data_list=[]
    value_coordinates_list=[]
    kv_data={}
    confidence_data={}
    value_coordinates_data={}
    previous_list_of_keys=[]
    key_val_conf_coordinates=[]
    previous_key_x, previous_key_y,previous_key_y_top,previous_key_x_left=0,0,0,0

    for key in json_data:
        if key not in non_keys:
            kv_data[key]=''
            list_of_keys=json_data[key][0] 
            if previous_list_of_keys==list_of_keys:
                key_x, key_y,key_y_top,key_x_left=previous_key_x, previous_key_y,previous_key_y_top,previous_key_x_left
            else: 
                key_x, key_y,key_y_top,key_x_left= get_xy_key(df,list_of_keys) 
            
            # print(json_data)
            kv,value_coordinates,confidence=mapping_single_key_multiple_row(df,json_data,key,list_of_keys,delta,end_row,key_x, key_y,key_y_top,key_x_left)
            kv_data[key],value_coordinates_data[key+'- Value Coordinates'],confidence_data[key]=kv,value_coordinates,confidence
            
            
            previous_list_of_keys=list_of_keys
            previous_key_x,previous_key_y,previous_key_y_top,previous_key_x_left =key_x, key_y,key_y_top,key_x_left
            
            #key_val_conf_coordinates.append({'key':key,'value':kv,'confidence':confidence,'xmin':value_coordinates[0],'ymin':value_coordinates[1],'xmax':value_coordinates[2],'ymax':value_coordinates[3],'regex':None})

    kv_data_list=split_list_to_dict(kv_data)
    value_coordinates_list=split_list_to_dict(value_coordinates_data)
    confidence_data_list=split_list_to_dict(confidence_data)
    
    for kv,vc,cc in zip(kv_data_list,value_coordinates_list,confidence_data_list):
        for key in kv:
            key_val_conf_coordinates.append({'key':key,'value':kv[key],'confidence':cc[key],'xmin':vc[key+'- Value Coordinates'][0],'ymin':vc[key+'- Value Coordinates'][1],'xmax':vc[key+'- Value Coordinates'][2],'ymax':vc[key+'- Value Coordinates'][3],'regex':None})
    
    return kv_data_list,value_coordinates_list,confidence_data_list,key_val_conf_coordinates 
     
    
    
#function to get all data of single_key with multiple_values
def table_variable_height_extract_data(df, json_data, height_template,width_template,non_keys):
    
    reference_key='Reference Key'
    stop_key=json_data['Stop Key'][0]

    kv_data_list=[]
    value_coordinates_list=[]
    confidence_data_list=[]
    kv_data={}
    confidence_data={}
    value_coordinates_data={}
    previous_list_of_keys=[]
    key_val_conf_coordinates=[]
    previous_key_x, previous_key_y,previous_key_y_top,previous_key_x_left=0,0,0,0
    
    if stop_key:
        _, _,stop_key_y_top,_= get_xy_key(df,stop_key) 
        if stop_key_y_top ==0:
            stop_key_y_top=height_template        
    else:
        stop_key_y_top=height_template

    for key in json_data:
        if key not in non_keys:
            kv_data[key]=''
            list_of_keys=json_data[key][0] 
            if previous_list_of_keys==list_of_keys:
                key_x, key_y,key_y_top,key_x_left=previous_key_x, previous_key_y,previous_key_y_top,previous_key_x_left
            else: 
                key_x, key_y,key_y_top,key_x_left= get_xy_key(df,list_of_keys) 
            
            # print(json_data)
            kv,value_coordinates,confidence=mapping_table_with_variable_height(df,json_data,key,list_of_keys,reference_key,stop_key_y_top,key_x, key_y,key_y_top,key_x_left)
            kv_data[key],value_coordinates_data[key+'- Value Coordinates'],confidence_data[key]=kv,value_coordinates,confidence
            previous_list_of_keys=list_of_keys
            previous_key_x,previous_key_y,previous_key_y_top,previous_key_x_left =key_x, key_y,key_y_top,key_x_left
            
            # print(kv)
#             print(value_coordinates)
            

    kv_data_list=split_list_to_dict(kv_data)
    value_coordinates_list=split_list_to_dict(value_coordinates_data)
    confidence_data_list=split_list_to_dict(confidence_data)
    
    kv_data_list,value_coordinates_list,confidence_data_list=filter_empty_dict(kv_data_list, value_coordinates_list,confidence_data_list)
    
    for kv,vc,cc in zip(kv_data_list,value_coordinates_list,confidence_data_list):
        for key in kv:
            key_val_conf_coordinates.append({'key':key,'value':kv[key],'confidence':cc[key],'xmin':vc[key+'- Value Coordinates'][0],'ymin':vc[key+'- Value Coordinates'][1],'xmax':vc[key+'- Value Coordinates'][2],'ymax':vc[key+'- Value Coordinates'][3],'regex':None})
            
    return kv_data_list,value_coordinates_list,confidence_data_list,key_val_conf_coordinates
    
    

# retrive single values if key is not present
def no_key_single_value_extract_data(df,json_data,non_keys):
    
    
    extracted_data={} 
    value_coordinates_data={}
    confidence_data={}
    key_val_conf_coordinates=[]
    
    
    for key,value in json_data.items():
        if key not in non_keys:
            dx_min,dx_max=json_data[key][0] 
            dy_min,dy_max= json_data[key][1] 
            
            count=0
            text = ''
            left,top,right,bottom=0,0,0,0
            
            extracted_data_no_key,value_coordinates,confidence=get_dxy_values(df,dx_min,dx_max,dy_min,dy_max)
                        
            value_coordinates_data[key+'- Value Coordinates']=value_coordinates
            extracted_data[key] = extracted_data_no_key  
            confidence_data[key]=confidence
        # print(text)
            key_val_conf_coordinates.append({'key':key,'value':extracted_data_no_key,'confidence':confidence,'xmin':value_coordinates_data[key+'- Value Coordinates'][0],'ymin':value_coordinates_data[key+'- Value Coordinates'][1],'xmax':value_coordinates_data[key+'- Value Coordinates'][2],'ymax':value_coordinates_data[key+'- Value Coordinates'][3],'regex':None})
    return extracted_data,value_coordinates_data,confidence_data,key_val_conf_coordinates


def get_page_value_coordinates(value_coordinates_data,object_x_min,object_y_min,height_object,width_object,height_template,width_template):
    value_coordinates_scaled={}
    for vc in value_coordinates_data:
        left,top,width,height=value_coordinates_data[vc]        
        left=(left*width_object/width_template)+object_x_min
        top=(top*height_object/height_template)+object_y_min
        width=width*width_object/width_template
        height=height*height_object/height_template        
        value_coordinates_scaled[vc]=[np.floor(left),np.floor(top),np.ceil(width),np.ceil(height)]
    return value_coordinates_scaled
                                

#updated
def extract_kv_data(obj_position,json_data,object_x_min,object_y_min,df_req,non_keys,height_object,width_object):
    
    kv_object=[]
    vc_object=[]
    confidence=[]
    key_val_conf_coordinates=[]
    
    width_template,height_template=json_data['Image Size']
            
    temp_type=json_data['Type']

    kv_data={}
    if temp_type=='Single Key Single Value':

        height_resized_template,width_resized_template=object_resize_with_aspect_ratio(height_object,width_object, width = width_template)
        df_norm=dataframe_normalisation(df_req,height_resized_template,width_resized_template,height_object,width_object,object_x_min,object_y_min)

        kv_data,value_coordinates_data,conf,kv_conf_coord=single_key_single_value_extract_data(df_norm,json_data,height_resized_template,width_resized_template,non_keys)
        kv_object.append(kv_data)
        value_coordinates_data_rescaled=get_page_value_coordinates(value_coordinates_data,object_x_min,object_y_min,height_object,width_object,height_resized_template,width_resized_template)
        vc_object.append(value_coordinates_data_rescaled)
        confidence.append(conf)
        key_val_conf_coordinates.append(kv_conf_coord)

    
    elif temp_type=='Single Key Multiple Row Variable Height':  
                
        height_resized_template,width_resized_template=object_resize_with_aspect_ratio(height_object,width_object, width = width_template)
        df_norm=dataframe_normalisation(df_req,height_resized_template,width_resized_template,height_object,width_object,object_x_min,object_y_min)
        # try:
        kv_data_list,value_coordinates_list,conf_list,kv_conf_coord_list=table_variable_height_extract_data(df_norm, json_data, height_resized_template,width_resized_template,non_keys)
        # print(kv_data_list)
        for kv in kv_data_list:
            kv_object.append(kv)
        for vc in value_coordinates_list:
            vc_scaled=get_page_value_coordinates(vc,object_x_min,object_y_min,height_object,width_object,height_resized_template,width_resized_template)
            vc_object.append(vc_scaled)
        for cf in conf_list:
            confidence.append(cf)
            
        for kvcc in kv_conf_coord_list:
            key_val_conf_coordinates.append(kvcc)
        # except:
        #     pass
                
    elif temp_type=='Single Key Multiple Row':  
                
        height_resized_template,width_resized_template=object_resize_with_aspect_ratio(height_object,width_object, width = width_template)
        # try:
        df_norm=dataframe_normalisation(df_req,height_resized_template,width_resized_template,height_object,width_object,object_x_min,object_y_min)
        kv_data_list,value_coordinates_list,conf_list,kv_conf_coord_list=single_key_multiple_row_extract_data(df_norm, json_data, height_resized_template,width_resized_template,non_keys)
        # print(kv_data_list)
        for kv in kv_data_list:
            kv_object.append(kv)
        for vc in value_coordinates_list:
            vc_scaled=get_page_value_coordinates(vc,object_x_min,object_y_min,height_object,width_object,height_resized_template,width_resized_template)
            vc_object.append(vc_scaled)
        for cf in conf_list:
            confidence.append(cf)
        for kvcc in kv_conf_coord_list:
            key_val_conf_coordinates.append(kvcc)
        # except:
        #     pass


            
    elif temp_type=='No Key Single Value':
                
        # width_object=right_max-left_min
        # height_object=bottom_max-top_min  
        # object_x_min,object_y_min=left_min,top_min
        df_norm=dataframe_normalisation(df_req,height_template,width_template,height_object,width_object,object_x_min,object_y_min)


        kv_data,value_coordinates_data,conf,kv_conf_coord=no_key_single_value_extract_data(df_norm,json_data,non_keys)
        kv_object.append(kv_data)
        value_coordinates_data_rescaled=get_page_value_coordinates(value_coordinates_data,object_x_min,object_y_min,height_object,width_object,height_template,width_template)
        vc_object.append(value_coordinates_data_rescaled)
        
        confidence.append(conf)
        key_val_conf_coordinates.append(kv_conf_coord)
        
        
    elif temp_type=='No Key Multiple Value':
                
        # width_object=right_max-left_min
        # height_object=bottom_max-top_min  
        # object_x_min,object_y_min=left_min,top_min
        # height_resized_template,width_resized_template=object_resize_with_aspect_ratio(height_object,width_object, width = width_template)
        df_norm=dataframe_normalisation(df_req,height_template,width_template,height_object,width_object,object_x_min,object_y_min)


        kv_data,value_coordinates_data,conf,kv_conf_coord=no_key_multiple_value_extract_data(df_norm,json_data,non_keys,height_template,obj_position)
        kv_object.append(kv_data)
        value_coordinates_data_rescaled=get_page_value_coordinates(value_coordinates_data,object_x_min,object_y_min,height_object,width_object,height_template,width_template)
        vc_object.append(value_coordinates_data_rescaled)
        
        confidence.append(conf)
        key_val_conf_coordinates.append(kv_conf_coord)

                # except:
                #     pass
    return kv_object,vc_object,confidence,key_val_conf_coordinates
    
    
#updated
# (df,data_dict,key,list_of_keys,reference_key,stop_key_y_top,key_x, key_y,key_y_top,key_x_left):
def no_key_multiple_value_extract_data(df,json_data,non_keys,stop_key_y_top,obj_position):
    
    extracted_data={} 
    value_coordinates_data={}
    confidence_data={}
    key_val_conf_coordinates=[]
    dy_min_list=[]
    dy_max_list=[]
    reference_key='Reference Key'
    no_of_lines=json_data['Lines Count']
    
    
    dx_min,dx_max=json_data[reference_key][0]
    
    dy_min,dy_max=json_data[reference_key][1]    
    
    dy_top_list=generate_dymin_dy_max_table(df,dx_min,dx_max,dy_min,stop_key_y_top)
    
    for k in range(0,len(dy_top_list)-1):
        dy_min_list.append(dy_top_list[k])
    
    for k in range(1,len(dy_top_list)):
        dy_max_list.append(dy_top_list[k])    
    
    # print(dy_min_list)
    # print(dy_max_list)
    

    for key,value in json_data.items():
        if key not in non_keys:
            dx_min,dx_max=json_data[key][0] 
            dy_min,dy_max= json_data[key][1] 
            line_no=json_data[key][2]

            # print(obj_position)

            count=0
            text = ''
            left,top,right,bottom=0,0,0,0
            # print(len(dy_min_list),len(dy_max_list),no_of_lines)
            if len(dy_min_list)==no_of_lines or len(dy_max_list)==no_of_lines:

                extracted_data_no_key,value_coordinates,confidence=get_dxy_values(df,dx_min,dx_max,dy_min_list[line_no-1],dy_max_list[line_no-1])

                value_coordinates_data[key+'- Value Coordinates']=value_coordinates
                extracted_data[key] = extracted_data_no_key  
                confidence_data[key]=confidence
                # print(text)
                key_val_conf_coordinates.append({'key':key,'value':extracted_data_no_key,'confidence':confidence,'xmin':value_coordinates_data[key+'- Value Coordinates'][0],'ymin':value_coordinates_data[key+'- Value Coordinates'][1],'xmax':value_coordinates_data[key+'- Value Coordinates'][2],'ymax':value_coordinates_data[key+'- Value Coordinates'][3],'regex':None})
            else:
                if obj_position=='Last':
                    if line_no<=len(dy_min_list):
                        extracted_data_no_key,value_coordinates,confidence=get_dxy_values(df,dx_min,dx_max,dy_min_list[line_no-1],dy_max_list[line_no-1])

                        value_coordinates_data[key+'- Value Coordinates']=value_coordinates
                        extracted_data[key] = extracted_data_no_key  
                        confidence_data[key]=confidence
                        # print(text)
                        key_val_conf_coordinates.append({'key':key,'value':extracted_data_no_key,'confidence':confidence,'xmin':value_coordinates_data[key+'- Value Coordinates'][0],'ymin':value_coordinates_data[key+'- Value Coordinates'][1],'xmax':value_coordinates_data[key+'- Value Coordinates'][2],'ymax':value_coordinates_data[key+'- Value Coordinates'][3],'regex':None})

                if obj_position=='First':
                    
                    # print('Line number is ',line_no)
                    # # print('len is ',len(dy_min_list))
                    # # print('dy_min list is ',dy_min_list)
                    # # print('dy_max list is ',dy_max_list)
                    # print('index is',(no_of_lines-len(dy_min_list)))
                    dy_min_list_extnd = [0] * (no_of_lines-len(dy_min_list))
                    dy_min_list_extnd.extend(dy_min_list)
                    dy_max_list_extnd = [0] * (no_of_lines-len(dy_min_list))
                    dy_max_list_extnd.extend(dy_max_list)
                    
                    if line_no>(no_of_lines-len(dy_min_list)):
                        
#                         print(line_no)
                        
#                         print(dy_max_list_extnd)
#                         print(dy_min_list_extnd)
                        extracted_data_no_key,value_coordinates,confidence=get_dxy_values(df,dx_min,dx_max,dy_min_list_extnd[line_no-1],dy_max_list_extnd[line_no-1])

                        value_coordinates_data[key+'- Value Coordinates']=value_coordinates
                        extracted_data[key] = extracted_data_no_key  
                        confidence_data[key]=confidence
                        # print(text)
                        key_val_conf_coordinates.append({'key':key,'value':extracted_data_no_key,'confidence':confidence,'xmin':value_coordinates_data[key+'- Value Coordinates'][0],'ymin':value_coordinates_data[key+'- Value Coordinates'][1],'xmax':value_coordinates_data[key+'- Value Coordinates'][2],'ymax':value_coordinates_data[key+'- Value Coordinates'][3],'regex':None})

    return extracted_data,value_coordinates_data,confidence_data,key_val_conf_coordinates
    

#updated
#returns each page objects
def get_kv_loss_runs_page(pages_path,object_list,input_bucket,page,row_list):    
    kv_object=[]
    vc_object=[]
    conf_object=[]
    kv_conf_object=[]
    # object_name_list=[]
    non_keys=['Image Size','Delta','Type','Remove End','Reference Key','Stop Key','Lines Count']
    blob = storage_client.bucket(input_bucket).get_blob(pages_path)
    blob.download_to_filename(page)
    read_page = Image.open(page)
    df_full=get_ocr_tesstract_df(read_page)
    
    total_object_row=[d for d in object_list if d['name'] in row_list]
    total_object=len(total_object_row)
    # print('Total object',total_object)
    # print('Object',object_list)
    first_obj=1
    
    count=1
    for each_object in object_list:
        obj_position=''
        if each_object['name'] in row_list:
            if count==total_object:
                obj_position='Last'
            if count==first_obj:
                obj_position='First'
            
            count=count+1
            # print('In loop',count,total_object)
            
            
        # print('aa',count,total_object)
        # print(obj_position,pages_path)
        
        df_req=df_full.loc[(df_full['x_center']>=each_object['xmin']) & (df_full['x_center']<=each_object['xmax']) & (df_full['y_center']>=each_object['ymin']) & (df_full['y_center']<=each_object['ymax'])]
        object_x_min=each_object['xmin']
        object_y_min=each_object['ymin']  
        left_min=df_req['left'].min()
        right_max=df_req['right'].max()
        top_min=df_req['top'].min()
        bottom_max=df_req['bottom'].max()

        
        height_object=each_object['ymax']-each_object['ymin']
        width_object=each_object['xmax']-each_object['xmin']
        object_name=each_object['name']     
        
         
        # if object_name in available_templates:
            
        json_data=load_json('json_data/'+object_name+'.json')

        temp_type=json_data['Type']

        if temp_type=='No Key Single Value':

            width_object=right_max-left_min
            height_object=bottom_max-top_min  
            object_x_min,object_y_min=left_min,top_min

        kv,vc,confidence,key_val_conf_coordinates=extract_kv_data(obj_position,json_data,object_x_min,object_y_min,df_req,non_keys,height_object,width_object)
        
        for d1,d2 in zip(kv,vc): 
            kv_object.append([object_name,d1])
            vc_object.append([object_name,d2])
        
        
        # count=count+1
            
    return kv_object,vc_object
            
#Extract data from pdf
def extract_dataframe_from_pdf_json(input_json,template_json_path,submission_id,doc_id,image_dir,input_bucket,doc_type):   
    
    df_loss_summary = []
    df_loss_details = []
    
    template_dictionary=load_json(template_json_path)

    id_list=[]
    row_list=[]
    available_templates=[]
    for data in template_dictionary:
        available_templates.append(data)
        if (template_dictionary[data]['category']=='id'):
            id_list.append(data)
        elif (template_dictionary[data]['category']=='row'):
            row_list.append(data)

    Objects=input_json['Objects']
    
    for pages in Objects:    
        page_path=image_dir+pages

        object_list=Objects[pages] 

        if isinstance(object_list, dict):
            if object_list['name']=='Unknown':
                pass
        else:
            object_list_filtered = [d for d in object_list if d['name'] in available_templates]
            object_list_sorted = sorted(object_list_filtered, key=lambda d: d['ymin'])   
            kv_object,vc_object=get_kv_loss_runs_page(page_path,object_list_sorted,input_bucket,pages,row_list)
            
            df_details={}
            df_details_id_ld={}
            df_details_id_coordinates_ld={}
            df_details_id_ls={}
            df_details_id_coordinates_ls={}
            count=0 
            for kv_obj,vc_obj in zip(kv_object,vc_object): 
                obj_name=kv_obj[0]
                kv_data=kv_obj[1]
                vc_data=vc_obj[1]

                flag_ld=0
                flag_ls=0
                table=template_dictionary[obj_name]['table']

                details={'submission_id': submission_id, 'document_id':doc_id, 'document_name':doc_type , 'page_id':pages }
                
                if table=='loss_details':  
                    
                    df_details_id_ld=merge_two_dicts(details,df_details_id_ld)

                    if obj_name in id_list:            
                        flag_ld=1
                        df_details_id_ld=merge_two_dicts(details,df_details_id_ld)                             
                        df_details_id_ld=merge_two_dicts(df_details_id_ld,kv_data)
                        df_details_id_coordinates_ld=merge_two_dicts(df_details_id_coordinates_ld,vc_data)

                    else:
                        df_details_row_ld=kv_data
                        df_details_row_coordinates_ld=vc_data
                        # print(df_details_row)

                        df_details_row_ld=merge_two_dicts(df_details_id_ld,df_details_row_ld)
                        df_details_coordinates_row_ld=merge_two_dicts(df_details_id_coordinates_ld,df_details_row_coordinates_ld)
                        df_details_ld=merge_two_dicts(df_details_row_ld,df_details_coordinates_row_ld)

                    if flag_ld==0:                  
                        
                        df_loss_details.append(df_details_ld)
                        
                elif table=='loss_summary':  
                    
                    df_details_id_ls=merge_two_dicts(details,df_details_id_ls)

                    if obj_name in id_list:            
                        flag_ls=1
                        df_details_id_ls=merge_two_dicts(details,df_details_id_ls)                             
                        df_details_id_ls=merge_two_dicts(df_details_id_ls,kv_data)
                        df_details_id_coordinates_ls=merge_two_dicts(df_details_id_coordinates_ls,vc_data)

                    else:
                        df_details_row_ls=kv_data
                        df_details_row_coordinates_ls=vc_data
                        # print(df_details_row)

                        df_details_row_ls=merge_two_dicts(df_details_id_ls,df_details_row_ls)
                        df_details_coordinates_row_ls=merge_two_dicts(df_details_id_coordinates_ls,df_details_row_coordinates_ls)
                        df_details_ls=merge_two_dicts(df_details_row_ls,df_details_coordinates_row_ls)

                    if flag_ls==0:                  
                        df_loss_summary.append(df_details_ls)
             
    df_loss_details=pd.DataFrame(df_loss_details) 
    df_loss_summary=pd.DataFrame(df_loss_summary)

    return  df_loss_details, df_loss_summary
          
#Extract data from pdf
def extract_dataframe_from_pdf_json_parallel(pages,object_list,template_json_path,submission_id,doc_id,image_dir,input_bucket,doc_type):   
    
    template_dictionary=load_json(template_json_path)

    id_list=[]
    row_list=[]
    available_templates=[]
    for data in template_dictionary:
        available_templates.append(data)
        if (template_dictionary[data]['category']=='id'):
            id_list.append(data)
        elif (template_dictionary[data]['category']=='row'):
            row_list.append(data)

    #Objects=input_json['Objects']
    
    #for pages in Objects:  
    df_loss_summary = []
    df_loss_details = []
    page_path=image_dir+pages

    #object_list=Objects[pages] 

    if isinstance(object_list, dict):
        if object_list['name']=='Unknown':
            pass
    else:
        object_list_filtered = [d for d in object_list if d['name'] in available_templates]
        object_list_sorted = sorted(object_list_filtered, key=lambda d: d['ymin'])   
        kv_object,vc_object=get_kv_loss_runs_page(page_path,object_list_sorted,input_bucket,pages,row_list)
        df_details={}
        df_details_id_ld={}
        df_details_id_coordinates_ld={}
        df_details_id_ls={}
        df_details_id_coordinates_ls={}
        count=0 
        for kv_obj,vc_obj in zip(kv_object,vc_object): 
            obj_name=kv_obj[0]
            kv_data=kv_obj[1]
            vc_data=vc_obj[1]

            flag_ld=0
            flag_ls=0
            table=template_dictionary[obj_name]['table']

            details={'submission_id': submission_id, 'document_id':doc_id, 'document_name':doc_type , 'page_id':pages }
            
            if table=='loss_details':  
                
                df_details_id_ld=merge_two_dicts(details,df_details_id_ld)

                if obj_name in id_list:            
                    flag_ld=1
                    df_details_id_ld=merge_two_dicts(details,df_details_id_ld)                             
                    df_details_id_ld=merge_two_dicts(df_details_id_ld,kv_data)
                    df_details_id_coordinates_ld=merge_two_dicts(df_details_id_coordinates_ld,vc_data)

                else:
                    df_details_row_ld=kv_data
                    df_details_row_coordinates_ld=vc_data
                    # print(df_details_row)

                    df_details_row_ld=merge_two_dicts(df_details_id_ld,df_details_row_ld)
                    df_details_coordinates_row_ld=merge_two_dicts(df_details_id_coordinates_ld,df_details_row_coordinates_ld)
                    df_details_ld=merge_two_dicts(df_details_row_ld,df_details_coordinates_row_ld)

                if flag_ld==0:                  
                    
                    df_loss_details.append(df_details_ld)
                    
                
            elif table=='loss_summary':  
                
                df_details_id_ls=merge_two_dicts(details,df_details_id_ls)

                if obj_name in id_list:            
                    flag_ls=1
                    df_details_id_ls=merge_two_dicts(details,df_details_id_ls)                             
                    df_details_id_ls=merge_two_dicts(df_details_id_ls,kv_data)
                    df_details_id_coordinates_ls=merge_two_dicts(df_details_id_coordinates_ls,vc_data)

                else:
                    df_details_row_ls=kv_data
                    df_details_row_coordinates_ls=vc_data
                    # print(df_details_row)

                    df_details_row_ls=merge_two_dicts(df_details_id_ls,df_details_row_ls)
                    df_details_coordinates_row_ls=merge_two_dicts(df_details_id_coordinates_ls,df_details_row_coordinates_ls)
                    df_details_ls=merge_two_dicts(df_details_row_ls,df_details_coordinates_row_ls)

                if flag_ls==0:                  
                    df_loss_summary.append(df_details_ls)
             
    df_loss_details=pd.DataFrame(df_loss_details) 
    df_loss_summary=pd.DataFrame(df_loss_summary)

    return  df_loss_details, df_loss_summary      
    

def loss_summary_col_mapping():
    v_col = { 'Address':'address', 'Address- Value Coordinates': 'address_value_coordinates', 'Cancel Date' : 'cancel_date' , 'Cancel Date- Value Coordinates' : 'cancel_date_value_coordinates' , 'Carrier' : 'carrier' , 'Carrier- Value Coordinates' : 'carrier_value_coordinates' , 'Closed Claims' : 'closed_claims' , 'Closed Claims- Value Coordinates' : 'closed_claims_value_coordinates' , 'Earned Premium' : 'earned_premium' , 'Earned Premium- Value Coordinates' : 'earned_premium_value_coordinates' , 'Effective Date' : 'effective_date' , 'Effective Date- Value Coordinates' : 'effective_date_value_coordinates' , 'Expiration Date' : 'expiration_date' , 'Expiration Date- Value Coordinates' : 'expiration_date_value_coordinates' , 'Insured' : 'insured' , 'Insured- Value Coordinates' : 'insured_value_coordinates' , 'Loss Ratio' : 'loss_ratio' , 'Loss Ratio- Value Coordinates' : 'loss_ratio_value_coordinates' , 'Open Claims' : 'open_claims' , 'Open Claims- Value Coordinates' : 'open_claims_value_coordinates' , 'Outstanding Losses and Expenses' : 'outstanding_losses_and_expenses' , 'Outstanding Losses and Expenses- Value Coordinates' : 'outstanding_losses_and_expenses_value_coordinates' , 'Paid Loss Adjustment Expenses' : 'paid_loss_adjustment_expenses' , 'Paid Loss Adjustment Expenses- Value Coordinates' : 'paid_loss_adjustment_expenses_value_coordinates' , 'Paid Losses' : 'paid_losses' , 'Paid Losses and Expenses' : 'paid_losses_and_expenses' , 'Paid Losses and Expenses- Value Coordinates' : 'paid_losses_and_expenses_value_coordinates' , 'Paid Losses- Value Coordinates' : 'paid_losses_value_coordinates' , 'Policy Effective' : 'policy_effective' , 'Policy Effective- Value Coordinates' : 'policy_effective_value_coordinates' , 'Policy Number' : 'policy_number' , 'Policy Number Insured' : 'policy_number_insured' , 'Policy Number Insured- Value Coordinates' : 'policy_number_insured_value_coordinates' , 'Policy Number- Value Coordinates' : 'policy_number_value_coordinates' , 'Policy Period' : 'policy_period' , 'Policy Period- Value Coordinates' : 'policy_period_value_coordinates' ,'Reserved Losses and Expenses' : 'reserved_losses_and_expenses' , 'Reserved Losses and Expenses- Value Coordinates' : 'reserved_losses_and_expenses_value_coordinates' , 'Total Claims' : 'total_claims' , 'Total Claims- Value Coordinates' : 'total_claims_value_coordinates' , 'Total Incurred' : 'total_incurred' , 'Total Incurred Losses and Expenses' : 'total_incurred_losses_and_expenses' , 'Total Incurred Losses and Expenses- Value Coordinates' : 'total_incurred_losses_and_expenses_value_coordinates' , 'Total Incurred- Value Coordinates' : 'total_incurred_value_coordinates' , 'Value as of' : 'value_as_of' , 'Value as of- Value Coordinates' : 'value_as_of_value_coordinates' , 'Written Premium' : 'written_premium' , 'Written Premium- Value Coordinates' : 'written_premium_value_coordinates',
    'Agency Name' : 'agency_name','Agency Name- Value Coordinates' : 'agency_name_value_coordinates','Current As of Date' : 'current_as_of_date','Current As of Date- Value Coordinates' : 'current_as_of_date_value_coordinates','Expenses Paid' : 'expenses_paid','Expenses Paid- Value Coordinates' : 'expenses_paid_value_coordinates','Insured Name' : 'insured_name','Insured Name- Value Coordinates' : 'insured_name_value_coordinates','Loss & Expense Reserves' : 'loss_expense_reserves','Loss & Expense Reserves- Value Coordinates' : 'loss_expense_reserves_value_coordinates','Losses & Expenses Paid' : 'losses_expenses_paid','Losses & Expenses Paid- Value Coordinates' : 'losses_expenses_paid_value_coordinates','Losses Paid' : 'losses_paid','Losses Paid- Value Coordinates' : 'losses_paid_value_coordinates','No of Claims/Occurrences' : 'no_of_claims_occurrences','No of Claims/Occurrences- Value Coordinates' : 'no_of_claims_occurrences_value_coordinates','Number of Loss Year Requested' : 'number_of_loss_year_requested','Number of Loss Year Requested- Value Coordinates' : 'number_of_loss_year_requested_value_coordinates','Policy #' : 'policy_number','Policy #- Value Coordinates' : 'policy_number_value_coordinates','Policy Term' : 'policy_term','Policy Term- Value Coordinates' : 'policy_term_value_coordinates','Policy Type' : 'policy_type','Policy Type- Value Coordinates' : 'policy_type_value_coordinates','Recovery' : 'recovery','Recovery- Value Coordinates' : 'recovery_value_coordinates',
    'Claims Open' : 'open_claims', 'Claims Open- Value Coordinates' : 'open_claims_value_coordinates', 'Claims Total' : 'total_claims', 'Claims Total- Value Coordinates' : 'total_claims_value_coordinates', 'Cliams Close' : 'closed_claims', 'Cliams Close- Value Coordinates' : 'closed_claims_value_coordinates', 'Current Experience Mod' : 'current_experience_mod', 'Current Experience Mod- Value Coordinates' : 'current_experience_mod_value_coordinates', 'Disability' : 'disability', 'Disability- Value Coordinates' : 'disability_value_coordinates', 'Estimated Annual' : 'estimated_annual', 'Estimated Annual- Value Coordinates' : 'estimated_annual_value_coordinates', 'Estimated Compensation' : 'estimated_compensation', 'Estimated Compensation- Value Coordinates' : 'estimated_compensation_value_coordinates', 'Estimated Medical' : 'estimated_medical', 'Estimated Medical- Value Coordinates' : 'estimated_medical_value_coordinates', 'Governing Class' : 'governing_class', 'Governing Class- Value Coordinates' : 'governing_class_value_coordinates', 'Litigated' : 'litigated', 'Litigated- Value Coordinates' : 'litigated_value_coordinates', 'Non-Disability' : 'non_disability', 'Non-Disability- Value Coordinates' : 'non_disability_value_coordinates', 'Paid Compensation' : 'paid_compensation', 'Paid Compensation- Value Coordinates' : 'paid_compensation_value_coordinates', 'Paid Medical' : 'paid_medical', 'Paid Medical- Value Coordinates' : 'paid_medical_value_coordinates', 'Policy Year' : 'policy_period', 'Policy Year- Value Coordinates' : 'policy_period_value_coordinates', 'Total Estimated/Incurred' : 'total_incurred', 'Total Estimated/Incurred- Value Coordinates' : 'total_incurred_value_coordinates', 'Total Paid' : 'total_paid', 'Total Paid- Value Coordinates' : 'total_paid_value_coordinates',
    'Expense Incurred' : 'expense_incurred', 'Ind Claims' : 'indemnity_claims', 'Indemnity Incurred' : 'indemnity_incurred', 'Med Claims' : 'medical_claims', 'Medical Incurred' : 'medical_incurred', 'Open Claims' : 'open_claims', 'Oth Claims' : 'other_claims', 'Policy State' : 'policy_state', 'Remove end' : 'remove_end', 'Total Claims' : 'total_claims', 'Expense Incurred- Value Coordinates' : 'expense_incurred_value_coordinates', 'Ind Claims- Value Coordinates' : 'indemnity_claims_value_coordinates', 'Indemnity Incurred- Value Coordinates' : 'indemnity_incurred_value_coordinates', 'Med Claims- Value Coordinates' : 'medical_claims_value_coordinates', 'Medical Incurred- Value Coordinates' : 'medical_incurred_value_coordinates', 'Open Claims- Value Coordinates' : 'open_claims_value_coordinates', 'Oth Claims- Value Coordinates' : 'other_claims_value_coordinates', 'Policy State- Value Coordinates' : 'policy_state_value_coordinates', 'Remove end- Value Coordinates' : 'remove_end_value_coordinates', 'Total Claims- Value Coordinates' : 'total_claims_value_coordinates','Expense Paid' : 'expenses_paid','Expense Paid- Value Coordinates' : 'expenses_paid_value_coordinates','Inception Date' : 'inception_date', 'Inception Date- Value Coordinates' : 'inception_date_value_coordinates','Indemnity Paid' : 'indemnity_paid' , 'Indemnity Paid- Value Coordinates' : 'indemnity_paid_value_coordinates','Medical Paid' : 'paid_medical', 'Medical Paid- Value Coordinates' : 'paid_medical_value_coordinates','Indemnity paid' : 'indemnity_paid' , 'Indemnity paid- Value Coordinates' : 'indemnity_paid_value_coordinates'    }
    return v_col
    
    
def loss_detail_col_mapping():
    v_col = { 'Accident Date' : 'accident_date' , 'Accident Date- Value Coordinates' : 'accident_date_value_coordinates' , 'Accident Description' : 'accident_description' , 'Accident Description- Value Coordinates' : 'accident_description_value_coordinates' , 'Adj Off' : 'adj_off' , 'Adj Off- Value Coordinates' : 'adj_off_value_coordinates' , 'Agent' : 'agent' , 'Agent- Value Coordinates' : 'agent_value_coordinates' , 'Body Part' : 'body_part' , 'Body Part- Value Coordinates' : 'body_part_value_coordinates' , 'Cause' : 'cause' , 'Cause- Value Coordinates' : 'cause_value_coordinates' , 'Claim Inc' : 'claim_incurred' , 'Claim Inc- Value Coordinates' : 'claim_incurred_value_coordinates' , 'Claim Number' : 'claim_number' , 'Claim Number- Value Coordinates' : 'claim_number_value_coordinates' , 'Claim O/S' : 'claim_outstanding' , 'Claim O/S- Value Coordinates' : 'claim_outstanding_value_coordinates' , 'Claim Status' : 'claim_status' , 'Claim Status- Value Coordinates' : 'claim_status_value_coordinates' , 'Claim pd' : 'claim_paid' , 'Claim pd- Value Coordinates' : 'claim_paid_value_coordinates' , 'Claimant' : 'claimant' , 'Claimant Name' : 'claimant_name' , 'Claimant Name- Value Coordinates' : 'claimant_name_value_coordinates' , 'Claimant- Value Coordinates' : 'claimant_value_coordinates' , 'Class Code' : 'class_code' , 'Class Code- Value Coordinates' : 'class_code_value_coordinates' , 'Close Date' : 'close_date' , 'Close Date- Value Coordinates' : 'close_date_value_coordinates' , 'Date Reported to Carrier' : 'date_reported_to_carrier' , 'Date Reported to Carrier- Value Coordinates' : 'date_reported_to_carrier_value_coordinates' , 'Date of Injury' : 'date_of_injury' , 'Date of Injury- Value Coordinates' : 'date_of_injury_value_coordinates' , 'Expense Inc' : 'expense_incurred' , 'Expense Inc- Value Coordinates' : 'expense_incurred_value_coordinates' , 'Expense Incured' : 'expense_incurred' , 'Expense Incured- Value Coordinates' : 'expense_incurred_value_coordinates' , 'Expense O/S:' : 'expense_outstanding' , 'Expense O/S:- Value Coordinates' : 'expense_outstanding_value_coordinates' , 'Expense Paid' : 'expense_paid' , 'Expense Paid- Value Coordinates' : 'expense_paid_value_coordinates' , 'Expense Reserve' : 'expense_reserve' , 'Expense Reserve- Value Coordinates' : 'expense_reserve_value_coordinates' , 'Expense paid' : 'expense_paid' , 'Expense paid- Value Coordinates' : 'expense_paid_value_coordinates' , 'Expense pd:' : 'expense_paid' , 'Expense pd:- Value Coordinates' : 'expense_paid_value_coordinates' , 'FP' : 'fp' , 'FP- Value Coordinates' : 'fp_value_coordinates' , 'In Litigation' : 'in_litigation' , 'In Litigation- Value Coordinates' : 'in_litigation_value_coordinates' , 'Indemnity Incured' : 'indemnity_incurred' , 'Indemnity Incured- Value Coordinates' : 'indemnity_incurred_value_coordinates' , 'Indemnity Paid' : 'indemnity_paid' , 'Indemnity Paid- Value Coordinates' : 'indemnity_paid_value_coordinates' , 'Indemnity Reserve' : 'indemnity_reserve' , 'Indemnity Reserve- Value Coordinates' : 'indemnity_reserve_value_coordinates' , 'Indemnity paid' : 'indemnity_paid' , 'Indemnity paid- Value Coordinates' : 'indemnity_paid_value_coordinates' , 'Insured' : 'insured' , 'Insured- Value Coordinates' : 'insured_value_coordinates' , 'Line of Insurance' : 'line_of_insurance' , 'Line of Insurance- Value Coordinates' : 'line_of_insurance_value_coordinates' , 'Loss Date' : 'loss_date' , 'Loss Date- Value Coordinates' : 'loss_date_value_coordinates' , 'Medical Inc:' : 'medical_incurred' , 'Medical Inc:- Value Coordinates' : 'medical_incurred_value_coordinates' , 'Medical Incured' : 'medical_incurred' , 'Medical Incured- Value Coordinates' : 'medical_incurred_value_coordinates' , 'Medical O/S' : 'medical_outstanding' , 'Medical O/S- Value Coordinates' : 'medical_outstanding_value_coordinates' , 'Medical Paid' : 'medical_paid' , 'Medical Paid- Value Coordinates' : 'medical_paid_value_coordinates' , 'Medical Reserve' : 'medical_reserve' , 'Medical Reserve- Value Coordinates' : 'medical_reserve_value_coordinates' , 'Medical pd' : 'medical_paid' , 'Medical pd- Value Coordinates' : 'medical_paid_value_coordinates' , 'Nature' : 'nature' , 'Nature of Injury' : 'nature_of_injury' , 'Nature of Injury- Value Coordinates' : 'nature_of_injury_value_coordinates' , 'Nature- Value Coordinates' : 'nature_value_coordinates' , 'Notice Date' : 'notice_date' , 'Notice Date- Value Coordinates' : 'notice_date_value_coordinates' , 'Policy' : 'policy' , 'Policy Effective' : 'policy_effective' , 'Policy Effective- Value Coordinates' : 'policy_effective_value_coordinates' , 'Policy Number' : 'policy_number' , 'Policy Number Insured' : 'policy_number_insured' , 'Policy Number Insured- Value Coordinates' : 'policy_number_insured_value_coordinates' , 'Policy Number- Value Coordinates' : 'policy_number_value_coordinates' , 'Policy Period' : 'policy_period' , 'Policy Period- Value Coordinates' : 'policy_period_value_coordinates' , 'Policy- Value Coordinates' : 'policy_value_coordinates' , 'Reported date' : 'reported_date' , 'Reported date- Value Coordinates' : 'reported_date_value_coordinates' , 'Total Inc' : 'total_incurred' , 'Total Inc- Value Coordinates' : 'total_incurred_value_coordinates' , 'Total Incured' : 'total_incurred' , 'Total Incured- Value Coordinates' : 'total_incurred_value_coordinates' , 'Total O/S' : 'total_outstanding' , 'Total O/S- Value Coordinates' : 'total_outstanding_value_coordinates' , 'Total Paid' : 'total_paid' , 'Total Paid- Value Coordinates' : 'total_paid_value_coordinates' , 'Total Reserve' : 'total_reserve' , 'Total Reserve- Value Coordinates' : 'total_reserve_value_coordinates' , 'Total pd' : 'total_paid' , 'Total pd- Value Coordinates' : 'total_paid_value_coordinates' , 'Value as of' : 'value_as_of' , 'Value as of- Value Coordinates' : 'value_as_of_value_coordinates','Traveler Customer' : 'traveler_customer' ,'Traveler Customer- Value Coordinates' : 'traveler_customer_value_coordinates' , 'Expense Reserved' : 'expense_reserve' , 'Indemnity Reserved' : 'indemnity_reserve' , 'Medical Reserved' : 'medical_reserve' , 'Total Reserved' : 'total_reserve','Expense Reserved- Value Coordinates' : 'expense_reserve_value_coordinates' , 'Indemnity Reserved- Value Coordinates' : 'indemnity_reserve_value_coordinates' , 'Medical Reserved- Value Coordinates' : 'medical_reserve_value_coordinates' , 'Total Reserved- Value Coordinates' : 'total_reserve_value_coordinates' ,' O/C' : 'occupation' , 'SAI Number' : 'sai_number',' O/C- Value Coordinates' : 'occupation_value_coordinates' , 'SAI Number- Value Coordinates' : 'sai_number_value_coordinates',
    'Carrier' : 'carrier', 'Carrier- Value Coordinates' : 'carrier_value_coordinates', 'Cause of Injury' : 'cause_of_injury', 'Cause of Injury- Value Coordinates' : 'cause_of_injury_value_coordinates', 'Claim Professional' : 'claim_professional', 'Claim Professional- Value Coordinates' : 'claim_professional_value_coordinates', 'Claim Type' : 'claim_type', 'Claim Type- Value Coordinates' : 'claim_type_value_coordinates', 'Claimant Status' : 'claimant_status', 'Claimant Status- Value Coordinates' : 'claimant_status_value_coordinates', 'Date Closed' : 'date_closed', 'Date Closed- Value Coordinates' : 'date_closed_value_coordinates', 'Date Reopened' : 'date_reopened', 'Date Reopened- Value Coordinates' : 'date_reopened_value_coordinates', 'Date Reported' : 'date_reported', 'Date Reported- Value Coordinates' : 'date_reported_value_coordinates', 'Date of Birth' : 'date_of_birth', 'Date of Birth- Value Coordinates' : 'date_of_birth_value_coordinates', 'Date of Hire' : 'date_of_hire', 'Date of Hire- Value Coordinates' : 'date_of_hire_value_coordinates', 'Date of Incident' : 'date_of_incident', 'Date of Incident- Value Coordinates' : 'date_of_incident_value_coordinates', 'Expense Outstanding' : 'expense_outstanding', 'Expense Outstanding- Value Coordinates' : 'expense_outstanding_value_coordinates', 'Injury Detail' : 'injury_detail', 'Injury Detail- Value Coordinates' : 'injury_detail_value_coordinates', 'Jurisdiction' : 'jurisdiction', 'Jurisdiction- Value Coordinates' : 'jurisdiction_value_coordinates', 'Location' : 'location', 'Location- Value Coordinates' : 'location_value_coordinates', 'Medical Outstanding' : 'medical_outstanding', 'Medical Outstanding- Value Coordinates' : 'medical_outstanding_value_coordinates', 'Occupation' : 'occupation', 'Occupation- Value Coordinates' : 'occupation_value_coordinates', 'PD Incured' : 'pd_incured', 'PD Incured- Value Coordinates' : 'pd_incured_value_coordinates', 'PD Outstanding' : 'pd_outstanding', 'PD Outstanding- Value Coordinates' : 'pd_outstanding_value_coordinates', 'PD Paid' : 'pd_paid', 'PD Paid- Value Coordinates' : 'pd_paid_value_coordinates', 'State' : 'state', 'State- Value Coordinates' : 'state_value_coordinates', 'TD Incured' : 'td_incured', 'TD Incured- Value Coordinates' : 'td_incured_value_coordinates', 'TD Outstanding' : 'td_outstanding', 'TD Outstanding- Value Coordinates' : 'td_outstanding_value_coordinates', 'TD Paid' : 'td_paid', 'TD Paid- Value Coordinates' : 'td_paid_value_coordinates', 'Total Outstanding' : 'total_outstanding', 'Total Outstanding- Value Coordinates' : 'total_outstanding_value_coordinates', 'Voc Rehab Incured' : 'voc_rehab_incured', 'Voc Rehab Incured- Value Coordinates' : 'voc_rehab_incured_value_coordinates', 'Voc Rehab Outstanding' : 'voc_rehab_outstanding', 'Voc Rehab Outstanding- Value Coordinates' : 'voc_rehab_outstanding_value_coordinates', 'Voc Rehab Paid' : 'voc_rehab_paid', 'Voc Rehab Paid- Value Coordinates' : 'voc_rehab_paid_value_coordinates', 'Reporting Group 1' : 'reporting_group_1', 'Reporting Group 1- Value Coordinates' : 'reporting_group_1_value_coordinates', 'Reporting Group 2' : 'reporting_group_2', 'Reporting Group 2- Value Coordinates' : 'reporting_group_2_value_coordinates', 
    'Adjuster' : 'adjuster', 'Adjuster- Value Coordinates' : 'adjuster_value_coordinates', 'Category' : 'category', 'Category- Value Coordinates' : 'category_value_coordinates', 'Claim No' : 'claim_number', 'Claim No- Value Coordinates' : 'claim_number_value_coordinates', 'Class Cd' : 'class_code', 'Class Cd- Value Coordinates' : 'class_code_value_coordinates', 'Converted' : 'converted', 'Converted- Value Coordinates' : 'converted_value_coordinates', 'DOL' : 'dol', 'DOL- Value Coordinates' : 'dol_value_coordinates', 'Date Revd' : 'date_revd', 'Date Revd- Value Coordinates' : 'date_revd_value_coordinates', 'Department' : 'department', 'Department- Value Coordinates' : 'department_value_coordinates', 'Employee Lag' : 'employee_lag', 'Employee Lag- Value Coordinates' : 'employee_lag_value_coordinates', 'First Aware' : 'first_aware', 'First Aware- Value Coordinates' : 'first_aware_value_coordinates', 'Incurred Indem' : 'incurred_indem', 'Incurred Indem- Value Coordinates' : 'incurred_indem_value_coordinates', 'Incurred LAE' : 'incurred_lae', 'Incurred LAE- Value Coordinates' : 'incurred_lae_value_coordinates', 'Incurred Madical' : 'medical_incurred', 'Incurred Madical- Value Coordinates' : 'medical_incurred_value_coordinates', 'Incurred Total' : 'total_incurred', 'Incurred Total- Value Coordinates' : 'total_incurred_value_coordinates', 'Juris St' : 'juris_st', 'Juris St- Value Coordinates' : 'juris_st_value_coordinates', 'Loss Description' : 'loss_description', 'Loss Description- Value Coordinates' : 'loss_description_value_coordinates', 'Loss Location' : 'loss_location', 'Loss Location- Value Coordinates' : 'loss_location_value_coordinates', 'Part Injured' : 'part_injured', 'Part Injured- Value Coordinates' : 'part_injured_value_coordinates', 'Payments Indem' : 'payments_indem', 'Payments Indem- Value Coordinates' : 'payments_indem_value_coordinates', 'Payments LAE' : 'payments_lae', 'Payments LAE- Value Coordinates' : 'payments_lae_value_coordinates', 'Payments Medical' : 'payments_medical', 'Payments Medical- Value Coordinates' : 'payments_medical_value_coordinates', 'Payments Total' : 'payments_total', 'Payments Total- Value Coordinates' : 'payments_total_value_coordinates', 'Pol. Eff Date' : 'policy_eff_date', 'Pol. Eff Date- Value Coordinates' : 'policy_eff_date_value_coordinates', 'Recoveries Indem' : 'recoveries_indem', 'Recoveries Indem- Value Coordinates' : 'recoveries_indem_value_coordinates', 'Recoveries LAE' : 'recoveries_lae', 'Recoveries LAE- Value Coordinates' : 'recoveries_lae_value_coordinates', 'Recoveries Medical' : 'recoveries_medical', 'Recoveries Medical- Value Coordinates' : 'recoveries_medical_value_coordinates', 'Recoveries Total' : 'recoveries_total', 'Recoveries Total- Value Coordinates' : 'recoveries_total_value_coordinates', 'Reporting Lag' : 'reporting_lag', 'Reporting Lag- Value Coordinates' : 'reporting_lag_value_coordinates', 'Reserves Indem' : 'reserves_indem', 'Reserves Indem- Value Coordinates' : 'reserves_indem_value_coordinates', 'Reserves LAE' : 'reserves_lae', 'Reserves LAE- Value Coordinates' : 'reserves_lae_value_coordinates', 'Reserves Medical' : 'medical_reserve', 'Reserves Medical- Value Coordinates' : 'medical_reserve_value_coordinates', 'Reserves Total' : 'total_reserve', 'Reserves Total- Value Coordinates' : 'total_reserve_value_coordinates', 'Status' : 'status', 'Status- Value Coordinates' : 'status_value_coordinates' , 
    'Agency Name' : 'agency_name', 'Agency Name- Value Coordinates' : 'agency_name_value_coordinates', 'Claim Reference #' : 'claim_reference', 'Claim Reference #- Value Coordinates' : 'claim_reference_value_coordinates', 'Claim/Occurrence' : 'claim_occurrence', 'Claim/Occurrence- Value Coordinates' : 'claim_occurrence_value_coordinates', 'Claimant #' : 'claimant', 'Claimant #- Value Coordinates' : 'claimant_value_coordinates', 'Claimant Name:' : 'claimant_name', 'Claimant Name:- Value Coordinates' : 'claimant_name_value_coordinates', 'Current As of Date' : 'current_as_of_date', 'Current As of Date- Value Coordinates' : 'current_as_of_date_value_coordinates', 'Deductible Amount' : 'deductible_amount', 'Deductible Amount- Value Coordinates' : 'deductible_amount_value_coordinates', 'Expenses Paid' : 'expense_paid', 'Expenses Paid- Value Coordinates' : 'expense_paid_value_coordinates', 'Insured Name' : 'insured_name', 'Insured Name- Value Coordinates' : 'insured_name_value_coordinates', 'Loss Reserve' : 'loss_reserve', 'Loss Reserve- Value Coordinates' : 'loss_reserve_value_coordinates', 'Losses Paid' : 'losses_paid', 'Losses Paid- Value Coordinates' : 'losses_paid_value_coordinates', 'Lossess and Expenses Paid' : 'lossess_and_expenses_paid', 'Lossess and Expenses Paid- Value Coordinates' : 'lossess_and_expenses_paid_value_coordinates', 'Number of Loss Year Requested' : 'number_of_loss_year_requested', 'Number of Loss Year Requested- Value Coordinates' : 'number_of_loss_year_requested_value_coordinates', 'Policy #' : 'policy_number', 'Policy #- Value Coordinates' : 'policy_number_value_coordinates', 'Policy Term' : 'policy_term', 'Policy Term- Value Coordinates' : 'policy_term_value_coordinates', 'Policy Type' : 'policy_type', 'Policy Type- Value Coordinates' : 'policy_type_value_coordinates', 'Recovery' : 'recovery', 'Recovery- Value Coordinates' : 'recovery_value_coordinates', 'Reported Date' : 'reported_date', 'Reported Date- Value Coordinates' : 'reported_date_value_coordinates', 'Total Incurred' : 'total_incurred', 'Total Incurred- Value Coordinates' : 'total_incurred_value_coordinates', 'Writing Company' : 'writing_company', 'Writing Company- Value Coordinates' : 'writing_company_value_coordinates' , 
    'Accident / Loss Description' : 'accident_description', 'Accident / Loss Description- Value Coordinates' : 'accident_description_value_coordinates', 'Account D&B Number' : 'account_db_number', 'Account D&B Number- Value Coordinates' : 'account_db_number_value_coordinates', 'Account Name' : 'account_name', 'Account Name- Value Coordinates' : 'account_name_value_coordinates', 'Alloc Exp Paid' : 'expense_paid', 'Alloc Exp Paid- Value Coordinates' : 'expense_paid_value_coordinates', 'Claim # / OneClaim #' : 'claim_number', 'Claim # / OneClaim #- Value Coordinates' : 'claim_number_value_coordinates', 'Close of Date' : 'close_date', 'Close of Date- Value Coordinates' : 'close_date_value_coordinates', 'Currency' : 'currency', 'Currency- Value Coordinates' : 'currency_value_coordinates', 'Customer name' : 'customer_name', 'Customer name- Value Coordinates' : 'customer_name_value_coordinates', 'Div / H.O.' : 'div_ho', 'Div / H.O.- Value Coordinates' : 'div_ho_value_coordinates', 'Ind/BI Paid' : 'indemnity_paid', 'Ind/BI Paid- Value Coordinates' : 'indemnity_paid_value_coordinates', 'Loss Type' : 'loss_type', 'Loss Type- Value Coordinates' : 'loss_type_value_coordinates', 'Major Class Code/Description' : 'major_class_code_description', 'Major Class Code/Description- Value Coordinates' : 'major_class_code_description_value_coordinates', 'Med/PD Paid' : 'medical_paid', 'Med/PD Paid- Value Coordinates' : 'medical_paid_value_coordinates', 'Receipt Date' : 'receipt_date', 'Receipt Date- Value Coordinates' : 'receipt_date_value_coordinates', 'Report Date/Time' : 'reported_date', 'Report Date/Time- Value Coordinates' : 'reported_date_value_coordinates', 'St/Terr/Ctry' : 'st_terr_ctry', 'St/Terr/Ctry- Value Coordinates' : 'st_terr_ctry_value_coordinates', 'Total Recoveries' : 'recoveries_total', 'Total Recoveries- Value Coordinates' : 'recoveries_total_value_coordinates', 'Total Reserves' : 'reserves_total', 'Total Reserves- Value Coordinates' : 'reserves_total_value_coordinates', 'Valuation Date' : 'valuation_date', 'Valuation Date- Value Coordinates' : 'valuation_date_value_coordinates', 'Account Number' : 'account_db_number','Account Number- Value Coordinates' : 'account_db_number_value_coordinates', 'Client Name' : 'client_name' , 'Client Name- Value Coordinates' : 'client_name_value_coordinates' ,'Client Number' : 'client_number' , 'Client Number- Value Coordinates' : 'client_number_value_coordinates', 'Filters' : 'filters' , 'Filters- Value Coordinates' : 'filters_value_coordinates', 'Report Name' : 'report_name' , 'Report Name- Value Coordinates' : 'report_name_value_coordinates' , 'Requested ID' : 'requested_id' , 'Requested ID- Value Coordinates' : 'requested_id_value_coordinates' , 'Source' : 'source' , 'Source- Value Coordinates' : 'source_value_coordinates','Account / D&B Number / Name' : 'account_db_number_and_name','Account / D&B Number / Name- Value Coordinates' : 'account_db_number_and_name_value_coordinates', 'Policy Details' : 'policy_details' , 'Policy Details- Value Coordinates' : 'policy_details_value_coordinates' , 
    'Class' : 'class_code', 'Class- Value Coordinates' : 'class_code_value_coordinates', 'Company Name' : 'company_name', 'Company Name- Value Coordinates' : 'company_name_value_coordinates', 'Deductible' : 'deductible_amount', 'Deductible- Value Coordinates' : 'deductible_amount_value_coordinates', 'Fatalities' : 'fatalities', 'Fatalities- Value Coordinates' : 'fatalities_value_coordinates', 'Injured Employee' : 'injured_employee', 'Injured Employee- Value Coordinates' : 'injured_employee_value_coordinates', 'Injury Date & Time' : 'injury_date_time', 'Injury Date & Time- Value Coordinates' : 'injury_date_time_value_coordinates', 'Injury Source' : 'injury_source', 'Injury Source- Value Coordinates' : 'injury_source_value_coordinates', 'Injury Type' : 'injury_type', 'Injury Type- Value Coordinates' : 'injury_type_value_coordinates', 'Median Days to Report Claim' : 'median_days_to_report_claim', 'Median Days to Report Claim- Value Coordinates' : 'median_days_to_report_claim_value_coordinates', 'Net Expense' : 'net_expense', 'Net Expense- Value Coordinates' : 'net_expense_value_coordinates','Policy Holder:' : 'policy_holder' , 'Policy Holder:- Value Coordinates' : 'policy_holder_value_coordinates',
    'A.R.D' : 'ard', 'A.R.D- Value Coordinates' : 'ard_value_coordinates', 'Address' : 'address', 'Address- Value Coordinates' : 'address_value_coordinates', 'Cancellation Code' : 'cancellation_code', 'Cancellation Code- Value Coordinates' : 'cancellation_code_value_coordinates', 'Claim ID' : 'claim_number', 'Claim ID- Value Coordinates' : 'claim_number_value_coordinates', 'Company Name & Address' : 'company_name', 'Company Name & Address- Value Coordinates' : 'company_name_value_coordinates', 'District Office' : 'district_office', 'District Office- Value Coordinates' : 'district_office_value_coordinates', 'Est. Comp' : 'est_comp', 'Est. Comp- Value Coordinates' : 'est_comp_value_coordinates', 'Est. Medical' : 'est_medical', 'Est. Medical- Value Coordinates' : 'est_medical_value_coordinates', 'Expiration Date' : 'expiration_date', 'Expiration Date- Value Coordinates' : 'expiration_date_value_coordinates', 'Inception Date' : 'inception_date', 'Inception Date- Value Coordinates' : 'inception_date_value_coordinates', 'Injury Date' : 'injury_date_time', 'Injury Date- Value Coordinates' : 'injury_date_time_value_coordinates', 'Name' : 'customer_name', 'Name- Value Coordinates' : 'customer_name_value_coordinates', 'Paid Comp' : 'paid_comp', 'Paid Comp- Value Coordinates' : 'paid_comp_value_coordinates', 'Paid Medical' : 'medical_paid', 'Paid Medical- Value Coordinates' : 'medical_paid_value_coordinates', 'Phone Number' : 'phone_number', 'Phone Number- Value Coordinates' : 'phone_number_value_coordinates', 'Quote ID' : 'quote_id', 'Quote ID- Value Coordinates' : 'quote_id_value_coordinates', 'Quote Type' : 'quote_type', 'Quote Type- Value Coordinates' : 'quote_type_value_coordinates','Brokerage' : 'brokerage' , 'Brokerage- Value Coordinates' : 'brokerage_value_coordinates',
    'Accident' : 'accident', 'Accident- Value Coordinates' : 'accident_value_coordinates', 'Expense Incurred' : 'expense_incurred', 'Expense Incurred- Value Coordinates' : 'expense_incurred_value_coordinates', 'Indemnity Incurred' : 'indemnity_incurred', 'Indemnity Incurred- Value Coordinates' : 'indemnity_incurred_value_coordinates', 'Indemnity Outstanding' : 'indemnity_outstanding', 'Indemnity Outstanding- Value Coordinates' : 'indemnity_outstanding_value_coordinates', 'Injury' : 'injury_source', 'Injury- Value Coordinates' : 'injury_source_value_coordinates', 'Last Closed Date' : 'close_date', 'Last Closed Date- Value Coordinates' : 'close_date_value_coordinates', 'Litigation Flag' : 'in_litigation', 'Litigation Flag- Value Coordinates' : 'in_litigation_value_coordinates', 'Medical Incurred' : 'medical_incurred', 'Medical Incurred- Value Coordinates' : 'medical_incurred_value_coordinates', 'Part of Body' : 'body_part', 'Part of Body- Value Coordinates' : 'body_part_value_coordinates', 'Pol State & Inc Yr' : 'policy_state', 'Pol State & Inc Yr- Value Coordinates' : 'policy_state_value_coordinates','Customer Name' : 'customer_name', 'Customer Name- Value Coordinates' : 'customer_name_value_coordinates',
    'Closed Date' : 'close_date', 'Closed Date- Value Coordinates' : 'close_date_value_coordinates', 'Date of Loss' : 'loss_date', 'Date of Loss- Value Coordinates' : 'loss_date_value_coordinates', 'Description of Loss' : 'loss_description', 'Description of Loss- Value Coordinates' : 'loss_description_value_coordinates', 'Driver/Class Cd' : 'class_code', 'Driver/Class Cd- Value Coordinates' : 'class_code_value_coordinates',  'Days Between' : 'days_between', 'Days Between- Value Coordinates' : 'days_between_value_coordinates', 'Deductible/Recovery' : 'deductible_recovery', 'Deductible/Recovery- Value Coordinates' : 'deductible_recovery_value_coordinates', 'LOB' : 'lob', 'LOB- Value Coordinates' : 'lob_value_coordinates', 'Cause of Loss' : 'cause_of_loss' , 'Cause of Loss- Value Coordinates' : 'cause_of_loss_value_coordinates', 'Lines Count' : 'lines_count', 'Lines Count- Value Coordinates' : 'lines_count_value_coordinates','Loss State/Loc-Veh Num' : 'loss_state', 'Loss State/Loc-Veh Num- Value Coordinates' : 'loss_state_value_coordinates', 'Loss Subline' : 'loss_subline', 'Loss Subline- Value Coordinates' : 'loss_subline_value_coordinates', 'Subro/Salvage' : 'subro_salvage' , 'Subro/Salvage- Value Coordinates' : 'subro_salvage_value_coordinates' ,
    'Accident Type' : 'accident_type', 'Claim Class(LOCS)' : 'claim_class', 'ClaimNumber' : 'claim_number', 'Client' : 'client', 'Client Claim #' : 'client_claim', 'Const Def?' : 'const_def', 'Controverted?' : 'controverted', 'Date' : 'date', 'Ded/SIR' : 'ded_sir', 'Desc of Accident' : 'accident_description', 'EXPENSE Incured' : 'expense_incurred', 'EXPENSE Outstanding' : 'expense_outstanding', 'EXPENSE Paid' : 'expense_paid', 'Hire Date' : 'hire_date', 'IND/BI Incured' : 'indemnity_incurred', 'IND/BI Outstanding' : 'indemnity_outstanding', 'IND/BI Paid' : 'indemnity_paid', 'Juris/Acc State' : 'juris_st', 'Litigation?' : 'in_litigation', 'Location Codes(1-3)' : 'location_codes_1_to_3', 'Location Codes(4-6)' : 'location_codes_4_to_6', 'MED/PropDmg Incured' : 'medical_incurred', 'MED/PropDmg Outstanding' : 'medical_outstanding', 'MED/PropDmg Paid' : 'medical_paid', 'NET Incured' : 'net_incured', 'NET Outstanding' : 'net_outstanding', 'NET Paid' : 'net_paid', 'Occurrence Id' : 'occurrence_id', 'Policy Cancel Dates' : 'policy_cancel_dates', 'Policy Effective Dates' : 'policy_eff_date', 'RECOVERED Incured' : 'recovered_incured', 'RECOVERED Outstanding' : 'recovered_outstanding', 'RECOVERED Paid' : 'recovered_paid', 'REIMBURSED Paid' : 'reimbursed_paid', 'Social Security' : 'social_security', 'Subrogation?' : 'subrogation', 'TOTAL Incured' : 'total_incurred', 'TOTAL Outstanding' : 'total_outstanding', 'TOTAL Paid' : 'total_paid', 'Valued At' : 'valued_at', 'Accident Type- Value Coordinates' : 'accident_type_value_coordinates' , 'Claim Class(LOCS)- Value Coordinates' : 'claim_class_value_coordinates' , 'ClaimNumber- Value Coordinates' : 'claim_number_value_coordinates' , 'Client- Value Coordinates' : 'client_value_coordinates' , 'Client Claim #- Value Coordinates' : 'client_claim_value_coordinates' , 'Const Def?- Value Coordinates' : 'const_def_value_coordinates' , 'Controverted?- Value Coordinates' : 'controverted_value_coordinates' , 'Date- Value Coordinates' : 'date_value_coordinates' , 'Ded/SIR- Value Coordinates' : 'ded_sir_value_coordinates' , 'Desc of Accident- Value Coordinates' : 'accident_description_value_coordinates' , 'EXPENSE Incured- Value Coordinates' : 'expense_incurred_value_coordinates' , 'EXPENSE Outstanding- Value Coordinates' : 'expense_outstanding_value_coordinates' , 'EXPENSE Paid- Value Coordinates' : 'expense_paid_value_coordinates' , 'Hire Date- Value Coordinates' : 'hire_date_value_coordinates' , 'IND/BI Incured- Value Coordinates' : 'indemnity_incurred_value_coordinates' , 'IND/BI Outstanding- Value Coordinates' : 'indemnity_outstanding_value_coordinates' , 'IND/BI Paid- Value Coordinates' : 'indemnity_paid_value_coordinates' , 'Juris/Acc State- Value Coordinates' : 'juris_st_value_coordinates' , 'Litigation?- Value Coordinates' : 'in_litigation_value_coordinates' , 'Location Codes(1-3)- Value Coordinates' : 'location_codes_1_to_3_value_coordinates' , 'Location Codes(4-6)- Value Coordinates' : 'location_codes_4_to_6_value_coordinates' , 'MED/PropDmg Incured- Value Coordinates' : 'medical_incurred_value_coordinates' , 'MED/PropDmg Outstanding- Value Coordinates' : 'medical_outstanding_value_coordinates' , 'MED/PropDmg Paid- Value Coordinates' : 'medical_paid_value_coordinates' , 'NET Incured- Value Coordinates' : 'net_incured_value_coordinates' , 'NET Outstanding- Value Coordinates' : 'net_outstanding_value_coordinates' , 'NET Paid- Value Coordinates' : 'net_paid_value_coordinates' , 'Occurrence Id- Value Coordinates' : 'occurrence_id_value_coordinates' , 'Policy Cancel Dates- Value Coordinates' : 'policy_cancel_dates_value_coordinates' , 'Policy Effective Dates- Value Coordinates' : 'policy_eff_date_value_coordinates' , 'RECOVERED Incured- Value Coordinates' : 'recovered_incured_value_coordinates' , 'RECOVERED Outstanding- Value Coordinates' : 'recovered_outstanding_value_coordinates' , 'RECOVERED Paid- Value Coordinates' : 'recovered_paid_value_coordinates' , 'REIMBURSED Paid- Value Coordinates' : 'reimbursed_paid_value_coordinates' , 'Social Security- Value Coordinates' : 'social_security_value_coordinates' , 'Subrogation?- Value Coordinates' : 'subrogation_value_coordinates' , 'TOTAL Incured- Value Coordinates' : 'total_incurred_value_coordinates' , 'TOTAL Outstanding- Value Coordinates' : 'total_outstanding_value_coordinates' , 'TOTAL Paid- Value Coordinates' : 'total_paid_value_coordinates' , 'Valued At- Value Coordinates' : 'valued_at_value_coordinates',
    'Acc Description- Value Coordinates' : 'accident_description_value_coordinates', 'Expensed Paid- Value Coordinates' : 'expense_paid_value_coordinates', 'Expensed Reserved- Value Coordinates' : 'expense_reserve_value_coordinates', 'Loss State- Value Coordinates' : 'loss_state_value_coordinates', 'Policy Name- Value Coordinates' : 'policy_name_value_coordinates', 'Policy Year- Value Coordinates' : 'policy_year_value_coordinates', 'Valn Id- Value Coordinates' : 'vlan_id_value_coordinates','Acc Description' : 'accident_description', 'Expensed Paid' : 'expense_paid' ,'Expensed Reserved' : 'expense_reserve', 'Loss State' : 'loss_state', 'Policy Name' : 'policy_name', 'Policy Year' : 'policy_year', 'Valn Id' : 'vlan_id'    }
    return v_col
   