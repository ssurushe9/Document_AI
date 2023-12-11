import io
import re
import pandas as pd
import base64
import requests
import json
from PIL import Image
import trp
from trp import Document
from google.cloud import storage
from difflib import SequenceMatcher

storage_client = storage.Client()

def get_table_trp(response, doc, width, height, template):
    """
     This function return tables data for\
     given textract response and desired table

     arguments:
         input: textract response, trp document width and height of image,\
         template having keys JSON

     return:
         Extracts the target  tables from given\
          input response : list
    """
    if '' in [response, width, height, template]:
        return ''
    doc = Document(response)
    table_details = []
    # try:
    for page in doc.pages:
        for table in page.tables:
            full_table = []
            for row in table.rows:
                table_rows = []
                for cell in row.cells:
                    table_rows.append(cell.text)
                full_table.append(table_rows)
            bbx = [table.geometry.boundingBox.left * width,
                   table.geometry.boundingBox.top * height,
                   table.geometry.boundingBox.width * width,
                   table.geometry.boundingBox.height * height]
            full_table.append(bbx)
            table_out = pd.DataFrame()
            #empty_header_str = [
            #    i for i, x in enumerate(
            #        full_table[0]) if x == '']
            #if empty_header_str:
            #    cnt = 0
            #    for i in empty_header_str:
            #        if cnt > 0:
            #            i = i - cnt
            #        for ind in range(len(full_table) - 1):
            #            full_table[ind].pop(i)
            #        cnt = cnt + 1
            if len(template['table']['table_with_header']) > 0:
                for heads in template['table']['table_with_header']:
                    math_per = SequenceMatcher(None,''.join([x.strip().lower()
                                for x in heads['headers']]),''.join([x.strip().lower()
                                    for x in full_table[0]]))
                    math_per_second = SequenceMatcher(None,''.join([x.strip().lower()
                                for x in heads['headers']]),''.join([x.strip().lower()
                                    for x in full_table[1]]))
                    if "HIGHEST FLOOR" in heads['extract']:
                        table_name = "Location"
                    elif "office phone" in heads['extract']:
                        table_name = "Contact Information"
                    elif "DUTIES" in heads['extract']:
                        table_name = "INDIVIDUALS INCLUDED / EXCLUDED"
                    elif "RATE" in heads['extract']:
                        table_name = "RATING INFORMATION - STATE"
                    elif "CARRIER & POLICY NUMBER" in heads['extract']:
                        table_name = "PRIOR CARRIER INFORMATION / LOSS HISTORY"
                    else:
                        table_name = "Table Data"
                    if float(math_per.ratio()) > 0.94 :
                        # cols = [x.lower().strip() for x  in full_table[0]]
                        table_out = pd.DataFrame(
                            data = full_table[:-1], columns=full_table[0])
                        # table_out = table_out[heads['extract']]
                        table_details.append({
                            'table_name': table_name,
                            'header': full_table[0],
                            'rows': [{"data": row.tolist()}\
                                     for _i, row in table_out.iterrows()\
                                     if ''.join([x.lower().strip()
                                                 for x in row.to_list()])\
                                     != ''.join([x.lower().strip()
                                                 for x in\
                                                 full_table[0]])],
                            'bounding_box': full_table[-1],
                        })
                    elif float(math_per_second.ratio()) > 0.94 : 
                        table_out = pd.DataFrame(
                            data=full_table[2:-1], columns=full_table[1])
                        # table_out = table_out[heads['extract']]
                        table_details.append({
                            'table_name': table_name,
                            'header': full_table[1],
                            'rows': [{"data": row.tolist()}\
                                     for _i, row in table_out.iterrows()\
                                     if ''.join([x.lower().strip()
                                                 for x in row.to_list()])\
                                     != ''.join([x.lower().strip()
                                                 for x in\
                                                 full_table[1]])],
                            'bounding_box': full_table[-1],
                        })                                            
                                                                     
        if 'table_from_bbx' in template['table']:
            table_details = sp_table(page,template,width, height,table_details)
        if 'table_from_bbx_2' in template['table']:
            table_details = sp_table_130_table(page,template,width, height,table_details)
    return table_details
    # except BaseException:
    #     return table_details

def sp_table(page,template,width, height,table_details):
    list_rep = {}
    key_match= False
    if len(template['table']['table_from_bbx']) > 0:
        for each in template['table']['table_from_bbx']:
            if 'key_word' in each:
                for lines in page.lines:
                    # print(lines.text,"lines")
                    if SequenceMatcher(None,str(lines.text).lower().strip(),each['key_word'].lower().strip()\
                                    ).ratio() > 0.94:
                        key_match = each['key_word']
                        key_match_top = lines.geometry.boundingBox.top*100
                        key_match_left = lines.geometry.boundingBox.left*100
                        key_match_left = key_match_left+lines.geometry.boundingBox.width*100/2
                        break
            if key_match:
                tab_cols = []
                for field in  page.lines:

                    if ( field.geometry.boundingBox.top*100 - key_match_top < each['range'][-1])\
                        and ( field.geometry.boundingBox.top*100 - key_match_top > 0)\
                        and (abs(field.geometry.boundingBox.left*100-key_match_left)  \
                        <= each['range'][0]):
                        tab_cols.append(str(field.text))
                list_rep[each['prefix']] = tab_cols
    list_rep = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in list_rep.items() ]))
    list_rep = list_rep.fillna('')
    bbvx = [.028* width,
                   0.13 * height,
                   .89 * width,
                   .3 * height]
    if len(list_rep)>0:
        table_details.append({
                # pass from input or...
                'table_name': 'RATING INFORMATION - STATE:',
                'header': list_rep.columns.to_list(),
                'rows': [{"data": row.tolist()}\
                         for _i, row in list_rep.iterrows()],
                'bounding_box': bbvx,
                            })                                 
    return table_details 
    
def sp_table_130_table(page,template,width, height,table_details):
    list_rep = {}
    key_match= False
    def combine_list(le_list):
        new_list = []
        while len(le_list)>0:
            new_list.append(le_list.pop(0)+ ", " +le_list.pop(0))
        return new_list
    if len(template['table']['table_from_bbx_2']) > 0:
        for each in template['table']['table_from_bbx_2']:
            if 'key_word' in each:
                for lines in page.lines:
                    if SequenceMatcher(None,str(lines.text).lower().strip(),each['key_word'].lower().strip()\
                                    ).ratio() > 0.94:
                        key_match = each['key_word']
                        key_match_top = lines.geometry.boundingBox.top*100
                        key_match_left = lines.geometry.boundingBox.left*100
                        break
            if key_match:
                tab_cols = []
                for field in  page.lines:
                    if ( field.geometry.boundingBox.top*100 - key_match_top < each['range'][-1])\
                        and ( field.geometry.boundingBox.top*100 - key_match_top > 0)\
                        and (abs(field.geometry.boundingBox.left*100-key_match_left)  \
                        <= each['range'][0]):
                        # print(field.text,"yyyyyy")
                        tab_cols.append(str(field.text))
                list_rep[each['prefix']] = tab_cols
    try:            
        list_rep['CARRIER & POLICY NUMBER'] =  combine_list(list_rep['CARRIER & POLICY NUMBER'])
    except:
        pass
    list_rep = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in list_rep.items() ]))
    list_rep = list_rep.fillna('')
    bbvx = [.028* width,
                   0.0625 * height,
                   .89 * width,
                   .179 * height]
    if len(list_rep)>0:
        table_details.append({
                # pass from input or...
                'table_name': 'PRIOR CARRIER INFORMATION / LOSS HISTORY',
                'header': list_rep.columns.to_list(),
                'rows': [{"data": row.tolist()}\
                         for _i, row in list_rep.iterrows()],
                'bounding_box': bbvx,
                            })                               
    return table_details
    
    
def key_value_check_box(
        key,
        value,
        new_bb_key,
        new_bb_value,
        template,
        box_check):
    """
     This function return key-value information for simple checkboxes

     arguments:
         input: key,bb info for keys,values,\
         input template and box_check dictionary

     return:
         Updated dictionary of box_check : dict
    """
    try:
        for each in template['check_box_fields']['check_box']:
            if key.lower().strip(
            ) in template['check_box_fields']['check_box'][each]\
                    and value[0] == 'SELECTED':
                if each in box_check:
                    box_check[each].append(
                        [key,
                         new_bb_key,
                         new_bb_value,
                         value[-2] / 100,
                         value[-1] / 100])
                else:
                    box_check[each] = [
                        [key,
                         new_bb_key,
                         new_bb_value,
                         value[-2] / 100,
                         value[-1] / 100]]
        return box_check
    except BaseException:
        if '' in [key, value, new_bb_key, new_bb_value, template, box_check]:
            return ''
        else:
            return box_check
    
    
def extract_child_key(input_string, possible_list):
    """
     This Finds similar string frrom input list

     arguments:
         input: string,possible list of values.
     return:
         matched string from input list:str

    """

    # try:
    # address = input_string.replace(",", " ").rstrip()
    regex = re.compile(
        r"\b(" + "|".join(possible_list) + r")\b",
        re.IGNORECASE)
    result = regex.findall(input_string)
    if result:
        return result[-1]
    else:
        return ""
    # except BaseException:
    #     return ""
    
def bb_box(values, key=True):
    """Taking out bbox in proper formted dict

     input: list of values from kv extraction

     return:
         bounding box in a formatted dictionary: dict
    """
    # try:
    bbx_dic = {}
    if key:
        bbx_dic['Width'] = 0
        if len(values[1]) > 0:
            for each in values[1]:
                bbx_dic['Width'] += each[1]['Width']
            bbx_dic['Left'] = values[1][0][1]['Left']
            bbx_dic['Top'] = values[1][0][1]['Top']
            bbx_dic['Height'] = values[1][0][1]['Height']
    else:
        if len(values[2]) > 0:
            bbx_dic['Width'] = 0
            for each in values[2]:
                bbx_dic['Width'] += each[1]['Width']
            bbx_dic['Left'] = values[2][0][1]['Left']
            bbx_dic['Top'] = values[2][0][1]['Top']
            bbx_dic['Height'] = values[2][0][1]['Height']
        else:
            bbx_dic = {}
    return bbx_dic
    # except BaseException:
    #     if '' in [key, values]:
    #         return ''
    #     else:
    #         return {}

def split_checkbox(key, value, new_bb_key, new_bb_value, template, box_check):
    """
     This function return key-value information for paired checkboxes

     arguments:
         input: key,bb info for keys,values,\
         input template and box_check dixtionary

     return:
         Updated dictionary of box_check : dict
    """

    # try:
    check_bx_fields = \
        template['check_box_fields']['check_box_splitted']
    for each in\
            check_bx_fields:
        if key.lower().strip() in\
            check_bx_fields[each]\
                and value[0] == 'SELECTED':
            for keys_splitted in\
                    check_bx_fields[each]:
                if key == keys_splitted and value[0] == 'SELECTED':
                    if each.split('/')[0] in box_check:
                        box_check[each.split('/')[0]].append(
                            [value[1][0][0].split('/')[0],
                             new_bb_key,
                             new_bb_value,
                             value[-2] / 100,
                             value[-1] / 100])
                    else:
                        box_check[each.split('/')[0]]\
                            = [[value[1][0][0].split('/')[0],
                                new_bb_key,
                                new_bb_value,
                                value[-2] / 100,
                                value[-1] / 100]]
                    if each.split('/')[1] in box_check:
                        box_check[each.split('/')[1]].append(
                            [value[1][0][0].split('/')[-1],
                             new_bb_key,
                             new_bb_value,
                             value[-2] / 100,
                             value[-1] / 100])
                    else:
                        box_check[each.split('/')[1]]\
                            = [[value[1][0][0].split('/')[-1],
                                new_bb_key,
                                new_bb_value,
                                value[-2] / 100,
                                value[-1] / 100]]
                    # dict_entity[each] = value[1][0][0].split('/')[0],
                else:
                    pass
    return box_check
    # except BaseException:
    #     if '' in [key, value, new_bb_key, new_bb_value, template, box_check]:
    #         return ''
    #     else:
    #         return box_check


def extract_kv_filter(kvs,kvs_helper_dict,width, height, template):
    """
     This function filter out desired key-value information

     arguments:
         input: key value dictionary,width and height of image,\
         template having keys JSON

     return:
         list of desired output :list
         dictionary of updated key values :dict
    """

    # try:
#     print(kvs,"kvs")
    res_dict = []
    dict_entity = {}
    # accord_dict = []
    box_check = {}
    accord_list = []
    if template.get('accord_rule') and \
            len(template.get('accord_rule')) > 0:
        for each in template.get(
                'accord_rule'):
            for items in each['parent']:
                accord_list.append(items.lower().strip())


    for key, value in kvs.items():
    
        full_list = template["field_keys"] + sum(template["check_box_fields"]["check_box"].values(), []) + sum(template["check_box_fields"]["check_box_splitted"].values(), []) + accord_list
        
        for every_k in full_list:
            if SequenceMatcher(None,key.lower().strip(),every_k).ratio() > 0.94:
        
                bb_key = bb_box(value, key=True)
                bb_value = bb_box(value, key=False)
                try:
                    left_key = width * bb_key["Left"]
                    top_key = height * bb_key["Top"]
                    left_value = width * bb_value["Left"]
                    top_value = height * bb_value["Top"]
                    new_bb_key = [
                        left_key,
                        top_key,
                        bb_key["Width"] * width,
                        bb_key["Height"] * height,
                    ]
                    new_bb_value = [
                        left_value,
                        top_value,
                        bb_value["Width"] * width,
                        bb_value["Height"] * height,
                    ]
                except BaseException:
                    new_bb_key = []
                    new_bb_value = []
                try:
                    dis_name = template["display_name"][key.lower().strip()][1]
                    key_name = template["display_name"][key.lower().strip()][0]
                except BaseException:
                    dis_name = key.lower().strip()
                    key_name = key.lower().strip()
                if key.lower().strip() in sum(
                    template["check_box_fields"]["check_box"].values(), []
                ):
                    if (
                        key.lower().strip() == "none"
                        and "Registered (lowa Only)".lower().strip()
                        not in kvs.items()
                    ):
                        pass
                    else:
                        box_check = key_value_check_box(
                            key, value, new_bb_key,
                            new_bb_value, template, box_check)
                if key.lower().strip() in sum(
                        template["check_box_fields"]
                                ["check_box_splitted"].values(), []):
                    box_check = split_checkbox(
                        key, value, new_bb_key,
                        new_bb_value, template, box_check)
                if key.lower().strip() in accord_list:
                    dict_entity[key] = value[0]

                for fild_keys in template["field_keys"]:
                    if SequenceMatcher(None,key.lower().strip(),fild_keys).ratio() > 0.94 and (value[0] not in [None, 'None', "", " "]):
            
                        dict_entity[key] = value[0]
                        res_dict.append(
                            {
                                "value": value[0],
                                "key": key_name,
                                "display_key": re.sub('[^a-zA-Z0-9 \n\.]', '', dis_name).title(),
                                "key_bounding_box": new_bb_key,
                                "value_bounding_box": new_bb_value,
                                "key_confidence_score": value[-2] / 100,
                                "value_confidence_score": value[-1] / 100,
                                "hitl_confirmed": False,
                            }
                        )
    for each in box_check:
        try:
            dis_name = template["display_name"][each][1]
            key_name = template["display_name"][each][0]
        except BaseException:
            dis_name = each
            key_name = each

        if len(box_check[each]) > 1:
            for enu, items in enumerate(box_check[each]):
                dict_entity[each + "_" + str(enu)] = items[0]
                # print(key_name + "_" + str(enu),"key_name")
                res_dict.append(
                    {
                        "key": key_name + "_" + str(enu),
                        "value": items[0],
                        "display_key": re.sub('[^a-zA-Z0-9 \n\.]', '', dis_name).title(),
                        "key_bounding_box": items[1],
                        "value_bounding_box": items[2],
                        "key_confidence_score": items[3],
                        "value_confidence_score": items[4],
                        "hitl_confirmed": False,
                    }
                )
        else:
            dict_entity[each] = box_check[each][0][0]
            res_dict.append(
                {
                    "key": key_name,
                    "value": box_check[each][0][0],
                    "display_key": re.sub('[^a-zA-Z0-9 \n\.]', '', dis_name).title(),
                    "key_bounding_box": box_check[each][0][1],
                    "value_bounding_box": box_check[each][0][2],
                    "key_confidence_score": box_check[each][0][3],
                    "value_confidence_score": box_check[each][0][4],
                    "hitl_confirmed": False,
                }
            )

    # 3/17/2022
    if template.get('accord_rule') and \
            len(template.get('accord_rule')) > 0:
        str_list = []
        key_map = {}
        for each in template.get(
                'accord_rule'):
            for items in each['parent']:
                if dict_entity.get(items) not in [None, "", " "]:
                    child_value = extract_child_key(
                        dict_entity.get(items), each['possible_values'])
                    str_list.append(child_value)
                    key_map[child_value] = items
                try:
                    child_value = max(str_list, key=len)
                    key_identified = key_map[child_value]
                    dict_entity[each['child']] = child_value
                    try:
                        child_name =\
                            template['display_name'][each['child']]
                        dis_name = child_name[1]
                        key_name = child_name[0]
                    except BaseException:
                        child_name = each['child']
                        dis_name = child_name
                        key_name = child_name
                    if child_value not in ["", " "]:
                        res_dict.append({'display_key': re.sub('[^a-zA-Z0-9 \n\.]', '', dis_name).title(),
                                         'key': key_name,
                                         'value': child_value,
                                         'key_bounding_box': [],
                                         'value_bounding_box':
                                         [kvs.get(key_identified)
                                             [2][0][1]['Left'] * width,
                                          kvs.get(key_identified)
                                             [2][0][1]['Top'] * height,
                                          kvs.get(key_identified)
                                             [2][0][1]['Width'] * width,
                                          kvs.get(key_identified)
                                             [2][0][1]['Height'] * height],
                                         'key_confidence_score':
                                         kvs.get(key_identified)[-2] / 100,
                                         'value_confidence_score':
                                         kvs.get(key_identified)[-1] / 100,
                                         'hitl_confirmed': False})
                except BaseException:
                    pass
    for key,value in kvs_helper_dict.items():
        bb_key = bb_box(value, key=True)
        bb_value = bb_box(value, key=False)
        try:
            left_key = width * bb_key["Left"]
            top_key = height * bb_key["Top"]
            left_value = width * bb_value["Left"]
            top_value = height * bb_value["Top"]
            new_bb_key = [
                left_key,
                top_key,
                bb_key["Width"] * width,
                bb_key["Height"] * height,
            ]
            new_bb_value = [
                left_value,
                top_value,
                bb_value["Width"] * width,
                bb_value["Height"] * height,
            ]
        except BaseException:
            new_bb_key = []
            new_bb_value = []
        try:
            dis_name = template["display_name"][key.lower().strip()][1]
            key_name = template["display_name"][key.lower().strip()][0]
        except BaseException:
            dis_name = key.lower().strip()
            key_name = key.lower().strip()
        res_dict.append(
                    {
                        "value": value[0],
                        "display_key": re.sub('[^a-zA-Z0-9 \n\.]', '', dis_name).capitalize(),
                        "key": key_name,
                        "key_bounding_box": new_bb_key,
                        "value_bounding_box": new_bb_value,
                        "key_confidence_score": value[-2] / 100,
                        "value_confidence_score": value[-1] / 100,
                        "hitl_confirmed": False,
                    })

    return res_dict, dict_entity
    # except BaseException:
    #     if '' in [kvs, width, height, template]:
    #         return '', ''
    #     else:
    #         return [], {}

def get_repeated_entity(template,page):
    list_rep = {}
    for each in template['repeated_entity']:
        word_list = each['word_list']
        range_buffer = each["range"]
        key_match = False
        check_box = False
        if 'key_word_string' in each:
            for lines in page.lines:
                # print(lines.text,"llll",lines.geometry.boundingBox.top*100,lines.geometry.boundingBox.left*100)
                if str(lines.text).lower().strip() == each['key_word_string'].lower().strip():
                    key_match = each['key_word_string']
                    key_match_top = lines.geometry.boundingBox.top*100
                    key_match_left = lines.geometry.boundingBox.left*100
                    check_box = True
                    # print("matched","123",key_match)
                    break
        if 'key_word' in each:
            for field in page.form.fields:#for item in response["Blocks"]:
                if (each['key_word'].lower().strip() == str(field.key).lower().strip()):
                    key_match = each['key_word']
                    key_match_top = field.key.geometry.boundingBox.top*100
                    key_match_left = field.key.geometry.boundingBox.left*100
                    check_box = False
                    break
        if key_match:
            for field in page.form.fields:
                if (str(field.key).strip().lower() in word_list) and (field.key.geometry.boundingBox.top*100-key_match_top  <= each['range'][-1]) and (field.key.geometry.boundingBox.top*100-key_match_top >= -0.07) and (abs(field.key.geometry.boundingBox.left*100-key_match_left) <= each['range'][0]):
                    try:
                        key_width = field.key.geometry.boundingBox.width
                        key_height = field.key.geometry.boundingBox.height
                        key_left = field.key.geometry.boundingBox.left
                        key_top = field.key.geometry.boundingBox.top
                    except BaseException:
                        key_width, key_height, key_left, key_top = 1, 1, 1, 1
                    try:
                        value_width = field.value.geometry.boundingBox.width
                        value_height = field.value.geometry.boundingBox.height
                        value_left = field.value.geometry.boundingBox.left
                        value_top = field.value.geometry.boundingBox.top
                    except BaseException:
                        value_width, value_height, value_left, value_top\
                            = 1, 1, 1, 1
                    try:
                        key_confidence = field.key.confidence
                    except BaseException:
                        key_confidence = 43  # if value is none
                    try:
                        value_confidence = field.value.confidence
                    except BaseException:
                        value_confidence = 43
                    if not check_box and (str(field.value) not in [None, "","None"]):
                        key_name = each['prefix']+str(field.key).strip().lower()
                        value = str(field.value)
                        list_rep[key_name] = [value,[(key_name,{
                                            "Width": key_width,
                                            "Height": key_height,
                                            "Left": key_left,
                                            "Top": key_top
                                        })],[(value,
                                            {"Width": value_width,
                                            "Height": value_height,
                                            "Left": value_left,
                                            "Top": value_top
                                            })
                                        ],key_confidence,value_confidence]
                    
                    elif (str(field.value)=='SELECTED'):
                        key_name = each['prefix']
                        value = str(field.key)
                        list_rep[key_name] = [value,[(key_name,{
                                            "Width": key_width,
                                            "Height": key_height,
                                            "Left": key_left,
                                            "Top": key_top
                                        })],[(value,
                                            {"Width": value_width,
                                            "Height": value_height,
                                            "Left": value_left,
                                            "Top": value_top
                                            })
                                        ],key_confidence,value_confidence]

    return list_rep       

def get_kv_pair(doc,template):
    # Finding key and value from document
    """
     This function Finds key and value from document

     arguments:
         input: AWS response

     return:

         dictionary of key-values pairs :dict
    """
    # try:
    del_list = []
    kvs_dict = {}
    kvs_helper_dict ={}
    for page in doc.pages:
        if 'repeated_entity' in template:
            list_rep = get_repeated_entity(template,page)
        if template.get('from_bbox') and \
                        len(template.get('from_bbox')) > 0:
            for each in template['from_bbox']:
                #print(template['from_bbox'][each])
                bbox = trp.BoundingBox(template['from_bbox'][each][0],
                                       template['from_bbox'][each][1],
                                       template['from_bbox'][each][2],
                                       template['from_bbox'][each][3])
                req_lines = doc.pages[0].getLinesInBoundingBox(bbox)
                cninit = []
                for contnt in req_lines:
                    #print(str(contnt.text),"l>")
                    cninit.append(str(contnt.text))
                cninit = template['from_bbox'][each][-1].join(cninit)
                kvs_dict[str(each)] = [
                        str(cninit),
                        [
                            (
                                str(each),
                                {
                                    "Width": template['from_bbox'][each][0],
                                    "Height": template['from_bbox'][each][1],
                                    "Left": template['from_bbox'][each][2],
                                    "Top": template['from_bbox'][each][3],
                                },
                            )
                        ],
                        [
                            (
                                str(cninit),
                                {
                                    "Width": template['from_bbox'][each][0],
                                    "Height": template['from_bbox'][each][1],
                                    "Left": template['from_bbox'][each][2],
                                    "Top": template['from_bbox'][each][3],
                                },
                            )
                        ],
                        98.9,
                        81.9,
                    ]    
            # bbox = [width, height, left, top]
        # for field in page.line:    
        for field in page.form.fields:
            try:
                key_width = field.key.geometry.boundingBox.width
                key_height = field.key.geometry.boundingBox.height
                key_left = field.key.geometry.boundingBox.left
                key_top = field.key.geometry.boundingBox.top
            except BaseException:
                key_width, key_height, key_left, key_top = 1, 1, 1, 1
            try:
                value_width = field.value.geometry.boundingBox.width
                value_height = field.value.geometry.boundingBox.height
                value_left = field.value.geometry.boundingBox.left
                value_top = field.value.geometry.boundingBox.top
            except BaseException:
                value_width, value_height, value_left, value_top\
                    = 1, 1, 1, 1
                # some time value is empty so no confidence score and bbx,
            try:
                key_confidence = field.key.confidence
            except BaseException:
                key_confidence = 43  # if value is none
            try:
                value_confidence = field.value.confidence
            except BaseException:
                value_confidence = 43

            try:
                if (
                    str(field.key) not in kvs_dict
                    and str(field.value) != "NOT_SELECTED"
                ):
                    kvs_dict[str(field.key)] = [
                        str(field.value),
                        [
                            (
                                str(field.key),
                                {
                                    "Width": key_width,
                                    "Height": key_height,
                                    "Left": key_left,
                                    "Top": key_top,
                                },
                            )
                        ],
                        [
                            (
                                str(field.value),
                                {
                                    "Width": value_width,
                                    "Height": value_height,
                                    "Left": value_left,
                                    "Top": value_top,
                                },
                            )
                        ],
                        key_confidence,
                        value_confidence,
                    ]
            except BaseException:
                pass
    return kvs_dict,list_rep
    # except BaseException:
    #     if doc == '':
    #         return ''
    #     else:
    #         return {}
def extraction_pipe_kv(doc, width, height, template,inp_res):
    """
     This function call other function in order

     arguments:
         input: TRP Document, width and height of image, target keys dict

     return:

         dictionary of key-values pairs :dict
         list of all extraction :list
    """
    # try:
    if '' in [doc, width, height, template]:
        return '', ''
    kvs,kvs_helper_dict = get_kv_pair(doc,template)
    result, entity_dict = extract_kv_filter(kvs,kvs_helper_dict, width, height, template)
    return result, entity_dict
    # except BaseException:
        # if '' in [doc, width, height, template]:
            # return '', ''
        # else:
            # return [], {}
        
def document_extract_entity(inp_res, width, height, template,page):  # mohd
    """
     This function return the page wise extraction data for\
     given textract response and desired  input keys

     arguments:
         input: textract response, width and height of image,\
         template having keys JSON

     return:
         Extracts the target KV-pairs and tables from given\
         input response : dict
    """
    if '' in [inp_res, width, height, template]:
        return ''
    # try:
    key = int(page.split('_')[-1].split('.')[0].replace('page',''))
    entire_dict = []
    entity_dict = {}
    page_dict = {}
    # for eavj res :
    page_dict["page_number"] = key
    trp_doc = Document(inp_res)
    res, page_vise_dict = extraction_pipe_kv(
        trp_doc, width, height, template,inp_res)
    entity_dict.update(page_vise_dict)
    # table
    if "table" in template:
        table_details = get_table_trp(
            inp_res, trp_doc, width, height, template)
    else:
        table_details = []
    page_dict["table"] = table_details
    page_dict["data"] = res
    #entire_dict.append(page_dict)
    #print(res, "out")
    return page_dict
    # except BaseException:
    #     return {}


def get_pdf_content_textract(File_Path,input_bucket,page):
    
    blob = storage_client.bucket(input_bucket).get_blob(File_Path)
    blob.download_to_filename(page)
    image = Image.open(page)
    w,h = image.size
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    payload = [{ "content": str(base64.b64encode(byte_im))[2:-1]}]
    url = "https://udj0ogrn19.execute-api.us-east-1.amazonaws.com/test"
    headers = {
    'content-type': "application/json",
    'x-api-key': "ASipZo6Xp22djmsxmX72435epWTV7PWg9mMo24Xc"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    data_icw = json.dumps(response.json())
    data_icw = json.loads(data_icw)
    
    return data_icw,w,h 
    
def read_json(json_path):
    f = open(json_path)
    data = json.load(f)
    f.close()
    return data
    
def process_extraction_accord_130(img_path,img,input_bucket):   
    file_path = img_path + img
    template = read_json('./Accord_Form_130.json')
    aws_res,width, height = get_pdf_content_textract(file_path,input_bucket,img)
    # if 'page1' in img:
        # with open("./130_page1.json", "w+") as f:
            # json.dump(aws_res, f)
    out_json = document_extract_entity(aws_res,width, height,template,img)
    out_json['data'] = sorted( out_json['data'] , key = lambda x: ( x['value_bounding_box'][1], x['value_bounding_box'][0] ) )

    for dt in out_json['data']:
        if dt['display_key'] == 'Phone':
            if dt['key_bounding_box'][0] < 500 :
                dt['display_key'] = 'Employer Phone Number'

        elif dt['display_key'] == 'Phone ':
            if dt['key_bounding_box'][0] > 500 :
                dt['display_key'] = 'Carrier Phone Number'
    
    # To Remove the blank empty rows from the Table
    for dt in out_json['table']:
        tmp_ls = []
        for inner_data in dt['rows'] :
            if len(''.join(inner_data['data']) ) != 0:
                tmp_ls.append(inner_data)
        dt['rows'] = tmp_ls
    
    # To remove the empty data in Key Value extracted
    for dt in out_json['data']:   
        if dt['value'] == "(A/C, No, Ext):" :
            dt['value'] = ''

        elif dt.get('value') == "None" :
            dt['value'] = ''
    
    # To remove multiple state value from key value
    cnt = 1
    for i in out_json['data']:
        if i['display_key'] == 'State' :
            if cnt > 1 :
                out_json['data'].remove(i)
            cnt = cnt + 1
    
    v_cnt = 1
    for i in out_json['data']:
        if i['display_key'] == 'Agency Name And Address' :
            if v_cnt > 1 :
                out_json['data'].remove(i)
            v_cnt = v_cnt + 1
    
    # To remove the multiple check boxes information
    insurer_score = 0
    insurer_cnt = 1
    for i in out_json['data']:
        if i['display_key'] == 'Insurer Type':
            if insurer_cnt > 1 and i['key_confidence_score'] < insurer_score :
                out_json['data'].remove(i)
            insurer_score = i['key_confidence_score']
            insurer_cnt = insurer_cnt + 1
            
    # To Remove NOT_SELECTED data in all Table cells
    for ele in out_json['table']:
        for data in ele['rows'] :
            data['data'] = list(map(lambda x: x.replace('NOT_SELECTED', '').strip().replace(',','').replace('@ g','@g'), data['data']))
    
    # TO ADD State column in Table Rating Information
    v_data = []
    for ele in out_json['data']:
        if 'rating' in ele['key'] and 'information' in ele['key'] :
            v_data.append(ele['value'])
            out_json['data'].remove(ele) 
    
    for tab in out_json['table']:
        if tab['table_name'] == 'RATING INFORMATION - STATE':
            for row in tab['rows'] :
                row['data'] = v_data + row['data']
            tab['header'] = ['State'] +  tab['header']
    
    return out_json
    
def process_extraction_accord_4(img_path,img,input_bucket):
    
    template = read_json('./Accord_Form_4.json')
    
    file_path = img_path + img
    aws_res,width, height = get_pdf_content_textract(file_path,input_bucket,img)
    out_json = document_extract_entity(aws_res,width, height,template,img)
    out_json['data'] = sorted( out_json['data'] , key = lambda x: ( x['value_bounding_box'][1], x['value_bounding_box'][0] ) )
    
    for dt in out_json['data']:
        if dt.get('value') == "(A/C, No, Ext):" :
            dt['value'] = ''
        elif dt.get('value') == "None" :
            dt['value'] = ''
    
    for dt in out_json['data']:
        if dt['display_key'] == 'Phone':
            if dt['key_bounding_box'][0] < 500 :
                dt['display_key'] = 'Employer Phone Number'
                
        elif dt['display_key'] == 'Phone ':
            if dt['key_bounding_box'][0] > 500 :
                dt['display_key'] = 'Carrier Phone Number'
    #remove Duplicate state keys            
    cnt = 1
    for i in out_json['data']:
        if i['display_key'] == 'State' :
            if cnt > 1 :
                out_json['data'].remove(i)
            cnt = cnt + 1
                
    return out_json