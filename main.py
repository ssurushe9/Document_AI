import base64
from flask import Flask, request
import json
import os
from google.cloud import storage
from db import run_query,set_db,run_select_query
import extract_func
import acord_extraction
import warnings
import datetime
import pytz
import time
import concurrent.futures as cf
import pandas as pd
warnings.filterwarnings('ignore')
import re

app = Flask(__name__)
storage_client = storage.Client()

def update_document_table(status,file_name,submission_id,document_id):
    query = "update dev_bhhc.documents set status = '{}' , extraction_output ='{}' where document_id='{}'".format(status,file_name,document_id)
    output = run_query("UPDATE",query)
    print(output)
    sub_query = "select status from dev_bhhc.documents where submission_id ='{}'".format(submission_id)
    result = run_query("SELECT",sub_query)
    result_list = [dict(row) for row in result]
    status = ''
    for item in result_list:
        if ((item['status'] == 'classified' or item['status'] == 'processed') and (status != 'processing')):
            status = 'under_review'
        elif(item['status'] == 'classifying' or item['status'] == 'extracting'):
            status = 'processing'
    update_query = "update dev_bhhc.submission_table set status = '{}' where submission_id = '{}'".format(status,submission_id)
    output = run_query("UPDATE",update_query)
    print(output)
    
mapping_dct = {'Atlas General Insurance Services' : 'B-01' , 'TRAVELERS': 'B-02' , 'ICW Group Insurance Companies' : 'B-03' , 'BHHC' : 'B-04' , 'Amtrust' : 'B-05' , 'AIG' : 'B-06',  'CHUBB' : 'B-07' , 'EMPLOYERS' : 'B-08' , 'State Compensation Insurance Fund' : 'B-09' , 'Zenith' : 'B-10', 'Benchmark' : 'B-11' , 'Zurich' : 'B-12' , 'EVEREST' : 'B-13'}

def update_broker_id(broker_id,submission_id,document_id):
    query = "update dev_bhhc.documents set broker_id = '{}' where document_id='{}' and submission_id='{}' ".format(broker_id,document_id,submission_id)
    output = run_query("UPDATE",query)
    print(output)
    
def helper(input_list):
    page,object_list,template_json_path,submission_id = input_list[0] , input_list[1], input_list[2], input_list[3]
    document_id, img_path,input_bucket,doc_type = input_list[4], input_list[5],input_list[6],input_list[7]
    df_loss_details, df_loss_summary = extract_func.extract_dataframe_from_pdf_json_parallel(page,object_list,template_json_path,submission_id,document_id,img_path,input_bucket,doc_type)
    return df_loss_details, df_loss_summary

def helper_130(input_list):
    img_path, img , input_bucket = input_list[0] , input_list[1] , input_list[2]
    out_json = acord_extraction.process_extraction_accord_130(img_path, img , input_bucket)
    return out_json 

def upload_blob(bucket_name, blob_text, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(blob_text)
    print('File uploaded to gs bucket')

def get_broker_id(doc_type):
    for k in mapping_dct.keys():
        if k == doc_type :
            return mapping_dct.get(k)

@app.route("/", methods=["POST"])
def index():
    envelope = request.get_json()
    start = time.perf_counter()
    
    if not envelope:
        msg = "no Pub/Sub message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        message = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
   
    data = json.loads(message)
    output_bucket = data['output_bucket']
    input_bucket = data['input_bucket']
    submission_id = data['submission_id']
    document_id = data['document_id']
    
    out_bucket = storage_client.get_bucket(output_bucket)
    in_bucket = storage_client.get_bucket(input_bucket)
    
    print('#####################################',submission_id)
    print('#####################################',document_id)
    
    file_name = submission_id +'/attachments/'+ document_id + '/'+ document_id +'_object_detection' + '.json'
       
    blob = out_bucket.blob(file_name)
    input_json = json.loads(blob.download_as_string().decode())
    img_path = submission_id +'/attachments/'+ document_id + '/images/' 
    inp_img_list = []
    for blob in storage_client.list_blobs(input_bucket, prefix=img_path):
        inp_img_list.append(blob.name.split('/')[-1])
    
    if input_json.get('type') == 'Loss Run':
        doc_type = input_json.get('subtype')
        template_json_path = "json_data/" + doc_type + ".json"
        
        try:
            input_list = []
            for ele in input_json['Objects']:
                input_list.append((ele, input_json['Objects'][ele] , template_json_path , submission_id,document_id,img_path,input_bucket,doc_type))

            try:
                loss_details_list = []
                loss_summary_list = []
                with cf.ProcessPoolExecutor() as executor:
                    for loss_detail, loss_summary in executor.map( helper ,[ ele for ele in input_list ] ):
                        loss_details_list.append(loss_detail)
                        loss_summary_list.append(loss_summary)
                        
                df_loss_details, df_loss_summary = pd.concat(loss_details_list), pd.concat(loss_summary_list)
                zip_df_loss_details = df_loss_details.columns.tolist()
            except:
                print('************ Extraction code has some issues *********************')
                update_document_table('processed','',submission_id,document_id)
            loss_summary_cols = extract_func.loss_summary_col_mapping()
            df_loss_summary.rename(columns = loss_summary_cols , inplace = True)
            try:
                df_loss_summary.drop(['Policy','Policy- Value Coordinates'], axis = 1,inplace = True)
                print('Loss Summary column name Policy got dropped')
            except:
                print('Column Policy not found in given Doc sub type')

            loss_detail_cols = extract_func.loss_detail_col_mapping()
            df_loss_details.rename(columns = loss_detail_cols , inplace = True)
            
            connection = set_db()
            cursor=connection.cursor()
            V_df_loss_summary = json.dumps(list(df_loss_summary.columns))
            v_df_loss_details = json.dumps(list(df_loss_details.columns))
            
            if len(list(df_loss_details.columns)) != 0 :
                if doc_type == 'Amtrust' :
                    pass
                else:
                    v_sql = "UPDATE dev_bhhc.broker_metadata SET loss_detail_columns='{}', loss_summary_columns='{}' WHERE broker_name = '{}' ".format(v_df_loss_details,V_df_loss_summary,doc_type)
                    cursor.execute(v_sql)
                    connection.commit()
                    affected = cursor.rowcount
                    print('Number of records affected in Broker Metadata table is : ', affected)
            
            v_sql = "SELECT COUNT(*) from dev_bhhc.loss_details where document_id='{}' and submission_id='{}' ".format(document_id,submission_id)
            record_cnt = run_select_query(v_sql)
            print('Total records for submission and document_id : ',record_cnt[0])
            
            if record_cnt[0] != 0:
                v_sql_loss_summary = "Delete from dev_bhhc.loss_summary where document_id='{}' and submission_id='{}' ".format(document_id,submission_id)
                output1 = run_query("DELETE",v_sql_loss_summary)
                
                v_sql_loss_detail = "Delete from dev_bhhc.loss_details where document_id='{}' and submission_id='{}' ".format(document_id,submission_id)
                output2 = run_query("DELETE",v_sql_loss_detail)
                print('Number of records deleted with submission and document_id is: ',output2)
            
            #Loss Summary insertion to DB starts here
            summary_cols = ",".join([str(i) for i in df_loss_summary.columns.tolist()])
            for i,row in df_loss_summary.iterrows():
                sql = "INSERT INTO dev_bhhc.loss_summary (" + summary_cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
                temp_summary_ls = []
                for ele in range(len(tuple(row))):
                    if isinstance(tuple(row)[ele] , list):
                        temp_summary_ls.append( str(tuple(row)[ele]) )
                    else:
                        temp_summary_ls.append( str(tuple(row)[ele]).strip() )
                cursor.execute(sql, tuple(temp_summary_ls))
            connection.commit()
            
            #Loss detail insertion to DB starts here
            data_row = []
            for i,row in df_loss_details.iterrows():
                data_row.append(list(zip(zip_df_loss_details ,tuple(row))))
            
            detail_cols = ",".join([str(i) for i in df_loss_details.columns.tolist()])
            
            for row in data_row:
                temp_detail_ls = []
                sql = "INSERT INTO dev_bhhc.loss_details (" + detail_cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
                for ele in row:
                    if isinstance(ele[1] , list):
                        temp_detail_ls.append( str(ele[1]) )
                    else:
                        v_str = str(ele[1]).strip()
                        try:
                            for i in (ele[0].split()):
                                pat = '^' + i
                                v_str = re.sub(pat,'',v_str.strip())
                        except:
                            print('This elelment row got error ',ele)
                        temp_detail_ls.append( v_str.replace(':','').replace('/','').strip() )
                        
                cursor.execute(sql, tuple(temp_detail_ls))
                connection.commit()
                   
            broker_id = get_broker_id(doc_type)
            update_broker_id(broker_id,submission_id,document_id)
            update_document_table('processed','',submission_id,document_id)
            
        except:
            print('************ Extraction Pipeline has some issues *********************')
            update_document_table('processed','',submission_id,document_id)
    
    elif input_json.get('subtype') == 'ACORD 4':
        doc_dict = []
        for img in inp_img_list:
            if 'page1.jpeg' in img:
                out_json = acord_extraction.process_extraction_accord_4(img_path,img,input_bucket)
                doc_dict.append(out_json)
        
        json_object = json.dumps(doc_dict)
        file_name = submission_id + '/attachments/' + document_id + '/' + document_id + '_extraction.json'
        upload_blob( output_bucket,json_object,file_name )
        status = 'processed'
        update_document_table(status,file_name,submission_id,document_id)
            
    elif input_json.get('subtype') == 'ACORD 130':
        doc_dict = []
        inp_img_list_130 = []
        for blob in storage_client.list_blobs(input_bucket, prefix=img_path):
            inp_img_list_130.append( (img_path , blob.name.split('/')[-1] ,input_bucket) )
        
        with cf.ProcessPoolExecutor() as executor:
            for out_json in executor.map( helper_130  ,[ img for img in inp_img_list_130 ] ):
                doc_dict.append(out_json)

        json_object = json.dumps(doc_dict)
        file_name = submission_id + '/attachments/' + document_id + '/' + document_id + '_extraction.json'
        upload_blob( output_bucket,json_object,file_name )
        status = 'processed'
        update_document_table(status,file_name,submission_id,document_id)
        
    end = time.perf_counter()
    print('Took ****************', end - start , '***************** seconds to finsh the job')
    
    return "", 204
    
if __name__ == "__main__":
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8080

    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host="127.0.0.1", port=PORT, debug=True)
