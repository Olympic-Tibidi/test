import streamlit as st
import streamlit as st
from streamlit_profiler import Profiler
#from st_files_connection import FilesConnection

import streamlit.components.v1 as components
import cv2
import numpy as np
#from st_files_connection import FilesConnection
import gcsfs
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
import streamlit_authenticator as sa
from google.cloud import storage
import os
import io
from io import StringIO
from io import BytesIO
import time
from PIL import Image
import random
import base64
import streamlit_authenticator as stauth
#from camera_input_live import camera_input_live
import pandas as pd
import datetime
from requests import get
from collections import defaultdict
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime as dt
from docx import Document
import pickle
import yaml
from yaml.loader import SafeLoader
#from streamlit_extras.dataframe_explorer import dataframe_explorer
import math
import plotly.express as px               #to create interactive charts
import plotly.graph_objects as go         #to create interactive charts
from plotly.subplots import make_subplots
import zipfile
import requests
from bs4 import BeautifulSoup
import csv

import re
import tempfile
import plotly.graph_objects as go

from pandas.tseries.offsets import BDay
import calendar
from google.cloud import documentai
from google.oauth2 import service_account
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

import google.auth
from rapidfuzz import process as processs
import warnings
warnings.simplefilter("ignore", UserWarning)
#from google.cloud import bigquery

#credentials = service_account.Credentials.from_service_account_info(st.secrets["gcs_connections"])

#client = documentai.DocumentProcessorServiceClient(credentials=credentials)

# credentials, _ = google.auth.default()
# credentials = google.auth.credentials.with_scopes_if_required(credentials, bigquery.Client.SCOPE)
# project_id = "newsuzano"
# storage_client= storage.Client(project=project_id,credentials=credentials)
# authed_http = google.auth.transport.requests.AuthorizedSession(credentials)


gcp_service_account_info = st.secrets["gcs_connections"]

def get_storage_client():
    # Create a storage client using the credentials
    storage_client = storage.Client.from_service_account_info(gcp_service_account_info)
    return storage_client
project_id = "Newsuz"


st.set_page_config(layout="wide")

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets['private_key']

target_bucket="new_suzano_spare"
utc_difference=8

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        user=st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return "No",user
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        user=st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return "No",user
    else:
        # Password correct.
        return "Yes",user



def output():
    #with open(fr'Suzano_EDI_{a}_{release_order_number}.txt', 'r') as f:
    with open('placeholder.txt', 'r') as f:
        output_text = f.read()
    return output_text

def send_email(subject, body, sender, recipients, password):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    # Attach the body of the email as text
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    #print("Message sent!")


def send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    # Attach the body of the email as text
    msg.attach(MIMEText(body, 'plain'))

    # Read the file content and attach it as a text file
    with open(file_path, 'r') as f:
        attachment = MIMEText(f.read())
    attachment.add_header('Content-Disposition', 'attachment', filename=file_name)
    msg.attach(attachment)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")

def gcp_download(bucket_name, source_file_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_text()
    return data
# def gcp_download_new(bucket_name, source_file_name):
#     conn = st.connection('gcs', type=FilesConnection)
#     a = conn.read(f"{bucket_name}/{source_file_name}", ttl=600)
#     return a
def gcp_download_x(bucket_name, source_file_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_bytes()
    return data

def gcp_csv_to_df(bucket_name, source_file_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data),index_col=None)
    print(f'Pulled down file from bucket {bucket_name}, file name: {source_file_name}')
    return df
@st.cache(allow_output_mutation=True)
def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a model from the bucket."""
    # Initialize a storage client
    storage_client = get_storage_client()
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Construct a blob
    blob = bucket.blob(source_blob_name)
    # Download the blob to a temporary local file
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")
    # Load the model
    model = load_model(destination_file_name)
    return model


def upload_cs_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = get_storage_client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)
    return True
    
def upload_json_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = get_storage_client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name,content_type="application/json")
    return True
def upload_xl_file(bucket_name, uploaded_file, destination_blob_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    uploaded_file.seek(0)

    # Upload the file from the file object provided by st.file_uploader
    blob.upload_from_file(uploaded_file)

def list_cs_files(bucket_name): 
    storage_client = get_storage_client()

    file_list = storage_client.list_blobs(bucket_name)
    file_list = [file.name for file in file_list]

    return file_list

def list_cs_files_f(bucket_name, folder_name):
    storage_client = get_storage_client()

    # List all blobs in the bucket
    blobs = storage_client.list_blobs(bucket_name)

    # Filter blobs that are within the specified folder
    folder_files = [blob.name for blob in blobs if blob.name.startswith(folder_name)]

    return folder_files

def list_files_in_folder(bucket_name, folder_name):
    storage_client = get_storage_client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)

    # Extract only the filenames without the folder path
    filenames = [blob.name.split("/")[-1] for blob in blobs if "/" in blob.name]

    return filenames

def list_files_in_subfolder(bucket_name, folder_name):
    storage_client = get_storage_client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name, delimiter='/')

    # Extract only the filenames without the folder path
    filenames = [blob.name.split('/')[-1] for blob in blobs]

    return filenames
def store_release_order_data(data,release_order_number,destination,po_number,sales_order_item,vessel,batch,ocean_bill_of_lading,grade,dryness,carrier_code,unitized,total):
       
    # Create a dictionary to store the release order data
    data[release_order_number]={
        
        
        'destination':destination,
        "po_number":po_number,
        "complete":False,
        sales_order_item: {
        "vessel":vessel,
        "batch": batch,
        "ocean_bill_of_lading": ocean_bill_of_lading,
        "grade": grade,
        "dryness":dryness,
        "carrier_code": carrier_code,
        "unitized":unitized,
        "total":total,
         "shipped":0,
        "remaining":total       
        }}              
    
    
                         

    # Convert the dictionary to JSON format
    json_data = json.dumps(data)
    return json_data
#release_order_database,release_order_number,sales_order_item_add,vessel_add,batch_add,ocean_bill_of_lading_add,grade_add,dryness_add,carrier_code_add,unitized_add,quantity_add,shipped_add,remaining_add
def add_release_order_data(data,release_order_number,sales_order_item,vessel,batch,ocean_bill_of_lading,grade,dryness,carrier_code,unitized,total,shipped,remaining):
       
    
    if sales_order_item not in data[release_order_number]:
        data[release_order_number][sales_order_item]={}
    data[release_order_number][sales_order_item]["vessel"]= vessel
    data[release_order_number][sales_order_item]["batch"]= batch
    data[release_order_number][sales_order_item]["ocean_bill_of_lading"]= ocean_bill_of_lading
    data[release_order_number][sales_order_item]["grade"]= grade
    data[release_order_number][sales_order_item]["dryness"]= dryness
    data[release_order_number][sales_order_item]["carrier_code"]= carrier_code
    data[release_order_number][sales_order_item]["unitized"]= unitized
    data[release_order_number][sales_order_item]["total"]= total
    data[release_order_number][sales_order_item]["shipped"]= 0
    data[release_order_number][sales_order_item]["remaining"]= total
    
    
       

    # Convert the dictionary to JSON format
    json_data = json.dumps(data)
    return json_data

def edit_release_order_data(data,release_order_number,destination,po_number,sales_order_item,vessel,batch,ocean_bill_of_lading,grade,dryness,carrier_code,unitized,total,shipped,remaining):
       
    # Edit the loaded current dictionary.
    data[release_order_number][sales_order_item]["destination"]= destination
    data[release_order_number][sales_order_item]["po_number"]=po_number
    data[release_order_number][sales_order_item]["vessel"]= vessel
    data[release_order_number][sales_order_item]["batch"]= batch
    data[release_order_number][sales_order_item]["ocean_bill_of_lading"]= ocean_bill_of_lading
    data[release_order_number][sales_order_item]["grade"]= grade
    data[release_order_number][sales_order_item]["dryness"]= dryness
    data[release_order_number][sales_order_item]["carrier_code"]= carrier_code
    data[release_order_number][sales_order_item]["unitized"]= unitized
    data[release_order_number][sales_order_item]["total"]= total
    data[release_order_number][sales_order_item]["shipped"]= shipped
    data[release_order_number][sales_order_item]["remaining"]= remaining
    
    
       

    # Convert the dictionary to JSON format
    json_data = json.dumps(data)
    return json_data

def process():
           
    line1="1HDR:"+a+b+terminal_code
    tsn="01" if medium=="TRUCK" else "02"
    
    tt="0001" if medium=="TRUCK" else "0002"
    if double_load:
        line21="2DTD:"+current_release_order+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(first_quantity*2000)+" "*(16-len(str(first_quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
        line22="2DTD:"+next_release_order+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(second_quantity*2000)+" "*(16-len(str(second_quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
    if mf_mix:
        line2="2DTD:"+release_order_number+" "*(10-len(release_order_number))+"000"+sales_order_item+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(int(quantity*2000))+" "*(16-len(str(int(quantity*2000))))+"USD"+" "*16+mf_number_split+" "*(20-len(mf_number_split))+"1"+" "*9+"|"+otm_number+" "*(49-len(otm_number))+c
    else:
        line2="2DTD:"+release_order_number+" "*(10-len(release_order_number))+"000"+sales_order_item+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(int(quantity*2000))+" "*(16-len(str(int(quantity*2000))))+"USD"+" "*16+" "*20+"1"+" "*9+"|"+otm_number+" "*(49-len(otm_number))+c
    
                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                             
               
    loadls=[]
    bale_loadls=[]
    if double_load:
        for i in first_textsplit:
            loadls.append("2DEV:"+current_release_order+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+i[:load_digit]+" "*(10-len(i[:load_digit]))+"0"*16+str(2000))
        for k in second_textsplit:
            loadls.append("2DEV:"+next_release_order+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+k[:load_digit]+" "*(10-len(k[:load_digit]))+"0"*16+str(2000))
    else:
        for k in loads:
            loadls.append("2DEV:"+release_order_number+" "*(10-len(release_order_number))+"000"+sales_order_item+a+tsn+k+" "*(10-len(k))+"0"*(20-len(str(int(loads[k]*2000))))+str(int(loads[k]*2000)))
        
        
    if double_load:
        number_of_lines=len(first_textsplit)+len(second_textsplit)+4
    else:
        number_of_lines=len(loads)+3
    end_initial="0"*(4-len(str(number_of_lines)))
    end=f"9TRL:{end_initial}{number_of_lines}"
     
    with open(f'placeholder.txt', 'w') as f:
        f.write(line1)
        f.write('\n')
        if double_load:
            f.write(line21)
            f.write('\n')
            f.write(line22)
        else:
            f.write(line2)
        f.write('\n')
        
        for i in loadls:
            
            f.write(i)
            f.write('\n')
       
        f.write(end)
def gen_bill_of_lading():
    data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
    bill_of_ladings=json.loads(data)
    list_of_ladings=[]
    try:
        for key in [k for k in bill_of_ladings if len(k)==8]:
            if int(key) % 2 == 0:
                list_of_ladings.append(int(key))
        bill_of_lading_number=max(list_of_ladings)+2
    except:
        bill_of_lading_number=11502400
    return bill_of_lading_number,bill_of_ladings

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
def get_weather():
    weather=defaultdict(int)
    headers = { 
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
            'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
            'Accept-Language' : 'en-US,en;q=0.5', 
            'Accept-Encoding' : 'gzip', 
            'DNT' : '1', # Do Not Track Request Header 
            'Connection' : 'close' }

    url='http://api.weatherapi.com/v1/forecast.json?key=5fa3f1f7859a415b9e6145743230912&q=98501&days=7'
    #response = get(url,headers=headers)
    response=get(url,headers=headers)
    #data = json.loads(response.text)
    data=json.loads(response.text)
    print(data)
    return data
def vectorize(direction,speed):
    Wind_Direction=direction
    Wind_Speed=speed
    wgu = 0.1*Wind_Speed * np.cos((270-Wind_Direction)*np.pi/180)
    wgv= 0.1*Wind_Speed*np.sin((270-Wind_Direction)*np.pi/180)
    return(wgu,wgv)
                
def parse_angle(angle_str):
    angle= mpcalc.parse_angle(angle_str)
    angle=re.findall(f'\d*\.?\d?',angle.__str__())[0]
    return float(angle)

def prep_ledger(dosya,yr1,month1,yr2,month2):
    
    df=pd.read_csv(dosya,header=None)
        
    checkdate=datetime.datetime.strptime(df.loc[1,14].split(" ")[-1],"%m/%d/%Y")

    a=df.iloc[:,41:45]
    b=df.iloc[:,49:59]

    df=pd.concat([a,b],axis=1)
    df.drop(columns=[43,54,57],inplace=True)

    columns=["Account","Name","Sub_Cat","Bat_No","Per_Entry","Ref_No","Date","Description","Debit","Credit","Job_No"]
    df.columns=columns
    df.dropna(subset="Date",inplace=True)

    temp=[]
    for i in df.Credit:
        try:
            temp.append(int(i.split(",")[0])*1000+float(i.split(",")[1]))
            #print(int(i.split(",")[0])*1000+float(i.split(",")[1]))
        except:
            temp.append(float(i))
    df.Credit=temp
    
    temp=[]
    for i in df.Debit:
        try:
            temp.append(int(i.split(",")[0])*1000+float(i.split(",")[1]))
            #print(int(i.split(",")[0])*1000+float(i.split(",")[1]))
        except:
            temp.append(float(i))
    df.Debit=temp
    
    df["Date"]=pd.to_datetime(df["Date"])
    df=df[(df["Date"]>= pd.Timestamp(datetime.date(yr1,month1,1)))&
         (df["Date"]< pd.Timestamp(datetime.date(yr2,month2,1)))]   #########################################DATE!@!@
    df["Net"]=df["Credit"]-df['Debit']
    
    df,weyco_df=apply_corrections(df)
    
    return df,weyco_df


def apply_corrections(df):
    df["Acc"]=[f"{i}-{j}" for i,j in zip(df["Account"],df["Sub_Cat"])]
    
    #### WEYCO CREDIT
    df.loc[df["Acc"]=="6313002-32","Account"]=6341000
    df.loc[df["Acc"]=="6313002-32","Name"]="Real Prop Rent - Land"
    df.loc[df["Acc"]=="6313002-32","Acc"]="6341000-32"
    #df=df[~df["Acc"]=="6313002-32"]
    
    
    ### SUZANO CORRECTIONS
    df.loc[df['Ref_No'].isin(["069998","070097","070138","070108","070268","070293","070499","070485",
                       "070592","070634","070713","070929","070980","071150"
                       ]),"Job_No"]="23SUZANO / 14"

    df.loc[df['Ref_No'].isin(["071462","071711","071746","071921","071922","071923"
                       ]),"Job_No"]="23SUZANO / 20"

    df.loc[df['Ref_No'].isin(["071931","072270","072279"
                       ]),"Job_No"]="24SUZANO / 03"
    df.loc[df['Ref_No'].isin(["072501","072505"
                       ]),"Job_No"]="24SUZANO / 08"
    df.loc[df['Ref_No']=="070485","Job_No"]="NP"

    df.loc[(df["Description"]=="Rev Accr. SSA Pacific 12.23")&(df["Name"]=="Equipment Rentals"),"Net"]=0
    df.loc[(df["Description"]=="Rev Accr. SSA Pacific 12.23")&(df["Name"]=="Equipment Rentals"),"Debit"]=0

    #### LOADOUTS TO HANDLING

    df.loc[df['Ref_No'].isin(["058498","058283","058710","058923","058924","059168","059386","059624",
                                      "059625","059881","060097","060298","060096"
                       ]),"Name"]="Handling"
    df.loc[df['Ref_No'].isin(["058498","058283","058710","058923","058924","059168","059386","059624",
                                      "059625","059881","060097","060298","060096"
                       ]),"Account"]=6316000

    #### UNITED FROM 03 to 05

    df.loc[df['Ref_No'].isin(["059426"
                       ]),"Job_No"]="24UNITED / 05"

    #### SAGA ENVIRONMENTAL TO SF

    df.loc[(df['Ref_No']=="060106")&(df['Description']=="Environmental Fee"),"Name"]="SWTF Facility Charge"
    df.loc[(df['Ref_No']=="060106")&(df['Description']=="Environmental Fee"),"Account"]=6318540
    
    #### MAKE FOREMAN OPERATOR REVENUE TO LOADING UNLOADING REVENUE
    
    df.loc[df["Acc"]=="6315100-32","Account"]=6315000
    df.loc[df["Acc"]=="6315100-32","Name"]="Loading & Unloading"
    df.loc[df["Acc"]=="6315100-32","Acc"]="6315000-32"
    df.loc[df["Acc"]=="6315200-32","Account"]=6315000
    df.loc[df["Acc"]=="6315200-32","Name"]="Loading & Unloading"
    df.loc[df["Acc"]=="6315200-32","Acc"]="6315000-32"

    weyco_df=df.copy()
    weyco_df.loc[((weyco_df["Account"].isin(ship_accounts))&(df["Description"].str.contains("Vessel Bark Clean Up"))),"Job_No"]="WEYCO"
    weyco_df.loc[weyco_df['Account'].isin([6315000,6316000,6317030,7313015,7311015]),"Name"]="Loading & Unloading"
    weyco_df.loc[weyco_df['Account'].isin([6315000,6316000,6317030,7313015,7311015]),"Account"]=6315000
    
    weyco_df.loc[(weyco_df["Account"].isin(ship_accounts))&(weyco_df["Job_No"].str.contains("WEYCO")),"Account"]=6999999
    weyco_df.loc[(weyco_df["Account"].isin(ship_accounts))&(weyco_df["Job_No"].str.contains("WEYCO")),"Name"]="Tenant Ship Income"
    
    weyco_df.loc[(weyco_df["Account"].isin(ship_accounts))&(weyco_df["Job_No"].str.contains("SUZANO")),"Account"]=6888888
    weyco_df.loc[(weyco_df["Name"]=="Handling")&(weyco_df["Job_No"].str.contains("SUZANO")),"Account"]=6888889
    
    weyco_df.loc[(weyco_df["Account"].isin(ship_accounts))&(weyco_df["Job_No"].str.contains("SUZANO")),"Name"]="Suzano Vessels"
    weyco_df.loc[(weyco_df["Name"]=="Handling")&(weyco_df["Job_No"].str.contains("SUZANO")),"Name"]="Suzano Warehouse"
    
    
    df["Acc"]=[f"{i}-{j}" for i,j in zip(df["Account"],df["Sub_Cat"])]
    weyco_df["Acc"]=[f"{i}-{j}" for i,j in zip(weyco_df["Account"],weyco_df["Sub_Cat"])]
    
    
    ####  LABOR NORMALIZING
#     df.loc[df['Account'].isin([7311015,6315000,6316000,6317030,7313015,7311015]),"Name"]="Loading & Unloading"
#     df.loc[df['Account'].isin([7311015,6315000,6316000,6317030,7313015,7311015]),"Account"]=6315000
#     df["Acc"]=[f"{i}-{j}" for i,j in zip(df["Account"],df["Sub_Cat"])]
    
    return df,weyco_df




def find_account_path(budget_dict, account_number):
    def recurse(d, path):
        for key, value in d.items():
            new_path = path + [key]  # Extend the path with the current key
            if isinstance(value, dict):
                # If the value is another dictionary, recurse into it
                result = recurse(value, new_path)
                if result:
                    return result
            elif key == account_number:
                # If the key is the account number we're looking for, return the path
                return new_path
        return None

    # Start the recursive search from the top of the dictionary
    return recurse(budget_dict, [])

# def populate_main(main,df):
    
#     for i in df["Account"].unique():
#         cost_center=df[df["Account"]==i]["Sub_Cat"].values[0]
#         #print(df[df["Account"]==i]["Name"].unique()[0])
#         acc=f"{i}-{cost_center}"
#         main[f"{i}-{cost_center}"]={"Name":None,"Cat":None,"Sub_Cat":None,"Net":0}
#         main[f"{i}-{cost_center}"]["Name"]=df[df["Account"]==i]["Name"].unique()[0]
#         try:
#             main[f"{i}-{cost_center}"]["Sub_Cat"]=find_account_path(Budget_Dict,acc)[1]
#         except:
#             main[f"{i}-{cost_center}"]["Sub_Cat"]=None
#         try:
#             main[f"{i}-{cost_center}"]["Cat"]=find_account_path(Budget_Dict,acc)[0]
#         except:
#             main[f"{i}-{cost_center}"]["Cat"]=None
#         main[f"{i}-{cost_center}"]["Net"]=round(df[df["Account"]==i]["Net"].sum(),2)
#     return main


def populate_main(main,df):
    
    for i in df["Acc"].unique():
        #cost_center=df[df["Account"]==i]["Sub_Cat"].values[0]
        #print(df[df["Account"]==i]["Name"].unique()[0])
        #acc=f"{i}-{cost_center}"
        main[i]={"Name":None,"Cat":None,"Sub_Cat":None,"Net":0}
        main[i]["Name"]=df[df["Acc"]==i]["Name"].unique()[0]
        try:
            main[i]["Sub_Cat"]=find_account_path(Budget_Dict,i)[1]
        except:
            main[i]["Sub_Cat"]=None
        try:
            main[i]["Cat"]=find_account_path(Budget_Dict,i)[0]
        except:
            main[i]["Cat"]=None
        main[i]["Net"]=round(df[df["Acc"]==i]["Net"].sum(),2)
    return main

def add_stat_rows(df):
    df=df.copy()
    df.loc[("_","Total"),:]=df.sum(axis=0)
    df=df.copy()
    df.loc[:,"Mean"]=round(df.mean(axis=1),1)
    return df

def dollar_format(x):
    if isinstance(x, (int, float)):  
        return f"${x:,.2f}"  
    return x


def get_gov_weather():
    weather=defaultdict(int)
    headers = { 
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
            'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
            'Accept-Language' : 'en-US,en;q=0.5', 
            'Accept-Encoding' : 'gzip', 
            'DNT' : '1', # Do Not Track Request Header 
            'Connection' : 'close' }
    url ='https://api.weather.gov/gridpoints/SEW/117,51/forecast/hourly'
    #url='https://api.weather.gov/points/47.0379,-122.9007'   #### check for station info with lat/long
    durl='https://api.weather.gov/alerts?zone=WAC033'
    response = get(url,headers=headers)
    desponse=get(durl)
    data = json.loads(response.text)
    datan=json.loads(desponse.text)
    #print(data)

    for period in data['properties']['periods']:
        #print(period)
        date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00')
        date_f=datetime.datetime.strftime(datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00'),"%b-%d,%a %H:%M")
        weather[date_f]={'Wind_Direction':f'{period["windDirection"]}','Wind_Speed':f'{period["windSpeed"]}',
                      'Temperature':f'{period["temperature"]}','Sky':f'{period["shortForecast"]}',
                       'Rain_Chance':f'{period["probabilityOfPrecipitation"]["value"]}'
                      }
        

    forecast=pd.DataFrame.from_dict(weather,orient='index')
    forecast.Wind_Speed=[int(re.findall(f'\d+',i)[0]) for i in forecast.Wind_Speed.values]
    #forecast['Vector']=[vectorize(parse_angle(i),j) for i,j in zip(forecast.Wind_Direction.values,forecast.Wind_Speed.values)]
    return forecast
def style_row(row):
    location = row["Location"]
    shipment_status = row["Status"]
    
    # Define colors for different locations
    colors = {
        "CLATSKANIE": "background-color: #d1e7dd;",  # light green
        "LEWISTON": "background-color: #ffebcd;",    # light coral
        "HALSEY": "background-color: #add8e6;",      # light blue
    }
    
    # Base style for the entire row based on location
    base_style = colors.get(location, "")
    
    # Apply styles based on shipment status
    if shipment_status == "SHIPPED":
        base_style += "font-weight: lighter; font-style: italic; text-decoration: line-through;"  # Less bold, italic, and strikethrough
    else:
        base_style += "font-weight: bold;"  # Slightly bolder for other statuses
    
    # Apply the style to all cells in the row
    return [base_style] * len(row)


def apply_grouping_mode(df, mode_col='original'):
    """
    Flattens Group/Subgroup from the specified mode column and adds them to the DataFrame.
    Drops the original nested column after expansion.
    
    Parameters:
    - df: input DataFrame
    - mode_col: name of the column containing nested {'Group': ..., 'Subgroup': ...} dict
    """
    df = df.copy()
    df['Group'] = df[mode_col].apply(lambda x: x.get('Group') if isinstance(x, dict) else None)
    df['Subgroup'] = df[mode_col].apply(lambda x: x.get('Subgroup') if isinstance(x, dict) else None)
    df = df.drop(columns=[mode_col])
    return df
@st.cache_data(show_spinner="Loading Ledger Data...")
def load_main_json(target_bucket, filename="main.json"):
    raw_json = gcp_download(target_bucket, filename)

    try:
        main_json = json.loads(raw_json)
        if isinstance(main_json, str):
            main_json = json.loads(main_json)
    except Exception as e:
        st.error(f"Failed to parse main.json: {e}")
        st.stop()

    return main_json

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'


       
    
with open('configure.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(fields={'PORT OF OLYMPIA TOS LOGIN', 'main'})


if authentication_status:
    authenticator.logout('Logout', 'main')
    if username == 'ayilmaz' or username=='gatehouse':
        st.subheader("PORT OF OLYMPIA TOS")
        st.write(f'Welcome *{name}*')
        
        #forecast=get_weather()
        
        
        select=st.sidebar.radio("SELECT FUNCTION",
            ('ADMIN', 'LOADOUT', 'INVENTORY','LABOR','FINANCE',"GATE CONV.NETWORK","STORAGE"))
        custom_style = """
                    <style>
                        .custom-container {
                            background-color: #f0f0f0;  /* Set your desired background color */
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        }
                    </style>
                """
        st.markdown(custom_style, unsafe_allow_html=True)
        local_model_path = 'temp_model.keras'
        #model = download_model(target_bucket, 'mygatemodel2.keras', local_model_path)
        # if select=="GATE CONV.NETWORK":
        #     def custom_preprocessing_function(img):
        #         # Crop the image first to ensure the gate is included
        #         #cropped_img = img[:,round(img.shape[1]*0.4):]
            
        #         # Now apply any further transformations you want
        #         # For example, manually apply rescaling and a mild zoom if necessary
        #         # Note: You might need additional libraries or write more complex transformations manually
            
        #         # Resize the cropped image to the desired input size of the model
        #         resized_img = cv2.resize(img, (150, 150))
        #         return resized_img
        #     #local_model_path = 'temp_model.keras'
        #     index_to_class={0: 'both_closed',
        #                      1: 'both_open',
        #                      2: 'inbound_closed_outbound_open',
        #                      3: 'inbound_open_outbound_closed'}
        #     index_to_class={0: {'inbound':0,'outbound':0},
        #                  1:  {'inbound':1,'outbound':1},
        #                  2:  {'inbound':0,'outbound':1},
        #                  3:  {'inbound':1,'outbound':0}}
        #     #model = download_model(target_bucket, 'mygatemodel2.keras', local_model_path)
        #     st.title('SOUTH GATE OPEN/CLOSE DETECTION')

        #     # Assuming you have a function `prepare_image` to process images
        #     uploaded_file = st.file_uploader("Upload an image", type="jpg")
        #     if uploaded_file is not None:
        #         img = image.load_img(uploaded_file, color_mode='rgb')
        #         img_array = image.img_to_array(img)
            
        #         # Apply custom preprocessing function
        #         img_array = custom_preprocessing_function(img_array)
            
        #         # Expand dimensions to match the batch shape and rescale pixel values
        #         test_image = np.expand_dims(img_array, axis=0) / 255.0
            
        #         # Predict using your model
        #         prediction = model.predict(test_image)
        #         predicted_class = np.argmax(prediction, axis=1)[0]
            
        #         # Print debug information
        #         st.write(f"Predicted Class: {predicted_class}")
        #         st.write(f"Prediction Probabilities: {prediction}")
        #         st.write(f"Gate Status: {index_to_class[predicted_class]}")
            
        #         gate1, gate2 = st.columns([5, 5])
            
        #         def gate_status_html(gate_name, status):
        #             background_color = "#32CD32" if status else "#FF6347"
        #             text = "OPEN" if status else "CLOSED"
        #             html_str = f"""
        #             <div style='background-color: {background_color}; padding: 10px; border-radius: 8px;'>
        #                 <h4 style='color: white; text-align: center; font-weight: bold;'>{gate_name}</h4>
        #                 <p style='color: white; text-align: center; font-size: 24px; font-weight: bold;'>{text}</p>
        #             </div>
        #             """
        #             return html_str
            
        #         # Display gate status in Streamlit
        #         with gate1:
        #             st.subheader("INBOUND")
        #             inbound_status = index_to_class[predicted_class]['inbound']
        #             inbound_html = gate_status_html("INBOUND", inbound_status)
        #             st.markdown(inbound_html, unsafe_allow_html=True)
            
        #         with gate2:
        #             st.subheader("OUTBOUND")
        #             outbound_status = index_to_class[predicted_class]['outbound']
        #             outbound_html = gate_status_html("OUTBOUND", outbound_status)
        #             st.markdown(outbound_html, unsafe_allow_html=True)
        
        if select=="FINANCE":
            hadi=True
            fin_password=st.sidebar.text_input("Enter Password",type="password",key="sas")
            if fin_password=="marineterm98501!":
                hadi=True
            if hadi:
                ttab1,ttab2,ttab3=st.tabs(["MT LEDGERS","UPLOAD CSV LEDGER UPDATES","BUDGET PERFORMANCE"])
                main_json = load_main_json(target_bucket)

                with ttab3:
                    pass
                    
                        
                    # upto_month=st.selectbox("Choose End Month",range(2,13))
                    
                    # ledger_b=gcp_download_x(target_bucket,rf"FIN/NEW/ledger_b.ftr")
                    # ledger=gcp_download_x(target_bucket,rf"FIN/NEW/ledger-2024.ftr")
                    # weyco_ledger=gcp_download_x(target_bucket,rf"FIN/NEW/weyco_ledger-2024.ftr")
                    
                    # budget=json.loads(gcp_download(target_bucket,rf"FIN/NEW/budget.json"))
                    # budget1=json.loads(gcp_download(target_bucket,rf"FIN/NEW/budget1.json"))
                    # budget_2024=json.loads(gcp_download(target_bucket,rf"FIN/NEW/budget_2023.json"))
                    # pure_budget=json.loads(gcp_download(target_bucket,rf"FIN/NEW/pure_budget.json"))
                    # weyco_suzano_budget=json.loads(gcp_download(target_bucket,rf"FIN/NEW/weyco_suzano_budget.json"))
                    
                    # ledger_b = pd.read_feather(io.BytesIO(ledger_b))
                    # ledger_b = ledger_b.set_index("index", drop=True).reset_index(drop=True)
                    # ledger = pd.read_feather(io.BytesIO(ledger))
                    # ledger = ledger.set_index("index", drop=True).reset_index(drop=True)
                    # weyco_ledger = pd.read_feather(io.BytesIO(weyco_ledger))
                    # weyco_ledger = weyco_ledger.set_index("index", drop=True).reset_index(drop=True)
                    
                    # ledger_b=ledger_b[ledger_b["Date"]<pd.Timestamp(datetime.date(2024,upto_month,1))]
                    # ledger=ledger[ledger["Date"]<pd.Timestamp(datetime.date(2024,upto_month,1))]
                    # weyco_ledger=weyco_ledger[weyco_ledger["Date"]<pd.Timestamp(datetime.date(2024,upto_month,1))]


                    # ledger_b.reset_index(drop=True,inplace=True)
                    # ledger_b=ledger_b.copy()
                    # as_of=f"End of {calendar.month_name[upto_month-1]} 2024"
                                        
                    
                    
                    # year="2024"
                    
                    # if year=="2024":
                    #     budget_year=budget_2024.copy()
                    #     month_count=9
                    # else:
                    #     budget_year=budget_2023.copy()
                    #     month_count=12
                    
                                                      
                    
                    # def get_key(d,item):
                        
                    #     for k, v in d.items():
                    #         if isinstance(v, dict):
                    #             for a, b in v.items():
                    #                 if isinstance(b, dict):
                    #                     for c, d in b.items():
                    #                         if c==item:
                    #                             return k,a
                    #                 else:
                    #                     if a==item:
                    #                         return k,b
                    #         else:
                    #             if v==item:
                    #                 return k,v
                    # for i in ledger_b.index:
                    #     try:
                    #         ledger_b.loc[i,"Group"]=get_key(budget1,ledger_b.loc[i,"Acc"])[0]
                    #     except:
                    #         pass
                    #     try:
                    #         ledger_b.loc[i,"Sub_Group"]=get_key(budget1,ledger_b.loc[i,"Acc"])[1]
                    #     except:
                    #         pass
                    #    # ledger_b.loc[i,"Sub_Group"]=get_key(budget,ledger_b.loc[i,"Account"])
                    
                    
                    # #### LABOR ADJUSTMENT
                    # ledger_b.loc[ledger_b["Name"]=="Longshore Services","Group"]="Revenues"
                    # ledger_b.loc[ledger_b["Name"]=="Longshore Fringe Benefits","Group"]="Revenues"
                    
                    
                    # ignore=set(ledger_b[ledger_b["Group"]==0].Account.to_list())
                    # ledger_b=ledger_b[~ledger_b["Account"].isin(ignore)]
                    # ### LEDGER-B READY
                    # grouped_ledger=round(ledger_b.groupby(["Group","Sub_Group","Acc","Name"])[["Net"]].sum(),1)
                    # grouped_ledger["Monthly"]=round(grouped_ledger["Net"]/month_count,1)
                    # #grouped_ledger.to_excel(fr"C:\Users\AfsinY\Desktop\LEDGERS\grouped.xlsx", sheet_name='Sheet1', index=True)
                    # grouped_ledger.reset_index(inplace=True)
                    # #grouped_ledger
                    
                    # grouped_ledger[f"Budget {year}"]=0
                    # for i in grouped_ledger.index:
                    #     a=grouped_ledger.loc[i,"Group"]
                        
                    #     b=grouped_ledger.loc[i,"Sub_Group"]
                    #     c=grouped_ledger.loc[i,"Acc"]
                    #     try:
                    #         grouped_ledger.loc[i,f"Budget {year}"]=budget_year[a][b][c]
                    #     except:
                    #         grouped_ledger.loc[i,f"Budget {year}"]=budget_year[a][c]
                    # grouped_ledger[f"Budget {year} YTD"]=round(grouped_ledger[f"Budget {year}"]/12*month_count,1)
                    # grouped_ledger["Variance"]=round(-grouped_ledger[f"Budget {year} YTD"]+grouped_ledger["Net"])
                    # ### GROUPED LEDGER READY
                    
                    # grouped_ledger_todate=grouped_ledger.groupby(["Group","Sub_Group"])[['Net', 'Monthly', f'Budget {year}',
                    #        f'Budget {year} YTD', 'Variance']].sum()
                    # ### GROUPED LEDGER TODATE READY
                    
                    
                    # order = {'Revenues': 1, 'Operating Expenses': 2, 'Maintenance Expenses': 3,"G & A Overhead":4,"Depreciation":5}
                    # grouped_ledger_todate['Sort_Key'] = grouped_ledger_todate.index.get_level_values('Group').map(order)
                    
                    # # Sort the DataFrame using the sort key
                    # grouped_ledger_todate.sort_values(by='Sort_Key',ascending=True, inplace=True)
                    
                    # # Drop the sort key column after sorting
                    # grouped_ledger_todate.drop(columns='Sort_Key', inplace=True)
                    # #grouped_ledger_todate.sort_index(level='Group', inplace=True)  # Sort by Group
                    
                    # ####   PREPARE NEW_DF
                    
                    # new_df = pd.DataFrame()
                    
                    # # Iterate through each group to calculate and insert totals
                    # for group, group_df in grouped_ledger_todate.groupby(level=0):
                    #     # Append the group DataFrame to the new DataFrame
                    #     new_df = pd.concat([new_df, group_df])
                        
                    #     # Calculate the total for the current group
                    #     total_row = group_df.sum()
                    #     total_row.name = (group, f'Total {group}', '')  # Setting up the name tuple to match the MultiIndex
                        
                    #     # Append the total row to the new DataFrame
                    #     new_df = pd.concat([new_df, pd.DataFrame([total_row], index=pd.MultiIndex.from_tuples([total_row.name]))])
                    
                    # # Display the new DataFrame with totals inserted
                    
                    # order = {'Revenues': 1, 'Operating Expenses': 2, 'Maintenance Expenses': 3,"G & A Overhead":4,"Depreciation":5}
                    # new_df['Sort_Key'] = new_df.index.get_level_values('Group').map(order)
                    
                    # # Sort the DataFrame using the sort key
                    # new_df.sort_values(by='Sort_Key',ascending=True, inplace=True)
                    
                    # # Drop the sort key column after sorting
                    # new_df.drop(columns='Sort_Key', inplace=True)
                    # #grouped_ledger_todate.sort_index(level='Group', inplace=True)  # Sort by Group
                    # total_rows = new_df.loc[[idx for idx in new_df.index if idx[1].startswith('Total')]]
                    
                    # # Calculate the overall total from these rows
                    # overall_total = total_rows.sum()
                    # overall_total.name = ('OVERALL', 'TOTAL', '')  # Naming the total row
                    
                    # # Append the overall total row to the new DataFrame
                    # new_df = pd.concat([new_df, pd.DataFrame([overall_total], index=pd.MultiIndex.from_tuples([overall_total.name]))])
                    # new_df=new_df[["Net","Budget 2024 YTD","Variance","Budget 2024"]]
                    # budget1,budget2,budget3,budget4=st.tabs(["TABULAR","VERSUS BUDGETED","SUNBURST CHART","MONTHLY PIVOT TABLE"])
                    # with budget1:
                    #     st.write(new_df)
                    # with budget2:
                    
                    #     new_df.reset_index(inplace=True)
                    #     fig, ax = plt.subplots(figsize=(15, 10))
                        
                    #     # Plotting the data first to define axes limits
                    #     bars_actual = ax.barh(new_df.Sub_Group, new_df['Net'], color='blue',alpha=0.3 ,label='Actual', zorder=3)
                    #     bars_budget = ax.barh(new_df.Sub_Group, new_df[f'Budget {year} YTD'], color='red', alpha=0.2, label='Budgeted ', zorder=3)
                        
                    #     # Load and show background image after plotting to get the correct extent
                    #     #bg_image = plt.imread('salish.png')  # Adjust this path if necessary
                    #    # ax.imshow(bg_image, aspect='auto', extent=[ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]], zorder=1, alpha=0.2)
                        
                    #     # Formatting the x-axis with commas and a dollar sign
                    #     ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        
                    #     # Title and other settings
                    #     current_date = datetime.datetime.now().date()
                    #     ax.set_title(f'Marine Terminal\nBUDGET {year} PERFORMANCE\nAs of {as_of}', fontsize=18)
                        
                    #     ax.set_xlabel('Amount ($)')
                    #     ax.legend()
                        
                    #     # Adding light shade grids and formatting labels
                    #     ax.grid(True, color='grey', alpha=0.3)
                    #     labels = ax.get_yticklabels()
                    #     for label in labels:
                    #         if "Total" in label.get_text() or "TOTAL" in label.get_text():
                    #             label.set_fontsize(12)
                    #             label.set_fontweight('bold')
                                
                    #     ax.tick_params(axis='y', labelsize=12)
                    #     ax.text(
                    #         0.95, 0.95, f'Overall Budgeted YTD NET - ${round(overall_total["Budget 2024 YTD"],1)}', 
                    #         transform=ax.transAxes, 
                    #         fontsize=14, 
                    #         verticalalignment='top', 
                    #         horizontalalignment='right', 
                    #         bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
                    #     )
                    #     ax.text(
                    #         0.95, 0.85, f'Actual YTD NET- ${round(overall_total["Net"],1)}', 
                    #         transform=ax.transAxes, 
                    #         fontsize=14, 
                    #         verticalalignment='top', 
                    #         horizontalalignment='right', 
                    #         bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
                    #     )
                    #     plt.style.use("fivethirtyeight")
                        
                    #     plt.rcParams['font.size'] = 5
                    #     plt.rcParams['grid.color'] = "black"
                    #     fig.patch.set_facecolor('lightblue')
                    #             #ax.patch.set_facecolor('yellow')
                    #     plt.tight_layout()
                    #     st.pyplot(fig)
                    
                    # with budget3:
                    #     weyco= st.checkbox("WEYCO SUZANO NORMALIZED")
                    #     if weyco:
                    #         Budget_Dict=weyco_suzano_budget.copy()
                    #         df=weyco_ledger.copy()
                    #     else:
                    #         Budget_Dict=budget.copy()
                    #         df=ledger.copy()
                        
                    #     ledger_p=pd.DataFrame()
                    #     main={}
                    #     dep=df[(df['Account']>=1712000)&(df["Account"]<=1865000)]
                    #     capital=df[(df['Account']<1712000)]
                    #     pma=df[(df['Account']>1865000)&(df["Account"]<=2131001)]
                    #     deposits=df[(df['Account']>2131001)&(df["Account"]<=2391030)]
                    #     df=df[((df['Account']>=1712000)&(df['Account']<=1865000))|(df["Account"]>2391040)]
                    #     # main=populate_main(main,main_30,accso)
                    #     # ledger=pd.concat([ledger,main_30])
                    #     #df=df[df["Account"]>2391030]
                    #     # df=prep_ledger(accso,yr)
                    #     df=df[~df["Account"].isin([7370000,7370010,7470000])]
                    #     labor=df[df['Account'].isin([7311015,6315000,6317030,7313015])]
                        
                    #     ledger_p=pd.concat([ledger_p,df])
                    #     combined_main=populate_main(main,ledger_p)
                        
                        
                        
                        
                    #     df=pd.DataFrame(combined_main).T
                    #     st.write(df)
                    #     df.drop(df[df["Net"]==0].index,inplace=True)
                    #     #df.loc[df["Cat"]=="Depreciation","Net"]=-df.loc[df["Cat"]=="Depreciation","Net"]
                    #     net_amount=df.Net.sum()
                    #     df["Net"]=abs(df["Net"])
                    #     #st.write(df)
                    #     fig = px.sunburst(
                    #         df, 
                    #         path=['Cat','Sub_Cat','Name'],  # Path is used to define the hierarchy of the chart
                    #         values='Net', 
                    #         color='Cat',  # Coloring based on the net value
                    #         color_continuous_scale='Blues',  # You can choose other color scales
                    #         #color_continuous_midpoint=np.average(df['Net'], weights=df['Net']),
                    #         title='Terminal Categories Sunburst Chart',
                    #         hover_data={'Net': ':,.2f'}  # Format 'Net' as a float with two decimal places
                    #     )
                    #     fig.update_traces(hovertemplate='<b>%{label}</b><br>Net: $%{customdata[0]:,.2f}')
                    #     fig.add_annotation(
                    #         x=0.95,  # x position, centered
                    #         y=0.9,  # y position, centered
                    #         text=f'Total Net: ${net_amount:,.2f}',  # text to display
                    #         showarrow=False,  # no arrow, just the text
                    #         font=dict(
                    #             family="Arial",
                    #             size=16,
                    #             color="black"
                    #         ),
                    #         align="center",
                    #         bgcolor="white",  # background color for better readability
                    #         opacity=0.8  # slightly transparent background
                    #     )
                        
                    #     fig.update_layout(
                    #         hoverlabel=dict(
                    #             bgcolor="white",  # Background color of the hover labels
                    #             font_size=16,     # Font size of the text in hover labels
                    #             font_family="Arial" ), # Font of the text in hover labels
                    #         margin=dict(t=40, l=10, r=10, b=10),  # Adjust margins to ensure the title and labels fit
                    #         width=800,  # Width of the figure in pixels
                    #         height=600,  # Height of the figure in pixels
                    #         #uniformtext=dict(minsize=15, mode='hide')
                    #     )

                    #     st.plotly_chart(fig)
                    # with budget4:
                    #     weycom= st.checkbox("WEYCO SUZANO NORMALIZED",key="fsa")
                    #     if weycom:
                    #         combined_ledger=weyco_ledger.copy()
                    #     else:
                    #         combined_ledger=ledger.copy()
                    #     combined_ledger["Month"]=combined_ledger["Date"].dt.month
                    #     combined_ledger=combined_ledger[combined_ledger["Month"]<upto_month]
                    #     combined_ledger["Net"] = combined_ledger["Net"].map('{:.2f}'.format)
                    #     combined_ledger["Net"]=combined_ledger["Net"].astype('float')
                        
                    #     ledger_pivot = combined_ledger.pivot_table(index=["Account", "Name"], columns="Month", values="Net", aggfunc="sum")
                    #     #ledger_pivot.to_excel("Ledger_Pivot.xlsx")
                    #     #ledger_pivot
                    #     temp=ledger_pivot.loc[:,:10]
                    #     temp.iloc[:20,:]=-temp.iloc[:20,:]
                    #     #temp = temp.append(temp.sum(axis=0).rename("Total"))
                        
                    #     ship_accounts=[6311000,6312000,6313000,6314000,6313950,6315000,6315100,6315200,6316000,6317030,6318540,6318600,
                    #                    6318900,6329000,6373000,6381000,6389000,7313015,7311015,7314300,7313949,7315000,7338700]
                    #     overhead=[7350080,7350082,7350083,7350085,7350087,7350088]
                        
                    #     weyco_non_ship_income=[6313001,6313002,6313003,6313955,6318101,6318301,6318501,6319040,6341000,6341010,6351000,6418500]
                    #     suzano_income=[6888888]
                    #     weyco_ship_income=[6999999]

                    #     non_vessel_expense = temp[(~temp.index.get_level_values("Account").isin(ship_accounts))&
                    #                               (temp.index.get_level_values("Account")>7000000)&
                    #                               (~temp.index.get_level_values("Account").isin(overhead))]
                                            
                        
                    #     vessel_expense = temp[temp.index.get_level_values("Account").isin(ship_accounts)&
                    #                               (temp.index.get_level_values("Account")>7000000)]
                        
                    #     depreciation=temp[temp.index.get_level_values("Account")<2000000]
                    #     overhead=temp[temp.index.get_level_values("Account").isin(overhead)]
                    #     vessel_ops=temp[temp.index.get_level_values("Account").isin(ship_accounts)]
                    #     weyco_static=temp[temp.index.get_level_values("Account").isin(weyco_non_ship_income)]
                    #     other_income=temp[(~temp.index.get_level_values("Account").isin(weyco_non_ship_income))&
                    #                       (~temp.index.get_level_values("Account").isin(ship_accounts))&
                    #                       (~temp.index.get_level_values("Account").isin(suzano_income))&
                    #                       (~temp.index.get_level_values("Account").isin(weyco_ship_income))&
                    #                       (temp.index.get_level_values("Account")<7000000)&
                    #                       (temp.index.get_level_values("Account")>2000000)]
                    #     suzano_income=temp[temp.index.get_level_values("Account").isin(suzano_income)]
                    #     weyco_ship_income=temp[temp.index.get_level_values("Account").isin(weyco_ship_income)]
                        
                        
                        
                    #     temp=add_stat_rows(temp)
                    #     non_vessel_expense = add_stat_rows(non_vessel_expense)
                    #     vessel_expense = add_stat_rows(vessel_expense)
                    #     depreciation=add_stat_rows(depreciation)
                    #     overhead=add_stat_rows(overhead)
                    #     vessel_ops=add_stat_rows(vessel_ops)
                    #     weyco_static=add_stat_rows(weyco_static)
                    #     other_income=add_stat_rows(other_income)
                    #     suzano_income=add_stat_rows(suzano_income)
                    #     weyco_ship_income=add_stat_rows(weyco_ship_income)

                    #     if weycom:
                    #         a=pd.DataFrame(columns=range(1,upto_month))
                    #         a.loc["Revenue - Other Vessel Ops"]=vessel_ops.iloc[-1,:-1]
                    #         a.loc["Revenue - Weyco Static"]=weyco_static.iloc[-1,:-1]
                    #         a.loc["Revenue - Weyco Ship Income"]=weyco_ship_income.iloc[-1,:-1]
                    #         a.loc["Revenue - Suzano Income"]=suzano_income.iloc[-1,:-1]
                    #         a.loc["Revenue - Other"]=other_income.iloc[-1,:-1]
                    #         a.loc["Expense - Running Cost"]=non_vessel_expense.iloc[-1,:-1]
                    #         a.loc["Expense - Overhead"]=overhead.iloc[-1,:-1]
                    #         a.loc["Expense - Depreciation"]=-depreciation.iloc[-1,:-1]
                    #     else:
                            
                    #         a=pd.DataFrame(columns=range(1,upto_month))
                    #         a.loc["Revenue - Vessel Ops"]=vessel_ops.iloc[-1,:-1]
                    #         a.loc["Revenue - Weyco Static"]=weyco_static.iloc[-1,:-1]
                    #         a.loc["Revenue - Other"]=other_income.iloc[-1,:-1]
                    #         a.loc["Expense - Running Cost"]=non_vessel_expense.iloc[-1,:-1]
                    #         a.loc["Expense - Overhead"]=overhead.iloc[-1,:-1]
                    #         a.loc["Expense - Depreciation"]=-depreciation.iloc[-1,:-1]
                        
                    #     a=a.copy()
                    #     a.loc[:,"Mean"]=round(a.mean(axis=1),1)
                    #     a=a.copy()
                    #     a.loc["Total",:]=a.sum(axis=0)
                    #     formatted_a = a.applymap(dollar_format)
                    #     formatted_a.columns=[calendar.month_abbr[int(i)] for i in formatted_a.columns[:-1]]+["Per Month"]
                    #     st.write(formatted_a)
                
                with ttab2:
                    
                    if st.checkbox("UPLOAD LEDGER CSV",key="fsdsw"):
                        led_col1,led_col2,led_col3,led_col4=st.columns([3,2,2,2])
                        with led_col1:
                            year=st.selectbox("SELECT YEAR TO UPLOAD",["2024","2023","2022"])
                            terminal = st.file_uploader("**Upload Ledgers for 030-032-036 csv**", type=["csv"],key="34wss")
                            stormwater= st.file_uploader("**Upload Ledger 040 csv**", type=["csv"],key="34ws2ss")
                            
                            if terminal and stormwater:
                                ship_accounts=[6311000,6312000,6313000,6314000,6313950,6315000,6315100,6315200,6316000,6317030,6318540,6318600,
                                       6318900,6329000,6373000,6381000,6389000,7313015,7311015,7314300,7313949,7315000,7338700]
                                
                                mnth=datetime.date.today().month
                                #t = pd.read_csv(terminal)
                                ter,weyco_ter=prep_ledger(terminal,2024,1,2024,mnth+1)
                                storm,weyco_storm=prep_ledger(stormwater,2024,1,2024,mnth+1)
                                ledger=pd.concat([ter,storm])
                                weyco_ledger=pd.concat([weyco_ter,weyco_storm])
                                # wey_ledger=prep_weyco_ledger(terminal,2024,1,2024,mnth+1)
                                # ledger=pd.concat([ledger,prep_weyco_ledger(stormwater,2024,1,2024,mnth+1)])
                                ledger.loc[ledger["Acc"].isin(["1712000-30","1721000-30","1721100-30","1722000-30","1731000-30","1758000-30","1765000-30",
                                   "1777000-30","1778000-30","1782000-30","1782001-30","1785000-30","1786001-30","1787000-30",
                                   "1796000-30","1861000-30","1865000-30","1738000-40","1757000-40","1761000-40","1796000-40",
                                   "1861000-40","7439010-40"]),"Net"]=-(ledger[ledger["Acc"].isin(["1712000-30","1721000-30","1721100-30","1722000-30","1731000-30","1758000-30","1765000-30",
                                   "1777000-30","1778000-30","1782000-30","1782001-30","1785000-30","1786001-30","1787000-30",
                                   "1796000-30","1861000-30","1865000-30","1738000-40","1757000-40","1761000-40","1796000-40",
                                   "1861000-40","7439010-40"])]["Net"])
                                weyco_ledger.loc[weyco_ledger["Acc"].isin(["1712000-30","1721000-30","1721100-30","1722000-30","1731000-30","1758000-30","1765000-30",
                                   "1777000-30","1778000-30","1782000-30","1782001-30","1785000-30","1786001-30","1787000-30",
                                   "1796000-30","1861000-30","1865000-30","1738000-40","1757000-40","1761000-40","1796000-40",
                                   "1861000-40","7439010-40"]),"Net"]=-(weyco_ledger[weyco_ledger["Acc"].isin(["1712000-30","1721000-30","1721100-30","1722000-30","1731000-30","1758000-30","1765000-30",
                                   "1777000-30","1778000-30","1782000-30","1782001-30","1785000-30","1786001-30","1787000-30",
                                   "1796000-30","1861000-30","1865000-30","1738000-40","1757000-40","1761000-40","1796000-40",
                                   "1861000-40","7439010-40"])]["Net"])
                                
                                ledger_detail=ledger[~ledger["Acc"].isin(["7370000-32","7370010-32","7470000-40"])]
                                ledger_detail=ledger_detail[(ledger_detail["Account"]>1690000)&(~ledger_detail["Account"].isin([2131001,2391030]))]
                                weyco_ledger=weyco_ledger[~weyco_ledger["Acc"].isin(["7370000-32","7370010-32","7470000-40"])]
                                weyco_ledger=weyco_ledger[(weyco_ledger["Account"]>1690000)&(~weyco_ledger["Account"].isin([2131001,2391030]))]
                                ledger_b=ledger[ledger["Account"]>6000000]
                                feather_data = BytesIO()
                                ledger_detail.reset_index().to_feather(feather_data)
                                # Create a temporary local file to store Feather data
                                temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
                                ledger_detail.reset_index().to_feather(temp_file_path)
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"FIN/NEW/ledger-{year}.ftr")
                                blob.upload_from_filename(temp_file_path)

                                feather_data = BytesIO()
                                weyco_ledger.reset_index().to_feather(feather_data)
                                # Create a temporary local file to store Feather data
                                temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
                                weyco_ledger.reset_index().to_feather(temp_file_path)
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"FIN/NEW/weyco_ledger-{year}.ftr")
                                blob.upload_from_filename(temp_file_path)
                                
                               
                                    
                                set=pd.read_feather(feather_data).set_index("index",drop=True).reset_index(drop=True)
                                       
                                with led_col3:
                                    st.write("Processing Depreciation/Overhead")
                                    set["Net"]=set["Credit"]-set["Debit"]
                                    depreciation=set[set.Account.astype(str).str.startswith("17")]#.resample("M",on="Date")[["Debit","Credit"]].sum()
                                    overhead=set[set.Account.astype(str).str.startswith("735")]#.resample("M",on="Date")[["Debit","Credit"]].sum()
                                    main30=set[set.Account.astype(str).str.startswith("731")]#.resample("M",on="Date")[["Debit","Credit"]].sum()
                                with led_col3:
                                    st.markdown(f"**Depreciation**")
                                    st.write("Total Credit :${:,}".format(round(depreciation.Credit.sum(),2)))
                                    st.write("Total Debit  :${:,}".format(round(depreciation.Debit.sum(),2)))
                                    st.write("Net          :${:,}".format(round(depreciation.Credit.sum()-depreciation.Debit.sum(),2)))
                                with led_col3:
                                    st.markdown(f"**Overhead**")
                                    st.write("Total Credit :${:,}".format(round(overhead.Credit.sum(),2)))
                                    st.write("Total Debit  :${:,}".format(round(overhead.Debit.sum(),2)))
                                    st.write("Net          :${:,}".format(round(overhead.Credit.sum()-overhead.Debit.sum(),2)))
                                with led_col3:
                                    st.markdown(f"**Main Ledger 030**")
                                    st.write("Total Credit :${:,}".format(round(main30.Credit.sum(),2)))
                                    st.write("Total Debit  :${:,}".format(round(main30.Debit.sum(),2)))
                                    st.write("Net          :${:,}".format(round(main30.Credit.sum()-main30.Debit.sum(),2)))
                                st.success(f"**SUCCESS. {year} Ledger has been updated!", icon="✅") 
                         
                    
                with ttab1:
                    
                    class Account:
                        def __init__(self, code, description,parent,root):
                            self.code = code
                            self.description = description
                            self.parent=parent
                            self.root=root
                    def find_main_root(dictionary, target_key, path=[]):
                        """
                        Recursively search the nested dictionary for the given target key and return the main root key and the path to the target key.
                        """
                        for key, value in dictionary.items():
                            new_path = path + [key]
                            if isinstance(value, dict):
                                subresult, subpath = find_main_root(value, target_key, new_path)
                                if subresult:
                                    return subresult, subpath
                            elif key == target_key:
                                return dictionary, new_path
                    
                        return None, None  # target key not found
                    def get_all_keys_(d,keys):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                get_all_keys(v,keys)
                            else:
                                keys.append(k)
                        return keys
        
                    def get_all_keys(d,keys):
                        
                        for k, v in d.items():
                            if isinstance(v, dict):
                                get_all_keys(v,keys)
                            else:
                                keys[k]=v
                        return keys
                    def dollar_format(x):
                        if isinstance(x, (int, float)):
                            return '${:,.1f}'.format(x)
                        else:
                            return x
                    @st.cache_data(show_spinner="Loading Ledger Data...")
                    def load_main_json(target_bucket, filename="main.json"):
                        raw_json = gcp_download(target_bucket, filename)
                    
                        try:
                            main_json = json.loads(raw_json)
                            if isinstance(main_json, str):
                                main_json = json.loads(main_json)
                        except Exception as e:
                            st.error(f"Failed to parse main.json: {e}")
                            st.stop()
                    
                        return main_json
                    tt=f"MARINE TERMINAL FINANCIALS"
                    original_title = f'<p style="font-family:Arial;font-weight: bold; color:Black; font-size: 20px;">{tt}</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                
                    fintab1,fintab2,fintab3,fintab4,fintab5,fintab6=st.tabs(["LEDGERS-MONTHLY","YEAR NET","BUDGET",
                                                                     "SEARCH BY CUSTOMER/VENDOR STRING","DEPRECIATION","ALL LEDGERS"])
                
                    #### LOAD BUDGET CODES, has information on 2022 and 2023 budget by account number
                
                    # budget_codes=gcp_download_x(target_bucket,rf"FIN/budget_codes.feather")
                    # budget_codes=pd.read_feather(io.BytesIO(budget_codes))
                    # budget_codes.set_index("index",drop=True,inplace=True)
                    # budget=json.loads(gcp_download(target_bucket,rf"FIN/NEW/budget.json"))
                    # weyco_normalized_budget=json.loads(gcp_download(target_bucket,rf"FIN/NEW/weyco_suzano_budget.json"))
                    

                    budget=json.loads(gcp_download(target_bucket,rf"main_budget.json"))

                                 
                     
                
                    with fintab1: 

                        weyco_normalized=st.checkbox("CLICK FOR CLIENT NORMALIZED VIEW")
                        year=st.selectbox("Select Year",["2025","2024","2023","2022","2021","2020","2019","2018", "2017","2016"])
                        
                        ### LETS PUT YEAR in st.session state to use later.
                        if year not in st.session_state:
                            st.session_state.year=year
                        # if "main_json" not in st.session_state:
                        #     raw_main_json = gcp_download(target_bucket, "main.json")
                            
                        #     # Safe loading
                        #     try:
                        #         main_json = json.loads(raw_main_json)
                        #         if isinstance(main_json, str):
                        #             main_json = json.loads(main_json)
                        #     except Exception as e:
                        #         st.error(f"Failed to load JSON: {e}")
                        #         st.stop()
                        
                        #     st.session_state.main_json = main_json
                        # Now it's safely a dict of entries like {"0": {...}, "1": {...}}


                        main_json = load_main_json(target_bucket)

                        if "main" not in st.session_state:
                            st.session_state.main=pd.DataFrame.from_dict(main_json, orient="index").T
                     
                        ledgers=st.session_state.main[st.session_state.main["Period_Year"]==int(year[-2:])]
                        ledgers["Account"]=ledgers["Account"].astype("str")
                        ledgers["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d") for i in ledgers["Date"]]
                        ledgers["Period_Date"]=[datetime.datetime.strptime(i,"%Y-%m") for i in ledgers["Period_Date"]]

                        ledgers["Credit"]=ledgers["Credit"].astype(float)
                        ledgers["Debit"]=ledgers["Debit"].astype(float)
                        ledgers["Net"]=ledgers["Net"].astype(float)
                        
                        # ledgers.set_index("index",drop=True,inplace=True)
                        
                        ### MAKE A COPY OF LEDGERS to change Account column to our structure : 6311000-32
                        ledgers_b=ledgers.copy()
                        ledgers_b.Account=[str(i)+"-"+str(j) for i,j in zip(ledgers_b.Account,ledgers_b.Sub_Cat)]
                        
                        
                        ##### LOAD THE MARINE ACCOUNT STRUCTURE dictionary from pickle file - AFSIN budget structure
                       
                            
                        st.session_state.category=None
                        st.session_state.sub_category=None
                        st.session_state.sub_item=None
                        ###START LEVELS
                        ###CHOOSE AND RECORD CAT in st session state
                        category=st.selectbox("Select Ledger",["Revenues","Operating Expenses","Maintenance Expenses","General & Administrative Overhead","Depreciation"])
                                   
                        if category not in st.session_state:
                            st.session_state.category=category
                
                        # Lets check if Deep categories or shallow (Revenue versus Depreciation)
                        deep=True if category in ["Revenues","Operating Expenses","Maintenance Expenses","Depreciation","General & Administrative Overhead"] else False
                        if weyco_normalized:
                            structure=weyco_normalized_budget.copy()
                        else:
                            structure=budget.copy()

                        depreciation_style="normal"
                        
                        structure=pd.DataFrame(budget).T
                        structure=apply_grouping_mode(structure, mode_col='original')
                        structure.drop(columns=['afsin'], inplace=True)
                        structure=structure[['Account','Name','Group','Subgroup','ship','2024','2025']]
                        if depreciation_style=="normal":
                            structure=structure[~structure["Account"].isin(["6999999-32","6888888-32","6888889-32"])]
                            structure=structure[~structure["Account"].str.startswith('1')]
                            
                            

                        
                        # LOAD a list of sub_cats to display. If not deep the keys becomes the names of subcats due to shallow depth.(last nodes)
                        liste=[f"ALL {category.upper()}"]+list(structure[structure["Group"]==category]["Subgroup"].unique()) if deep else [f"ALL {category.upper()}"]+list(structure[category].values())
                            
                        ##CHOSE AND RECORD SUB_CAT in st session state
                        sub_category=st.selectbox("Select Sub_Category",liste)
                        
                        if sub_category not in st.session_state:
                            st.session_state.sub_category=sub_category
                        
                        # boolean for if all categories are selected like All revenues or All Operating Expenses
                        display_allsubcat=True if sub_category==f"ALL {category.upper()}" else False
                          
                          
                        ### if all categories are selected like All revenues or All Operating Expenses 
                        if display_allsubcat:
                            level=1  
                            ###THIS IS TITLE FOR the MONTHLY option. Like MONTHLY REVENUES
                            monthly_label=category
                            # if it is revenues, operating expenses or maintenance expenses
                            if deep:
                                
                                final=ledgers_b[ledgers_b["Acc"].isin(structure[(structure["Group"]==category)]["Account"].unique())]
                                if final not in st.session_state:
                                    st.session_state["final"]=final
                            # if it is depreciation or overhead   
                            else:
                                ###  WE CHOOSE DICTIONARY keys (like codes) cause we cant go deeper
                                final=ledgers_b[ledgers_b["Account"].isin([i for i in structure[category].keys()])]
                                if final not in st.session_state:
                                    st.session_state["final"]=final
                        
                        ### if individual categories are selected like All revenues or All Operating Expenses           
                        else:
                            level=2
                            if deep:
                                sub_item=st.selectbox("Select Item",[f"ALL {sub_category.upper()}"]+list(structure[(structure["Group"]==category)&(structure["Subgroup"]==sub_category)]["Name"].values))
                                monthly_label=sub_item
                                
                                display_allsubitem=True if sub_item==f"ALL {sub_category.upper()}" else False
                                                    
                                if display_allsubitem:
                                    final=ledgers_b[ledgers_b["Acc"].isin(list(structure[(structure["Group"]==category)&(structure["Subgroup"]==sub_category)]["Account"].values))]
                                else:
                                    level=3
                                    level_3_acc=structure[(structure["Group"]==category)&(structure["Subgroup"]==sub_category)&(structure["Name"]==sub_item)]["Account"].unique()[0]
                                    final=ledgers_b[ledgers_b["Acc"]==level_3_acc]
                                if sub_item not in st.session_state:
                                    st.session_state.sub_item=sub_item
                            else:
                                #sub_category=st.selectbox("Select Sub_Category",structure[category].values(),key="shallow")
                                account=[i for i in list(structure[category]) if structure[category][i]==sub_category][0]
                               
                                final=ledgers_b[ledgers_b["Account"]==account]
                                                    
                                monthly_label=sub_category
                            
                        options = ['WHOLE', 'MONTHLY'] 
                
                        selected_option = st.radio('**Select an option:**', options)
                        
                        if selected_option == 'WHOLE':
                            try:
                                st.subheader(sub_item+" - "+'${:,.1f}'.format(final.Net.sum()))
                            except:
                                try:
                                    st.subheader(sub_category+" - "+'${:,.1f}'.format(final.Net.sum()))
                                except:
                                    st.subheader(monthly_label+" - "+'${:,.1f}'.format(final.Net.sum()))
                            st.write(final)
                        else:
                            
                            a=st.session_state.category
                            b=st.session_state.sub_category
                            c=st.session_state.sub_item
                            
                            #st.write(f"Level={level}")
                            
                            #if deep:
                                #st.write("DEEP")
                            
                            #st.write(a,"-",b,"-",c)
                            
                            if level==3:
                                if deep:
                                    for key, value in structure[a][b].items():
                                        if value == c:
                                            #st.write(key,":",value)
                                            monthly=ledgers_b[ledgers_b["Acc"]==level_3_account]
                                            accounts=[level_3_account]         
                                    
                            elif level==2:
                                if deep:
                                    #st.write(structure[a][b])
                                    monthly=ledgers_b[ledgers_b["Acc"].isin(list(structure[(structure["Group"]==category)&(structure["Subgroup"]==sub_category)]["Account"].values))]
                                    accounts=list(structure[(structure["Group"]==category)&(structure["Subgroup"]==sub_category)]["Account"].values)
                                else:
                                    for key, value in structure[a].items():
                                        if value == b:
                                            #st.write(key,":",value)
                                            accounts=[key]
                            elif level==1:
                                #st.write(structure[a])
                                keys={}
                                accounts=structure[structure["Group"]==a]["Account"].unique()
                                
                            
                                           
                            #st.write(accounts)
                            st.subheader(f'Monthly {monthly_label} {st.session_state.year}')
                            
                           
                            #accounts=structure[a][b].keys()
                            
                    #                 for i in accounts:
                    #                     st.write(budget_codes.loc[budget_codes["Account"]==i]["2023 Adopted"])
                            #st.write(accounts)
                             
                            #st.write(ledgers_b)
                            monthly=ledgers_b[ledgers_b["Acc"].isin(accounts)]
                            monthly.loc[:,"Period_Month"]=[i.month for i in monthly["Period_Date"]]
                            monthly_=monthly.groupby("Period_Month")["Debit","Credit","Net"].sum()
                            #st.write(monthly)
                            # monthly.set_index("Date",drop=True, inplace=True)
        #                     #st.write(monthly)
                            # monthly_=monthly.resample("M")["Debit","Credit","Net"].sum()
                            # monthly_.index=[i.month_name() for i in monthly_.index]
                            avg=round(monthly_.Net.sum()/12 if year=="2024" else 6)
                            total=round(monthly_.Net.sum(),1)
                            
                               
                            annual_budget=structure[structure["Account"].isin(accounts)][str(year)].sum()
                                #st.write(annual_budget)
                            
                            
                            
                                
                               
                            budgeted_monthly=annual_budget/12
                            monthly_=monthly_.applymap(dollar_format)
                            #st.write(annual_budget)
                            #st.write(accounts)
                            col1, col2,col3= st.columns([3,2,4])
                            
                            with col1:
                                st.table(monthly_)
                            with col2:
                                text='${:,.1f}'.format(total)
                                st.markdown('**Average Monthly    :      ${:,.1f}**'.format(avg))
                                st.markdown(f"**Budgeted {st.session_state.year} Monthly:  {'${:,.1f}**'.format(budgeted_monthly)}")
                                st.markdown("##")
                                st.markdown(f"**Budgeted {st.session_state.year} Annual:  {'${:,.1f}**'.format(annual_budget)}")
                                st.markdown(f'**TOTAL so far in {st.session_state.year}:   {text}**')
                            
                                
                                percent_spent = avg / budgeted_monthly * 100
                
                                #st_gauge(percent_spent, label='Monthly Budget Status', min_value=0, max_value=100)
                            with col3:
                                agree = st.checkbox('**CHECK FOR YTD GAUGE INSTEAD OF MONTHLY**')
                                
                                if agree:
                                    value=total
                                    budgeted=annual_budget
                                else:
                                    value=avg
                                    budgeted=budgeted_monthly
                                gauge_text=f"MONTHLY BUDGET STATUS FOR {monthly_label}"
                                fig = go.Figure(go.Indicator(
                                        mode = "gauge+number+delta",
                                        value = abs(round(value,1)),
                                        number = { 'prefix': '$', 'font': {'size':50}},
                                        domain = {'x': [0,1], 'y': [0, 1]},
                                        title = {'text': gauge_text, 'font': {'size': 24}},
                                        delta = {'position':'bottom','reference': abs(round(budgeted,1)), 'increasing': {'color': "RebeccaPurple"}},
                                        gauge = {
                                            'axis': {'range': [None, 2*abs(budgeted)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                            'bar': {'color': "darkblue"},
                                            'bgcolor': "white",
                                            'borderwidth': 2,
                                            'bordercolor': "gray",
                                            'steps': [
                                                {'range': [0, abs(budgeted)], 'color': 'cyan'},
                                                {'range': [abs(budgeted),
                                                          abs(2*budgeted)], 'color': 'royalblue'}],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': abs(budgeted*1.25)}}))
                                fig.add_annotation(
                                                    x=0.5,
                                                    y=0.35,
                                                    text="Monthly Average",
                                                    showarrow=False,
                                                    font=dict(
                                                        size=16,
                                                        color="darkblue",
                                                        family="Arial"   )  )
                                fig.add_annotation(
                                                    x=0,
                                                    y=-0.2,
                                                    text="<b>**DARK BLUE:Running Monthly Average**<b>",
                                                    showarrow=False,
                                                    font=dict(
                                                        size=12,
                                                        color="darkblue",
                                                        family="Arial"   )  )
                                fig.add_annotation(
                                                    x=0,
                                                    y=-0.1,
                                                    text="<b>**CYAN : MONTHLY BUDGET**<b>",
                                                    showarrow=False,
                                                    font=dict(
                                                        size=12,
                                                        color="darkblue",
                                                        family="Arial"   )  )
                                fig.add_annotation(
                                                    x=0,
                                                    y=-0.3,
                                                    text="**<b>RED LINE : 1.5 X MONTHLY BUDGET**<b>",
                                                    showarrow=False,
                                                    font=dict(
                                                        size=12,
                                                        color="darkblue",
                                                        family="Arial"   )  )
                                fig.add_annotation(
                                                    x=0.5,
                                                    y=-0.05,
                                                    text="From Budgeted Monthly",
                                                    showarrow=False,
                                                    font=dict(
                                                        size=16,
                                                        color="darkblue",
                                                        family="Arial"
                                                    )
                                                )
                                fig.update_layout(paper_bgcolor = "lavender",
                                                  font = {'color': "darkblue", 'family': "Arial"})
                
                                st.plotly_chart(fig)
                            col5, col6= st.columns([2,2])
                            with col5:
                                
                                st.subheader(f"Monthly Bar Graph for {st.session_state.year} - {monthly_label}")
                                fig1 = go.Figure(data=[go.Bar(x=monthly_.index, y=monthly_.Net)])
                                st.plotly_chart(fig1)
                            with col6:
                                st.subheader(f"{monthly_label} Across Years")
                                yillar=["2024","2023","2022","2021","2020","2019","2018", "2017","2016"]
                                results=[]
                                for k in yillar:
                                    if k!="2024":
                                        
                                        temp=gcp_download_x(target_bucket,rf"FIN/main{k}.ftr")
                                        temp=pd.read_feather(io.BytesIO(temp))
                                        
                                    
                                        temp.set_index("index",drop=True,inplace=True)
                    
                                        ### MAKE A COPY OF LEDGERS to change Account column to our structure : 6311000-32
                                        temp1=temp.copy()
                                        temp1.Account=[str(i)+"-"+str(j) for i,j in zip(temp1.Account,temp1.Sub_Cat)]
                                        result=temp1[temp1["Account"].isin(accounts)]["Net"].sum()
                                    else:
                                        result=ledgers_b[ledgers_b["Acc"].isin(accounts)]["Net"].sum()
                                    results.append(result)
                                fig2 = go.Figure(data=[go.Bar(x=yillar, y=results)])
                                st.plotly_chart(fig2)
                            #st.write(monthly)
                            
                    with fintab2:
                        year=st.selectbox("Select Year",["2025","2024","2023","2022","2021","2020","2019","2018", "2017","2016"],key="second")
                        month=st.selectbox("Select Month",range(1,13),index=11,key="secsond")
                        ### LETS PUT YEAR in st.session state to use later.
                        
                            
                        ### LOAD LEDGERS by year
                        

                        # Try loading once
                        # try:
                        #     main_json = json.loads(main_json)
                        #     # Check: was it a string again?
                        #     if isinstance(main_json, str):
                        #         main_json = json.loads(main_json)
                        # except Exception as e:
                        #     st.error(f"Failed to load JSON: {e}")
                        #     st.stop()
                        main_json = load_main_json(target_bucket)

                        # Now it's safely a dict of entries like {"0": {...}, "1": {...}}
                        main = pd.DataFrame.from_dict(main_json, orient="index").T
                        ledgers=main[main["Period_Year"]==int(year[-2:])]
                        ledgers["Account"]=ledgers["Account"].astype("str")
                        ledgers["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d") for i in ledgers["Date"]]
                        ledgers["Period_Date"]=[datetime.datetime.strptime(i,"%Y-%m") for i in ledgers["Period_Date"]]
                        ledger_b=ledgers.copy()
                        ledger_b=ledger_b[ledger_b["Period_Date"]<=pd.Timestamp(datetime.date(int(year),int(month),1))]
                        
                        ### MAKE A COPY OF LEDGERS to change Account column to our structure : 6311000-32
                        
                        ledger_b.Account=[str(i)+"-"+str(j) for i,j in zip(ledger_b.Account,ledger_b.Sub_Cat)]

                        revenues_codes=list(structure[structure["Group"]=="Revenues"]["Account"].unique())
                        operations_codes=list(structure[structure["Group"]=="Operating Expenses"]["Account"].unique())
                        maintenance_codes=list(structure[structure["Group"]=="Maintenance Expenses"]["Account"].unique())
                        depreciation_codes=list(structure[structure["Group"]=="Depreciation"]["Account"].unique())
                        overhead_codes=list(structure[structure["Group"]=="General & Administrative Overhead"]["Account"].unique())
                        # st.write(revenues_codes)
                        expenses=operations_codes+maintenance_codes
                        expenses_dep=expenses+depreciation_codes
                    
                        ins=ledger_b[ledger_b["Acc"].isin(revenues_codes)].Net.sum()
                        outs=ledger_b[ledger_b["Acc"].isin(expenses)].Net.sum()
                        outs_overhead=ledger_b[ledger_b["Acc"].isin(overhead_codes)].Net.sum()
                        outs_dep=ledger_b[ledger_b["Acc"].isin(expenses_dep)].Net.sum()
                        dep=ledger_b[ledger_b["Acc"].isin(depreciation_codes)].Net.sum()
                        
                        a1, a2,= st.columns([2,7])
                        with a1:
                            
                            st.write(f"**REVENUES     :  {'${:,.1f}**'.format(ins)}")
                            st.write(f"**EXPENSES      :  {'${:,.1f}**'.format(outs)}")
                            st.write(f"**OVERHEAD      :  {'${:,.1f}**'.format(outs_overhead)}")
                            if ins+outs<0:
                                tt=f"NET BEFORE DEPRECIATION:  {'${:,.1f}'.format(ins+outs+outs_overhead)}"
                                original_title = f'<p style="font-family:Arial;font-weight: bold; color:Red; font-size: 15px;">{tt}</p>'
                                st.markdown(original_title, unsafe_allow_html=True)
                            else:
                                st.write(f"**NET BEFORE DEPRECIATION:  {'${:,.1f}**'.format(ins+outs+outs_overhead)}")
                            st.write(f"**DEPRECIATION:  {'${:,.1f}**'.format(dep)}")
                            if ins+outs+outs_overhead+dep<0:
                                tt=f"NET AFTER DEPRECIATION:  {'${:,.1f}'.format(ins+outs+dep+outs_overhead)}"
                                original_title = f'<p style="font-family:Arial;font-weight: bold; color:Red; font-size: 15px;">{tt}</p>'
                                st.markdown(original_title, unsafe_allow_html=True)
                            else:
                                st.write(f"**NET AFTER DEPRECIATION:  {'${:,.1f}**'.format(ins+outs+dep+outs_overhead)}")
                        
                        with a2:
                            
                        # Define the list of values and labels for the waterfall chart
                            values = [ins,outs,outs_overhead,ins+outs, dep,ins+outs+outs_overhead+dep]
                            labels = ['Revenues', 'Expenses','Overhead', 'Net Before Depreciation', 'Depreciation', 'Net After Depreciation']
                            
                            # Define the colors for each bar in the waterfall chart
                            end=ins+outs+outs_overhead+dep
                            #st.write(end)
                            if end<0:
                                #colors = ['#4CAF50', '#FFC107', '#2196F3', '#F44336', '#F44336']
                                totals={"marker":{"color":"maroon", "line":{"color":"rgb(63, 63, 63)", "width":1}}}
                            else:
                                #colors=['#4CAF50', '#FFC107', '#2196F3', '#F44336', '#EF9A9A']
                                totals={"marker":{"color":f"#2196F3", "line":{"color":"rgb(63, 63, 63)", "width":1}}}
                           
                
                            # Define the text for each bar in the waterfall chart
                            text = ['<b>${:,.1f}<b>'.format(value) for value in values]
                            text_font = {'size': 14,'color':['black','red','red','black','red','black']}
                            this_year=f"So Far This Year" if year=="2024" else ""
                            # Create the trace for the waterfall chart
                            trace = go.Waterfall(
                                name = "Net Result",
                                orientation = "v",
                                measure = ['absolute', 'relative','relative','total', 'relative', 'total'],
                                x = labels,
                                text = text,
                                textfont=text_font,
                                 
                                y = values,
                                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                                decreasing = {"marker":{"color":"#FFC107"}},
                                increasing = {"marker":{"color":"#4CAF50"}},
                                totals={"marker":{"color":f"#2196F3", "line":{"color":"rgb(63, 63, 63)", "width":1}}},
                                cliponaxis = False,  
                            )
                
                            # Define the layout for the waterfall chart
                    #                 layout = go.Layout(
                    #                     title = f'MARINE TERMINAL FINANCIALS-WATERFALL-{year}',
                    #                     xaxis = {'title': 'Components'},
                    #                     yaxis = {'title': 'Amount ($)'},
                    #                 )
                    #                 layout = go.Layout(
                    #                                     title = 'Waterfall Chart',
                    #                                     titlefont=dict(size=24, family='Arial', color='black',),
                    #                                     xaxis = {'title': 'Components', 'titlefont': dict(size=18, family='Arial', color='black'),
                    #                                              'tickfont': dict(size=16, family='Arial', color='black',)},
                    #                                     yaxis = {'title': 'Amount ($)', 'titlefont': dict(size=18, family='Arial', color='black'),
                    #                                              'tickfont': dict(size=16, family='Arial', color='black',)},
                    #                                 )
                            
                            layout = go.Layout(
                                title=dict(
                                    text=f'MARINE TERMINAL FINANCIALS-WATERFALL-{year}<br>{this_year}',
                                    font=dict(size=20, family='Arial', color='black'),
                                    x=0.5
                                ),
                                xaxis=dict(
                                   
                                    tickfont=dict(size=16, family='Arial', color='black'),
                                ),
                                yaxis=dict(
                                    title='Amount ($)',
                                    
                                    tickfont=dict(size=16, family='Arial', color='black'),
                                ),
                                shapes=[
                                    {'type': 'line', 'x0': -0.5, 'y0': 0, 'x1': len(labels)-0.5, 'y1': 0, 'line': {'color': 'red', 'width': 2}}
                                ],
                                height=600,
                            )
                
                            fig = go.Figure(data = trace, layout = layout)
                            
                
                            st.plotly_chart(fig)
                            
                        def key_from_value(d,val):
                            keys={}
                            for k, v in d.items():
                                
                                if isinstance(v, dict):
                                    get_all_keys(v,keys)
                                else:
                                    if d[k]==val:
                                        return(k)
                        def get_all_keys(d,keys):
                            keys={}
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    get_all_keys(v,keys)
                                else:
                                    keys[k]=v
                            return keys

                        temp=ledger_b.copy()
                        vlabels=list(temp[temp["Group"]=="Revenues"]["Sub_Group"].unique())+['TOTAL REVENUE','OPERATING EXPENSES']+\
                               list(temp[temp["Group"]=="Operating Expenses"]["Sub_Group"].unique())+\
                                ["MAINTENANCE EXPENSES"]+\
                                list(temp[temp["Group"]=="Maintenance Expenses"]["Sub_Group"].unique())+\
                                ["DEPRECIATION"]+\
                                list(temp[temp["Group"]=="Depreciation"]["Sub_Group"].unique())+\
                                ["G&A OVERHEAD"]+\
                                list(temp[temp["Group"]=="General & Administrative Overhead"]["Sub_Group"].unique())+\
                                ["Excess Revenue",
                                "Loss After Depreciation"]


                        def extract_values(temp, labels, section_name_start, section_name_end):
                            start = 0 if section_name_start is None else labels.index(section_name_start) + 1
                            end = labels.index(section_name_end)
                            return [round(abs(temp[temp["Sub_Group"] == i]["Net"].sum()), 1) for i in labels[start:end]]
                        
                        
                        def build_sankey_links(revs, ops, maint, dep, overhead, overall, labels):
                            source = []
                            target = []
                            values = []
                            link_colors = []
                        
                            palettes = {
                                'revenue': '#FFA500',
                                'ops': '#87CEFA',
                                'maint': '#90EE90',
                                'dep': '#DDA0DD',
                                'overhead': '#808080',
                                'net': '#C0C0C0'
                            }
                        
                            def add_section_links(section_values, from_node, to_nodes_start_idx, color_key):
                                for i, val in enumerate(section_values):
                                    source.append(from_node)
                                    target.append(to_nodes_start_idx + i)
                                    values.append(val)
                                    link_colors.append(palettes[color_key])
                        
                            # Revenue → TOTAL REVENUE
                            for i, val in enumerate(revs):
                                source.append(i)
                                target.append(labels.index("TOTAL REVENUE"))
                                values.append(val)
                                link_colors.append(palettes['revenue'])
                        
                            idx = labels.index
                        
                            # TOTAL REVENUE → main sections
                            for section, section_data, color_key in zip(
                                ["OPERATING EXPENSES", "MAINTENANCE EXPENSES", "DEPRECIATION", "G&A OVERHEAD"],
                                [ops, maint, dep, overhead],
                                ['ops', 'maint', 'dep', 'overhead']):
                                source.append(idx("TOTAL REVENUE"))
                                target.append(idx(section))
                                values.append(sum(section_data))
                                link_colors.append(palettes[color_key])
                        
                            # Section headers → individual categories
                            offset = len(revs) + 1
                            for section, section_data, color_key in zip(
                                ["OPERATING EXPENSES", "MAINTENANCE EXPENSES", "DEPRECIATION", "G&A OVERHEAD"],
                                [ops, maint, dep, overhead],
                                ['ops', 'maint', 'dep', 'overhead']):
                                parent_idx = idx(section)
                                for i, val in enumerate(section_data):
                                    source.append(parent_idx)
                                    target.append(parent_idx + 1 + i)
                                    values.append(val)
                                    link_colors.append(palettes[color_key])
                        
                            # Final surplus or deficit
                            final_target = idx("Excess Revenue") if overall >= 0 else idx("Loss After Depreciation")
                            final_source = idx("TOTAL REVENUE") if overall >= 0 else idx("Excess Revenue")
                            source.append(final_source)
                            target.append(final_target)
                            values.append(abs(overall))
                            link_colors.append(palettes['net'])
                        
                            return source, target, values, link_colors
                        
                        
                        def build_sankey_chart(temp, year):
                            labels = list(temp[temp["Group"] == "Revenues"]["Sub_Group"].unique()) + [
                                'TOTAL REVENUE', 'OPERATING EXPENSES'] + \
                                     list(temp[temp["Group"] == "Operating Expenses"]["Sub_Group"].unique()) + \
                                     ['MAINTENANCE EXPENSES'] + \
                                     list(temp[temp["Group"] == "Maintenance Expenses"]["Sub_Group"].unique()) + \
                                     ['DEPRECIATION'] + \
                                     list(temp[temp["Group"] == "Depreciation"]["Sub_Group"].unique()) + \
                                     ['G&A OVERHEAD'] + \
                                     list(temp[temp["Group"] == "General & Administrative Overhead"]["Sub_Group"].unique()) + \
                                     ['Excess Revenue', 'Loss After Depreciation']
                        
                            revs=[round(abs(temp[(temp["Sub_Group"]==i)&(temp["Group"]=="Revenues")]["Net"].sum()),1) for i in labels[:labels.index("TOTAL REVENUE")]]
                            ops=[round(abs(temp[(temp["Sub_Group"]==i)&(temp["Group"]=="Operating Expenses")]["Net"].sum()),1) for i in labels[labels.index("OPERATING EXPENSES")+1:labels.index("MAINTENANCE EXPENSES")]]
                            maint=[round(abs(temp[(temp["Sub_Group"]==i)&(temp["Group"]=="Maintenance Expenses")]["Net"].sum()),1) for i in labels[labels.index("MAINTENANCE EXPENSES")+1:labels.index("DEPRECIATION")]]
                            dep=[round(abs(temp[temp["Sub_Group"]==i]["Net"].sum()),1) for i in labels[labels.index("DEPRECIATION")+1:labels.index("G&A OVERHEAD")]]
                            overhead=[round(abs(temp[temp["Sub_Group"]==i]["Net"].sum()),1) for i in labels[labels.index("G&A OVERHEAD")+1:labels.index("Excess Revenue")]]
                            overall=sum(revs)-sum(ops)-sum(maint)-sum(dep)-sum(overhead)
                            valerians = revs + [sum(revs), sum(ops)] + ops + [sum(maint)] + maint + [sum(dep)] + dep + [sum(overhead)] + overhead + [overall, overall]
                            valerians_fmt = ['<b>${:,.1f}</b>'.format(round(i, 1)) for i in valerians]
                            flabels = [f'<b>{i}</b>' for i in labels]
                        
                            source, target, values, link_colors = build_sankey_links(revs, ops, maint, dep, overhead, overall, labels)
                        
                            fig = go.Figure(data=[go.Sankey(
                                node=dict(
                                    thickness=10,
                                    label=[f'{flabels[i]} - {valerians_fmt[i]}' for i in range(len(flabels))],
                                    color="#F5F5F5"
                                ),
                                link=dict(
                                    source=source,
                                    target=target,
                                    value=values,
                                    color=link_colors
                                )
                            )])
                        
                            title_ = f'{year}-YTD' if year == "2023" else year
                            fig.update_layout(
                                width=1200,
                                height=800,
                                title=title_,
                                hovermode='x',
                                font=dict(size=12, color='black'),
                                paper_bgcolor='#FCE6C9',
                                margin=dict(l=50, r=350, t=50, b=50)
                            )
                            return fig
                        fig = build_sankey_chart(temp, year)
                        st.plotly_chart(fig)
                    #fig.write_html(fr'c:\Users\{loc}\Desktop\OldBudget.html')
                    #fig.show()
                
                
                    with fintab3:   ## BUDGET

                        def apply_grouping_mode(df, mode_col='original'):
                            """
                            Flattens Group/Subgroup from the specified mode column and adds them to the DataFrame.
                            Drops the original nested column after expansion.
                            
                            Parameters:
                            - df: input DataFrame
                            - mode_col: name of the column containing nested {'Group': ..., 'Subgroup': ...} dict
                            """
                            df = df.copy()
                            df['Group'] = df[mode_col].apply(lambda x: x.get('Group') if isinstance(x, dict) else None)
                            df['Subgroup'] = df[mode_col].apply(lambda x: x.get('Subgroup') if isinstance(x, dict) else None)
                            df = df.drop(columns=[mode_col])
                            return df

                        
                        
                        
                        budget_df_json = json.loads(gcp_download(target_bucket,rf"main_budget.json"))
                        st.session_state.budget_df=pd.DataFrame(budget_df_json).T

                        
                        
                        df_a = apply_grouping_mode(st.session_state.budget_df, mode_col='original')
                        df_a.drop(columns=['afsin'], inplace=True)
                        df_a=df_a[['Account','Name','Group','Subgroup','ship','2024','2025']]
                        # Apply option "b"
                        df_b = apply_grouping_mode(st.session_state.budget_df, mode_col='afsin')
                        df_b.drop(columns=['original'], inplace=True)
                        df_b=df_b[['Account','Name','Group','Subgroup','ship','2024','2025']]

                        st.title("📊 Editable Budget Table")
                        st.caption("Edit 2025 and 2026 budget projections and save your changes.")
                        if "account_notes" not in st.session_state:
                           st.session_state.account_notes = {}


                        genre = st.radio(
                            "CHOOSE BUDGET STYLE",
                            [":rainbow[ORIGINAL]", "***AFSIN***"],
                            captions=[
                                "Original Terminal Budget Structure",
                                "Re-Organized for Clearer View",
                            ],
                            horizontal=True
                        )
                        if genre==":rainbow[ORIGINAL]":
                            df=df_a.copy()
                        else:
                            df=df_b.copy()
                        if '2026' not in df.columns:
                            df['2026'] = 0
                        df['Account Link'] = df['Account'].apply(lambda x: f"[{x}](#account-{x})")
                        # Now select only 2025 and 2026 as editable
                        edited_df = st.data_editor(
                            df,
                            num_rows="dynamic", # Allows adding rows if needed
                            column_config={
                                "2025": st.column_config.NumberColumn("2025 Budget"),
                                "2026": st.column_config.NumberColumn("2026 Budget"),
                            },
                            disabled=["Account", "Name", "Group", "Subgroup", "ship", "2024", "2024 Results", "Variance"] # Disable others
                        )
                        save_changes, download, upload=st.columns(3)
                        with save_changes:
                            
                            if st.button("💾 Save Changes"):
                                st.session_state.budget_df = edited_df
                                st.success("Changes saved to session!")
                        
                        # ---- SHOW UPDATED TABLE ----
                        # st.markdown("### 🔄 Current Saved Budget")
                        # st.dataframe(st.session_state.budget_df, use_container_width=True)

                        with download:
                            # ---- DOWNLOAD BUTTON ----
                            csv = st.session_state.budget_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name='edited_budget.csv',
                                mime='text/csv'
                            )

                        with upload:
                            
                            if st.button("📤 Upload to GCS"):
                                edited = edited_df.set_index("Account")
                            
                                # Reload full original from session
                                original_df = st.session_state.budget_df
                                original_dict = original_df.T.to_dict()
                            
                                # Loop through edited rows
                                for acct in edited.index:
                                    # Get new values
                                    new_2025 = edited.loc[acct, "2025"]
                                    new_2026 = edited.loc[acct, "2026"]
                            
                                    # Update only 2025 and 2026 in original
                                    if acct in original_dict:
                                        original_dict[acct]["2025"] = new_2025
                                        original_dict[acct]["2026"] = new_2026
                            
                                # Optional: save to session state too
                                st.session_state.budget_df = pd.DataFrame(original_dict).T
                            
                                # Dump and upload
                                # Convert all values to native Python types (like int, float, str)
                                def convert_numpy(obj):
                                    if isinstance(obj, (np.int64, np.int32)):
                                        return int(obj)
                                    elif isinstance(obj, (np.float64, np.float32)):
                                        return float(obj)
                                    return obj
                                
                                updated_json = json.dumps(original_dict, default=convert_numpy)
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"main_budget.json")
                                blob.upload_from_string(updated_json)
                       
                            
                                st.success("✅ Budget successfully updated and uploaded to GCS!")
                            # Capture clicked account manually with selectbox
                        selected_account = st.selectbox("Pick an account to view details:", df['Account'])

                        if selected_account:
                            row = df[df["Account"] == selected_account].iloc[0]
                            main = pd.DataFrame.from_dict(main_json, orient="index").T
                            ledgers = main[main["Period_Year"] == int(24)]
                            ledgers["Account"] = ledgers["Account"].astype("str")
                            ledgers["Date"] = [datetime.datetime.strptime(i, "%Y-%m-%d") for i in ledgers["Date"]]
                            ledgers["Period_Date"] = [datetime.datetime.strptime(i, "%Y-%m") for i in ledgers["Period_Date"]]
                            ledger_df = ledgers.copy()
                            ledger_df = ledger_df[ledger_df["Period_Date"] <= pd.Timestamp(datetime.date(int(year), int(month), 1))]
                        
                            ledger_df["Account"] = [str(i) + "-" + str(j) for i, j in zip(ledger_df["Account"], ledger_df["Sub_Cat"])]
                            ledger_entries = ledger_df[ledger_df["Account"] == selected_account]
                            
                        
                            with st.expander(f"📋 Account Info: {selected_account}", expanded=True):
                                info, entries=st.columns([2,8])
                                with info:
                                    st.markdown(
                                        f"""
                                        <div style="background-color:#f0f5ff; padding:15px; border-radius:10px;">
                                            <h4 style="margin-bottom:0;">📋 Account: {selected_account}</h4>
                                            <p><strong>Name:</strong> {row['Name']}<br>
                                            <strong>Group:</strong> {row['Group']}<br>
                                            <strong>Subgroup:</strong> {row['Subgroup']}<br>
                                            <strong>2024 Budget:</strong> ${row['2024']:,.0f}<br>
                                            <strong>2024 Total Net:</strong> ${ledger_entries.Net.sum():,.0f}<br>
                                            <strong>2024 Monthly Net:</strong> ${round(ledger_entries.Net.sum()/12,2):,.0f}<br>
                                            <strong>2025 Budget:</strong> ${row['2025']:,.0f}<br>
                                            <strong>2026 Budget:</strong> ${row['2026']:,.0f}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                            
                                    note_key = f"note_{selected_account}"
                                    current_note = st.session_state.account_notes.get(selected_account, "")
                                    updated_note = st.text_area("📝 Notes for this Account", value=current_note, height=120)
                                    if updated_note != current_note:
                                        st.session_state.account_notes[selected_account] = updated_note
                                with entries:
                        
                                    # 📖 Ledger View inside the same expander
                                    if st.button("📖 View Ledger Entries") or st.session_state.get("show_ledger_entries", False):
                                        st.session_state.show_ledger_entries = True  # Set the flag when button clicked
                                        st.markdown("## 📘 Ledger Entries")
                            
                                        
                                        
                                    
                                        if not ledger_entries.empty:
                                            st.dataframe(ledger_entries, use_container_width=True)
                                    
                                            # 🎯 Calculate percentiles
                                            q25 = np.percentile(ledger_entries["Net"], 25)
                                            q50 = np.percentile(ledger_entries["Net"], 50)
                                            q75 = np.percentile(ledger_entries["Net"], 75)
                                            q90 = np.percentile(ledger_entries["Net"], 90)  # Threshold
                                    
                                            def assign_color(amount):
                                                if amount <= q25:
                                                    return "red"
                                                elif amount <= q50:
                                                    return "orange"
                                                elif amount <= q75:
                                                    return "lightblue"
                                                else:
                                                    return "blue"
                                    
                                            ledger_entries["Color"] = ledger_entries["Net"].apply(assign_color)
                                            ledger_entries["Above Threshold"] = ledger_entries["Net"] > q90
                                    
                                            # 🎯 Create scatter and histogram separately
                                            scatter = go.Scatter(
                                                x=ledger_entries["Per_Entry"],
                                                y=ledger_entries["Net"],
                                                mode='markers',
                                                marker=dict(
                                                    color=ledger_entries["Color"],
                                                    size=ledger_entries["Above Threshold"].apply(lambda x: 14 if x else 8),
                                                    opacity=0.8,
                                                    line=dict(width=1, color='DarkSlateGrey')
                                                ),
                                                text=ledger_entries["Description"],
                                                hovertemplate="Date: %{x}<br>Amount: %{y}<br>Description: %{text}<extra></extra>",
                                                name="Transactions"
                                            )
                                    
                                            hist = go.Histogram(
                                                x=ledger_entries["Net"],
                                                nbinsx=30,
                                                marker_color='gray',
                                                opacity=0.6,
                                                name="Amount Distribution"
                                            )
                                    
                                            # 🎯 Build figure with subplots
                                            fig = make_subplots(
                                                rows=2, cols=1,
                                                shared_xaxes=False,
                                                row_heights=[0.7, 0.3],
                                                vertical_spacing=0.1,
                                                subplot_titles=(f"💰 Payment Scatter for {selected_account}", "📊 Payment Amount Histogram")
                                            )
                                    
                                            fig.add_trace(scatter, row=1, col=1)
                                            fig.add_trace(hist, row=2, col=1)
                                    
                                            # Threshold line
                                            fig.add_shape(
                                                type="line",
                                                x0=min(ledger_entries["Per_Entry"]),
                                                x1=max(ledger_entries["Per_Entry"]),
                                                y0=q90,
                                                y1=q90,
                                                line=dict(color="green", width=2, dash="dash"),
                                                row=1, col=1
                                            )
                                    
                                            fig.update_layout(
                                                height=850,
                                                plot_bgcolor='white',
                                                showlegend=False,
                                                hovermode="closest",
                                            )
                                    
                                            fig.update_xaxes(title_text="Transaction Date", row=1, col=1)
                                            fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
                                            fig.update_xaxes(title_text="Amount ($)", row=2, col=1)
                                            fig.update_yaxes(title_text="Number of Transactions", row=2, col=1)
                                    
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                            # 🎯 Monthly Aggregation (Sum of Payments)
                                            ledger_entries["YearMonth"] = ledger_entries["Date"].dt.to_period('M').astype(str)
                                            monthly_sum = ledger_entries.groupby("YearMonth")["Net"].sum().reset_index()
                                    
                                            bar_fig = px.bar(
                                                monthly_sum,
                                                x="YearMonth",
                                                y="Net",
                                                title=f"📅 Monthly Sum of Payments for {selected_account}",
                                                labels={"YearMonth": "Month", "Amount": "Total Amount ($)"},
                                                text_auto='.2s'
                                            )
                                    
                                            bar_fig.update_layout(
                                                plot_bgcolor='white',
                                                height=500,
                                                xaxis_tickangle=-45
                                            )
                                    
                                            st.plotly_chart(bar_fig, use_container_width=True)
                                    
                                        else:
                                            st.info("No ledger entries found for this account.")
                            

                   
                   
                    with fintab4:
                        ear=st.selectbox("Select Year",["2025","2024","2023","2022","2021"],key="yeartab2")

                        main_json = gcp_download(target_bucket, "main.json")
                        try:
                            main_json = json.loads(main_json)
                            # Check: was it a string again?
                            if isinstance(main_json, str):
                                main_json = json.loads(main_json)
                        except Exception as e:
                            st.error(f"Failed to load JSON: {e}")
                            st.stop()
                
                        ledgers=st.session_state.main[st.session_state.main["Period_Year"]==int(ear[-2:])]
                        ledgers["Account"]=ledgers["Account"].astype("str")
                        ledgers["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d") for i in ledgers["Date"]]
                        ledgers["Period_Date"]=[datetime.datetime.strptime(i,"%Y-%m") for i in ledgers["Period_Date"]]

                        ledgers["Credit"]=ledgers["Credit"].astype(float)
                        ledgers["Debit"]=ledgers["Debit"].astype(float)
                        ledgers["Net"]=ledgers["Net"].astype(float)
                        #st.write(ledgers)
        
                        for_search_ledger=ledgers.fillna("")
                        
                        vendor,job=st.tabs(["SEARCH BY VENDOR","SEARCH BY JOB"])
                        with vendor:
                            
                            pattern = r'^([A-Z&]{3}\d{3})\s+(.+)$'
                            vendors={}
                            tata=[]
                            # Loop over the strings and print the vendor codes and names
                            for s in ledgers["Description"].values.tolist():
                                s=str(s)
                                tata.append(s)
                                try:
                                    match = re.match(pattern, s)
                                    if match:
                                        vendor_code = match.group(1)
                                        vendor_name = match.group(2)
                                        vendors[vendor_name]=vendor_code
                                        #print(f'{vendor_code} {vendor_name}')
                                except:
                                    pass
                            
                            
                            
                            #match = re.match(pattern, tata[1500])
                       
                
                            string_=st.selectbox("Select Vendor",vendors.keys(),key="vendor")
                       
                            
                            if string_:
                                st.subheader(f"{vendors[string_]} - {string_} {ear} Expenses")
                                temp=ledgers[ledgers["Description"].str.contains(string_).fillna(False)]
                                
                                total='${:,.1f}'.format(temp.Net.sum())
                                total=f"<b>TOTAL = {total}</b>"
                                try:
                                    st.table(temp)
                                    st.markdown(total,unsafe_allow_html=True)
                                except:
                                    st.write("NO RESULTS")
                        with job:
                            
                        
                            jobs=[]
                            pattern = r"\b\d+\b"
                            # Loop over the strings and print the vendor codes and names
                            for s in ledgers["Job_No"].values.tolist():
                                
                                try:
                                    match = re.match(pattern, s)
                                except:
                                    pass
                                if match:
                                    jobs.append(s)
                                    
                            
                            jobs=ledgers["Job_No"].unique().tolist()
                            string_=st.selectbox("Select Job",jobs,key="job")
                            if string_:
                                st.subheader(f"{string_} {ear} Records")
                                temp=ledgers[ledgers["Job_No"].str.contains(string_).fillna(False)]
                                
                                total='${:,.1f}'.format(temp.Net.sum())
                                total=f"<b>TOTAL = {total}</b>"
                                try:
                                    st.table(temp)
                                    st.markdown(total,unsafe_allow_html=True)
                                except:
                                    st.write("NO RESULTS")
                            st.write(jobs)
                #                 print(f'{vendor_code} {vendor_name}')
                #                 filtered=[]
                #                 for i in for_search_ledger.index:
                #                     #st.write(i)
                #                     result=re.findall(fr'{string_}',for_search_ledger.loc[i,"Job_No"],re.IGNORECASE)
                #                     #st.write(result)
                #                     #st.write(for_search_ledger.loc[i,"Description"])
                # #                     if string_ in for_search_ledger.loc[i,"Description"]:
                # #                         st.write("ysy")
                #                     if len(result)>0:
                #                         filtered.append(i)
                #                         temp=for_search_ledger.loc[filtered]
                                
        #                 #st.write(final)
        #             with fintab5:
        #                 year=st.selectbox("SELECT YEAR",["2023","2022","2021","2020","2019","2018","2017"],key="depreciation")
        #                 terminal_depreciation=gcp_download_x(target_bucket,rf"FIN/main{year}-30.ftr")
        #                 terminal_depreciation=pd.read_feather(io.BytesIO(terminal_depreciation)).set_index("index",drop=True).reset_index(drop=True)
                        
        #                 a=terminal_depreciation[terminal_depreciation["Account"].isin( [i for i in terminal_depreciation.Account.unique().tolist() if i>1700000 and  i<2000000])]
        #                 a=a.groupby(["Account"])[["Credit"]].sum()
        #                 a.insert(0,"Name",[terminal_depreciation.loc[terminal_depreciation["Account"]==i,"Name"].values[0] for i in a.index])
                        
        #                 divisor=3 if year=="2023" else 12
                        
                        
                        
        #                 th_props = [
        #                       ('font-size', '16px'),
        #                       ('text-align', 'center'),
        #                       ('font-weight', 'bold'),
        #                       ('color', '#6d6d6d'),
        #                       ('background-color', '#f7ffff')
        #                       ]
                                                           
        #                 td_props = [
        #                   ('font-size', '15px'),
        #                   ('background-color', '#r9f9ff')
        #                   ]
        #                 def highlight_total(val):
        #                     return 'font-weight: bold; font-size: 30px;'
                           
        #                 styles = [
        #                           dict(selector="th", props=th_props),
        #                           dict(selector="td", props=td_props)
        #                           ]
                        
        #             #             sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        #             #                      
        #             #             colors = ['#FFD700', '#32CD32', '#FF69B4', '#ADD8E6', '#FFA07A']
        #             #             
        #             #             width = st.sidebar.slider("plot width", 1, 25, 3)
        #             #             height = st.sidebar.slider("plot height", 1, 25, 1)
        #             #             selection=st.sidebar.checkbox("I agree")
        #             #             fig, ax = plt.subplots(figsize=(6,4))
        #             #             
        #             #             ax.axis('off')
        #             #             labels = [f'{name[5:]}\n${credit:,.2f}' if credit>50000 else "*" for name, credit in zip(a.index, a['Credit'])]
        #             #             
        #             #             squarify.plot(sizes=a['Credit'], label=labels, color=colors, alpha=0.8, text_kwargs={'fontsize':6, 'fontweight':'bold'})
        #             #             plt.title(f'TERMINAL DEPRECIATION - {year}', fontweight='bold', fontsize=14, y=1.08)
        #             #             small_items = a[a['Credit'] <= 50000][['Credit']]
        #             #             small_ax = fig.add_axes([0.905, 0.3, 0.1, 0.5])
        #             #             small_ax.barh(y=[i[5:] for i in small_items.index], width=small_items['Credit'], color='#808080')
        #             #             small_ax.set_xlabel('Amount')
        #             #             small_ax.yaxis.set_label_position("right")
        #             #             small_ax.yaxis.tick_right()
        #             #             #fig.set_size_inches(4, 6)
        #                 labels = [f'{name[5:]}\n${credit:,.2f}' if credit>50000 else "*" for name, credit in zip(a["Name"], a['Credit'])]
        #                 fig = px.treemap(a, 
        #                      path=["Name"], 
        #                      values='Credit',
        #                                  labels={"Name":labels},
        #                      color='Credit',
        #                      color_continuous_scale='Blues')
                
        #             # Update the layout
        #                 fig.update_layout(
        #                     margin=dict(t=50, l=0, r=0, b=0),
        #                     font=dict(size=16),
        #                     title='TERMINAL DEPRECIATION',
        #                     title_font_size=24,
        #                     title_font_family='Arial')
                
        #             # Show the plot
                
                        
        #                 st.plotly_chart(fig)
        #                 a.set_index("Name",drop=True,inplace=True)
        #                 a.loc["TOTAL"]=a.sum()
        #                 a["Monthly"]=['${:,.1f}'.format(i/divisor) for i in a["Credit"]]          
        #                 a["Credit"]=['${:,.1f}'.format(i) for i in a["Credit"]]
                        
        #                 a=a.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles).applymap(highlight_total, subset=pd.IndexSlice["TOTAL", ["Credit","Monthly"]])
        #                 st.table(a)    
        #             with fintab6:
        #                 ledgers=gcp_download_x(target_bucket,rf"FIN/all_ledgers.ftr")
        #                 ledgers=pd.read_feather(io.BytesIO(ledgers))
        #                 ledgers["Account"]=ledgers["Account"].astype("str")
        #                 #ledgers.set_index("index",drop=True,inplace=True)
        #                 for_search_ledger=ledgers.fillna("")
                        
        #                 vendor,job=st.tabs(["SEARCH BY VENDOR","SEARCH BY JOB"])
        #                 with vendor:
                            
        #                     pattern = r'^([A-Z&]{3}\d{3})\s+(.+)$'
        #                     vendors={}
        #                     tata=[]
        #                     # Loop over the strings and print the vendor codes and names
        #                     for s in ledgers["Description"].values.tolist():
        #                         s=str(s)
        #                         tata.append(s)
        #                         try:
        #                             match = re.match(pattern, s)
        #                             if match:
        #                                 vendor_code = match.group(1)
        #                                 vendor_name = match.group(2)
        #                                 vendors[vendor_name]=vendor_code
        #                                 #print(f'{vendor_code} {vendor_name}')
        #                         except:
        #                             pass
                            
                            
                            
        #                     #match = re.match(pattern, tata[1500])
                       
                
        #                     string_=st.selectbox("Select Vendor",vendors.keys(),key="vendoeer")
                       
                            
        #                     if string_:
        #                         st.subheader(f"{vendors[string_]} - {string_} {ear} Expenses")
        #                         temp=ledgers[ledgers["Description"].str.contains(string_).fillna(False)]
                                
        #                         total='${:,.1f}'.format(temp.Net.sum())
        #                         total=f"<b>TOTAL = {total}</b>"
        #                         try:
        #                             st.write(temp)
        #                             st.markdown(total,unsafe_allow_html=True)
        #                         except:
        #                             st.write("NO RESULTS")
        #                 with job:
                            
                        
        #                     jobs=[]
        #                     pattern = r"\b\d+\b"
        #                     # Loop over the strings and print the vendor codes and names
        #                     for s in ledgers["Job_No"].values.tolist():
                                
        #                         try:
        #                             match = re.match(pattern, s)
        #                         except:
        #                             pass
        #                         if match:
        #                             jobs.append(s)
                                    
                            
        #                     jobs=ledgers["Job_No"].unique().tolist()
        #                     string_=st.selectbox("Select Job",jobs,key="jssob")
        #                     if string_:
        #                         st.subheader(f"{string_} {ear} Records")
        #                         temp=ledgers[ledgers["Job_No"].str.contains(string_).fillna(False)]
                                
        #                         total='${:,.1f}'.format(temp.Net.sum())
        #                         total=f"<b>TOTAL = {total}</b>"
        #                         try:
        #                             st.table(temp)
        #                             st.markdown(total,unsafe_allow_html=True)
        #                         except:
        #                             st.write("NO RESULTS")
        #                     st.write(jobs)
        #         #                 print(f'{vendor_code} {vendor_name}')
        #         #                 filtered=[]
        #         #                 for i in for_search_ledger.index:
        #         #                     #st.write(i)
        #         #                     result=re.findall(fr'{string_}',for_search_ledger.loc[i,"Job_No"],re.IGNORECASE)
        #         #                     #st.write(result)
        #         #                     #st.write(for_search_ledger.loc[i,"Description"])
        #         # #                     if string_ in for_search_ledger.loc[i,"Description"]:
        #         # #                         st.write("ysy")
        #         #                     if len(result)>0:
        #         #                         filtered.append(i)
        #         #                         temp=for_search_ledger.loc[filtered]
                                
        #                 #st.write(final)
        
            
            
              
        # if select=="STORAGE" :
        #     maintenance=False
        #     if not maintenance:
        #         def calculate_balance(start_tons, daily_rate, storage_rate):
        #             balances={}
        #             tons_remaining = start_tons
        #             accumulated=0
        #             day=1
        #             while tons_remaining>daily_rate:
        #                 #print(day)
        #                 balances[day]={"Remaining":tons_remaining,"Charge":0,"Accumulated":0}
        #                 if day % 7 < 5:  # Consider only weekdays
        #                     tons_remaining-=daily_rate58
        #                     #print(tons_remaining)
                            
        #                     balances[day]={"Remaining":tons_remaining,"Charge":0,"Accumulated":0}
                
        #                     # If storage free days are over, start applying storage charges
        #                 elif day % 7 in ([5,6]):
        #                     balances[day]={"Remaining":tons_remaining,"Charge":0,"Accumulated":accumulated}
        #                 if day >free_days_till:
        #                     charge = round(tons_remaining*storage_rate,2)  # You can adjust the storage charge after the free days
        #                     accumulated+=charge
        #                     accumulated=round(accumulated,2)
        #                     balances[day]={"Remaining":tons_remaining,"Charge":charge,"Accumulated":accumulated}
                        
        #                 day+=1
        #             return balances
                    
        #         here1,here2,here3=st.columns([2,5,3])
                
        #         with here1:
        #             with st.container(border=True):
        #                 initial_tons =st.number_input("START TONNAGE", min_value=1000, help=None, on_change=None,step=50, disabled=False, label_visibility="visible",key="fas2aedseq")
        #                 daily_rate=st.slider("DAILY SHIPMENT TONNAGE",min_value=248, max_value=544, step=10,key="fdee2a")
        #                 storage_rate = st.number_input("STORAGE RATE DAILY ($)",value=0.15, help="dsds", on_change=None, disabled=False, label_visibility="visible",key="fdee2dsdseq")
        #                 free_days_till = st.selectbox("FREE DAYS",[15,30,45,60])
                
        #         with here3:
        #             with st.container(border=True):    
        #                 balances = calculate_balance(initial_tons, daily_rate, storage_rate)
        #                 d=pd.DataFrame(balances).T
        #                 start_date = pd.to_datetime('today').date()
        #                 end_date = start_date + pd.DateOffset(days=120)  # Adjust as needed
        #                 date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                        
        #                 d.columns=["Remaining Tonnage","Daily Charge","Accumulated Charge"]
        #                 d.rename_axis("Days",inplace=True)
        #                 total=round(d.loc[len(d),'Accumulated Charge'],1)
        #                 st.dataframe(d)

        #         with here2:
        #             with st.container(border=True):     
        #                 st.write(f"######  Cargo: {initial_tons} - Loadout Rate/Day: {daily_rate} Tons - Free Days : {free_days_till}" )
        #                 st.write(f"##### TOTAL CHARGES:  ${total}" )
        #                 st.write(f"##### DURATION OF LOADOUT:  {len(d)} Days")
        #                 st.write(f"##### MONTHLY REVENUE: ${round(total/len(d)*30,1)} ")
        #                 fig = px.bar(d, x=d.index, y="Accumulated Charge", title="Accumulated Charges Over Days")
    
        #                 # Add a horizontal line for the monthly average charge
        #                 average_charge = round(total/len(d)*30,1)
        #                 fig.add_shape(
        #                     dict(
        #                         type="line",
        #                         x0=d.index.min(),
        #                         x1=d.index.max(),
        #                         y0=average_charge,
        #                         y1=average_charge,
        #                         line=dict(color="red", dash="dash"),
        #                     )
        #                 )
                        
        #                 # Add annotation with the average charge value
        #                 fig.add_annotation(
        #                     x=d.index.max()//3,
        #                     y=average_charge,
        #                     text=f'Monthly Average Income: <b><i>${average_charge:.2f}</b></i> ',
        #                     showarrow=True,
        #                     arrowhead=4,
        #                     ax=-50,
        #                     ay=-30,
        #                     font=dict(size=16),
        #                     bgcolor='rgba(255, 255, 255, 0.6)',
        #                 )
                        
        #                 # Set layout options
        #                 fig.update_layout(
        #                     xaxis_title="Days",
        #                     yaxis_title="Accumulated Charge",
        #                     sliders=[
        #                         {
        #                             "steps": [
        #                                 {"args": [[{"type": "scatter", "x": d.index, "y": d["Accumulated Charge"]}], "layout"], "label": "All", "method": "animate"},
        #                             ],
        #                         }
        #                     ],
        #                 )
        #                 st.plotly_chart(fig)
        
        
        
        if select=="LABOR":
            gate_entries=json.loads(gcp_download(target_bucket,rf"gate_entries.json"))
            combined_entries = gate_entries['paper'] + gate_entries['log']
            combined_entries=pd.DataFrame(combined_entries)
            combined_entries.set_index("date",drop=True,inplace=True)
            st.write(combined_entries)
            
                                    
        #     labor_issue=False
        #     secondary=True
            
        #     if secondary:
        #         pma_rates=gcp_download(target_bucket,rf"LABOR/pma_dues.json")
        #         pma_rates=json.loads(pma_rates)
        #         assessment_rates=gcp_download(target_bucket,rf"LABOR/occ_codes2023.json")
        #         assessment_rates=json.loads(assessment_rates)
        #         lab_tab1,lab_tab2,lab_tab3,lab_tab4=st.tabs(["LABOR TEMPLATE", "JOBS","RATES","LOOKUP"])

        #         with lab_tab4:
        #             itsreadytab4=True
        #             if itsreadytab4:
                        
        #                 def dfs_sum(dictionary, key):
        #                     total_sum = 0
                        
        #                     for k, v in dictionary.items():
        #                         if k == key:
        #                             total_sum += v
        #                         elif isinstance(v, dict):
        #                             total_sum += dfs_sum(v, key)
                        
        #                     return total_sum
        #                 mt_jobs_=gcp_download(target_bucket,rf"LABOR/mt_jobs.json")
        #                 mt_jobs=json.loads(mt_jobs_)
        #                 c1,c2,c3=st.columns([2,2,6])
        #                 with c1:
        #                     with st.container(border=True):
        #                         by_year=st.selectbox("SELECT YEAR",mt_jobs.keys())
        #                         by_job=st.selectbox("SELECT JOB",mt_jobs[by_year].keys())
        #                         by_date=st.selectbox("SELECT DATE",mt_jobs[by_year][by_job]["RECORDS"].keys())
        #                         by_shift=st.selectbox("SELECT SHIFT",mt_jobs[by_year][by_job]["RECORDS"][by_date].keys())
                                
        #                 with c2:
        #                     info=mt_jobs[by_year][by_job]["INFO"]
        #                     st.dataframe(info)
        #                 with c3:
        #                     with st.container(border=True):
                                
        #                         d1,d2,d3=st.columns([4,4,2])
        #                         with d1:
        #                             by_choice=st.radio("SELECT INVOICE",["LABOR","EQUIPMENT","MAINTENANCE"])
        #                         with d2:
        #                             by_location=st.radio("SELECT INVOICE",["DOCK","WAREHOUSE","LINES"])
        #                     with st.container(border=True):
        #                         e1,e2,e3=st.columns([4,4,2])
        #                         with e1:
        #                             st.write(f"TOTAL {by_location}-{by_choice} COST for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],'TOTAL COST'),2)}")
        #                             st.write(f"TOTAL {by_location}-{by_choice} MARKUP for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],'Mark UP'),2)}")
        #                             if by_choice=="LABOR":
        #                                 st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],'INVOICE'),2)}")
        #                             else:
        #                                 st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],f'{by_choice} INVOICE'),2)}")
        #                         with e2:
        #                             st.write(f"TOTAL {by_location}-{by_choice} COST for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],'TOTAL COST') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
        #                             st.write(f"TOTAL {by_location}-{by_choice} MARKUP for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],'Mark UP') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
        #                             if by_choice=="LABOR":
        #                                 st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],'INVOICE') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
        #                             else:
        #                                 st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],f'{by_choice} INVOICE') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
        #                     #st.write(mt_jobs[by_year][by_job]["RECORDS"][by_date][by_shift][by_choice][by_location])
        #                 a=pd.DataFrame(mt_jobs[by_year][by_job]["RECORDS"][by_date][by_shift][by_choice][by_location]).T
        #                 if by_choice=="LABOR":
        #                     try:
        #                         a.loc["TOTAL FOR SHIFT"]=a[["Quantity","Hours","OT","Hour Cost","OT Cost","Total Wage","Benefits","PMA Assessments","TOTAL COST","Ind Ins","SIU","Mark UP","INVOICE"]].sum()
        #                     except:
        #                         pass
        #                 else:
        #                     try:
        #                         a.loc["TOTAL FOR SHIFT"]=a[[ "Quantity","Hours","TOTAL COST","Mark UP",f"{by_choice} INVOICE"]].sum()
        #                     except:
        #                         pass
        #                 st.write(a)
                            
                           
                        
        #         with lab_tab3:
        #             with st.container(border=True):
                        
        #                 tinker,tailor=st.columns([5,5])
        #                 with tinker:
        #                     select_year=st.selectbox("SELECT ILWU PERIOD",["JUL 2023","JUL 2022","JUL 2021"],key="ot1221")
        #                 with tailor:
        #                     select_pmayear=st.selectbox("SELECT PMA PERIOD",["JUL 2023","JUL 2022","JUL 2021"],key="ot12w21")
                    
        #             year=select_year.split(' ')[1]
        #             month=select_year.split(' ')[0]
        #             pma_year=select_pmayear.split(' ')[1]
        #             pma_rates_=pd.DataFrame(pma_rates).T
        #             occ_codes=pd.DataFrame(assessment_rates).T
        #             occ_codes=occ_codes.rename_axis('Occ_Code')
        #             shortened_occ_codes=occ_codes.loc[["0036","0037","0055","0065","0092","0101","0103","0115","0129","0213","0215"]]
        #             shortened_occ_codes=shortened_occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
        #             occ_codes=occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
        #             rates=st.checkbox("SELECT TO DISPLAY RATE TABLE FOR THE YEAR",key="iueis")
        #             if rates:
                        
        #                 lan1,lan2=st.columns([2,2])
        #                 with lan1:
        #                     st.write(occ_codes)
        #                 with lan2:
        #                     st.write(pma_rates[pma_year])
        #         with lab_tab2:
        #             lab_col1,lab_col2,lab_col3=st.columns([2,2,2])
        #             with lab_col1:
        #                 with st.container(border=True):
                            
        #                     job_vessel=st.text_input("VESSEL",disabled=False)
        #                     vessel_length=st.number_input("VESSEL LENGTH",step=1,disabled=False)
        #                     job_number=st.text_input("MT JOB NO",disabled=False)
        #                     shipper=st.text_input("SHIPPER",disabled=False)
        #                     cargo=st.text_input("CARGO",disabled=False)
        #                     agent=st.selectbox("AGENT",["TALON","ACGI","NORTON LILLY"],disabled=False)
        #                     stevedore=st.selectbox("STEVEDORE",["SSA","JONES"],disabled=False)
                            
        #                     alongside_date=st.date_input("ALONGSIDE DATE",disabled=False,key="arr")
        #                     alongside_date=datetime.datetime.strftime(alongside_date,"%Y-%m-%d")
                            
        #                     alongside_time=st.time_input("ALONGSIDE TIME",disabled=False,key="arrt")
        #                     alongside_time=alongside_time.strftime("%H:%M")
                            
        #                     departure_date=st.date_input("DEPARTURE DATE",disabled=False,key="dep")
        #                     departure_date=datetime.datetime.strftime(departure_date,"%Y-%m-%d")
                           
        #                     departure_time=st.time_input("DEPARTURE TIME",disabled=False,key="dept")
        #                     departure_time=departure_time.strftime("%H:%M")
                            
                            
        #                 if st.button("RECORD JOB"):
        #                     year="2023"
        #                     mt_jobs_=gcp_download(target_bucket,rf"LABOR/mt_jobs.json")
        #                     mt_jobs=json.loads(mt_jobs_)
        #                     if year not in mt_jobs:
        #                         mt_jobs[year]={}
        #                     if job_number not in mt_jobs[year]:
        #                         mt_jobs[year][job_number]={"INFO":{},"RECORDS":{}}
        #                     mt_jobs[year][job_number]["INFO"]={"Vessel":job_vessel,"Vessel Length":vessel_length,"Cargo":cargo,
        #                                         "Shipper":shipper,"Agent":agent,"Stevedore":stevedore,"Alongside Date":alongside_date,
        #                                         "Alongside Time":alongside_time,"Departure Date":departure_date,"Departure Time":departure_time}
        #                     mt_jobs=json.dumps(mt_jobs)
        #                     storage_client = get_storage_client()
        #                     bucket = storage_client.bucket(target_bucket)
        #                     blob = bucket.blob(rf"mt_jobs.json")
        #                     blob.upload_from_string(mt_jobs)
        #                     st.success(f"RECORDED JOB NO {job_number} ! ")
        #         with lab_tab1:
        #             equipment_tariff={"CRANE":908.51,"FORKLIFT":84.92,"TRACTOR":65,"KOMATSU":160,"GENIE MANLIFT":84.92,"Z135 MANLIFT":130.46}
        #             foreman=False
        #             with st.container(border=True):
                        
        #                 tinker,tailor=st.columns([5,5])
        #                 with tinker:
        #                     select_year=st.selectbox("SELECT ILWU PERIOD",["JUL 2023","JUL 2022","JUL 2021"])
        #                 with tailor:
        #                     select_pmayear=st.selectbox("SELECT PMA PERIOD",["JUL 2023","JUL 2022","JUL 2021"])
                    
        #             year=select_year.split(' ')[1]
        #             month=select_year.split(' ')[0]
        #             pma_year=select_pmayear.split(' ')[1]
        #             pma_rates_=pd.DataFrame(pma_rates).T
        #             occ_codes=pd.DataFrame(assessment_rates).T
        #             occ_codes=occ_codes.rename_axis('Occ_Code')
        #             shortened_occ_codes=occ_codes.loc[["0036","0037","0055","0065","0092","0101","0103","0115","0129","0213","0215","0277","0278"]]
        #             shortened_occ_codes=shortened_occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
        #             occ_codes=occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
                    
                    
                    
        #             if "scores" not in st.session_state:
        #                 st.session_state.scores = pd.DataFrame(
        #                     {"Code": [], "Shift":[],"Quantity": [], "Hours": [], "OT": [],"Hour Cost":[],"OT Cost":[],"Total Wage":[],"Benefits":[],"PMA Assessments":[],
        #                      "TOTAL COST":[],"Ind Ins":[],"SIU":[],"Mark UP":[],"INVOICE":[]})
        #             if "eq_scores" not in st.session_state:
        #                 st.session_state.eq_scores = pd.DataFrame(
        #                     {"Equipment": [], "Quantity":[],"Hours": [], "TOTAL COST":[],"Mark UP":[],"EQUIPMENT INVOICE":[]})
        #             if "maint_scores" not in st.session_state:
        #                 st.session_state.maint_scores = pd.DataFrame(
        #                     {"Quantity":[2],"Hours": [8], "TOTAL COST":[1272],"Mark UP":[381.6],"MAINTENANCE INVOICE":[1653.6]})
        #             if "maint" not in st.session_state:
        #                 st.session_state.maint=False
                    
        #             ref={"DAY":["1ST","1OT"],"NIGHT":["2ST","2OT"],"WEEKEND":["2OT","2OT"],"HOOT":["3ST","3OT"]}
                    
        #             def equip_scores():
        #                 equipment=st.session_state.equipment
        #                 equipment_qty=st.session_state.eqqty
        #                 equipment_hrs=st.session_state.eqhrs
        #                 equipment_cost=equipment_qty*equipment_hrs*equipment_tariff[equipment]
        #                 equipment_markup=equipment_cost*st.session_state.markup/100
        #                 eq_score=pd.DataFrame({ "Equipment": [equipment],
        #                         "Quantity": [equipment_qty],
        #                         "Hours": [equipment_hrs*equipment_qty],
        #                         "TOTAL COST":equipment_cost,
        #                         "Mark UP":[round(equipment_markup,2)],
        #                         "EQUIPMENT INVOICE":[round(equipment_cost+equipment_markup,2)]})
        #                 st.session_state.eq_scores = pd.concat([st.session_state.eq_scores, eq_score], ignore_index=True)
                   
        #             def new_scores():
                        
        #                 if num_code=='0129':
        #                     foreman=True
        #                 else:
        #                     foreman=False
                        
        #                 pension=pma_rates[pma_year]["LS_401k"]
        #                 if foreman:
        #                     pension=pma_rates[pma_year]["Foreman_401k"]
                                     
                        
        #                 qty=st.session_state.qty
        #                 total_hours=st.session_state.hours+st.session_state.ot
        #                 hour_cost=st.session_state.hours*occ_codes.loc[st.session_state.code,ref[st.session_state.shift][0]]
        #                 ot_cost=st.session_state.ot*occ_codes.loc[st.session_state.code,ref[st.session_state.shift][1]]
        #                 wage_cost=hour_cost+ot_cost
        #                 benefits=wage_cost*0.062+wage_cost*0.0145+wage_cost*0.0021792                 #+wage_cost*st.session_state.siu/100
                        
        #                 assessments=total_hours*pma_rates[pma_year]["Cargo_Dues"]+total_hours*pma_rates[pma_year]["Electronic_Input"]+total_hours*pma_rates[pma_year]["Benefits"]+total_hours*pension
        #                 total_cost=wage_cost+benefits+assessments
        #                 siu_choice=wage_cost*st.session_state.siu/100
        #                 ind_ins=total_hours*1.5
        #                 with_siu=total_cost+siu_choice
        #                 #markup=with_siu*st.session_state.markup/100   ##+benefits*st.session_state.markup/100+assessments*st.session_state.markup/100
        #                 markup=total_cost*st.session_state.markup/100
        #                 if foreman:
        #                     markup=total_cost*st.session_state.f_markup/100  ###+benefits*st.session_state.f_markup/100+assessments*st.session_state.f_markup/100
        #                 invoice=total_cost+siu_choice+ind_ins+markup
                        
                        
        #                 new_score = pd.DataFrame(
        #                     {
        #                         "Code": [st.session_state.code],
        #                         "Shift": [st.session_state.shift],
        #                         "Quantity": [st.session_state.qty],
        #                         "Hours": [st.session_state.hours*qty],
        #                         "OT": [st.session_state.ot*qty],
        #                         "Hour Cost": [hour_cost*qty],
        #                         "OT Cost": [ot_cost*qty],
        #                         "Total Wage": [round(wage_cost*qty,2)],
        #                         "Benefits":[round(benefits*qty,2)],
        #                         "PMA Assessments":[round(assessments*qty,2)],
        #                         "TOTAL COST":[round(total_cost*qty,2)],
        #                         "Ind Ins":[round(ind_ins*qty,2)],
        #                         "SIU":[round(siu_choice*qty,2)],
        #                         "Mark UP":[round(markup*qty,2)],
        #                         "INVOICE":[round(invoice*qty,2)]
                                
        #                     }
        #                 )
        #                 st.session_state.scores = pd.concat([st.session_state.scores, new_score], ignore_index=True)
                        
                        
                 
                    
                        
        #             # Form for adding a new score
                    
        #             with st.form("new_score_form"):
        #                 st.write("##### LABOR")
        #                 form_col1,form_col2,form_col3=st.columns([3,3,4])
        #                 with form_col1:
                            
        #                     st.session_state.siu=st.number_input("ENTER SIU (UNEMPLOYMENT) PERCENTAGE",step=1,key="kdsha")
        #                     st.session_state.markup=st.number_input("ENTER MARKUP",step=1,key="wer")
        #                     st.session_state.f_markup=st.number_input("ENTER FOREMAN MARKUP",step=1,key="wfder")
                            
        #                 with form_col2:
        #                     st.session_state.shift=st.selectbox("SELECT SHIFT",["DAY","NIGHT","WEEKEND DAY","WEEKEND NIGHT","HOOT"])
        #                     st.session_state.shift_record=st.session_state.shift
        #                     st.session_state.shift="WEEKEND" if st.session_state.shift in ["WEEKEND DAY","WEEKEND NIGHT"] else st.session_state.shift
                        
        #                     # Dropdown for selecting Code
        #                     st.session_state.code = st.selectbox(
        #                         "Occupation Code", options=list(shortened_occ_codes.index)
        #                     )
        #                     # Number input for Quantity
        #                     st.session_state.qty = st.number_input(
        #                         "Quantity", step=1, value=0, min_value=0
        #                 )
        #                 with form_col3:
                            
        #                     # Number input for Hours
        #                     st.session_state.hours = st.number_input(
        #                         "Hours", step=0.5, value=0.0, min_value=0.0
        #                     )
                        
        #                     # Number input for OT
        #                     st.session_state.ot = st.number_input(
        #                         "OT", step=0.5, value=0.0, min_value=0.0
        #                     )
                            
        #                     # Form submit button
        #                     submitted = st.form_submit_button("Submit")
        #                 # If form is submitted, add the new score
                    
        #             if submitted:
        #                 num_code=st.session_state.code[1].strip()
        #                 new_scores()
        #                 st.success("Rank added successfully!")
                    
                    
        #             with st.form("equipment_form"):
        #                 st.write("##### EQUIPMENT")
        #                 eqform_col1,eqform_col2,eqform_col3=st.columns([3,3,4])
        #                 with eqform_col1: 
        #                     st.session_state.equipment = st.selectbox(
        #                         "Equipment", options=["CRANE","FORKLIFT","TRACTOR","KOMATSU","GENIE MANLIFT","Z135 MANLIFT"],key="sds11")
        #                 with eqform_col2:
        #                     # Number input for Equipment Quantity
        #                     st.session_state.eqqty = st.number_input(
        #                         "Equipment Quantity", key="sds",step=1, value=0, min_value=0)
        #                 with eqform_col3:
        #                     st.session_state.eqhrs = st.number_input(
        #                         "Equipment Hours",key="sdsss", step=1, value=0, min_value=0)
        #                     eq_submitted = st.form_submit_button("Submit Equipment")
        #             if eq_submitted:
        #                 equip_scores()
        #                 st.success("Equipment added successfully!")
                    
                        
        #             with st.container(border=True):
                        
        #                 sub_col1,sub_col2,sub_col3=st.columns([3,3,4])
        #                 with sub_col1:
        #                     pass
        #                 with sub_col2:
        #                     template_check=st.checkbox("LOAD FROM TEMPLATE")
        #                     if template_check:
        #                         with sub_col3:
        #                             template_choice_valid=False
        #                             template_choice=st.selectbox("Select Recorded Template",["Pick From List"]+[i for i in list_files_in_subfolder(target_bucket, rf"labor_templates/")],
        #                                                           label_visibility="collapsed")
        #                             if template_choice!="Pick From List":
        #                                 template_choice_valid=True 
        #                             if template_choice_valid:
        #                                 loaded_template=gcp_csv_to_df(target_bucket,rf"labor_templates/{template_choice}")
                                
                                
                       
                   
        #                 display=pd.DataFrame(st.session_state.scores)
        #                 display.loc["TOTAL FOR SHIFT"]=display[["Quantity","Hours","OT","Hour Cost","OT Cost","Total Wage","Benefits","PMA Assessments","TOTAL COST","Ind Ins","SIU","Mark UP","INVOICE"]].sum()
        #                 display=display[["Code","Shift","Quantity","Hours","OT","Hour Cost","OT Cost","Total Wage","Benefits","PMA Assessments","TOTAL COST","Ind Ins","SIU","Mark UP","INVOICE"]]
        #                 display.rename(columns={"SIU":f"%{st.session_state.siu} SIU"},inplace=True)
        #                 eq_display=pd.DataFrame(st.session_state.eq_scores)
        #                 eq_display.loc["TOTAL FOR SHIFT"]=eq_display[[ "Quantity","Hours","TOTAL COST","Mark UP","EQUIPMENT INVOICE"]].sum()
        #                 if template_check and template_choice_valid:
        #                     st.dataframe(loaded_template)
        #                 else:
        #                     st.write("##### LABOR")
        #                     st.dataframe(display)
        #                     part1,part2=st.columns([5,5])
        #                     with part1:
        #                         st.write("##### EQUIPMENT")
        #                         st.dataframe(eq_display)
        #                     maint1,maint2,maint3=st.columns([2,2,6])
        #                     st.session_state.maint=False
        #                     with maint1:
        #                         st.write("##### MAINTENANCE (IF NIGHT/WEEKEND SHIFT)")
        #                     with maint2:
        #                         maint=st.checkbox("Check to add maint crew")
        #                     if maint:
        #                         st.session_state.maint=True
        #                         st.dataframe(st.session_state.maint_scores)
        #                     else:
        #                         st.session_state.maint=False
        #                     with part2:
        #                         subpart1,subpart2,subpart3=st.columns([3,3,4])
        #                         with subpart1:
        #                             with st.container(border=True):
        #                                 st.write(f"###### COSTS")
        #                                 st.write(f"###### LABOR: {round(display.loc['TOTAL FOR SHIFT','TOTAL COST'],2)}")
        #                                 st.write(f"###### EQUIPMENT: {round(eq_display.loc['TOTAL FOR SHIFT','TOTAL COST'],2)}")
        #                                 if st.session_state.maint:
        #                                     st.write(f"###### MAINTENANCE: {round(st.session_state.maint_scores['TOTAL COST'].values[0],2)}")
        #                                     st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','TOTAL COST']+eq_display.loc['TOTAL FOR SHIFT','TOTAL COST']+st.session_state.maint_scores['TOTAL COST'].values[0],2)}")
        #                                 else:
        #                                     st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','TOTAL COST']+eq_display.loc['TOTAL FOR SHIFT','TOTAL COST'],2)}")

        #                         with subpart2:
        #                             with st.container(border=True):
        #                                 st.write(f"###### MARKUPS")
        #                                 st.write(f"###### LABOR: {round(display.loc['TOTAL FOR SHIFT','Mark UP'],2)}")
        #                                 st.write(f"###### EQUIPMENT: {round(eq_display.loc['TOTAL FOR SHIFT','Mark UP'],2)}")
        #                                 if st.session_state.maint:
        #                                     st.write(f"###### MAINTENANCE: {round(st.session_state.maint_scores['Mark UP'].values[0],2)}")
        #                                     st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','Mark UP']+eq_display.loc['TOTAL FOR SHIFT','Mark UP']+st.session_state.maint_scores['Mark UP'].values[0],2)}")
        #                                 else:
        #                                     st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','Mark UP']+eq_display.loc['TOTAL FOR SHIFT','Mark UP'],2)}")

        #                         with subpart3:
        #                             with st.container(border=True):
        #                                 st.write(f"###### TOTALS")
        #                                 st.write(f"###### TOTAL LABOR: {round(display.loc['TOTAL FOR SHIFT','INVOICE'],2)}")
        #                                 st.write(f"###### TOTAL EQUIPMENT: {round(eq_display.loc['TOTAL FOR SHIFT','EQUIPMENT INVOICE'],2)}")
        #                                 if st.session_state.maint:
        #                                     st.write(f"###### TOTAL MAINTENANCE: {round(st.session_state.maint_scores['MAINTENANCE INVOICE'].values[0],2)}")
        #                                     st.write(f"##### TOTAL INVOICE: {round(display.loc['TOTAL FOR SHIFT','INVOICE']+eq_display.loc['TOTAL FOR SHIFT','EQUIPMENT INVOICE']+st.session_state.maint_scores['MAINTENANCE INVOICE'].values[0],2)}")
        #                                 else:
        #                                     st.write(f"##### TOTAL INVOICE: {round(display.loc['TOTAL FOR SHIFT','INVOICE']+eq_display.loc['TOTAL FOR SHIFT','EQUIPMENT INVOICE'],2)}")
                                                
                            
                                
                           
        #             clear1,clear2,clear3=st.columns([2,2,4])
        #             with clear1:
        #                 if st.button("CLEAR LABOR TABLE"):
        #                     try:
        #                         st.session_state.scores = pd.DataFrame(
        #                         {"Code": [], "Shift":[],"Quantity": [], "Hours": [], "OT": [],"Hour Cost":[],"OT Cost":[],
        #                          "Total Wage":[],"Benefits":[],"PMA Assessments":[],"SIU":[],"TOTAL COST":[],"Mark UP":[],"INVOICE":[]})
        #                         st.rerun()
        #                     except:
        #                         pass
        #             with clear2:
        #                 if st.button("CLEAR EQUIPMENT TABLE",key="54332dca"):
        #                     try:
        #                        st.session_state.eq_scores = pd.DataFrame({"Equipment": [], "Quantity":[],"Hours": [], "TOTAL COST":[],"Mark UP":[],"EQUIPMENT INVOICE":[]})
        #                        st.rerun()
        #                     except:
        #                         pass
                    
        #             csv=convert_df(display)
        #             file_name=f'Gang_Cost_Report-{datetime.datetime.strftime(datetime.datetime.now(),"%m-%d,%Y")}.csv'
        #             down_col1,down_col2,down_col3,down_col4=st.columns([2,2,2,4])
        #             with down_col1:
        #                 #st.write(" ")
        #                 filename=st.text_input("Name the Template",key="7dr3")
        #                 template=st.button("SAVE AS TEMPLATE",key="srfqw")
        #                 if template:
        #                     temp=display.to_csv(index=False)
        #                     storage_client = get_storage_client()
        #                     bucket = storage_client.bucket(target_bucket)
                            
        #                     # Upload CSV string to GCS
        #                     blob = bucket.blob(rf"labor_templates/{filename}.csv")
        #                     blob.upload_from_string(temp, content_type="text/csv")
        #             with down_col2:
        #                 mt_jobs_=gcp_download(target_bucket,rf"LABOR/mt_jobs.json")
        #                 mt_jobs=json.loads(mt_jobs_)
        #                 #st.write(st.session_state.scores.T.to_dict())
        #                 job_no=st.selectbox("SELECT JOB NO",[i for i in mt_jobs["2023"]])
        #                 year="2023"
        #                 work_type=st.selectbox("SELECT JOB NO",["DOCK","WAREHOUSE","LINES"])
        #                 work_date=st.date_input("Work Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="work_date")
        #                 record=st.button("RECORD TO JOB",key="srfqwdsd")
        #                 if record:
                            
        #                     if year not in mt_jobs:
        #                         mt_jobs[year]={}
        #                     if job_no not in mt_jobs[year]:
        #                         mt_jobs[year][job_no]={}
        #                     if "RECORDS" not in mt_jobs[year][job_no]:
        #                         mt_jobs[year][job_no]["RECORDS"]={}
        #                     if str(work_date) not in mt_jobs[year][job_no]["RECORDS"]:
        #                         mt_jobs[year][job_no]["RECORDS"][str(work_date)]={}
        #                     if st.session_state.shift_record not in mt_jobs[year][job_no]["RECORDS"][str(work_date)]:
        #                         mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]={}
        #                     if "LABOR" not in mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]:
        #                         mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]["LABOR"]={"DOCK":{},"LINES":{},"WAREHOUSE":{}}
        #                     if 'EQUIPMENT' not in mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]:
        #                         mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]["EQUIPMENT"]={"DOCK":{},"LINES":{},"WAREHOUSE":{}}
        #                     if 'MAINTENANCE' not in mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]:
        #                         mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]["MAINTENANCE"]={"DOCK":{},"LINES":{},"WAREHOUSE":{}}
        #                     mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]['LABOR'][work_type]=st.session_state.scores.T.to_dict()
        #                     mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]['EQUIPMENT'][work_type]=st.session_state.eq_scores.T.to_dict()
        #                     if st.session_state.maint:
        #                         mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]['MAINTENANCE'][work_type]=st.session_state.maint_scores.T.to_dict()
        #                     mt_jobs_=json.dumps(mt_jobs)
        #                     storage_client = get_storage_client()
        #                     bucket = storage_client.bucket(target_bucket)
        #                     blob = bucket.blob(rf"mt_jobs.json")
        #                     blob.upload_from_string(mt_jobs_)
        #                     st.success(f"RECORDED JOB NO {job_no} ! ")
                        
                       
                        
                                               
                    
        #             index=st.number_input("Enter Index To Delete",step=1,key="1224aa")
        #             if st.button("DELETE BY INDEX"):
        #                 try:
        #                     st.session_state.scores=st.session_state.scores.drop(index)
        #                     st.session_state.scores.reset_index(drop=True,inplace=True)
        #                 except:
        #                     pass      
        if select=="ADMIN" :

            # conn = st.connection('gcs', type=FilesConnection)
            # a = conn.read(f"new_suzano/map.json", ttl=600)
            #st.write(a)
            admin_tab1,admin_tab2,admin_tab3,admin_tab4,admin_tab5=st.tabs(["RELEASE ORDERS","BILL OF LADINGS","EDI'S","AUDIT","VESSEL/MILL/CARRIER ENTRY"])
            
            with admin_tab5:   ###   Vessel Entry
                vessel=st.text_input("ENTER VESSEL NAME/VOY",key="dsd1eqx")
                uploaded_shipping_file = st.file_uploader("Upload the Shipping EDI txt file", type="txt")
                if uploaded_shipping_file:
                    def parse_edi_line(edi_line):
                        fields = {
                            'type': (0, 4),
                            'lot_number': (5, 15),
                            'grade': (15, 24),
                            'remarks': (24, 71),
                            'vessel_name': (61, 91),
                            'voyage_number': (91, 101),
                            'date': (101, 109),
                            'quantity': (109, 116),
                            'batch': (116, 129),
                            'ocean_bill_of_lading': (129, 179),
                            'admt': (179, 195)
                        }
                        
                        # Extract fields using slicing
                        parsed_data = {}
                        for field, (start, end) in fields.items():
                            if field=='quantity':
                                parsed_data[field] = int(edi_line[start:end].strip())
                            else:
                                parsed_data[field] = edi_line[start:end].strip()
                    
                        return parsed_data
                    tons=0
                    bols={}
                    #with open(uploaded_shipping_file, 'r') as infile:
                    for line in uploaded_shipping_file:
                        # Split the line into components based on space or another delimiter
                        line = line.decode('utf-8')
                        components = line.strip()  # Change split argument if using another delimiter
                        #st.write(parse_edi_line(line))
                        #st.write(components)
                        if components[0]=="2":
                            #st.write(components)
                            data=parse_edi_line(components)
                            tons+=data['quantity']
                            if data['ocean_bill_of_lading'] not in bols:
                                bols[data['ocean_bill_of_lading']]={}
                                bols[data['ocean_bill_of_lading']]['grade']=None
                                bols[data['ocean_bill_of_lading']]['grade']=data['grade']
                                bols[data['ocean_bill_of_lading']]['batch']=data['batch'].lstrip('0')
                                bols[data['ocean_bill_of_lading']]['qty']=0
                                bols[data['ocean_bill_of_lading']]['qty']+=data['quantity']/2
                                bols[data['ocean_bill_of_lading']]['admt']=float(data['admt'].lstrip("0"))/1000
                                bols[data['ocean_bill_of_lading']]['lots']=[]
                                bols[data['ocean_bill_of_lading']]['lots'].append(data['lot_number'])
                            else:
                                bols[data['ocean_bill_of_lading']]['qty']+=data['quantity']/2
                                bols[data['ocean_bill_of_lading']]['lots'].append(data['lot_number'])
                    bols_df=pd.DataFrame(bols).T
                    bols_df.columns=[["GRADE","BATCH","QUANTITY","DRYNESS","LOTS"]]
                    st.write(bols_df)
                    if st.button("REGISTER VESSEL AND LOTS",key="ddsaa"):
                        map_vessel=gcp_download(target_bucket,rf"map.json")
                        map_vessel=json.loads(map_vessel)
                        map_vessel['batch_mapping'][vessel]={i:{'batch':bols[i]['batch'],'dryness':bols[i]['admt'],
                                      'grade':bols[i]['grade'][:3],'fit':bols[i]['qty'] ,"damaged":0,"total":bols[i]['qty']} for i in bols}

                        for i in bols:
                            map_vessel['bol_mapping'][i]={'batch':bols[i]['batch'],'dryness':bols[i]['admt'],
                                                              'grade':bols[i]['grade'][:3],'fit':bols[i]['qty'] ,"damaged":0,"total":bols[i]['qty'],
                                                        'FSC':"FSC Certified Products. FSC Mix Credit IMA—COC-001470"} 
                        #st.write(map_vessel)
                        bill_mapping_vessel=gcp_download(target_bucket,rf"bill_mapping.json")
                        bill_mapping_vessel=json.loads(bill_mapping_vessel)
                        if vessel not in bill_mapping_vessel:
                            bill_mapping_vessel[vessel]={}
                        for bill,item in bols.items():
                            for i in item['lots']:
                                bill_mapping_vessel[vessel][i]={'Batch':bols[bill]['batch'],'Ocean_bl':bill}
                        storage_client = get_storage_client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"map.json")
                        blob.upload_from_string(json.dumps(map_vessel))
                        storage_client = get_storage_client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"bill_mapping.json")
                        blob.upload_from_string(json.dumps(bill_mapping_vessel))
                        st.success(f"Vessel and BOL data is registered")
                
            
            with admin_tab4:   ###   AUDIT
                if st.button("RUN RECORD AUDIT"):
                    
                    dfb=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                    dfb=json.loads(dfb)
                    dfb=pd.DataFrame.from_dict(dfb).T[1:]
                    suz=gcp_download(target_bucket,rf"suzano_report.json")
                    suz=json.loads(suz)
                    suz_frame=pd.DataFrame(suz).T
                    raw_ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                    raw_ro=json.loads(raw_ro)
                    def dict_compare(d1, d2):
                        d1_keys = set(d1.keys())
                        d2_keys = set(d2.keys())
                        shared_keys = d1_keys.intersection(d2_keys)
                        added = d1_keys - d2_keys
                        removed = d2_keys - d1_keys
                        modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
                        same = set(o for o in shared_keys if d1[o] == d2[o])
                        return added, removed, modified, same
                    
                    def extract_bol_shipped(data,bol):
                        qt=0
                        sales_group=["001","002","003","004","005"]
                        for ro in data:
                            for sale in data[ro]:
                                if sale in sales_group and data[ro][sale]['ocean_bill_of_lading']==bol:
                                    qt+=data[ro][sale]['shipped']
                        return qt
                    def compare_dict(a, b):
                        # Compared two dictionaries..
                        # Posts things that are not equal..
                        res_compare = []
                        for k in set(list(a.keys()) + list(b.keys())):
                            if isinstance(a[k], dict):
                                z0 = compare_dict(a[k], b[k])
                            else:
                                z0 = a[k] == b[k]
                    
                            z0_bool = np.all(z0)
                            res_compare.append(z0_bool)
                            if not z0_bool:
                                if a==rel_t:
                                    st.markdown(f"**:red[Discrepancy]**")
                                    st.markdown(f"**{k} - Inventory :{a[k]} Units Shipped - BOL Report : {b[k]} Units Shipped**")
                                    
                                else:
                                    st.markdown(f"**:red[Discrepancy]**")
                                    st.markdown(f"**{k} - Suzano Report :{a[k]} Units Shipped - BOL Report : {b[k]} Units Shipped**")
                                    diff = dfb[~dfb.index.isin([str(i) for i in suz_frame['Shipment ID #']]+['11502400', '11503345', 'MF01769573*', '11503871'])].index.to_list()
                                    for i in diff:
                                        st.markdown(f"**...Shipment {i} to {dfb.loc[i,'destination']} is in BOL but not Suzano Report**") 
                        return np.all(res_compare)
               
                    suz_frame["Quantity"]=[float(i) for i in suz_frame["Quantity"]]
                    suz_t=suz_frame.groupby("Ocean BOL#")["Quantity"].sum().to_dict()
                    df_t=dfb.groupby("ocean_bill_of_lading")["quantity"].sum().to_dict()
                    #corrections due to shipment MF01769573 and 1150344 on 12-15 between Kirkenes and Juventas mixed loads.
                    suz_t['GSSWKIR6013E']=suz_t['GSSWKIR6013E']+7
                    suz_t['GSSWKIR6013D']=suz_t['GSSWKIR6013D']+9
                    suz_t['GSSWJUV8556C']=suz_t['GSSWJUV8556C']-9
                    suz_t['GSSWJUV8556A']=suz_t['GSSWJUV8556A']-7
                    
                    rel_t={i:extract_bol_shipped(raw_ro,i) for i in suz_t}
                    aud_col1,aud_col2=st.columns(2)
                    with aud_col1:
                        
                        if compare_dict(suz_t,df_t):
                            st.markdown("**:blue[All Checks Complete !]**")
                            st.markdown("**:blue[SUZANO REPORT match BILL OF LADINGS]**")       
                            st.write(f"{len(suz_frame)} Shipments")
                           

                    with aud_col2:
                        
                        if compare_dict(rel_t,df_t):
                            st.markdown("**:blue[All Checks Complete !]**")
                            st.markdown("**:blue[INVENTORY match BILL OF LADINGS]**")       
                            st.write(f"{len(suz_frame)} Shipments")
                edi_audit=st.toggle("AUDIT TODAYS EDIs AND REPORTS")
                if edi_audit:
                    
                    with st.spinner("Wait for it"):
                        guilty=None
                        def list_files_uploaded_today(bucket_name, folder_name):
                        # Initialize Google Cloud Storage client
                            storage_client = get_storage_client()
                        
                            # Get the current date
                            today = (datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()
                            # Get the list of blobs in the specified folder
                            blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)
                        
                            # Extract filenames uploaded today only
                            filenames = []
                            for blob in blobs:
                                # Check if blob's last modification date is today
                                if blob.updated.date() == today:
                                    # Extract only the filenames without the folder path
                                    filename = blob.name.split("/")[-1]
                                    filenames.append(filename)
                                time.sleep(0.001)
                        
                            return filenames
                        today_uploaded_files = list_files_uploaded_today(target_bucket,rf"EDIS/")
                        st.markdown(f"**# of EDIs Uploaded : {len(today_uploaded_files)}**")
                        st.markdown(f"EDIs Uploaded Today : {today_uploaded_files}")
                       
                        
                        base=[]
                        for i in today_uploaded_files:
                            
                            lines=gcp_download(target_bucket, rf"EDIS/{i}").splitlines()
                            
                            
                            # Step 2: Process the contents
                            data = []
                            count=1
                            line_count=0
                            dev_count=0
                            line_tonnage=0
                            unit_count=0
                            for line in lines:
                                if line.startswith("1HDR"):
                                    prefix, data = line.split(':') 
                                    assert prefix=="1HDR" , "Prefix does not match the expected value '1HDR'"
                                    date_str = data[:8]  # YYYYMMDD
                                    time_str = data[8:14]  # HHMMSS
                                    terminal_code = data[14:18]  # 4 letters
                                    datetime_obj = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
                                    line_count+=1
                                elif line.startswith("2DTD"):
                                    prefix, data = line.split(':') 
                                    release_order = data[:10].strip() 
                                    sales_item = data[10:16].strip() 
                                    date=data[16:24].strip()
                                    date = datetime.datetime.strptime(date, '%Y%m%d').date()
                                    transport_type=data[24:26].strip()
                                    transport_sequential=data[26:30].strip()
                                    vehicle_id=data[30:50].strip()
                                    total_tonnage=int(data[50:66].strip())
                                    carrier_code=data[105:115].strip()
                                    bill_of_lading=data[115:165].strip()
                                    eta_date=data[165:].strip()
                                    eta_date = datetime.datetime.strptime(eta_date, '%Y%m%d').date()
                                    line_count+=1
                                elif line.startswith("2DEV"):
                                    prefix, data = line.split(':')
                                    line_release_order=data[:10]
                                    line_sales_item=data[10:16].strip()
                                    line_date=data[16:24].strip()
                                    line_date = datetime.datetime.strptime(line_date, '%Y%m%d').date()
                                    transport_type=data[24:26].strip()
                                    lot_number=data[26:36].strip()
                                    line_weight=int(data[51:56].strip())
                                    line_unit_count=line_weight/2000
                                    line_tonnage+=line_weight
                                    unit_count+=line_unit_count
                                    dev_count+=1
                                    line_count+=1
                        
                                elif line.startswith("9TRL"):
                                    prefix, data = line.split(':')
                                    edi_line_count=int(data[:4])
                                    line_count+=1
                                    assert edi_line_count==line_count,f"no, line_count is {line_count}"
                                    assert line_tonnage==total_tonnage
                            base.append({'Date Shipped':datetime_obj, 'Vehicle':vehicle_id, 'Shipment ID #':bill_of_lading, 'Release #':release_order,
                                             'Carrier':carrier_code, 'Quantity':unit_count, 'Metric Ton':total_tonnage/1000})
                        
                        edis=pd.DataFrame(base)
                        edis=edis.sort_values(by="Date Shipped")
                        edis.set_index("Shipment ID #",drop=True,inplace=True)
                        
                        suz=gcp_download(target_bucket,rf"suzano_report.json")
                        suz=json.loads(suz)
                        suz_frame=pd.DataFrame(suz).T
                        suz_frame["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in suz_frame["Date Shipped"]]
                        suz_frame_daily=suz_frame[suz_frame.Date==(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()]
                        suz_frame_daily=suz_frame_daily[['Date Shipped', 'Vehicle', 'Shipment ID #', 'Release #', 'Carrier',
                                      'Quantity', 'Metric Ton',]]
                        suz_frame_daily.set_index("Shipment ID #",drop=True,inplace=True)
                        # suz_frame_daily.loc["MF01799999"]={"Date Shipped":"2024-03-28 12:26:58","Vehicle":"3423C",
                        #                                            "Shipment ID #":"MF01799420","Release #":"3172295",
                        #                                        "Carrier":"123456","Quantity":14.0,"Metric Ton":28.0}
                        more=None
                        if len(edis)!=len(suz_frame_daily):
                            diff=abs(len(edis)-len(suz_frame_daily))
                            if len(edis)<len(suz_frame_daily):
                                more=suz_frame_daily.copy()
                                guilty="edis"
                                for i in range(diff):
                                    edis.loc[len(edis)]=None
                            else:
                                more=edis.copy()
                                guilty="suz_frame_daily"
                        
                        
                        difference = (edis.index!=suz_frame_daily.index)
                        
                        
                        
                        if guilty=="edis":
                            more=more[difference]
                            st.markdown("**:red[Following Shipment from Suzano Report is Missing an EDI]**")
                            st.write(more)
                        elif guilty=="suz_frame_daily":
                            more=more[difference]
                            st.markdown("**:red[Following EDI is Missing in Suzano Report]**")
                            st.write(more)
                        else:
                            st.success("All EDIs and Suzano Report Entries are accounted for!! ")  
                            
            with admin_tab2:   #### BILL OF LADINGS
                bill_data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                bill_data_reverse=json.loads(bill_data)
                admin_bill_of_ladings=pd.DataFrame.from_dict(bill_data_reverse).T[1:]
                admin_bill_of_ladings["St_Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in admin_bill_of_ladings["issued"]]
                release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                release_order_database=json.loads(release_order_database)
                suzano_report=gcp_download(target_bucket,rf"suzano_report.json")
                suzano_report=json.loads(suzano_report)
                try:
                    voided_shipments=gcp_download(target_bucket,rf"voided_shipments.json")
                    voided_shipments=json.loads(voided_shipments)
                except:
                    voided_shipments={}
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')
                use=True
                if use:
                    now=datetime.datetime.now()-datetime.timedelta(hours=utc_difference)
                    
                    choose = st.radio(
                                    "Select Today's Bill of Ladings or choose by Date or choose ALL",
                                    ["DAILY", "ACCUMULATIVE", "FIND BY DATE"],key="wewas")
                    if choose=="DAILY":
                        display_df=admin_bill_of_ladings[admin_bill_of_ladings["St_Date"]==now.date()]
                        st.dataframe(display_df)
                        file_name=f'OLYMPIA_DAILY_BILL_OF_LADINGS-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                    elif choose=="FIND BY DATE":
                        required_date=st.date_input("CHOOSE DATE",key="dssar")
                        display_df=admin_bill_of_ladings[admin_bill_of_ladings["St_Date"]==required_date]
                        st.dataframe(display_df)
                        file_name=f'OLYMPIA_BILL_OF_LADINGS_FOR-{datetime.datetime.strftime(required_date,"%m-%d,%Y")}.csv'
                    else:
                        display_df=admin_bill_of_ladings
                        st.write("DATA TOO LARGE, DOWNLOAD INSTEAD")
                        file_name=f'OLYMPIA_ALL_BILL_OF_LADINGS to {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                    csv=convert_df(display_df)
                    bilo1,bilo2,bilo3=st.columns([3,3,3])
                    with bilo1:
                        st.download_button(
                            label="DOWNLOAD BILL OF LADINGS",
                            data=csv,
                            file_name=file_name,
                            mime='text/csv')
                    with bilo2:
                        pass
                        # if display_df.shape[0]>0:
                        #     entry=st.selectbox("SELECT SHIPMENT TO CREATE THE EDI", [i for i in display_df.index])
                        #     entry=display_df.loc[entry].to_dict()
                        #     terminal_bill_of_lading=entry["edi_no"].split(".")[0]
                        #     def make_edi(entry):
                        #         loads={}
                        #         for load in entry['loads']:
                        #             load_=load[:-3]
                        #             if load_ not in loads:
                        #                 loads[load_]=0
                        #                 loads[load_]=entry["loads"][load]
                        #             else:
                        #                 loads[load_]+=entry["loads"][load]
                        #         double_load=False
                        #         terminal_bill_of_lading=entry["edi_no"].split(".")[0]
                        #         a=datetime.datetime.strptime(entry["issued"], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')#%H%M%S')
                        #         b=datetime.datetime.strptime(entry["issued"], '%Y-%m-%d %H:%M:%S').strftime('%H%M%S')
                                
                        #         line1="1HDR:"+a+b+"OLYM"
                        #         tsn="01" 
                        #         tt="0001"
                        #         # if double_load:
                        #         #     line21="2DTD:"+entry["release_order"]+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(first_quantity*2000)+" "*(16-len(str(first_quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
                        #         #     line22="2DTD:"+entry["release_order"]+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(second_quantity*2000)+" "*(16-len(str(second_quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
                        #         line2="2DTD:"+entry["release_order"]+" "*(10-len(entry["release_order"]))+"000"+entry["sales_order"]+a+tsn+tt+entry["vehicle"]+" "*(20-len(entry["vehicle"]))+str(int(entry["quantity"]*2000))+" "*(16-len(str(int(entry["quantity"]*2000))))+"USD"+" "*36+entry["carrier_id"]+" "*(10-len(str(entry["carrier_id"])))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+a
                            
                        #         loadls=[]
                        #         bale_loadls=[]
                        #         if double_load:
                        #             for i in first_textsplit:
                        #                 loadls.append("2DEV:"+current_release_order+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+i[:load_digit]+" "*(10-len(i[:load_digit]))+"0"*16+str(2000))
                        #             for k in second_textsplit:
                        #                 loadls.append("2DEV:"+next_release_order+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+k[:load_digit]+" "*(10-len(k[:load_digit]))+"0"*16+str(2000))
                        #         else:
                        #             for k in loads:
                            
                        #                 loadls.append("2DEV:"+entry["release_order"]+" "*(10-len(entry["release_order"]))+"000"+entry["sales_order"]+a+tsn+k+" "*(10-len(k))+"0"*(20-len(str(int(loads[k]*2000))))+str(int(loads[k]*2000)))
                            
                            
                        #         if double_load:
                        #             number_of_lines=len(first_textsplit)+len(second_textsplit)+4
                        #         else:
                        #             number_of_lines=len(loadls)+3
                        #         end_initial="0"*(4-len(str(number_of_lines)))
                        #         end=f"9TRL:{end_initial}{number_of_lines}"
                            
                        #         with open(f'{terminal_bill_of_lading}.txt', 'w') as f:
                        #             f.write(line1)
                        #             f.write('\n')
                        #             if double_load:
                        #                 f.write(line21)
                        #                 f.write('\n')
                        #                 f.write(line22)
                        #             else:
                        #                 f.write(line2)
                        #             f.write('\n')
                            
                        #             for i in loadls:
                            
                        #                 f.write(i)
                        #                 f.write('\n')
                            
                        #             f.write(end)
                        #         with open(f'{terminal_bill_of_lading}.txt', 'r') as f:
                        #             file_content=f.read()
                        #         file_name=f'{terminal_bill_of_lading}.txt'
                        #         return file_content,file_name
                        #     file_content,file_name=make_edi(entry)
                        #     st.download_button(
                        #         label="CREATE EDI FOR THIS SHIPMENT",
                        #         data=file_content,
                        #         file_name=file_name,
                        #         mime='text/csv',key="53432")
                    
                    with bilo3:
                        if display_df.shape[0]>0:
                            
                            to_reverse=st.selectbox("SELECT SHIPMENT TO VOID", [i if len(display_df)>0 else None for i in display_df.index ])
                            
                            if st.button("VOID SHIPMENT"):
                                voided_shipments[to_reverse]={}
                                voided_shipments[to_reverse]=display_df.loc[to_reverse].to_dict()
                                if to_reverse!=None:
                                    to_reverse_data=display_df.loc[to_reverse].to_dict()
                                    ro_to_reverse=to_reverse_data['release_order']
                                    so_to_reverse=to_reverse_data['sales_order']
                                    qty_to_reverse=to_reverse_data['quantity']
                                release_order_database[ro_to_reverse][so_to_reverse]['shipped']-=qty_to_reverse
                                release_order_database[ro_to_reverse][so_to_reverse]['remaining']+=qty_to_reverse
                                
                                
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                                blob.upload_from_string(json.dumps(release_order_database))
                                st.success(f"Release order {ro_to_reverse} updated with reversal!")
    
                                del bill_data_reverse[to_reverse]
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                                blob.upload_from_string(json.dumps(bill_data_reverse))
                                st.success(f"Terminal Bill of Ladings updated with reversal!")
    
                                suz_index=next(key for key, value in suzano_report.items() if value['Shipment ID #'] == to_reverse)
                                del suzano_report[suz_index]
                                suzano_report = {i + 1: v for i, (k, v) in enumerate(suzano_report.items())}
                                
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"suzano_report.json")
                                blob.upload_from_string(json.dumps(suzano_report))
                                st.success(f"Suzano Report updated with reversal!")

                                del voided_shipments[to_reverse]['St_Date']
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"voided_shipments.json")
                                blob.upload_from_string(json.dumps(voided_shipments))
                                st.success(f"Void recorded in voided shipments!")
    
                                if to_reverse[0]=="M":
                                    mf_numbers=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")
                                    mf_numbers=json.loads(mf_numbers)
                                    mf_numbers[ro_to_reverse].append(to_reverse)
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"release_orders/mf_numbers.json")
                                    blob.upload_from_string(json.dumps(mf_numbers))
                                    st.success(f"MF Numbers entered back into RO {ro_to_reverse}!")
                                try:
                                    delete_file_from_gcs(target_bucket, f'EDIS/{to_reverse}.txt')
                                    st.success(f"Deleted EDI {to_reverse}.txt!")
                                except:
                                    st.write("NO Edis found for this shipment")
                    to_print_loads=st.selectbox("SELECT SHIPMENT TO DISPLAY LOADS", [i if len(display_df)>0 else None for i in display_df.index ])
                            
                    if st.button("DISPLAY LOADS"):     
                        st.write([key for key in display_df.loc[to_print_loads]['loads']])

                    if st.button("DISPLAY VOIDED SHIPMENTS"):     
                        st.write(voided_shipments)


##  EDIS ###        
            with admin_tab3:
                edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                
                display_edi=st.toggle("DISPLAY EDI")
                if display_edi:
                    data=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}")
                    st.text_area("EDI",data,height=400)                                
               
                st.download_button(
                    label="DOWNLOAD EDI",
                    data=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}"),
                    file_name=f'{requested_edi_file}',
                    mime='text/csv')
               
                
                
                
                    
              
                  
            
                            

            
                          
            with admin_tab1:
                map=gcp_download(target_bucket,rf"map.json")
                map=json.loads(map)
                release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                release_order_database=json.loads(release_order_database)
                dispatch=gcp_download(target_bucket,rf"dispatched.json")
                dispatch=json.loads(dispatch)
                
                carrier_list=map['carriers']
                mill_info=map["mill_info"]
                      
                release_order_tab1,release_order_tab2,release_order_tab3=st.tabs(["RELEASE ORDER DATABASE","CREATE RELEASE ORDER","RELEASE ORDER STATUS"])
                
                with release_order_tab2:   ###CREATE RELEASE ORDER
                   
                    add=st.checkbox("CHECK TO ADD TO EXISTING RELEASE ORDER",disabled=False)
                    edit=st.checkbox("CHECK TO EDIT EXISTING RELEASE ORDER")
                    upload=st.checkbox("CHECK TO UPLOAD SUZANO RELEASE ORDER XLSX")
                    
                   
                    batch_mapping=map['batch_mapping']

                    if upload:
                        st.subheader("UPLOAD RELEASE ORDER")
                        release_order_upload=  st.file_uploader("Upload **SUZANO** Release Order", type="xlsx",key="dssds")
                        destinations = [ 'GP-Halsey,OR', 'GP-Clatskanie,OR',
                                        'KRUGER-New Westminster,BC', 'WILLAMETTE FALLS-West Linn,OR',
                                        'WILLAMETTE FALLS-Vancouver,WA', 'AHLSTROM-MUNKSJO-Kaukauna,WI',
                                        'AHLSTROM-MUNKSJO-De Pere,WI', 'CELLMARK-Gila Bend,AZ', 'BIORIGIN-Menominee,MI',
                                        'CATALYST-Surrey,BC', 'SOFIDEL-Lewiston,ID']
                                           
                        # Preprocessing function
                        def preprocess(text):
                            # Replace common terms and normalize
                            replacements = {
                                "GEORGIA-PACIFIC": "GP",
                                "WILLAMETTE FALLS": "WILLAMETTE",
                                "AHLSTROM-MUNKSJO": "AHLSTROM",
                                "NEW WESTMINSTER": "New Westminster",
                                # Add more rules as needed
                            }
                            for key, value in replacements.items():
                                text = text.replace(key, value)
                            return text.strip().upper()
                        
                        # Function to find the closest match
                        def find_closest_match(client_input, destinations):
                            # Preprocess both the input and destinations
                            processed_input = preprocess(client_input)
                            processed_destinations = [preprocess(dest) for dest in destinations]
                            # Find the best match
                            match = processs.extractOne(processed_input, processed_destinations)
                            # Return the original destination corresponding to the match
                            return destinations[processed_destinations.index(match[0])] if match else None
                        
                        # Find the closest match
                        if release_order_upload:
                            df=pd.read_excel(release_order_upload)
                            ro_payload = {}
                            required_columns = ["Order Base ID", "Order Base Line ID", "Destination City", "PO Number", "Weight"]
                            missing_columns = [col for col in required_columns if col not in df.columns]
                            
                            if missing_columns:
                                st.error(f"Missing columns in the uploaded file: {missing_columns}")
                            else:
                                # Proceed with processing
                                pass
    
                            for i in df.index:
                                # Extract data from the DataFrame
                                release_order_number_upload = str(df.loc[i, "Order Base ID"])
                                sales_order_item_upload = df.loc[i, "Order Base Line ID"][-3:]  # Extract the last 3 characters (e.g., "001")
                                destination = find_closest_match(df.loc[i, "Destination City"], destinations)
                            
                                # Ensure the release order exists in the dictionary
                                if release_order_number_upload not in ro_payload:
                                    ro_payload[release_order_number_upload] = {
                                        "po_number": str(df.loc[i, "PO Number"]),
                                        "destination": destination,
                                        "complete": False  # Set this based on logic if required
                                    }
                                cargo_prep="UNITIZED" if destination=='SOFIDEL-Lewiston,ID' else "DE-UNITIZED"

                                # Add or update the sales order item details directly under the release order number
                                ro_payload[release_order_number_upload][str(sales_order_item_upload)] = {
                                    "vessel": df.loc[i, "Vessel"],
                                    "batch": str(df.loc[i, "Batch"]),
                                    "ocean_bill_of_lading": df.loc[i, "Vessel BOL"],
                                    "grade": df.loc[i, "Grade"].split("-")[0],
                                    "dryness": str(df.loc[i, "Dryness"]),
                                    "unitized":cargo_prep,
                                    "total": int(df.loc[i, "Weight"]/2),               # Add relevant columns
                                    "shipped": 0,           # Add relevant columns
                                    "remaining": int(df.loc[i, "Weight"]/2),       # Add relevant columns
                                }
                            release_order_database.update(ro_payload)
                            st.write(ro_payload)
                            # Example Output
                            if st.button("UPLOAD RELEASE ORDER"):
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                                blob.upload_from_string(json.dumps(release_order_database))
                                st.success(f"UPLOADED {release_order_number_upload}!")

                    
                    elif edit:
                        
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in release_order_database]))
                        po_number_edit=st.text_input("PO No",release_order_database[release_order_number]["po_number"],disabled=False)
                        destination_edit=st.text_input("Destination",release_order_database[release_order_number]["destination"],disabled=False)
                        sales_order_item_edit=st.selectbox("Sales Order Item",[i for i in release_order_database[release_order_number] if i in ["001","002","003","004","005"]] ,disabled=False)
                        vessel_edit=vessel=st.selectbox("SELECT VESSEL",[i for i in map['batch_mapping']],key="poFpoa")
                        ocean_bill_of_lading_edit=st.selectbox("Ocean Bill Of Lading",batch_mapping[vessel_edit].keys(),key="trdfeerw") 
                        grade_edit=st.text_input("Grade",release_order_database[release_order_number][sales_order_item_edit]["grade"],disabled=True)
                        batch_edit=st.text_input("Batch No",release_order_database[release_order_number][sales_order_item_edit]["batch"],disabled=True)
                        dryness_edit=st.text_input("Dryness",release_order_database[release_order_number][sales_order_item_edit]["dryness"],disabled=True)
                        admt_edit=st.text_input("ADMT PER UNIT",round(float(batch_mapping[vessel_edit][ocean_bill_of_lading_edit]["dryness"])/90,6),True)
                        unitized_edit=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity_edit=st.number_input("Quantity of Units", 0, disabled=False, label_visibility="visible")
                        tonnage_edit=2*quantity_edit
                        shipped_edit=st.number_input("Shipped # of Units",release_order_database[release_order_number][sales_order_item_edit]["shipped"],disabled=True)
                        remaining_edit=st.number_input("Remaining # of Units",
                                                       quantity_edit-release_order_database[release_order_number][sales_order_item_edit]["shipped"],disabled=True)
                        carrier_code_edit=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()],key="dsfdssa")          
                        
                    elif add:
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in release_order_database]))
                        po_number_add=st.text_input("PO No",release_order_database[release_order_number]["po_number"],disabled=False)
                        destination_add=st.text_input("Destination",release_order_database[release_order_number]["destination"],disabled=False)
                        sales_order_item_add=st.text_input("Sales Order Item",disabled=False)
                        vessel_add=vessel=st.selectbox("SELECT VESSEL",[i for i in map['batch_mapping']],key="popoa")
                        ocean_bill_of_lading_add=st.selectbox("Ocean Bill Of Lading",batch_mapping[vessel_add].keys(),key="treerw")  
                        grade_add=st.text_input("Grade",batch_mapping[vessel_add][ocean_bill_of_lading_add]["grade"],disabled=True) 
                        batch_add=st.text_input("Batch No",batch_mapping[vessel_add][ocean_bill_of_lading_add]["batch"],disabled=False)
                        dryness_add=st.text_input("Dryness",batch_mapping[vessel_add][ocean_bill_of_lading_add]["dryness"],disabled=False)
                        admt_add=st.text_input("ADMT PER UNIT",round(float(batch_mapping[vessel_add][ocean_bill_of_lading_add]["dryness"])/90,6),disabled=False)
                        unitized_add=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity_add=st.number_input("Quantity of Units", 0, disabled=False, label_visibility="visible")
                        tonnage_add=2*quantity_add
                        shipped_add=0
                        remaining_add=st.number_input("Remaining # of Units", quantity_add,disabled=True)
                        carrier_code_add=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()])            
                    
                    else:  ### If creating new release order
                        
                        release_order_number=st.text_input("Release Order Number")
                        po_number=st.text_input("PO No")
                        destination_list=[i for i in mill_info.keys()]
                        #st.write(destination_list)
                        destination=st.selectbox("SELECT DESTINATION",destination_list)
                        sales_order_item=st.text_input("Sales Order Item")  
                        vessel=st.selectbox("SELECT VESSEL",[i for i in map['batch_mapping']],key="tre")
                        ocean_bill_of_lading=st.selectbox("Ocean Bill Of Lading",batch_mapping[vessel].keys())   #######
                        grade=st.text_input("Grade",batch_mapping[vessel][ocean_bill_of_lading]["grade"],disabled=True)   ##### batch mapping injection
                        batch=st.text_input("Batch No",batch_mapping[vessel][ocean_bill_of_lading]["batch"],disabled=True)   #####
                        dryness=st.text_input("Dryness",batch_mapping[vessel][ocean_bill_of_lading]["dryness"],disabled=True)   #####
                        admt=st.text_input("ADMT PER UNIT",round(float(batch_mapping[vessel][ocean_bill_of_lading]["dryness"])/90,6),disabled=True)  #####
                        unitized=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity=st.number_input("Quantity of Units", min_value=1, max_value=5000, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                        tonnage=2*quantity
                        carrier_code=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()])            
                    #if edit:
                    #    fsc_verified=st.checkbox(f"**VERIFY RELEASE ORDER FSC CERTIFICATE CODE MATCHES BATCH {batch_edit} STATEMENT : FSC  CERTIFIED PRODUCTS FSC MIX CREDIT SCS-COC-009938**",key="sasa")
                  #  else:
                    #    fsc_verified=st.checkbox(f"**VERIFY RELEASE ORDER FSC CERTIFICATE CODE MATCHES BATCH {batch} STATEMENT : FSC  CERTIFIED PRODUCTS FSC MIX CREDIT SCS-COC-009938**",key="sdasa")
                    create_release_order=st.button("SUBMIT")
                    fsc_verified=True
                    if create_release_order and fsc_verified:
                        
                        if add: 
                            temp=add_release_order_data(release_order_database,release_order_number,sales_order_item_add,vessel_add,batch_add,ocean_bill_of_lading_add,grade_add,dryness_add,carrier_code_add,unitized_add,quantity_add,shipped_add,remaining_add)
                            storage_client = get_storage_client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                            blob.upload_from_string(temp)
                            st.success(f"ADDED sales order item {sales_order_item_add} to release order {release_order_number}!")
                        elif edit:
                            temp=edit_release_order_data(release_order_database,release_order_number,destination_edit,po_number_edit,sales_order_item_edit,vessel_edit,batch_edit,ocean_bill_of_lading_edit,grade_edit,dryness_edit,carrier_code_edit,unitized_edit,quantity_edit,shipped_edit,remaining_edit)
                            storage_client = get_storage_client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                            blob.upload_from_string(temp)
                            st.success(f"Edited release order {release_order_number} successfully!")
                            
                        else:
                            temp=store_release_order_data(release_order_database,release_order_number,destination,po_number,sales_order_item,vessel,batch,ocean_bill_of_lading,grade,dryness,carrier_code,unitized,quantity)
                            storage_client = get_storage_client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                            blob.upload_from_string(temp)
                            st.success(f"Created release order {release_order_number} successfully!")
                        
             
                        
                with release_order_tab1:  ##   RELEASE ORDER DATABASE ##
                    
     
                    rls_tab1,rls_tab2,rls_tab3,rls_tab4=st.tabs(["ACTIVE RELEASE ORDERS","COMPLETED RELEASE ORDERS","SHIPMENT NUMBERS","SCHEDULE"])


### SCHEDULE ADMIN
                    # with rls_tab4:  #####  SCHEDULE
                    #     pass
                        # bill_for_schedule=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                        # bill_for_schedule=json.loads(bill_for_schedule)
                        # schedule=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
                        # schedule=json.loads(schedule)
                        # dfb=pd.DataFrame.from_dict(bill_for_schedule).T[1:]
                        # #dfb=bill.copy()
                        # dfb["St_Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in dfb["issued"]]
                        # #dfb=dfb[dfb["St_Date"]==(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()]
                        # scheduled=[]
                        # today=str((datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date())
                        # selected_date_datetime=st.date_input("SELECT DATE",(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date())
                        # selected_date=str(selected_date_datetime)
                        
                        # if selected_date in schedule:
                        #     dfb=dfb[dfb["St_Date"]==selected_date_datetime]
                        #     for dest in schedule[selected_date]:
                        #         for rel in schedule[selected_date][dest]:
                        #             for carrier in schedule[selected_date][dest][rel]:
                        #                 scheduled.append({"Destination":dest,
                        #                       "Release Order":rel,"Sales Item":"001",
                        #                       "ISP":release_order_database[rel]["001"]['grade'],
                        #                       "Prep":release_order_database[rel]["001"]['unitized'],
                        #                       "Carrier":carrier.split("-")[1],
                        #                       "Scheduled":int(len(schedule[selected_date][dest][rel][carrier])),
                        #                       "Loaded":int(dfb[(dfb["release_order"]==rel)&(dfb["sales_order"]=="001")&
                        #                        (dfb["carrier_id"]==str(carrier.split("-")[0]))].vehicle.count()),
                        #                       "Remaining":0})
                        #     scheduled=pd.DataFrame(scheduled)
                        #     scheduled["Scheduled"] = scheduled["Scheduled"].astype(int)
                        #     scheduled["Loaded"] = scheduled["Loaded"].astype(int)
                        #     scheduled["Remaining"] = scheduled["Remaining"].astype(int)
    
                        #     if len(scheduled)>0:
                                
                        #         scheduled["Remaining"]=[int(i) for i in scheduled["Scheduled"]-scheduled["Loaded"]]
                        #         scheduled.loc["Total",["Scheduled","Loaded","Remaining"]]=scheduled[["Scheduled","Loaded","Remaining"]].sum()
                        #         #scheduled.set_index('Destination',drop=True,inplace=True)
                        #         scheduled["Scheduled"] = scheduled["Scheduled"].astype(int)
                        #         scheduled["Loaded"] = scheduled["Loaded"].astype(int)
                        #         scheduled["Remaining"] = scheduled["Remaining"].astype(int)
                        #         scheduled.loc["Total",["Scheduled","Loaded","Remaining"]].astype(int)
                        #         scheduled.fillna("",inplace=True)
    
    
                        #         def style_row(row,code=1):
                                    
                        #             if code==2:
                        #                 shipment_status = row["Status"]
                        #                 location = row["Location"]
                        #             else:
                        #                 location = row['Destination']
                        #             # Define colors for different locations
                        #             colors = {
                        #                 "CLATSKANIE": "background-color: #d1e7dd;",  # light green
                        #                 "LEWISTON": "background-color: #ffebcd;",    # light coral
                        #                 "HALSEY": "background-color: #add8e6;",      # light blue
                        #             }
                                    
                        #             # Base style for the entire row based on location
                        #             base_style = colors.get(location, "")
                        #             if code==2:
                        #                 if shipment_status == "SHIPPED":
                        #                     base_style += "font-weight: lighter; font-style: italic; text-decoration: line-through;"  # Less bold, italic, and strikethrough
                        #                 else:
                        #                     base_style += "font-weight: bold;"  # Slightly bolder for other statuses
                            
                        #             return [base_style] * len(row)
                                
                        #         list_view=st.checkbox("LIST VIEW")
                        #         if list_view:
                        #             flattened_data = []
                        #             for date, locations in schedule.items():
                        #                 for location, location_data in locations.items():
                        #                     for order, carriers in location_data.items():
                        #                         for carrier, shipments in carriers.items():
                        #                             for shipment in shipments:
                        #                                 dfb_=dfb[dfb["St_Date"]==selected_date_datetime]
                        #                                 status="NONE"
                        #                                 # Split the shipment data if needed (separate IDs if joined by "|")
                        #                                 if shipment in dfb_.index:
                        #                                     status_="SHIPPED"
                        #                                 else:
                        #                                     status_="Scheduled"
                        #                                 shipment_parts = shipment.split("|") if "|" in shipment else [shipment]
                        #                                 carrier_=carrier.split("-")[1]
                        #                                 flattened_data.append({
                        #                                     "Date": date,
                        #                                     "Location": location,
                        #                                     "Order": order,
                        #                                     "Carrier": carrier_,
                        #                                     "EDI Bill Of Lading":shipment,
                        #                                     "Shipment ID": shipment_parts[0],
                        #                                     "MF Number": shipment_parts[1] if len(shipment_parts) > 1 else None,
                        #                                     "Status":status_
                        #                                 })
    
                        #         # Convert to DataFrame
                        #             flat_df = pd.DataFrame(flattened_data)
                        #             #flat_df["Status"]=["Scheduled"]*len(flat_df)
                        #             flat_df=flat_df[flat_df.Date==selected_date]
                        #             flat_df.reset_index(drop=True,inplace=True)
                        #             flat_df.index+=1
    
                        #         #styled_schedule =scheduled.style.apply(style_row, axis=1)
                        #         if list_view:
                        #             styled_schedule = flat_df.style.apply(lambda row: style_row(row, code=2), axis=1)
                        #         else:
                        #             styled_schedule = scheduled.style.apply(lambda row: style_row(row, code=1), axis=1)
    
                             
                        #         st.write(styled_schedule.to_html(), unsafe_allow_html=True)
                
                        # else:
                        #     st.write("Nothing Scheduled")


                        
                    with rls_tab1:
                        
                        destinations_of_release_orders=[f"{i} to {release_order_database[i]['destination']}" for i in release_order_database if release_order_database[i]["complete"]!=True]
                        if len(destinations_of_release_orders)==0:
                            st.markdown("**ALL RELEASE ORDERS COMPLETED, REACTIVATE A RELEASE ORDER OR ENTER NEW") 
                         ###       Dropdown menu
                        else:
                            nofile=0
                            requested_file_=st.selectbox("ACTIVE RELEASE ORDERS",destinations_of_release_orders)
                            requested_file=requested_file_.split(" ")[0]
    
                            target=release_order_database[requested_file]
                            destination=target['destination']
                            po_number=target['po_number']
                                        
                            number_of_sales_orders=len([i for i in target if i not in ["destination","po_number"]])   ##### WRONG CAUSE THERE IS NOW DESTINATION KEYS
                            
                            
                           
                            if nofile!=1 :         
                                with st.container(border=True):
                                    rel_ccol1,rel_ccol2,rel_ccol3,rel_ccol4=st.columns([2,2,2,2])
                                    targets=[i for i in target if i in ["001","002","003","004","005"]] 
                                    sales_orders_completed=[k for k in targets if target[k]['remaining']<=0]
                                    with rel_ccol1:
                                        st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                    with rel_ccol2:
                                        st.markdown(f"**:blue[Destination] : {target['destination']}**")
                                    with rel_ccol3:
                                        st.markdown(f"**:blue[PO Number] : {target['po_number']}**")
                                    with rel_ccol4:
                                        st.markdown(f"**:blue[...]**")
                                    st.divider()
                                    rel_col1,rel_col2,rel_col3,rel_col4=st.columns([2,2,2,2])
                                    with rel_col1:
                                        
                                        if targets[0] in sales_orders_completed:
                                            st.markdown(f"**:orange[Sales Order Item : {targets[0]} - COMPLETED]**")
                                            target0_done=True
                                            
                                        else:
                                            st.markdown(f"**:blue[Sales Order Item] : {targets[0]}**")
                                        
                                        st.write(f"        Total Quantity: {target[targets[0]]['total']} Units - {2*target[targets[0]]['total']} Tons")
                                        st.write(f"        Ocean Bill Of Lading : {target[targets[0]]['ocean_bill_of_lading']}")
                                        st.write(f"        Batch : {target[targets[0]]['batch']} WIRES : {target[targets[0]]['unitized']}")
                                        st.write(f"        Units Shipped : {target[targets[0]]['shipped']} Units - {2*target[targets[0]]['shipped']} Tons")
                                        if 0<target[targets[0]]['remaining']<=10:
                                            st.markdown(f"**:red[Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Tons]**")
                                        elif target[targets[0]]['remaining']<=0:
                                            st.markdown(f":orange[Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Tons]")                                                                        
                                        else:
                                            st.write(f"       Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Tons")
                                    with rel_col2:
                                        try:
                                            if targets[1] in sales_orders_completed:
                                                st.markdown(f"**:orange[Sales Order Item : {targets[1]} - COMPLETED]**")                                    
                                            else:
                                                st.markdown(f"**:blue[Sales Order Item] : {targets[1]}**")
                                            st.write(f"        Total Quantity : {target[targets[1]]['total']} Units - {2*target[targets[1]]['total']} Tons")                        
                                            st.write(f"        Ocean Bill Of Lading : {target[targets[1]]['ocean_bill_of_lading']}")
                                            st.write(f"        Batch : {target[targets[1]]['batch']} WIRES : {target[targets[1]]['unitized']}")
                                            st.write(f"        Units Shipped : {target[targets[1]]['shipped']} Units - {2*target[targets[1]]['shipped']} Tons")
                                            if 0<target[targets[1]]['remaining']<=10:
                                                st.markdown(f"**:red[Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Tons]**")
                                            elif target[targets[1]]['remaining']<=0:
                                                st.markdown(f":orange[Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Tons]")
                                            else:
                                                st.write(f"       Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Tons")
                                                
                                        except:
                                            pass
                        
                                    with rel_col3:
                                        try:
                                            if targets[2] in sales_orders_completed:
                                                st.markdown(f"**:orange[Sales Order Item : {targets[2]} - COMPLETED]**")
                                            else:
                                                st.markdown(f"**:blue[Sales Order Item] : {targets[2]}**")
                                            st.write(f"        Total Quantity : {target[targets[2]]['total']} Units - {2*target[targets[2]]['total']} Tons")
                                            st.write(f"        Ocean Bill Of Lading : {target[targets[2]]['ocean_bill_of_lading']}")
                                            st.write(f"        Batch : {target[targets[2]]['batch']} WIRES : {target[targets[2]]['unitized']}")
                                            st.write(f"        Units Shipped : {target[targets[2]]['shipped']} Units - {2*target[targets[2]]['shipped']} Tons")
                                            if 0<target[targets[2]]['remaining']<=10:
                                                st.markdown(f"**:red[Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Tons]**")
                                            elif target[targets[2]]['remaining']<=0:
                                                st.markdown(f":orange[Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Tons]")
                                            else:
                                                st.write(f"       Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Tons")
                                            
                                            
                                        except:
                                            pass
                    
                                    with rel_col4:
                                        try:
                                            if targets[3] in sales_orders_completed:
                                                st.markdown(f"**:orange[Sales Order Item : {targets[3]} - COMPLETED]**")
                                            else:
                                                st.markdown(f"**:blue[Sales Order Item] : {targets[3]}**")
                                            st.write(f"        Total Quantity : {target[targets[3]]['total']} Units - {2*target[targets[3]]['total']} Tons")
                                            st.write(f"        Ocean Bill Of Lading : {target[targets[3]]['ocean_bill_of_lading']}")
                                            st.write(f"        Batch : {target[targets[3]]['batch']} WIRES : {target[targets[3]]['unitized']}")
                                            st.write(f"        Units Shipped : {target[targets[3]]['shipped']} Units - {2*target[targets[2]]['shipped']} Tons")
                                            if 0<target[targets[3]]['remaining']<=10:
                                                st.markdown(f"**:red[Units Remaining : {target[targets[3]]['remaining']} Units - {2*target[targets[3]]['remaining']} Tons]**")
                                            elif target[targets[3]]['remaining']<=0:
                                                st.markdown(f":orange[Units Remaining : {target[targets[3]]['remaining']} Units - {2*target[targets[3]]['remaining']} Tons]")
                                            else:
                                                st.write(f"       Units Remaining : {target[targets[3]]['remaining']} Units - {2*target[targets[3]]['remaining']} Tons")
                                            
                                            
                                        except:
                                            pass
                                
                      
                                with st.container(border=True):
                                    dol1,dol2,dol3,dol4=st.columns([2,2,2,2])
                                    with dol1:
                                        hangisi=st.selectbox("**:green[SELECT SALES ORDER ITEM TO DISPATCH]**",([i for i in targets if i not in sales_orders_completed]))
                                        if st.button("DISPATCH TO WAREHOUSE",key="lala"):
                                            try:
                                                last=list(dispatch[requested_file].keys())[-1]
                                             
                                                dispatch[requested_file][hangisi]={"release_order":requested_file,"sales_order":hangisi,"destination":destination}
                                            except:
                                                dispatch[requested_file]={}
                                                dispatch[requested_file][hangisi]={"release_order":requested_file,"sales_order":hangisi,"destination":destination}
                    
                                            
                                            json_data = json.dumps(dispatch)
                                            storage_client =get_storage_client()
                                            bucket = storage_client.bucket(target_bucket)
                                            blob = bucket.blob(rf"dispatched.json")
                                            blob.upload_from_string(json_data)
                                            st.markdown(f"**DISPATCHED Release Order Number {requested_file} Item No : {hangisi} to Warehouse**")
                                    
                                                   
                                    with dol2:  
                                        try:
                                            liste=[]
                                            for i in dispatch.keys():
                                                for k in dispatch[i]:
                                                    liste.append(f"{i}-{k}")
                                            item=st.selectbox("CHOOSE ITEM",liste)
                                            undispatch_rel=item.split("-")[0]
                                            undispatch_sal=item.split("-")[1]
                                            if st.button("UN-DISPATCH ITEM"):                                       
                                                del dispatch[undispatch_rel][undispatch_sal]
                                                json_data = json.dumps(dispatch)
                                                storage_client = get_storage_client()
                                                bucket = storage_client.bucket(target_bucket)
                                                blob = bucket.blob(rf"dispatched.json")
                                                blob.upload_from_string(json_data)
                                                st.markdown(f"**CLEARED DISPATCH ITEM {item}**")   
                                        except:
                                            pass
                                        
                                    with dol3:
                                        if st.button("CLEAR DISPATCH QUEUE!"):
                                            dispatch={}
                                            json_data = json.dumps(dispatch)
                                            storage_client = get_storage_client()
                                            bucket = storage_client.bucket(target_bucket)
                                            blob = bucket.blob(rf"dispatched.json")
                                            blob.upload_from_string(json_data)
                                            st.markdown(f"**CLEARED ALL DISPATCHES**")   
                                        
                                with st.container(border=True):
                                    
                                    st.markdown("**CURRENT DISPATCH QUEUE**")
                                    try:
                                        for dispatched_release in dispatch.keys():
                                            for sales in dispatch[dispatched_release].keys():
                                                st.markdown(f'**Release Order = {dispatched_release}, Sales Item : {sales}, Destination : {dispatch[dispatched_release][sales]["destination"]} .**')
                                    except:
                                        st.write("NO DISPATCH ITEMS")
                            
                            else:
                                st.write("NO RELEASE ORDERS IN DATABASE")
                    with rls_tab2:
                        completed=[i for i in release_order_database if release_order_database[i]["complete"]==True]
                                                    
                        completed_release_order_dest_map={}
                        for i in completed:
                            for sale in [s for s in release_order_database[i] if s in ["001","002","003","004","005"]]:
                                completed_release_order_dest_map[f"{i}-{sale}"]=[sale,release_order_database[i]["destination"],
                                                                     release_order_database[i][sale]["ocean_bill_of_lading"],
                                                                     release_order_database[i][sale]["grade"],
                                                                     release_order_database[i][sale]["vessel"],
                                                                     release_order_database[i][sale]["total"],
                                                                                release_order_database[i][sale]["shipped"],
                                                                                release_order_database[i][sale]["remaining"]]
                      
                        column_names = ["Sale", "Destination", "Ocean Bill of Lading", "Grade", "Vessel", "Total", "Shipped", "Remaining"]

                        # Create DataFrame and assign column names
                        completed_frame = pd.DataFrame(completed_release_order_dest_map).T
                        completed_frame.columns = column_names

                        st.write(completed_frame)
                        activate_list=list(set([i.split("-")[0] for i in completed_release_order_dest_map.keys()]))
                        to_reactivate=st.selectbox("SELECT RELEASE ORDER TO CHANGE FROM COMPLETE TO UNCOMPLETE",activate_list,key="erfdaq")
                        if st.button("ACTIVATE RELEASE ORDER"):
                            release_order_database[to_reactivate]['complete']=False
                            storage_client = get_storage_client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                            blob.upload_from_string(json.dumps(release_order_database))
                            st.success(f"Reactivated {to_reactivate} successfully!")
                        not_completed=[i for i in release_order_database if release_order_database[i]["complete"]==False]
                        to_deactivate=st.selectbox("SELECT RELEASE ORDER TO CHANGE FROM NOT COMPLETE TO COMPLETE",not_completed,key="erfsdaq")
                        if st.button("DE-ACTIVATE RELEASE ORDER",key="sdasa"):
                            release_order_database[to_deactivate]['complete']=True
                            storage_client = get_storage_client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                            blob.upload_from_string(json.dumps(release_order_database))
                            st.success(f"Deactivated {to_deactivate} successfully!")
                        
#                     with rls_tab3:
#                         bill_for_mf=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
#                         bill_for_mf=json.loads(bill_for_mf)
#                         dfb=pd.DataFrame.from_dict(bill_for_mf).T[1:]
#                         #dfb=bill.copy()
#                         dfb["St_Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in dfb["issued"]]
# ###  MF NUMBERS
# ### EDIT - REMOVE
#                         mf1,mf2=st.tabs(["VIEW/EDIT MF NUMBERS","AUTO UPLOAD"])
                        
                        
                        
#                         with mf1:
#                             mf_numbers=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")
#                             mf_numbers=json.loads(mf_numbers)
#                             schedule=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
#                             schedule=json.loads(schedule)

                            
#                             def check_home(ro):
#                                 destination=release_order_database[ro]['destination']
#                                 keys=[sale for sale in release_order_database[ro] if sale in ["001","002","003","004","005"]]
#                                 remains=[release_order_database[ro][key]["remaining"] for key in keys]
#                                 if sum(remains)==0:
#                                     return False
#                                 return f"{ro} to {destination}"
                                
#                             destinations_of_release_orders=[check_home(i) for i in release_order_database if check_home(i) ]
#                             if len(destinations_of_release_orders)==0:
#                                 st.warning("NO GP RELEASE ORDERS FOR THIS VESSEL")
#                             else:
                                
#                                 release_order_number_mf_=st.selectbox("SELECT RELEASE ORDER FOR SHIPMENT NUMBERS",destinations_of_release_orders,key="tatata")
#                                 release_order_number_mf=release_order_number_mf_.split(" ")[0]
#                                 dest=release_order_number_mf_.split(" ")[2].split("-")[1].split(",")[0].upper()
#                                 mf_date_str=datetime.datetime.strftime((st.date_input("Shipment Date",datetime.datetime.today(),disabled=False,key="popddao3")),"%Y-%m-%d")
#                                 carrier_mf=st.selectbox("SELECT CARRIER",[f"{i}-{j}" for i,j in map["carriers"].items()],key="tatpota")
#                                 input_mf_numbers=st.text_area("**ENTER SHIPMENT NUMBERS**",height=100,key="juy")
#                                 if input_mf_numbers is not None:
#                                     input_mf_numbers = input_mf_numbers.splitlines()
#                                     input_mf_numbers=[i for i in input_mf_numbers]####### CAREFUL THIS ASSUMES SAME DIGIT MF EACH TIME
#                                 if st.button("SUBMIT SHIPMENT NUMBERS",key="ioeru" ):
                                    
#                                     if mf_date_str not in mf_numbers.keys():   
#                                         mf_numbers[mf_date_str]={}
#                                     if dest not in mf_numbers[mf_date_str]:
#                                         mf_numbers[mf_date_str][dest]={}
#                                     if release_order_number_mf not in mf_numbers[mf_date_str][dest]:
#                                         mf_numbers[mf_date_str][dest][release_order_number_mf]={}
#                                     if carrier_mf not in mf_numbers[mf_date_str][dest][release_order_number_mf]:
#                                        mf_numbers[mf_date_str][dest][release_order_number_mf][carrier_mf]=[]
#                                     mf_numbers[mf_date_str][dest][release_order_number_mf][carrier_mf]+=input_mf_numbers
#                                     mf_numbers[mf_date_str][dest][release_order_number_mf][carrier_mf]=list(set(mf_numbers[mf_date_str][dest][release_order_number_mf][carrier_mf]))
#                                     mf_data=json.dumps(mf_numbers)
#                                     #storage_client = storage.Client()
#                                     storage_client = get_storage_client()
#                                     bucket = storage_client.bucket(target_bucket)
#                                     blob = bucket.blob(rf"release_orders/mf_numbers.json")
#                                     blob.upload_from_string(mf_data)
#                                     st.success(f"MF numbers entered to {release_order_number_mf} successfully!")
                                    
#                                 if st.button("REMOVE SHIPMENT NUMBERS",key="ioerssu" ):
#                                     for i in input_mf_numbers:
#                                         try:
#                                             mf_numbers[mf_date_str][dest][release_order_number_mf][carrier_mf].remove(int(i))
#                                         except:
#                                             mf_numbers[mf_date_str][dest][release_order_number_mf][carrier_mf].remove(str(i))
#                                         st.success(f"MF numbers removed from {release_order_number_mf} successfully!")   
                                                
#                                     mf_data=json.dumps(mf_numbers)
#                                    # storage_client = storage.Client()
#                                     storage_client = get_storage_client()
#                                     bucket = storage_client.bucket(target_bucket)
#                                     blob = bucket.blob(rf"release_orders/mf_numbers.json")
#                                     blob.upload_from_string(mf_data)
#  ### MF NUMBERS                                   
                               
#                                 flattened_data = []
#                                 for date, locations in mf_numbers.items():
#                                     for location, location_data in locations.items():
#                                         for order, carriers in location_data.items():
#                                             for carrier, shipments in carriers.items():
#                                                 for shipment in shipments:
#                                                     dfb=dfb[dfb["St_Date"]==selected_date_datetime]
#                                                     status="NONE"
#                                                     if shipment in dfb.index:
#                                                         status_="SHIPPED"
#                                                     else:
#                                                         status_="Scheduled"
#                                                     shipment_parts = shipment.split("|") if "|" in shipment else [shipment]
#                                                     carrier_=carrier.split("-")[1]
#                                                     flattened_data.append({
#                                                         "Date": date,
#                                                         "Location": location,
#                                                         "Order": order,
#                                                         "Carrier": carrier_,
#                                                         "EDI Bill Of Lading":shipment,
#                                                         "MF Number": shipment_parts[0] if len(shipment_parts) > 1 else None,
#                                                         "Shipment ID": shipment_parts[1] if len(shipment_parts) > 1 else shipment_parts[0]
#                                                     })

                                
#                                 flat_df=pd.DataFrame(flattened_data)
#                                 flat_df["Date"] = pd.to_datetime(flat_df["Date"])#.dt.date
#                                 flat_df.insert(1,"Day",flat_df["Date"].dt.day_name())
#                                 flat_df["Status"]="None"
#                                 flat_df['Status'] = flat_df['EDI Bill Of Lading'].apply(lambda x: 'SHIPPED' if x in bill_for_mf else 'Scheduled')

                                
#                                 mf_display_tab1,mf_display_tab2,mf_display_tab3=st.tabs(["DAILY","WEEK","ALL SCHEDULE"])
                                
#                                 with mf_display_tab1:
                                    
#                                     display_flat_df=flat_df[flat_df.Date==mf_date_str]
#                                     display_flat_df.reset_index(drop=True,inplace=True)
#                                     display_flat_df.index+=1
#                                     display_flat_df["Date"] = display_flat_df["Date"].dt.date
                                    

#                                     styled_df = display_flat_df.style.apply(style_row, axis=1)
#                                     st.write(styled_df.to_html(), unsafe_allow_html=True)

                              
#                                 with mf_display_tab2:
                              
#                                     #flat_df["Status"] = ["Scheduled"] * len(flat_df)
#                                     flat_df.reset_index(drop=True, inplace=True)
#                                     flat_df.index += 1
                                    
#                                     # Convert "Date" to datetime and set as index
#                                     #flat_df["Date"] = pd.to_datetime(flat_df["Date"])
#                                     #flat_df.insert(1,"Day",flat_df["Date"].dt.day_name())
                                    
#                                     today=(datetime.datetime.today()-datetime.timedelta(hours=utc_difference)).date()
#                                     start_of_this_week = today - datetime.timedelta(days=today.weekday()) 
#                                     start_of_next_week = (start_of_this_week + datetime.timedelta(days=7))
#                                     end_of_next_week = (start_of_next_week + datetime.timedelta(days=7))
                                    
#                                     if today.weekday() < 5:  # 0 = Monday, 4 = Friday
#                                         # Filter to display only the current week
#                                         weekly_display = flat_df[(flat_df["Date"]<pd.Timestamp(start_of_next_week))&
#                                                                  (flat_df["Date"]>pd.Timestamp(start_of_this_week))]
#                                     else:
#                                         # Display the upcoming week
#                                         weekly_display = flat_df[(flat_df["Date"]>pd.Timestamp(start_of_next_week))&
#                                                                  (flat_df["Date"]<pd.Timestamp(end_of_next_week))]


                                    
#                                     weekly_counts = (
#                                         weekly_display.groupby([weekly_display["Date"].dt.day_name(), "Location"])
#                                         .size()
#                                         .unstack(fill_value=0)
#                                     )
                                    
                                    
#                                     # Define weekdays to display in the table (Monday to Friday)
#                                     weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
#                                     weekly_counts_ = weekly_counts.reindex(weekdays)
                                    
#                                     weekly_counts_.fillna("",inplace=True)
#                                     weekly_counts_ = weekly_counts_.applymap(lambda x: int(x) if isinstance(x, float) else x)

                                    
                                    
#                                     st.write(weekly_counts_)
#                                     # Generate HTML table with large squares
#                                     weekly_counts = weekly_counts.dropna(axis=0)
#                                     html_content = """
#                                     <style>
#                                         .day-table {
#                                             display: flex;
#                                             justify-content: space-around;
#                                             margin-top: 20px;
#                                         }
#                                         .day-cell {
#                                             width: 150px;
#                                             height: 150px;
#                                             display: flex;
#                                             flex-direction: column;
#                                             justify-content: center;
#                                             align-items: center;
#                                             border: 2px solid #ddd;
#                                             font-size: 20px;
#                                             font-weight: bold;
#                                             background-color: #f9f9f9;
#                                         }
#                                         .day-name {
#                                             font-size: 18px;
#                                             color: #333;
#                                             margin-bottom: 10px;
#                                         }
#                                         .shipment-info {
#                                             font-size: 16px;
#                                             color: #555;
#                                         }
#                                     </style>
                                    
#                                     <div class="day-table">
#                                     """
                                    
#                                     # Populate each weekday's cell with shipment information
#                                     for day in weekdays:
#                                         shipments = weekly_counts.loc[day] if day in weekly_counts.index else None
#                                         html_content += f"""
#                                         <div class="day-cell">
#                                             <div class="day-name">{day}</div>
#                                         """
#                                         if shipments is not None:
#                                             for destination, count in shipments.items():
#                                                 html_content += f'<div class="shipment-info">{count}  x  {destination.title()}</div>'
#                                         else:
#                                             html_content += '<div class="shipment-info">No Shipments</div>'
#                                         html_content += "</div>"
                                    
#                                     html_content += "</div>"
                                    
#                                     # Display the HTML table in Streamlit
#                                     st.components.v1.html(html_content, height=300, scrolling=False)
#                                 with mf_display_tab3:
#                                     #flat_df["Status"]=["Scheduled"]*len(flat_df)
#                                     flattened_data = []
#                                     for date, locations in schedule.items():
#                                         for location, location_data in locations.items():
#                                             for order, carriers in location_data.items():
#                                                 for carrier, shipments in carriers.items():
#                                                     for shipment in shipments:
#                                                         dfb=dfb[dfb["St_Date"]==selected_date_datetime]
#                                                         status="NONE"
#                                                         if shipment in dfb.index:
#                                                             status_="SHIPPED"
#                                                         else:
#                                                             status_="Scheduled"
#                                                         shipment_parts = shipment.split("|") if "|" in shipment else [shipment]
#                                                         carrier_=carrier.split("-")[1]
#                                                         flattened_data.append({
#                                                             "Date": date,
#                                                             "Location": location,
#                                                             "Order": order,
#                                                             "Carrier": carrier_,
#                                                             "EDI Bill Of Lading":shipment,
#                                                             "MF Number": shipment_parts[0] if len(shipment_parts) > 1 else None,
#                                                             "Shipment ID": shipment_parts[1] if len(shipment_parts) > 1 else shipment_parts[0]
#                                                         })
    
                                    
#                                     flat_df=pd.DataFrame(flattened_data)
#                                     flat_df["Date"] = pd.to_datetime(flat_df["Date"])#.dt.date
#                                     flat_df.insert(1,"Day",flat_df["Date"].dt.day_name())
#                                     flat_df["Status"]="None"
#                                     flat_df['Status'] = flat_df['EDI Bill Of Lading'].apply(lambda x: 'SHIPPED' if x in bill_for_mf else 'Scheduled')
#                                     flat_df.reset_index(drop=True,inplace=True)
#                                     flat_df.index+=1
#                                     styled_df =flat_df.style.apply(style_row, axis=1)
#                                     st.write(styled_df.to_html(), unsafe_allow_html=True)

                        
                        
#                         with mf2:
#                             col11,col22,col33=st.columns([3,3,4])
#                             button=True
#                             with col11:
                                
#                                 st.subheader("SELECT DATES TO UPLOAD")
#                                 dates1=st.date_input("FROM (INCLUSIVE)",datetime.date.today()-datetime.timedelta(hours=utc_difference),key="r3wsd")
#                                 dates2=st.date_input("TO (INCLUSIVE)",datetime.date.today()+datetime.timedelta(days=30),key="rz3wsd")
#                             with col22:
#                                 st.subheader("UPLOAD SHIPMENT CSV FILES")
#                                 suzano_shipment = st.file_uploader("Upload **SUZANO** Shipment CSV", type="xlsx",key="dsds")
#                                 kbx_shipment = st.file_uploader("Upload **KBX** Shipment CSV", type="xls",key="dsdfqa")
                            
#                             if suzano_shipment and kbx_shipment:
#                                 st.success("Files uploaded")
#                                 button=False
#                                 done=False
#                             done=False
                            
#                             if not button:
                            
                                
#                                 df=pd.read_excel(suzano_shipment)
#                                 df=df[['PK', 'Release Order','Start Time', 'Destination City',
#                                        'Destination Province Code', 'Weight', 'Unit Count', 
#                                        'Service Provider ID',  'Transit Status','BOL','Vehicle ID']]
#                                 df['Release Order']=[i[10:] for i in df['Release Order']]
#                                 df["Pickup"] = pd.to_datetime(df["Start Time"]).dt.date#.apply(lambda i: datetime.datetime.strptime(i, "%m/%d/%Y %I:%M %p").date())
                                
#                                 df["Service Provider ID"]=df["Service Provider ID"].astype(str)
#                                 df["PK"]=[i[7:] for i in df["PK"].values]
                                
                                
#                                 df1=pd.read_html(kbx_shipment)[0]
#                                 df1=df1[[ 'Load Number', 'Owner', 'Pro Number',
#                                            'SCAC', 'Pickup Date','Orig Loc Nbr', 'Orig Name', 'Delivery Date', 
#                                          'Dest Loc Nbr', 'Dest City', 'Dest Name', 'Movement Type',  'Miles',
#                                            'Total Weight']]
#                                 df1["Pickup Date"] = pd.to_datetime(df1["Pickup Date"]).dt.date
#                                 df1.rename(columns={"Dest City":"Destination City"},inplace=True)
                                
                                
                                
#                                 df=df[(df['Pickup']>=dates1)&(df['Pickup']<=dates2)].sort_values(by="Pickup")
#                                 df1=df1[df1['Pickup Date']>=dates1].sort_values(by="Pickup Date")
                               
#                                 matches={}
#                                 days_loads={}
#                                 kbx_loads={}
                                
#                                 #for i in sorted(df[df["Pickup"]>=datetime.date.today()]["Pickup"].unique()):  ### RULE FROM TODAY
#                                 for i in sorted(df["Pickup"].unique()):  ### RULE FROM TODAY
#                                     matches[str(i)]={}
                                   
                                    
#                                     for dest in df[df["Pickup"]==i]["Destination City"].unique():
#                                         matches[str(i)][dest]={}
#                                         for rel in df.loc[(df["Pickup"] == i) & (df["Destination City"] == dest), "Release Order"].unique():
#                                             matches[str(i)][dest][rel]={}
#                                             for trans in df.loc[(df["Release Order"] == rel) &(df["Pickup"] == i) & (df["Destination City"] == dest),
#                                                                 "Service Provider ID"]:
#                                                 if trans=="KBX":
#                                                     trans="123456-KBX"
#                                                     suz=sorted(df.loc[(df["Release Order"] == rel) &(df["Pickup"] == i) & (df["Destination City"] == dest), 
#                                                                       "PK"])
#                                                     kbx=sorted(df1.loc[(df1["Pickup Date"] == i) & (df1["Destination City"] == dest)&(~df1["SCAC"].isna()),
#                                                                        "Load Number"])
#                                                     mat=[f"{j}|{k}" for j,k in zip(kbx,suz)]
#                                                     matches[str(i)][dest][rel][trans]=mat
#                                                 else:
#                                                     trans_=f"{str(trans)}-{map['carriers'][str(trans)]}"
#                                                     suz=sorted(df.loc[(df["Service Provider ID"] == trans) &(df["Release Order"] == rel) &(df["Pickup"] == i) & (df["Destination City"] == dest), 
#                                                                       "PK"])
#                                                     mat=suz.copy()
#                                                     matches[str(i)][dest][rel][trans_]=mat
#                                 done=True
#                                 if matches not in st.session_state:
#                                     st.session_state.matches=matches
#                                 st.session_state.matches=matches
#                                 st.success("SHIPMENTS MATCHED AND REFRESHED!")
#                                 button_html = """
#                                 <div style="text-align: center; margin-top: 20px;">
#                                     <button onclick="triggerPython()" style="
#                                         background: linear-gradient(145deg, #ffffff, #d4d4d4);
#                                         border: none;
#                                         border-radius: 12px;
#                                         box-shadow: 5px 5px 10px #b8b8b8, -5px -5px 10px #ffffff;
#                                         color: #444;
#                                         font-size: 18px;
#                                         font-weight: bold;
#                                         padding: 10px 20px;
#                                         cursor: pointer;
#                                         transition: 0.2s;
#                                     " 
#                                     onmouseover="this.style.background='#e0e0e0';"
#                                     onmouseout="this.style.background='linear-gradient(145deg, #ffffff, #d4d4d4)';"
#                                     onmousedown="this.style.boxShadow='inset 3px 3px 5px #b8b8b8, inset -3px -3px 5px #ffffff'; this.style.transform='translateY(2px)';"
#                                     onmouseup="this.style.boxShadow='5px 5px 10px #b8b8b8, -5px -5px 10px #ffffff'; this.style.transform='translateY(0px)';">
#                                         RECORD SUZANO
#                                     </button>
#                                 </div>
#                                 <script>
#                                     function triggerPython() {
#                                         window.location.href = "/?button_clicked=true";
#                                     }
#                                 </script>
#                                 """
                                
#                                 st.components.v1.html(button_html, height=120)

# # Handle backend logic
#                                 if st.experimental_get_query_params().get("button_clicked") == ["true"]:
#                                     st.balloons()  # Celebrate the button click
#                                     st.success("3D Button Clicked and Python Action Triggered!")
                                
#                                 if st.button("RECORD SUZANO LIST",disabled=button,key="sdsqawds2"):
#                                     suz_=json.dumps(matches)
#                                     storage_client = get_storage_client()
#                                     bucket = storage_client.bucket(target_bucket)
#                                     blob = bucket.blob(rf"release_orders/suzano_shipments.json")
#                                     blob.upload_from_string(suz_)
#                                     st.success(f"Suzano list updated!")
#                             cor1,cor2=st.columns([5,5])
#                             with cor1:
#                                 if done:
#                                     st.write(dict(matches))
#                             with cor2:
#                                 try:
#                                     matches=st.session_state.matches
#                                 except:
#                                     pass
#                                 if st.button("UPLOAD SHIPMENTS TO SYSTEM",key="dsdsdads2"):
                                    
#                                     # for i in matches:
#                                     #     st.write("Processing date:", i)
#                                     #     if i not in mf_numbers:
#                                     #         mf_numbers[i]={}
#                                     #     mf_numbers[i]=matches[i].copy()
#                                     # st.write("MF Numbers Dictionary (after update):", mf_numbers)
#                                     mf_numbers=matches.copy()
#                                     mf_datam=json.dumps(mf_numbers)
#                                         #storage_client = storage.Client()
#                                     storage_client = get_storage_client()
#                                     bucket = storage_client.bucket(target_bucket)
#                                     blob = bucket.blob(rf"release_orders/mf_numbers.json")
#                                     blob.upload_from_string(mf_datam)
#                                     st.success(f"MF numbers updated with schedule!")
#                                     st.rerun()
#                                 #st.write(mf_numbers)

###  RELEASE ORDER STATUS                
                with release_order_tab3:  ### RELEASE ORDER STATUS
                    raw_ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                    raw_ro = json.loads(raw_ro)
                    status_dict={}
                    sales_group=["001","002","003","004","005"]
                    for ro in raw_ro:
                        for sale in [i for i in raw_ro[ro] if i in sales_group]:
                            status_dict[f"{ro}-{sale}"]={"Release Order #":ro,"Sales Order #":sale,
                                                "Destination":raw_ro[ro]['destination'],
                                                "Ocean BOL":raw_ro[ro][sale]['ocean_bill_of_lading'],
                                                "Total":raw_ro[ro][sale]['total'],
                                                "Shipped":raw_ro[ro][sale]['shipped'],
                                                "Remaining":raw_ro[ro][sale]['remaining']}
                    status_frame=pd.DataFrame(status_dict).T.set_index("Release Order #",drop=True)
                    active_frame_=status_frame[status_frame["Remaining"]>0]
                    status_frame.loc["Total"]=status_frame[["Total","Shipped","Remaining"]].sum()
                    active_frame=active_frame_.copy()
                    active_frame.loc["Total"]=active_frame[["Total","Shipped","Remaining"]].sum()
                    
                    # --- layout: two columns (table left, editor right) ---
                    col_table, col_edit = st.columns([2, 1])
                
                    with col_table:
                        st.markdown("### RELEASE ORDER STATUS (Active)")
                        st.markdown(active_frame.to_html(render_links=True), unsafe_allow_html=True)
                
                    # Build fig as you already do (left column is fine)
                    with col_table:
                        release_orders = status_frame.index[:-1]
                        release_orders = pd.Categorical(release_orders)
                        active_order_names = [f"{i} to {raw_ro[i]['destination']}" for i in active_frame_.index]
                        destinations = [raw_ro[i]['destination'] for i in active_frame_.index]
                        active_orders = [str(i) for i in active_frame.index]
                
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=active_orders, y=active_frame["Total"], name='Total', marker_color='lightgray'))
                        fig.add_trace(go.Bar(x=active_orders, y=active_frame["Shipped"], name='Shipped', marker_color='blue', opacity=0.7))
                        remaining_data = [r if r > 0 else None for r in active_frame_["Remaining"]]
                        fig.add_trace(go.Scatter(x=active_orders, y=remaining_data, mode='markers', name='Remaining',
                                                 marker=dict(color='red', size=10)))
                        st.plotly_chart(fig, use_container_width=True)
                
                    # --- EDITOR (right column) ---
                    with col_edit:
                        def safe_download_json(bucket, path, default):
                            try:
                                raw = gcp_download(bucket, path)
                                if not raw:
                                    return default
                                return json.loads(raw)
                            except Exception:
                                # file missing or invalid JSON → start fresh
                                return default
                        st.markdown("### Edit Release Order Numbers")
                
                        # Only allow editing rows that are actually displayed (exclude the summary row)
                        # We need both RO and Sales Order to uniquely identify the line
                        if len(active_frame_) == 0:
                            st.info("No active release orders to edit.")
                        else:
                            # Rebuild a tidy table with RO + SO as distinct rows for selection
                            tidy = (
                                active_frame_
                                .reset_index()  # brings "Release Order #" back as a column
                                .rename(columns={"Release Order #": "RO"})
                            )
                
                            # NOTE: active_frame_ has duplicate index values (RO) for each Sales Order.
                            # We pull the Sales Order from the original dict using the combined keys.
                            # Let's reconstruct by walking status_dict and filtering those present in active_frame_
                            rows = []
                            for key, row in status_dict.items():
                                ro = row["Release Order #"]
                                if ro in active_frame_.index:
                                    rows.append({
                                        "key": key,  # e.g. "RO-SO"
                                        "RO": ro,
                                        "SO": row["Sales Order #"],
                                        "Dest": row["Destination"],
                                        "Total": row["Total"],
                                        "Shipped": row["Shipped"],
                                        "Remaining": row["Remaining"]
                                    })
                            edit_df = pd.DataFrame(rows)
                
                            # If somehow nothing matched (shouldn't happen), guard:
                            if edit_df.empty:
                                st.info("No editable rows found for the current view.")
                            else:
                                enable_edit = st.checkbox("EDIT RELEASE ORDER NUMBERS", value=False)
                
                                if enable_edit:
                                    with st.form("ro_edit_form", clear_on_submit=False):

                                        try:
                                            ro_log = safe_download_json(target_bucket, r"release_orders/ro_log.json", default=[])
                                            if not isinstance(ro_log, list):
                                                ro_log = []
                                        except Exception:
                                            ro_log = []
                                        # Build human-friendly labels but keep (RO, SO) as the actual value
                                        options = [
                                            (r["RO"], r["SO"]) for _, r in edit_df.iterrows()
                                        ]
                                        labels = [
                                            f"RO {r['RO']} | SO {r['SO']} → {r['Dest']}"
                                            for _, r in edit_df.iterrows()
                                        ]
                                        # Map index to label/value
                                        label_to_value = {labels[i]: options[i] for i in range(len(labels))}
                                        choice_label = st.selectbox("Select Release Order / Sales Order", labels, index=0)
                                        sel_ro, sel_so = label_to_value[choice_label]
                    
                                        # Current values for the selection
                                        current = edit_df[(edit_df["RO"] == sel_ro) & (edit_df["SO"] == sel_so)].iloc[0]
                                        total_val = float(current["Total"])
                                        shipped_val = float(current["Shipped"])
                                        remaining_val = float(current["Remaining"])
                    
                                        st.caption(f"Total (read-only): **{total_val:,.0f}**")
                    
                                        new_shipped = st.number_input(
                                            "Shipped",
                                            min_value=0.0,
                                            max_value=float(total_val),
                                            value=float(shipped_val),
                                            step=1.0,
                                            help="Must be ≤ Total"
                                        )
                    
                                        # User can edit Remaining, but we'll reconcile on save
                                        new_remaining = st.number_input(
                                            "Remaining",
                                            min_value=0.0,
                                            max_value=float(total_val),
                                            value=float(remaining_val),
                                            step=1.0,
                                            help="Should equal Total − Shipped; adjusted on save if not."
                                        )
    
                                        # NEW: user + reason for logging
                                        user_name = st.text_input("Your name/initials (for log)", value="", placeholder="e.g., AY",key="inp_user")
                                        reason = st.text_area("Reason for change", value="", placeholder="Explain why you're updating the numbers",key="inp_reason")

                                        submitted = st.form_submit_button(
                                                "Save Changes", use_container_width=True)
                    
                                        if submitted:
                                            if user_name.strip() == "" or reason.strip() == "":
                                                st.error("Please enter your name/initials and a reason before saving.")
                                                st.stop()
                                            else:
                                                
                                                # Reconcile: enforce Remaining = Total − Shipped (and clamp at 0)
                                                new_shipped = float(new_shipped)
                                                fixed_remaining = max(0.0, float(total_val) - new_shipped)
                                                # Soft validation
                                                if abs((new_shipped + new_remaining) - total_val) > 1e-6:
                                                    st.warning("Shipped + Remaining ≠ Total. Remaining will be set to (Total − Shipped) on save.")
    
                                                try:
                                                    # --- capture old values BEFORE update ---
                                                    old_shipped = float(raw_ro[str(sel_ro)][str(sel_so)]["shipped"])
                                                    delta_shipped = new_shipped - old_shipped
                                                except:
                                                    pass
                        
                                                # Update underlying JSON
                                                try:
                                                    
                                                    # Update raw_ro structure
                                                    raw_ro[str(sel_ro)][str(sel_so)]["shipped"] = new_shipped
                                                    raw_ro[str(sel_ro)][str(sel_so)]["remaining"] = fixed_remaining
                        
                                                    # Upload back to GCS
                                                    storage_client = get_storage_client()
                                                    bucket = storage_client.bucket(target_bucket)
                                                    blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                                                    blob.upload_from_string(json.dumps(raw_ro))
                                                    
                                                   
                                                    try:
                                                        from zoneinfo import ZoneInfo  # Py3.9+
                                                        _tz = ZoneInfo("America/Los_Angeles")
                                                    except Exception:
                                                        import pytz
                                                        _tz = pytz.timezone("America/Los_Angeles")
                                                    
                                                    log_entry = {
                                                        "date": datetime.datetime.now(_tz).isoformat(timespec="seconds"),
                                                        "user": user_name.strip(),
                                                        "ro": str(sel_ro),
                                                        "sales_order": str(sel_so),
                                                        "prev_shipped": old_shipped,
                                                        "new_shipped": new_shipped,
                                                        "delta_shipped": delta_shipped,
                                                        "reason": reason.strip()
                                                    }
                                                    ro_log.append(log_entry)
                            
                                                    storage_client = get_storage_client()
                                                    bucket = storage_client.bucket(target_bucket)
                                                    blob = bucket.blob(rf"release_orders/ro_log.json")
                                                    blob.upload_from_string(json.dumps(ro_log))
                                                    st.success(
                                                        f"Saved: RO {sel_ro} / SO {sel_so} — "
                                                        f"Shipped {old_shipped:,.0f} → {new_shipped:,.0f} (Δ {delta_shipped:+,.0f}). "
                                                        f"Remaining={fixed_remaining:,.0f}. Log updated."
                                                    )
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Failed to save changes: {e}")
                                    # --- LOG VIEWER ---
                            # --- LOG VIEWER ---
                            st.markdown("---")
                            if st.checkbox("Show Edit Log"):
                                try:
                                    ro_log = safe_download_json(target_bucket, r"release_orders/ro_log.json", default=[])
                                    if not ro_log:
                                        st.info("No log entries found yet.")
                                    else:
                                        log_df = pd.DataFrame(ro_log)
                            
                                        # optional filter
                                        filter_current = st.checkbox("Filter to selected RO/SO", value=False)
                                        if filter_current and 'sel_ro' in locals() and 'sel_so' in locals():
                                            log_df = log_df[(log_df.get("ro","").astype(str) == str(sel_ro)) &
                                                            (log_df.get("sales_order","").astype(str) == str(sel_so))]
                            
                                        # clean types
                                        if "date" in log_df.columns:
                                            log_df["date"] = pd.to_datetime(log_df["date"], errors="coerce")
                                        for c in ["prev_shipped","new_shipped","delta_shipped"]:
                                            if c in log_df.columns:
                                                log_df[c] = pd.to_numeric(log_df[c], errors="coerce")
                            
                                        # backfill delta if missing
                                        if "delta_shipped" not in log_df.columns:
                                            log_df["delta_shipped"] = np.nan
                                        if "prev_shipped" in log_df.columns and "new_shipped" in log_df.columns:
                                            mask = log_df["delta_shipped"].isna()
                                            log_df.loc[mask, "delta_shipped"] = (
                                                log_df.loc[mask, "new_shipped"] - log_df.loc[mask, "prev_shipped"]
                                            )
                            
                                        def fmt_arrow(x):
                                            if pd.isna(x): return "—"
                                            x = float(x)
                                            if x > 0:  return f"↑ {x:,.0f}"
                                            if x < 0:  return f"↓ {abs(x):,.0f}"
                                            return "—"
                            
                                        log_df["Δ shipped"] = log_df["delta_shipped"].apply(fmt_arrow)
                                        if "date" in log_df.columns:
                                            log_df = log_df.sort_values("date", ascending=False)
                            
                                        display_cols = [c for c in ["date","user","ro","sales_order","reason",
                                                                    "prev_shipped","new_shipped","Δ shipped"]
                                                        if c in log_df.columns]
                                        st.dataframe(log_df[display_cols], use_container_width=True, height=320)
                            
                                except Exception as e:
                                    st.error(f"Could not load log: {e}")
                                    
                    
                    
                    duration=st.toggle("Duration Report")
                    if duration:
                        
                        temp_dict={}
                            
                        for rel_ord in raw_ro:
                            for sales in [i for i in raw_ro[rel_ord] if i in ["001","002","003","004","005"]]:
                                temp_dict[rel_ord,sales]={}
                                dest=raw_ro[rel_ord]['destination']
                                vessel=raw_ro[rel_ord][sales]['vessel']
                                total=raw_ro[rel_ord][sales]['total']
                                remaining=raw_ro[rel_ord][sales]['remaining']
                                temp_dict[rel_ord,sales]={'destination': dest,'vessel': vessel,'total':total,'remaining':remaining}
                        temp_df=pd.DataFrame(temp_dict).T
                      
                        temp_df= temp_df.rename_axis(['release_order','sales_order'], axis=0)
                    
                        temp_df['First Shipment'] = temp_df.index.map(admin_bill_of_ladings.groupby(['release_order','sales_order'])['issued'].first())
                        
                        for i in temp_df.index:
                            if temp_df.loc[i,'remaining']<=2:
                                try:
                                    temp_df.loc[i,"Last Shipment"]=admin_bill_of_ladings.groupby(['release_order','sales_order']).issued.last().loc[i]
                                except:
                                    temp_df.loc[i,"Last Shipment"]=datetime.datetime.now()
                                temp_df.loc[i,"Duration"]=(pd.to_datetime(temp_df.loc[i,"Last Shipment"])-pd.to_datetime(temp_df.loc[i,"First Shipment"])).days+1
                        
                        temp_df['First Shipment'] = temp_df['First Shipment'].fillna(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
                        temp_df['Last Shipment'] = temp_df['Last Shipment'].fillna(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
                        
                        ####
                        
                        def business_days(start_date, end_date):
                            return pd.date_range(start=start_date, end=end_date, freq=BDay())
                        temp_df['# of Shipment Days'] = temp_df.apply(lambda row: len(business_days(row['First Shipment'], row['Last Shipment'])), axis=1)
                        df_temp=admin_bill_of_ladings.copy()
                        df_temp["issued"]=[pd.to_datetime(i).date() for i in df_temp["issued"]]
                        for i in temp_df.index:
                            try:
                                temp_df.loc[i,"Utilized Shipment Days"]=df_temp.groupby(["release_order",'sales_order'])[["issued"]].nunique().loc[i,'issued']
                            except:
                                temp_df.loc[i,"Utilized Shipment Days"]=0
                        
                        temp_df['First Shipment'] = temp_df['First Shipment'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),'%d-%b,%Y'))
                        temp_df['Last Shipment'] = temp_df['Last Shipment'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),'%d-%b,%Y') if type(x)==str else None)
                        liste=['Duration','# of Shipment Days',"Utilized Shipment Days"]
                        for col in liste:
                            temp_df[col] = temp_df[col].apply(lambda x: f" {int(x)} days" if not pd.isna(x) else np.nan)
                        temp_df['remaining'] = temp_df['remaining'].apply(lambda x: int(x))
                        temp_df.columns=['Destination', 'Vessel', 'Total Units', 'Remaining Units', 'First Shipment',
                               'Last Shipment', 'Duration', '# of Calendar Shipment Days',
                               'Utilized Calendar Shipment Days']
                        st.dataframe(temp_df)
                        a=df_temp.groupby(["issued"])[['quantity']].sum()
                        a.index=pd.to_datetime(a.index)
                        labor=gcp_download(target_bucket,rf"trucks.json")
                        labor = json.loads(labor)
                        
                        labor=pd.DataFrame(labor).T
                        labor.index=pd.to_datetime(labor.index)
                        for index in a.index:
                            try:
                                a.loc[index,'cost']=labor.loc[index,'cost']
                            except:
                                pass
                        a['quantity']=2*a['quantity']
                        a['Per_Ton']=a['cost']/a['quantity']
                        trucks=df_temp.groupby(["issued"])[['vehicle']].count().vehicle.values
                        a.insert(0,'trucks',trucks)
                        a['Per_Ton']=round(a['Per_Ton'],1)
                        w=a.copy()
                        m=a.copy()
                        cost_choice=st.radio("Select Daily/Weekly/Monthly Cost Analysis",["DAILY","WEEKLY","MONTHLY"])
                        if cost_choice=="DAILY":
                            a['Per_Ton']=["${:.2f}".format(number) for number in a['Per_Ton']]
                            a['cost']=["${:.2f}".format(number) for number in a['cost']]
                            a.index=[i.date() for i in a.index]
                            a= a.rename_axis('Day', axis=0)
                            a.columns=["# of Trucks","Tons Shipped","Total Cost","Cost Per Ton"]
                            st.dataframe(a)
                        if cost_choice=="WEEKLY":
                            w.columns=["# of Trucks","Tons Shipped","Total Cost","Cost Per Ton"]
                            weekly=w.dropna()
                            weekly=weekly.resample('W').sum()
                            weekly['Cost Per Ton']=round(weekly['Total Cost']/weekly['Tons Shipped'],1)
                            weekly['Cost Per Ton']=["${:.2f}".format(number) for number in weekly['Cost Per Ton']]
                            weekly['Total Cost']=["${:.2f}".format(number) for number in weekly['Total Cost']]
                            weekly.index=[i.date() for i in weekly.index]
                            weekly= weekly.rename_axis('Week', axis=0)
                            st.dataframe(weekly)
                        if cost_choice=="MONTHLY":
                            m.columns=["# of Trucks","Tons Shipped","Total Cost","Cost Per Ton"]
                            monthly=m.dropna()
                            monthly=monthly.resample('M').sum()
                            monthly['Cost Per Ton']=round(monthly['Total Cost']/monthly['Tons Shipped'],1)
                            monthly['Cost Per Ton']=["${:.2f}".format(number) for number in monthly['Cost Per Ton']]
                            monthly['Total Cost']=["${:.2f}".format(number) for number in monthly['Total Cost']]
                            monthly.index=[calendar.month_abbr[k] for k in [i.month for i in monthly.index]]
                            monthly= monthly.rename_axis('Month', axis=0)
                            st.dataframe(monthly)
                        else:
                            #st.write("NO RELEASE ORDER FOR THIS VESSEL IN DATABASE")
                            pass
                        
                                
        
                        
        
        ##########  LOAD OUT  ##############
        
        
        
        if select=="LOADOUT" :
        

            map=gcp_download(target_bucket,rf"map.json")
            map=json.loads(map)
            release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
            release_order_database=json.loads(release_order_database)
            dispatched=gcp_download(target_bucket,rf"dispatched.json")
            dispatched=json.loads(dispatched)
            carrier_list=map['carriers']
            mill_info=map["mill_info"]
            
            bill_mapping=gcp_download(target_bucket,"bill_mapping.json")   ###### DOWNLOADS
            bill_mapping=json.loads(bill_mapping)
            mf_numbers_for_load=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")  ###### DOWNLOADS
            mf_numbers_for_load=json.loads(mf_numbers_for_load)

            bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")   ###### DOWNLOADS
            bill_of_ladings=json.loads(bill_of_ladings)
            
            no_dispatch=0
           
            number=None  
            if number not in st.session_state:
                st.session_state.number=number
                           
          
            double_load=False
            
            if len(dispatched.keys())>0 and not no_dispatch:
                loadout,schedule=st.tabs(["LOADOUT","SCHEDULE"])
                with schedule:
                    
                    
                    bill_for_schedule=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                    bill_for_schedule=json.loads(bill_for_schedule)
                    schedule=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
                    schedule=json.loads(schedule)
                    dfb=pd.DataFrame.from_dict(bill_for_schedule).T[1:]
                    #dfb=bill.copy()
                    dfb["St_Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in dfb["issued"]]
                    #dfb=dfb[dfb["St_Date"]==(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()]
                    scheduled=[]
                    today=str((datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date())
                    
                    selected_date_datetime=st.date_input("SELECT DATE",(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date())
                    selected_date=str(selected_date_datetime)
                    weekday_name = selected_date_datetime.strftime('%A')

                    schedule_display_tab1,schedule_display_tab2,schedule_display_tab3=st.tabs(["DAILY","WEEK","ALL SCHEDULE"])

                    
                    flattened_data = []
                    for date, locations in schedule.items():
                        for location, location_data in locations.items():
                            for order, carriers in location_data.items():
                                for carrier, shipments in carriers.items():
                                    for shipment in shipments:
                                        #dfb_=dfb[dfb["St_Date"]>datetime.datetime(2024,10,1)]
                                        dfb_=dfb.copy()
                                        
                                        status="NONE"
                                        # Split the shipment data if needed (separate IDs if joined by "|")
                                        if shipment in dfb_.index:
                                            status_="SHIPPED"
                                        else:
                                            status_="Scheduled"
                                        shipment_parts = shipment.split("|") if "|" in shipment else [shipment]
                                        carrier_=carrier.split("-")[1]
                                        flattened_data.append({
                                            "Date": date,
                                            "Location": location,
                                            "Order": order,
                                            "Carrier": carrier_,
                                            "EDI Bill Of Lading":shipment,
                                            "Shipment ID": shipment_parts[0],
                                            "MF Number": shipment_parts[1] if len(shipment_parts) > 1 else None,
                                            "Status":status_
                                        })

                            # Convert to DataFrame
                                flat_df = pd.DataFrame(flattened_data)
                                all_schedule_flat_df=flat_df.copy()
                                #flat_df["Status"]=["Scheduled"]*len(flat_df)
                    def style_row(row,code=1):
                                    
                        if code==2:
                            shipment_status = row["Status"]
                            location = row["Location"]
                        else:
                            location = row['Destination']
                        # Define colors for different locations
                        colors = {
                            "CLATSKANIE": "background-color: #d1e7dd;",  # light green
                            "LEWISTON": "background-color: #ffebcd;",    # light coral
                            "HALSEY": "background-color: #add8e6;",      # light blue
                        }
                        
                        # Base style for the entire row based on location
                        base_style = colors.get(location, "")
                        if code==2:
                            if shipment_status == "SHIPPED":
                                base_style += "font-weight: lighter; font-style: italic; text-decoration: line-through;"  # Less bold, italic, and strikethrough
                            else:
                                base_style += "font-weight: bold;"  # Slightly bolder for other statuses
                
                        return [base_style] * len(row)            
                    with schedule_display_tab1:    
                        if selected_date in schedule:
                            st.markdown(   f"""
                                        <div style="font-size:20px; font-weight:bold; color:#808000;">
                                            Action On {selected_date}, {weekday_name}
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            st.write("")
                            dfb=dfb[dfb["St_Date"]==selected_date_datetime]
                            for dest in schedule[selected_date]:
                                for rel in schedule[selected_date][dest]:
                                    for carrier in schedule[selected_date][dest][rel]:
                                        scheduled.append({"Destination":dest,
                                              "Release Order":rel,"Sales Item":"001",
                                              "ISP":release_order_database[rel]["001"]['grade'],
                                              "Prep":release_order_database[rel]["001"]['unitized'],
                                              "Carrier":carrier.split("-")[1],
                                              "Scheduled":int(len(schedule[selected_date][dest][rel][carrier])),
                                              "Loaded":int(dfb[(dfb["release_order"]==rel)&(dfb["sales_order"]=="001")&
                                               (dfb["carrier_id"]==str(carrier.split("-")[0]))].vehicle.count()),
                                              "Remaining":0})
                            scheduled=pd.DataFrame(scheduled)
                            scheduled["Scheduled"] = scheduled["Scheduled"].astype(int)
                            scheduled["Loaded"] = scheduled["Loaded"].astype(int)
                            scheduled["Remaining"] = scheduled["Remaining"].astype(int)
    
                            if len(scheduled)>0:
                                
                                scheduled["Remaining"]=[int(i) for i in scheduled["Scheduled"]-scheduled["Loaded"]]
                                scheduled.loc["Total",["Scheduled","Loaded","Remaining"]]=scheduled[["Scheduled","Loaded","Remaining"]].sum()
                                #scheduled.set_index('Destination',drop=True,inplace=True)
                                scheduled["Scheduled"] = scheduled["Scheduled"].astype(int)
                                scheduled["Loaded"] = scheduled["Loaded"].astype(int)
                                scheduled["Remaining"] = scheduled["Remaining"].astype(int)
                                scheduled.loc["Total",["Scheduled","Loaded","Remaining"]].astype(int)
                                scheduled.fillna("",inplace=True)
    
    
                                
                                
                                flat_df_for_day=flat_df[flat_df.Date==selected_date]
                                flat_df_for_day.reset_index(drop=True,inplace=True)
                                flat_df_for_day.index+=1
                                list_view=st.checkbox("LIST VIEW")
                                                                  
    
                                #styled_schedule =scheduled.style.apply(style_row, axis=1)
                                if list_view:
                                    styled_schedule = flat_df_for_day.style.apply(lambda row: style_row(row, code=2), axis=1)
                                else:
                                    styled_schedule = scheduled.style.apply(lambda row: style_row(row, code=1), axis=1)
    
                             
                                st.write(styled_schedule.to_html(), unsafe_allow_html=True)
            
                        else:
                            st.markdown(   f"""
                                        <div style="font-size:20px; font-weight:bold; color:#808000;">
                                            Nothing scheduled on {selected_date}, {weekday_name}
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            st.write("")
                    with schedule_display_tab2:
                                  
                        flat_df["Status"] = ["Scheduled"] * len(flat_df)
                        flat_df.reset_index(drop=True, inplace=True)
                        flat_df.index += 1
                        
                        # Convert "Date" to datetime and set as index
                        flat_df["Date"] = pd.to_datetime(flat_df["Date"])
                        flat_df.insert(1,"Day",flat_df["Date"].dt.day_name())
                        
                        today=(datetime.datetime.today()-datetime.timedelta(hours=utc_difference)).date()
                        start_of_this_week = today - datetime.timedelta(days=today.weekday()) 
                        start_of_next_week = (start_of_this_week + datetime.timedelta(days=7))
                        end_of_next_week = (start_of_next_week + datetime.timedelta(days=7))
                        
                        if today.weekday() < 5:  # 0 = Monday, 4 = Friday
                            # Filter to display only the current week
                            weekly_display = flat_df[(flat_df["Date"]<pd.Timestamp(start_of_next_week))&
                                                     (flat_df["Date"]>pd.Timestamp(start_of_this_week))]
                        else:
                            # Display the upcoming week
                            weekly_display = flat_df[(flat_df["Date"]>pd.Timestamp(start_of_next_week))&
                                                     (flat_df["Date"]<pd.Timestamp(end_of_next_week))]


                        
                        weekly_counts = (
                            weekly_display.groupby([weekly_display["Date"].dt.day_name(), "Location"])
                            .size()
                            .unstack(fill_value=0)
                        )
                        
                        
                        # Define weekdays to display in the table (Monday to Friday)
                        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                        weekly_counts_ = weekly_counts.reindex(weekdays)
                        
                        weekly_counts_.fillna("",inplace=True)
                        weekly_counts_ = weekly_counts_.applymap(lambda x: int(x) if isinstance(x, float) else x)

                        
                        
                        st.write(weekly_counts_)
                        # Generate HTML table with large squares
                        weekly_counts = weekly_counts.dropna(axis=0)
                        html_content = """
                        <style>
                            .day-table {
                                display: flex;
                                justify-content: space-around;
                                margin-top: 20px;
                            }
                            .day-cell {
                                width: 150px;
                                height: 150px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                align-items: center;
                                border: 2px solid #ddd;
                                font-size: 20px;
                                font-weight: bold;
                                background-color: #f9f9f9;
                            }
                            .day-name {
                                font-size: 18px;
                                color: #333;
                                margin-bottom: 10px;
                            }
                            .shipment-info {
                                font-size: 16px;
                                color: #555;
                            }
                        </style>
                        
                        <div class="day-table">
                        """
                        
                        # Populate each weekday's cell with shipment information
                        for day in weekdays:
                            shipments = weekly_counts.loc[day] if day in weekly_counts.index else None
                            html_content += f"""
                            <div class="day-cell">
                                <div class="day-name">{day}</div>
                            """
                            if shipments is not None:
                                for destination, count in shipments.items():
                                    html_content += f'<div class="shipment-info">{count}  x  {destination.title()}</div>'
                            else:
                                html_content += '<div class="shipment-info">No Shipments</div>'
                            html_content += "</div>"
                        
                        html_content += "</div>"
                        
                        # Display the HTML table in Streamlit
                        st.components.v1.html(html_content, height=300, scrolling=False)
                    with schedule_display_tab3:
                        #flat_df["Status"]=["Scheduled"]*len(flat_df)
                        all_schedule_flat_df.reset_index(drop=True,inplace=True)
                        all_schedule_flat_df.index+=1
                        styled_df =all_schedule_flat_df.style.apply(lambda row: style_row(row, code=2), axis=1)
                        st.write(styled_df.to_html(), unsafe_allow_html=True)

### ADMINL<OAOUDUT                
                with loadout:
                    
                    menu_destinations={}
                    
                    for rel_ord in dispatched.keys():
                        for sales in dispatched[rel_ord]:
                            try:
                                menu_destinations[f"{rel_ord} -{sales}"]=dispatched[rel_ord][sales]["destination"]
                            except:
                                pass
                    if 'work_order_' not in st.session_state:
                        st.session_state.work_order_ = None
                        
                    liste=[f"{i} to {menu_destinations[i]}" for i in menu_destinations.keys()]
                
                    work_order_=st.selectbox("**SELECT RELEASE ORDER/SALES ORDER TO WORK**",liste,index=0 if st.session_state.work_order_ else 0) 
                    st.session_state.work_order_=work_order_
                    if work_order_:
                        
                        work_order=work_order_.split(" ")[0]
                        order=["001","002","003","004","005","006"]
                        
                        current_release_order=work_order
                        current_sales_order=work_order_.split(" ")[1][1:]
                        vessel=release_order_database[current_release_order][current_sales_order]["vessel"]
                        destination=release_order_database[current_release_order]['destination']
                        
                        
                        
              
                        try:
                            next_release_order=dispatched['002']['release_order']    #############################  CHECK HERE ######################## FOR MIXED LOAD
                            next_sales_order=dispatched['002']['sales_order']
                            
                        except:
                            
                            pass
                        
        
                        
                            
                        
                        st.markdown(rf'**:blue[CURRENTLY WORKING] :**')
                        load_col1,load_col2=st.columns([9,1])
                        
                        with load_col1:
                            wrap_dict={"ISU":"UNWRAPPED","ISP":"WRAPPED","AEP":"WRAPPED"}
                            wrap=release_order_database[current_release_order][current_sales_order]["grade"]
                            ocean_bill_of_=release_order_database[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                            unitized=release_order_database[current_release_order][current_sales_order]["unitized"]
                            quant_=release_order_database[current_release_order][current_sales_order]["total"]
                            real_quant=int(math.floor(quant_))
                            ship_=release_order_database[current_release_order][current_sales_order]["shipped"]
                            ship_bale=(ship_-math.floor(ship_))*8
                            remaining=release_order_database[current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                            temp={f"<b>Release Order #":current_release_order,"<b>Destination":destination,"<b>VESSEL":vessel}
                            temp2={"<b>Ocean B/L":ocean_bill_of_,"<b>Type":wrap_dict[wrap],"<b>Prep":unitized}
                            temp3={"<b>Total Units":quant_,"<b>Shipped Units":ship_,"<b>Remaining Units":remaining}
                            #temp4={"<b>Total Bales":0,"<b>Shipped Bales":int(8*(ship_-math.floor(ship_))),"<b>Remaining Bales":int(8*(remaining-math.floor(remaining)))}
                            temp5={"<b>Total Tonnage":quant_*2,"<b>Shipped Tonnage":ship_*2,"<b>Remaining Tonnage":quant_*2-(ship_*2)}
        
        
                            
                            sub_load_col1,sub_load_col2,sub_load_col3,sub_load_col4=st.columns([3,3,3,3])
                            
                            with sub_load_col1:   
                                st.markdown(rf'**Release Order-{current_release_order}**')
                                st.markdown(rf'**Destination : {destination}**')
                                st.markdown(rf'**VESSEL-{vessel}**')
                                #st.write (pd.DataFrame(temp.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                         
                            with sub_load_col2:
                                st.markdown(rf'**Ocean B/L: {ocean_bill_of_}**')
                                st.markdown(rf'**Type : {wrap_dict[wrap]}**')
                                st.markdown(rf'**Prep : {unitized}**')
                                #st.write (pd.DataFrame(temp2.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                           
                                
                            with sub_load_col3:
            
                                st.markdown(rf'**Total Units : {quant_}**')
                                st.markdown(rf'**Shipped Units : {ship_}**')
                                if remaining<=10:
                                    st.markdown(rf'**:red[CAUTION : Remaining : {remaining} Units]**')
                                else:
                                    st.markdown(rf'**Remaining Units : {remaining}**')
                            with sub_load_col4:
                                
                            
                                #st.write (pd.DataFrame(temp5.items(),columns=["TONNAGE","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                                
                                st.markdown(rf'**Total Tonnage : {quant_*2}**')
                                st.markdown(rf'**Shipped Tonnage : {ship_*2}**')
                                st.markdown(rf'**Remaining Tonnage : {remaining*2}**')
                        
                        with load_col2:
                            if double_load:
                                
                                try:
                                    st.markdown(rf'**NEXT ITEM : Release Order-{next_release_order}**')
                                    st.markdown(rf'**Sales Order Item-{next_sales_order}**')
                                    st.markdown(f'**Ocean Bill Of Lading : {info[vessel][next_release_order][next_sales_order]["ocean_bill_of_lading"]}**')
                                    st.markdown(rf'**Total Quantity : {info[vessel][next_release_order][next_sales_order]["quantity"]}**')
                                except:
                                    pass
        
                        st.divider()
                        ###############    LOADOUT DATA ENTRY    #########
                        
                        col1, col2,col3,col4= st.columns([2,2,2,2])
                        yes=False
                      
                        release_order_number=current_release_order
                        medium="TRUCK"
                        
                        with col1:
                        ######################  LETS GET RID OF INPUT BOXES TO SIMPLIFY LONGSHORE SCREEN
                   
                            terminal_code="OLYM"
                            
                            file_date=(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()
                            if file_date not in st.session_state:
                                st.session_state.file_date=file_date
                            file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False)
                          
                            delivery_date=(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()
                            eta_date=delivery_date
                            
                            # carrier_code=release_order_database[current_release_order][current_sales_order]["carrier_code"]
                            vessel=release_order_database[current_release_order][current_sales_order]["vessel"]
                            transport_sequential_number="TRUCK"
                            transport_type="TRUCK"
                            placeholder = st.empty()
                            with placeholder.container():
                                today_str=str(st.date_input("Shipment Date",(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date(),disabled=False,key="popdo3"))
                                vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=7)
                                manual_bill=st.toggle("Toggle for Manual BOL")
                                if manual_bill:
                                    manual_bill_of_lading_number=st.text_input("ENTER BOL",key="eirufs")
                                mf=True
                                mf_mix=False
                                load_mf_number_issued=False
                                dest=destination.split("-")[1].split(",")[0].upper()
                                if destination in ["GP-Halsey,OR","GP-Clatskanie,OR"]:
                                    carrier_code=st.selectbox("Carrier Code",["311627-KBX"],disabled=True,key=29)
                                                                
                                else:
                                    carrier_code=st.selectbox("Carrier Code",list(mf_numbers_for_load[today_str][dest][release_order_number].keys()),disabled=False,key=29)
                                   
                                
                                
                                if 'load_mf_number' not in st.session_state:
                                    st.session_state.load_mf_number = None
                                    
                                if today_str in mf_numbers_for_load.keys():
                                    try:
                                        mf_liste=[i for i in mf_numbers_for_load[today_str][dest][release_order_number][f"{carrier_code.split('-')[0]}-{carrier_code.split('-')[1].upper()}"]]
                                    except:
                                        mf_liste=[]
                                    if len(mf_liste)>0:
                                        try:
                                            load_mf_number = st.selectbox("MF NUMBER", mf_liste, disabled=False, key=14551, index=mf_liste.index(st.session_state.load_mf_number) if st.session_state.load_mf_number else 0)
                                        except:
                                            load_mf_number = st.selectbox("MF NUMBER", mf_liste, disabled=False, key=14551)
                                        mf=True
                                        load_mf_number_issued=True
                                        yes=True
                                        st.session_state.load_mf_number = load_mf_number
                                        if "|" in load_mf_number:
                                            mf_mix=True
                                            otm_number=f"{load_mf_number.split('|')[1]}"
                                            mf_number_split=f"{load_mf_number.split('|')[0]}"
                                        else:
                                            otm_number=f"{load_mf_number}"
                                       
                                    else:
                                        st.write(f"**:red[ASK ADMIN TO PUT SHIPMENT NUMBERS]**")
                                        mf=False
                                        yes=False
                                        load_mf_number_issued=False  
                                else:
                                    st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
                                    mf=False
                                    yes=False
                                    load_mf_number_issued=False  
                                   
                                foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=8)
                                foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=123)
                            click_clear1 = st.button('CLEAR VEHICLE-QUANTITY INPUTS', key=34)
                            if click_clear1:
                                 with placeholder.container():
                                     vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=17)
                            
                                     mf=True
                                     load_mf_number_issued=False
                                     carrier_code=st.text_input("Carrier Code",info[current_release_order][current_sales_order]["carrier_code"],disabled=True,key=19)
                                    
                                     if carrier_code=="123456-KBX":
                                       if release_order_number in mf_numbers_for_load.keys():
                                           mf_liste=[i for i in mf_numbers_for_load[release_order_number]]
                                           if len(mf_liste)>0:
                                               load_mf_number=st.selectbox("SHIPMENT NUMBER",mf_liste,disabled=False,key=14551)
                                               mf=True
                                               load_mf_number_issued=True
                                               yes=True
                                           else:
                                               st.write(f"**:red[ASK ADMIN TO PUT SHIPMENT NUMBERS]**")
                                               mf=False
                                               yes=False
                                               load_mf_number_issued=False  
                                       else:
                                           st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
                                           mf=False
                                           yes=False
                                           load_mf_number_issued=False  
                                     foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=18)
                                     foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=1123)
                                     
                                   
                            
                                    
                            
                        with col2:
                            ocean_bol_to_batch = {"GSSWKIR6013D": 45302855,"GSSWKIR6013E": 45305548}
                            if double_load:
                                #release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                                release_order_number=current_release_order
                                #sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                                sales_order_item=current_sales_order
                                #ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                                ocean_bill_of_lading=info[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                                current_ocean_bill_of_lading=ocean_bill_of_lading
                                next_ocean_bill_of_lading=info[next_release_order][next_sales_order]["ocean_bill_of_lading"]
                                #batch=st.text_input("Batch",info[current_release_order][current_sales_order]["batch"],disabled=True)
                                batch=info[current_release_order][current_sales_order]["batch"]
                                #grade=st.text_input("Grade",info[current_release_order][current_sales_order]["grade"],disabled=True)
                                grade=info[current_release_order][current_sales_order]["grade"]
                                
                                #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                                pass
                            else:
                                release_order_number=current_release_order
                                sales_order_item=current_sales_order
                                ocean_bill_of_lading=release_order_database[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                                
                                batch=release_order_database[current_release_order][current_sales_order]["batch"]
                         
                                grade=release_order_database[current_release_order][current_sales_order]["grade"]
                                
                       
                            updated_quantity=0
                            live_quantity=0
                            if updated_quantity not in st.session_state:
                                st.session_state.updated_quantity=updated_quantity
                            load_digit=-2 if vessel=="KIRKENES-2304" else -3
                            def audit_unit(x):
                                if vessel=="LAGUNA-3142":
                                    return True
                                if vessel=="KIRKENES-2304":
                                    if len(x)>=10:
                                        if bill_mapping[vessel][x[:-2]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-2]]["Batch"]!=batch:
                                            st.write("**:red[WRONG B/L, DO NOT LOAD BELOW!]**")
                                            return False
                                        else:
                                            return True
                                else:
                                    if bill_mapping[vessel][x[:-3]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-3]]["Batch"]!=batch:
                                        return False
                                    else:
                                        return True
                            def audit_split(release,sales):
                                if vessel=="KIRKENES-2304":
                                    if len(x)>=10:
                                        if bill_mapping[vessel][x[:-2]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-2]]["Batch"]!=batch:
                                            st.write("**:red[WRONG B/L, DO NOT LOAD BELOW!]**")
                                            return False
                                        else:
                                            return True
                                else:
                                    if bill_mapping[vessel][x[:-3]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-3]]["Batch"]!=batch:
                                            return False
                                    else:
                                        return True
                            
                            flip=False 
                            first_load_input=None
                            second_load_input=None
                            load_input=None
                            bale_load_input=None
                            if double_load:
                                
                                try:
                                    next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                                    
                                    first_load_input=st.text_area("**FIRST SKU LOADS**",height=300)
                                    first_quantity=0
                                    second_quantity=0
                                    if first_load_input is not None:
                                        first_textsplit = first_load_input.splitlines()
                                        first_textsplit=[i for i in first_textsplit if len(i)>8]
                                        first_quantity=len(first_textsplit)
                                    second_load_input=st.text_area("**SECOND SKU LOADS**",height=300)
                                    if second_load_input is not None:
                                        second_textsplit = second_load_input.splitlines()
                                        second_textsplit = [i for i in second_textsplit if len(i)>8]
                                        second_quantity=len(second_textsplit)
                                    updated_quantity=first_quantity+second_quantity
                                    st.session_state.updated_quantity=updated_quantity
                                except Exception as e: 
                                    st.write(e)
                                    #st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                                    pass
                                
                            
                            else:
                                
                
                
                                placeholder1 = st.empty()
                                
                                
                                
                                load_input=placeholder1.text_area("**UNITS**",value="",height=300,key=1)#[:-2]
                            
                                
                        with col3:
                            placeholder2 = st.empty()
                            bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1111)#[:-2]
                            
                                
                        with col2:
                            click_clear = st.button('CLEAR SCANNED INPUTS', key=3)
                            if click_clear:
                                load_input = placeholder1.text_area("**UNITS**",value="",height=300,key=2)#
                                bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1121)#
                            if load_input is not None :
                                textsplit = load_input.splitlines()
                                textsplit=[i for i in textsplit if len(i)>8]
                                updated_quantity=len(textsplit)
                                st.session_state.updated_quantity=updated_quantity
                            if bale_load_input is not None:
                                bale_textsplit = bale_load_input.splitlines()
                                bale_textsplit=[i for i in bale_textsplit if len(i)>8]
                                bale_updated_quantity=len(bale_textsplit)
                                st.session_state.updated_quantity=updated_quantity+bale_updated_quantity*0.125
                           
                        with col4:
                            
                            if double_load:
                                first_faults=[]
                                if first_load_input is not None:
                                    first_textsplit = first_load_input.splitlines()
                                    first_textsplit=[i for i in first_textsplit if len(i)>8]
                                    #st.write(textsplit)
                                    for j,x in enumerate(first_textsplit):
                                        if audit_split(current_release_order,current_sales_order):
                                            st.text_input(f"Unit No : {j+1}",x)
                                            first_faults.append(0)
                                        else:
                                            st.text_input(f"Unit No : {j+1}",x)
                                            first_faults.append(1)
                                second_faults=[]
                                if second_load_input is not None:
                                    second_textsplit = second_load_input.splitlines()
                                    second_textsplit = [i for i in second_textsplit if len(i)>8]
                                    #st.write(textsplit)
                                    for i,x in enumerate(second_textsplit):
                                        if audit_split(next_release_order,next_sales_order):
                                            st.text_input(f"Unit No : {len(first_textsplit)+1+i}",x)
                                            second_faults.append(0)
                                        else:
                                            st.text_input(f"Unit No : {j+1+i+1}",x)
                                            second_faults.append(1)
                
                                loads=[]
                                
                                for k in first_textsplit:
                                    loads.append(k)
                                for l in second_textsplit:
                                    loads.append(l)
                                    
                            ####   IF NOT double load
                            else:
                                
                                past_loads=[]
                                for row in bill_of_ladings.keys():
                                    if row!="11502400":
                                        past_loads+=[load for load in bill_of_ladings[row]['loads'].keys()]
                                faults=[]
                                bale_faults=[]
                                fault_messaging={}
                                bale_fault_messaging={}
                                textsplit={}
                                bale_textsplit={}
                                if load_input is not None:
                                    textsplit = load_input.splitlines()
                                    
                                        
                                    textsplit=[i for i in textsplit if len(i)>8]
                               
                                    seen=set()
                                    alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                    for i,x in enumerate(textsplit):
                                        alternate_vessels=[ship for ship in bill_mapping if ship!=vessel]
                                        if x[:load_digit] in ["ds"]:
                                            st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                            faults.append(1)
                                            st.markdown("**:red[THIS LOT# IS FROM THE OTHER VESSEL!]**")
                                        else:
                                                
                                            if x[:load_digit] in bill_mapping[vessel] or vessel=="LAGUNA-3142" :
                                                if audit_unit(x):
                                                    if x in seen:
                                                        st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                        faults.append(1)
                                                        st.markdown("**:red[This unit has been scanned TWICE!]**")
                                                    if x in past_loads:
                                                        st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                        faults.append(1)
                                                        st.markdown("**:red[This unit has been SHIPPED!]**")   
                                                    else:
                                                        st.write(f"**Unit No : {i+1}-{x}**")
                                                        faults.append(0)
                                                else:
                                                    st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                    st.write(f"**:red[WRONG B/L, DO NOT LOAD UNIT {x}]**")
                                                    faults.append(1)
                                                seen.add(x)
                                            else:
                                                #st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                faults.append(1)
                                                #st.markdown("**:red[This LOT# NOT IN INVENTORY!]**")
                                                #st.info(f"VERIFY THIS UNIT CAME FROM {vessel} - {'Unwrapped' if grade=='ISU' else 'wrapped'} piles")
                                                with st.expander(f"**:red[Unit No : {i+1}-{x} This LOT# NOT IN INVENTORY!---VERIFY UNIT {x} CAME FROM {vessel} - {'Unwrapped' if grade=='ISU' else 'wrapped'} piles]**"):
                                                    st.write("Verify that the unit came from the pile that has the units for this release order and click to inventory")
                                                    if st.button("ADD UNIT TO INVENTORY",key=f"{x}"):
                                                        updated_bill=bill_mapping.copy()
                                                        updated_bill[vessel][x[:load_digit]]={"Batch":batch,"Ocean_bl":ocean_bill_of_lading}
                                                        updated_bill=json.dumps(updated_bill)
                                                        #storage_client = storage.Client()
                                                        storage_client = get_storage_client()
                                                        bucket = storage_client.bucket(target_bucket)
                                                        blob = bucket.blob(rf"bill_mapping.json")
                                                        blob.upload_from_string(updated_bill)
        
                                                        alien_units=gcp_download(target_bucket,rf"alien_units.json")
                                                        alien_units=json.loads(alien_units)
                                                        if vessel not in alien_units:
                                                            alien_units[vessel]={}
                                                        alien_units[vessel][x]={}
                                                        alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                                "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                                        alien_units=json.dumps(alien_units)
                                                        #storage_client = storage.Client()
                                                        storage_client = get_storage_client()
                                                        bucket = storage_client.bucket(target_bucket)
                                                        blob = bucket.blob(rf"alien_units.json")
                                                        blob.upload_from_string(alien_units)
                                                        
                                                        
                                                        subject=f"FOUND UNIT {x} NOT IN INVENTORY"
                                                        body=f"Clerk identified an uninventoried {'Unwrapped' if grade=='ISU' else 'wrapped'} unit {x}, and after verifying the physical pile, inventoried it into Ocean Bill Of Lading : {ocean_bill_of_lading} for vessel {vessel}. Unit has been put into alien unit list."
                                                        sender = "warehouseoly@gmail.com"
                                                        #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                                        recipients = ["afsiny@portolympia.com"]
                                                        password = "xjvxkmzbpotzeuuv"
                                                        send_email(subject, body, sender, recipients, password)
                                                        time.sleep(0.1)
                                                        st.success(f"Added Unit {x} to Inventory!",icon="✅")
                                                        st.rerun()
                                        
                                if bale_load_input is not None:
                                
                                    bale_textsplit = bale_load_input.splitlines()                       
                                    bale_textsplit=[i for i in bale_textsplit if len(i)>8]                           
                                    seen=set()
                                    alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                    for i,x in enumerate(bale_textsplit):
                                        alternate_vessel=[ship for ship in bill_mapping if ship!=vessel][0]
                                        if x[:load_digit] in bill_mapping[alternate_vessel]:
                                            st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                            faults.append(1)
                                            st.markdown("**:red[THIS BALE LOT# IS FROM THE OTHER VESSEL!]**")
                                        else:
                                            if x[:load_digit] in bill_mapping[vessel]:
                                                if audit_unit(x):
                                                    st.write(f"**Bale No : {i+1}-{x}**")
                                                    faults.append(0)
                                                else:
                                                    st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                    st.write(f"**:red[WRONG B/L, DO NOT LOAD BALE {x}]**")
                                                    faults.append(1)
                                                seen.add(x)
                                            else:
                                                faults.append(1)
                                                with st.expander(f"**:red[Bale No : {i+1}-{x} This LOT# NOT IN INVENTORY!---VERIFY BALE {x} CAME FROM {vessel} - {'Unwrapped' if grade=='ISU' else 'wrapped'} piles]**"):
                                                    st.write("Verify that the bale came from the pile that has the units for this release order and click to inventory")
                                                    if st.button("ADD BALE TO INVENTORY",key=f"{x}"):
                                                        updated_bill=bill_mapping.copy()
                                                        updated_bill[vessel][x[:load_digit]]={"Batch":batch,"Ocean_bl":ocean_bill_of_lading}
                                                        updated_bill=json.dumps(updated_bill)
                                                        #storage_client = storage.Client()
                                                        storage_client = get_storage_client()
                                                        bucket = storage_client.bucket(target_bucket)
                                                        blob = bucket.blob(rf"bill_mapping.json")
                                                        blob.upload_from_string(updated_bill)
        
                                                        alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                                        alien_units[vessel][x]={}
                                                        alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                                "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                                        alien_units=json.dumps(alien_units)
                                                       # storage_client = storage.Client()
                                                        storage_client = get_storage_client()
                                                        bucket = storage_client.bucket(target_bucket)
                                                        blob = bucket.blob(rf"alien_units.json")
                                                        blob.upload_from_string(alien_units)
                                                        
                                                        
                                                        subject=f"FOUND UNIT {x} NOT IN INVENTORY"
                                                        body=f"Clerk identified an uninventoried {'Unwrapped' if grade=='ISU' else 'wrapped'} unit {x}, and after verifying the physical pile, inventoried it into Ocean Bill Of Lading : {ocean_bill_of_lading} for vessel {vessel}. Unit has been put into alien unit list."
                                                        sender = "warehouseoly@gmail.com"
                                                        #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                                        recipients = ["afsiny@portolympia.com"]
                                                        password = "xjvxkmzbpotzeuuv"
                                                        send_email(subject, body, sender, recipients, password)
                                                        time.sleep(0.1)
                                                        st.success(f"Added Unit {x} to Inventory!",icon="✅")
                                                       # st.rerun()
                               
                                   
                                loads={}
                                pure_loads={}
                                yes=True
                                if 1 in faults or 1 in bale_faults:
                                    yes=False
                                
                                if yes:
                                    pure_loads={**{k:0 for k in textsplit},**{k:0 for k in bale_textsplit}}
                                    loads={**{k[:load_digit]:0 for k in textsplit},**{k[:load_digit]:0 for k in bale_textsplit}}
                                    for k in textsplit:
                                        loads[k[:load_digit]]+=1
                                        pure_loads[k]+=1
                                    for k in bale_textsplit:
                                        loads[k[:load_digit]]+=0.125
                                        pure_loads[k]+=0.125
                        with col3:
                            quantity=st.number_input("**Scanned Quantity of Units**",st.session_state.updated_quantity, key=None, help=None, on_change=None, disabled=True, label_visibility="visible")
                            st.markdown(f"**{quantity*2} TONS - {round(quantity*2*2204.62,1)} Pounds**")
                            
                            admt=round(float(release_order_database[current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
                            st.markdown(f"**ADMT= {admt} TONS**")
                        manual_time=False   
                        #st.write(faults)                  
                        if st.checkbox("Check for Manual Entry for Date/Time"):
                            manual_time=True
                            file_date=st.date_input("File Date",datetime.datetime.today(),disabled=False,key="popo3")
                            a=datetime.datetime.strftime(file_date,"%Y%m%d")
                            a_=datetime.datetime.strftime(file_date,"%Y-%m-%d")
                            file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False,key="popop")
                            b=file_time.strftime("%H%M%S")
                            b_=file_time.strftime("%H:%M:%S")
                            c=datetime.datetime.strftime(eta_date,"%Y%m%d")
                        else:     
                            
                            a=datetime.datetime.strftime(file_date,"%Y%m%d")
                            a_=datetime.datetime.strftime(file_date,"%Y-%m-%d")
                            b=file_time.strftime("%H%M%S")
                            b=(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).strftime("%H%M%S")
                            b_=(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).strftime("%H:%M:%S")
                            c=datetime.datetime.strftime(eta_date,"%Y%m%d")
                            
                            
                        
                        liste=[(i,datetime.datetime.strptime(bill_of_ladings[i]['issued'],"%Y-%m-%d %H:%M:%S")) for i in bill_of_ladings if i!="11502400"]
                        liste.sort(key=lambda a: a[1])
                        last_submitted=[i[0] for i in liste[-3:]]
                        last_submitted.reverse()
                        st.markdown(f"**Last Submitted Bill Of Ladings (From most recent) : {last_submitted}**")
                        
                        
                        
                        if yes and mf and quantity>0:
                            
                            if st.button('**:blue[SUBMIT EDI]**'):
                             
                                proceed=True
                                if not yes:
                                    proceed=False
                                             
                                if fault_messaging.keys():
                                    for i in fault_messaging.keys():
                                        error=f"**:red[Unit {fault_messaging[i]}]**"
                                        proceed=False
                                if remaining<0:
                                    proceed=False
                                    error="**:red[No more Items to ship on this Sales Order]"
                                    st.write(error)
                                if not vehicle_id: 
                                    proceed=False
                                    error="**:red[Please check Vehicle ID]**"
                                    st.write(error)
                                
                                if quantity!=foreman_quantity+int(foreman_bale_quantity)/8:
                                    proceed=False
                                    error=f"**:red[{updated_quantity} units and {bale_updated_quantity} bales on this truck. Please check. You planned for {foreman_quantity} units and {foreman_bale_quantity} bales!]** "
                                    st.write(error)
                                if proceed:
                                    carrier_code_mf=f"{carrier_code.split('-')[0]}-{carrier_code.split('-')[1]}"
                                    carrier_code=carrier_code.split("-")[0]
                                    
                                    suzano_report=gcp_download(target_bucket,rf"suzano_report.json")
                                    suzano_report=json.loads(suzano_report)
        
                                    consignee=destination.split("-")[0]
                                    consignee_city=mill_info[destination]["city"]
                                    consignee_state=mill_info[destination]["state"]
                                    vessel_suzano,voyage_suzano=vessel.split("-")
                                    if manual_time:
                                        eta=datetime.datetime.strftime(file_date+datetime.timedelta(hours=mill_info[destination]['Driving_Hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['Driving_Minutes']+30),"%Y-%m-%d  %H:%M:%S")
                                    else:
                                        eta=datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=mill_info[destination]['Driving_Hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['Driving_Minutes']+30),"%Y-%m-%d  %H:%M:%S")
            
                                    if double_load:
                                        bill_of_lading_number,bill_of_ladings=gen_bill_of_lading()
                                        edi_name= f'{bill_of_lading_number}.txt'
                                        bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                                     "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                                     "quantity":len(first_textsplit),"issued":f"{a_} {b_}","edi_no":edi_name} 
                                        bill_of_ladings[str(bill_of_lading_number+1)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":next_sales_order,
                                                                                     "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                                     "quantity":len(second_textsplit),"issued":f"{a_} {b_}","edi_no":edi_name} 
                                    
                                    else:
                                        bill_of_lading_number,bill_of_ladings=gen_bill_of_lading()
                                        if load_mf_number_issued:
                                            bill_of_lading_number=st.session_state.load_mf_number
                                        if manual_bill:
                                            bill_of_lading_number=manual_bill_of_lading_number
                                        edi_name= f'{bill_of_lading_number}.txt'
                                        bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                                     "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                                     "quantity":st.session_state.updated_quantity,"issued":f"{a_} {b_}","edi_no":edi_name,"loads":pure_loads} 
                                                        
                                    bill_of_ladings=json.dumps(bill_of_ladings)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                                    blob.upload_from_string(bill_of_ladings)
                                    
                                    
                                    
                                    terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",bill_of_lading_number,disabled=True)
                                    success_container=st.empty()
                                    success_container.info("Uploading Bill of Lading")
                                    time.sleep(0.1) 
                                    success_container.success("Uploaded Bill of Lading...",icon="✅")
                                    process()
                                    #st.toast("Creating EDI...")
                                    try:
                                        suzano_report_keys=[int(i) for i in suzano_report.keys()]
                                        next_report_no=max(suzano_report_keys)+1
                                    except:
                                        next_report_no=1
                                    if double_load:
                                        
                                        suzano_report.update({next_report_no:{"Date Shipped":f"{a_} {b_}","Vehicle":vehicle_id, "Shipment ID #": bill_of_lading_number, "Consignee":consignee,"Consignee City":consignee_city,
                                                             "Consignee State":consignee_state,"Release #":release_order_number,"Carrier":carrier_code,
                                                             "ETA":eta,"Ocean BOL#":ocean_bill_of_lading,"Warehouse":"OLYM","Vessel":vessel_suzano,"Voyage #":voyage_suzano,"Grade":wrap,"Quantity":quantity,
                                                             "Metric Ton": quantity*2, "ADMT":admt,"Mode of Transportation":transport_type}})
                                    else:
                                       
                                        suzano_report.update({next_report_no:{"Date Shipped":f"{a_} {b_}","Vehicle":vehicle_id, "Shipment ID #": bill_of_lading_number, "Consignee":consignee,"Consignee City":consignee_city,
                                                             "Consignee State":consignee_state,"Release #":release_order_number,"Carrier":carrier_code,
                                                             "ETA":eta,"Ocean BOL#":ocean_bill_of_lading,"Batch#":batch,
                                                             "Warehouse":"OLYM","Vessel":vessel_suzano,"Voyage #":voyage_suzano,"Grade":wrap,"Quantity":quantity,
                                                             "Metric Ton": quantity*2, "ADMT":admt,"Mode of Transportation":transport_type}})
                                        suzano_report=json.dumps(suzano_report)
                                        #storage_client = storage.Client()
                                        storage_client = get_storage_client()
                                        bucket = storage_client.bucket(target_bucket)
                                        blob = bucket.blob(rf"suzano_report.json")
                                        blob.upload_from_string(suzano_report)
                                        success_container1=st.empty()
                                        time.sleep(0.1)                            
                                        success_container1.success(f"Updated Suzano Report",icon="✅")
            
                                      
                                        
                                    if double_load:
                                        info[current_release_order][current_sales_order]["shipped"]=info[current_release_order][current_sales_order]["shipped"]+len(first_textsplit)
                                        info[current_release_order][current_sales_order]["remaining"]=info[current_release_order][current_sales_order]["remaining"]-len(first_textsplit)
                                        info[next_release_order][next_sales_order]["shipped"]=info[next_release_order][next_sales_order]["shipped"]+len(second_textsplit)
                                        info[next_release_order][next_sales_order]["remaining"]=info[next_release_order][next_sales_order]["remaining"]-len(second_textsplit)
                                    else:
                                        release_order_database[current_release_order][current_sales_order]["shipped"]=release_order_database[current_release_order][current_sales_order]["shipped"]+quantity
                                        release_order_database[current_release_order][current_sales_order]["remaining"]=release_order_database[current_release_order][current_sales_order]["remaining"]-quantity
                                    if release_order_database[current_release_order][current_sales_order]["remaining"]<=0:
                                        to_delete=[]
                                        for release in dispatched.keys():
                                            if release==current_release_order:
                                                for sales in dispatched[release].keys():
                                                    if sales==current_sales_order:
                                                        to_delete.append((release,sales))
                                        for victim in to_delete:
                                            del dispatched[victim[0]][victim[1]]
                                            if len(dispatched[victim[0]].keys())==0:
                                                del dispatched[victim[0]]
                                        
                                        json_data = json.dumps(dispatched)
                                        #storage_client = storage.Client()
                                        storage_client = get_storage_client()
                                        bucket = storage_client.bucket(target_bucket)
                                        blob = bucket.blob(rf"dispatched.json")
                                        blob.upload_from_string(json_data)       
                                    def check_complete(data,ro):
                                        complete=True
                                        for i in data[ro]:
                                            if i in ["001","002","003","004","005"]:
                                                if data[ro][i]['remaining']>0:
                                                    complete=False
                                        return complete
                                    if check_complete(release_order_database,current_release_order):
                                        release_order_database[current_release_order]['complete']=True
                                    
                                    json_data = json.dumps(release_order_database)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                                    blob.upload_from_string(json_data)
                                                              
                                    success_container3=st.empty()
                                    time.sleep(0.1)                            
                                    success_container3.success(f"Updated Release Order Database",icon="✅")
                                    with open('placeholder.txt', 'r') as f:
                                        output_text = f.read()
                                    
                                    #st.markdown("**EDI TEXT**")
                                    #st.text_area('', value=output_text, height=600)
                                    with open('placeholder.txt', 'r') as f:
                                        file_content = f.read()
                                    newline="\n"
                                    filename = f'{bill_of_lading_number}'
                                    file_name= f'{bill_of_lading_number}.txt'
                                    
                                    
                                    subject = f'Suzano_EDI_{a}_ R.O:{release_order_number}-Terminal BOL :{bill_of_lading_number}-Destination : {destination}'
                                    body = f"EDI for Below attached.{newline}Release Order Number : {current_release_order} - Sales Order Number:{current_sales_order}{newline} Destination : {destination} Ocean Bill Of Lading : {ocean_bill_of_lading}{newline}Terminal Bill of Lading: {terminal_bill_of_lading} - Grade : {wrap} {newline}{2*quantity} tons {unitized} cargo were loaded to vehicle : {vehicle_id} with Carried ID : {carrier_code} {newline}Truck loading completed at {a_} {b_}"
                                    #st.write(body)           
                                    sender = "warehouseoly@gmail.com"
                                    recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                    #recipients = ["afsiny@portolympia.com"]
                                    password = "xjvxkmzbpotzeuuv"
                            
                          
                            
                            
                                    with open('temp_file.txt', 'w') as f:
                                        f.write(file_content)
                            
                                    file_path = 'temp_file.txt'  # Use the path of the temporary file
                            
                                    
                                    upload_cs_file(target_bucket, 'temp_file.txt',rf"EDIS/{file_name}")
                                    success_container5=st.empty()
                                    time.sleep(0.1)                            
                                    success_container5.success(f"Uploaded EDI File",icon="✅")
                                    
                                    
                                    try:
                                        mf_numbers_for_load[a_][dest][release_order_number][carrier_code_mf.upper()].remove(str(bill_of_lading_number))
                                    except:
                                        mf_numbers_for_load[a_][dest][release_order_number][carrier_code_mf.upper()].remove(int(bill_of_lading_number))
                                    mf_numbers_for_load=json.dumps(mf_numbers_for_load)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"release_orders/mf_numbers.json")
                                    blob.upload_from_string(mf_numbers_for_load)
                                    st.write("Updated MF numbers...")
                               
                                    send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                                    success_container6=st.empty()
                                    time.sleep(0.1)                            
                                    success_container6.success(f"Sent EDI Email",icon="✅")
                                    st.markdown("**SUCCESS! EDI FOR THIS LOAD HAS BEEN SUBMITTED,THANK YOU**")
                                    st.write(filename,current_release_order,current_sales_order,destination,ocean_bill_of_lading,terminal_bill_of_lading,wrap)
                                    this_shipment_aliens=[]
                                    for i in pure_loads:
                                        try:
                                                
                                            if i in alien_units[vessel]:       
                                                alien_units[vessel][i]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                        "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S"),
                                                        "Destination":destination,"Release_Order":current_release_order,"Terminal_Bill_of Lading":terminal_bill_of_lading,"Truck":vehicle_id}
                                                this_shipment_aliens.append(i)
                                            if i not in alien_units[vessel]:
                                                if i[:-2] in [u[:-2] for u in alien_units[vessel]]:
                                                    alien_units[vessel][i]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                        "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S"),
                                                        "Destination":destination,"Release_Order":current_release_order,"Terminal_Bill_of Lading":terminal_bill_of_lading,"Truck":vehicle_id}
                                                    this_shipment_aliens.append(i)
                                        except:
                                            pass
                                    alien_units=json.dumps(alien_units)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"alien_units.json")
                                    blob.upload_from_string(alien_units)   
                                    
                                    if len(this_shipment_aliens)>0:
                                        subject=f"UNREGISTERED UNITS SHIPPED TO {destination} on RELEASE ORDER {current_release_order}"
                                        body=f"{len([i for i in this_shipment_aliens])} unregistered units were shipped on {vehicle_id} to {destination} on {current_release_order}.<br>{[i for i in this_shipment_aliens]}"
                                        sender = "warehouseoly@gmail.com"
                                        recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                        #recipients = ["afsiny@portolympia.com"]
                                        password = "xjvxkmzbpotzeuuv"
                                        send_email(subject, body, sender, recipients, password)
                                    
                                else:   ###cancel bill of lading
                                    pass
            
                        
    
        
                    
                    else:
                        st.subheader("**Nothing dispatched!**")
                    
            
                    
                        
        ##########################################################################
        
                
                        
        if select=="INVENTORY" :

            #map=gcp_download_new(target_bucket,rf"map.json")
            map=gcp_download(target_bucket,rf"map.json")
            map=json.loads(map)
            mill_info=map["mill_info"]
            inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
         
            raw_ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
            raw_ro = json.loads(raw_ro)
            bol_mapping = map["bol_mapping"]
            
            suzano,edi_bank,main_inventory,status,mill_progress=st.tabs(["SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","RELEASE ORDER STATUS","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS"])
            
            with suzano:
                
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')
                
                now=datetime.datetime.now()-datetime.timedelta(hours=utc_difference)
                suzano_report_=gcp_download(target_bucket,rf"suzano_report.json")
                suzano_report=json.loads(suzano_report_)
                suzano_report=pd.DataFrame(suzano_report).T
                suzano_report=suzano_report[["Date Shipped","Vehicle", "Shipment ID #", "Consignee","Consignee City","Consignee State","Release #","Carrier","ETA","Ocean BOL#","Batch#","Warehouse","Vessel","Voyage #","Grade","Quantity","Metric Ton", "ADMT","Mode of Transportation"]]
                suzano_report["Shipment ID #"]=[str(i) for i in suzano_report["Shipment ID #"]]
                suzano_report["Batch#"]=[str(i) for i in suzano_report["Batch#"]]
                daily_suzano=suzano_report.copy()
                daily_suzano["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in suzano_report["Date Shipped"]]
                daily_suzano_=daily_suzano[daily_suzano["Date"]==now.date()]
                
                
                choose = st.radio(
                                "Select Daily or Accumulative Report",
                                ["DAILY", "ACCUMULATIVE", "FIND BY DATE","FIND DATE RANGE","BY RELEASE ORDER","BY BATCH"])
                if choose=="DAILY":
                    daily_suzano_=daily_suzano_.reset_index(drop=True)
                    daily_suzano_.index=[i+1 for i in daily_suzano_.index]
                    daily_suzano_.loc["TOTAL"]=daily_suzano_[["Quantity","Metric Ton","ADMT"]].sum()
                    st.dataframe(daily_suzano_)
                    csv=convert_df(daily_suzano_)
                    file_name=f'OLYMPIA_DAILY_REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                elif choose=="FIND BY DATE":
                    required_date=st.date_input("CHOOSE DATE",key="dssar")
                    filtered_suzano=daily_suzano[daily_suzano["Date"]==required_date]
                    filtered_suzano=filtered_suzano.reset_index(drop=True)
                    filtered_suzano.index=[i+1 for i in filtered_suzano.index]
                    filtered_suzano.loc["TOTAL"]=filtered_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                    st.dataframe(filtered_suzano)
                    csv=convert_df(filtered_suzano)
                    file_name=f'OLYMPIA_SHIPMENT_REPORT-{datetime.datetime.strftime(required_date,"%m-%d,%Y")}.csv'

                elif choose=="BY RELEASE ORDER":
                    report_ro=st.selectbox("SELECT RELEASE ORDER",([i for i in raw_ro]),key="sdgerw")
                    ro_suzano=daily_suzano[daily_suzano["Release #"]==report_ro]
                    ro_suzano=ro_suzano.reset_index(drop=True)
                    ro_suzano.index=[i+1 for i in ro_suzano.index]
                    ro_suzano.loc["TOTAL"]=ro_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                    st.dataframe(ro_suzano)
                    csv=convert_df(ro_suzano)
                    file_name=f'OLYMPIA_SHIPMENT_REPORT-ReleaseOrder-{report_ro}.csv'

                elif choose=="BY BATCH":
                    report_batch=st.selectbox("SELECT BATCH",(daily_suzano["Batch#"].unique()),key="sddgerw")
                    batch_suzano=daily_suzano[daily_suzano["Batch#"]==report_batch]
                    batch_suzano=batch_suzano.reset_index(drop=True)
                    batch_suzano.index=[i+1 for i in batch_suzano.index]
                    batch_suzano.loc["TOTAL"]=batch_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                    st.dataframe(batch_suzano)
                    csv=convert_df(batch_suzano)
                    file_name=f'OLYMPIA_SHIPMENT_REPORT-Batch-{report_batch}.csv'

                elif choose=="FIND DATE RANGE":
                    datecol1,datecol2,datecol3=st.columns([3,3,4])
                    with datecol1:
                        tarih1=st.date_input("FROM",key="dsssaar")
                    with datecol2:
                        tarih2=st.date_input("TO",key="dssdar")
                        
                    range_suzano=daily_suzano[(daily_suzano["Date"]>=tarih1)&(daily_suzano["Date"]<=tarih2)]
                    range_suzano=range_suzano.reset_index(drop=True)
                    range_suzano.index=[i+1 for i in range_suzano.index]
                    range_suzano.loc["TOTAL"]=range_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                    st.dataframe(range_suzano)
                    csv=convert_df(range_suzano)
                    file_name=f'OLYMPIA_SHIPMENT_REPORT-daterange.csv'
                
                else:
                    st.dataframe(suzano_report)
                    csv=convert_df(suzano_report)
                    file_name=f'OLYMPIA_ALL_SHIPMENTS to {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                
                
                
               
                
            
                st.download_button(
                    label="DOWNLOAD REPORT AS CSV",
                    data=csv,
                    file_name=file_name,
                    mime='text/csv')
                

            with edi_bank:
                edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                
                display_edi=st.toggle("DISPLAY EDI")
                if display_edi:
                    data=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}")
                    st.text_area("EDI",data,height=400)                                
               
                st.download_button(
                    label="DOWNLOAD EDI",
                    data=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}"),
                    file_name=f'{requested_edi_file}',
                    mime='text/csv')
                
                
                
            with main_inventory:
                
                maintenance=False
                                
                if maintenance:
                    st.title("CURRENTLY UNDER MAINTENANCE, CHECK BACK LATER")
                               
                else:
                    inventory,daily,unregistered=st.tabs(["INVENTORY","DAILY SHIPMENT REPORT","UNREGISTERED LOTS FOUND"])
                    
                    with daily:
                        
                        amount_dict={"KIRKENES-2304":9200,"JUVENTAS-2308":10000,"LYSEFJORD-2308":10000,"LAGUNA-3142":453,"FRONTIER-55VC":9811,"BEIJA_FLOR-88VC":11335}
                        inv_vessel=st.selectbox("Select Vessel",[i for i in map['batch_mapping']])
                        kf=inv_bill_of_ladings.iloc[1:].copy()
                        kf['issued'] = pd.to_datetime(kf['issued'])
                        kf=kf[kf["vessel"]==inv_vessel]
                        
                        kf['Date'] = kf['issued'].dt.date
                        kf['Date'] = pd.to_datetime(kf['Date'])
                        # Create a date range from the minimum to maximum date in the 'issued' column
                        date_range = pd.date_range(start=kf['Date'].min(), end=kf['Date'].max(), freq='D')
                        
                        # Create a DataFrame with the date range
                        date_df = pd.DataFrame({'Date': date_range})
                        # Merge the date range DataFrame with the original DataFrame based on the 'Date' column
                        merged_df = pd.merge(date_df, kf, how='left', on='Date')
                        merged_df['quantity'].fillna(0, inplace=True)
                        merged_df['Shipped Tonnage']=merged_df['quantity']*2
                        merged_df_grouped=merged_df.groupby('Date')[['quantity','Shipped Tonnage']].sum()
                        merged_df_grouped['Accumulated_Quantity'] = merged_df_grouped['quantity'].cumsum()
                        merged_df_grouped["Accumulated_Tonnage"]=merged_df_grouped['Accumulated_Quantity']*2
                        merged_df_grouped["Remaining_Units"]=[amount_dict[inv_vessel]-i for i in merged_df_grouped['Accumulated_Quantity']]
                        merged_df_grouped["Remaining_Tonnage"]=merged_df_grouped["Remaining_Units"]*2
                        merged_df_grouped.rename(columns={'quantity':"Shipped Quantity", 'Accumulated_Quantity':"Shipped Qty To_Date",
                                                          'Accumulated_Tonnage':"Shipped Tonnage To_Date"},inplace=True)
                        merged_df_grouped=merged_df_grouped.reset_index()
                        merged_df_grouped["Date"]=merged_df_grouped['Date'].dt.strftime('%m-%d-%Y, %A')
                        #merged_df_grouped=merged_df_grouped.set_index("Date",drop=True)
                      
                        st.dataframe(merged_df_grouped)
                        csv_inventory=convert_df(merged_df_grouped)
                        st.download_button(
                            label="DOWNLOAD INVENTORY REPORT AS CSV",
                            data=csv_inventory,
                            file_name=f'INVENTORY REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                            mime='text/csv')   
                        
                            
                    with inventory:
                        def log_inventory_change(bol_to_edit, total_edit, damaged_edit, user, log_type):
                            change_log = {
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "user": user,
                                "log_type": log_type,
                                "ocean_bill_of_lading": bol_to_edit,
                                "new_total": total_edit,
                                "new_damaged": damaged_edit
                                }
                            
                            try:
                                inventory_log=gcp_download(target_bucket,rf"inventory_log.json")
                                inventory_log=json.loads(inventory_log)
                            except :
                                inventory_log = {}
                        
                            if bol_to_edit not in inventory_log:
                                inventory_log[bol_to_edit]=[]
                            inventory_log[bol_to_edit].append(change_log)
                        
                            return inventory_log




                        
                        bols = {}
                        for key, value in raw_ro.items():
                            for item_key, item_value in value.items():
                                if isinstance(item_value, dict):
                                    ocean_bill_of_lading = item_value.get('ocean_bill_of_lading')
                                    if ocean_bill_of_lading:
                                        if ocean_bill_of_lading not in bols:
                                            bols[ocean_bill_of_lading] = []
                                        bols[ocean_bill_of_lading].append(key)
                        for i in map['bol_mapping']:
                            if i not in bols:
                                bols[i]=[]
 
                        inventory={}
                        a=[i for i in map['bol_mapping']]
                        for bill in a:
                            inventory[bill]=[map['bol_mapping'][bill]['total'],map['bol_mapping'][bill]['damaged']]
                            
                        def extract_qt(data,ro,bol):
                            totals=[0,0,0]
                            sales_group=["001","002","003","004","005"]
                            for item in data[ro]:
                                if item in sales_group:
                                    if data[ro][item]['ocean_bill_of_lading']==bol:
                                        totals[0]+=data[ro][item]['total']
                                        totals[1]+=data[ro][item]['shipped']
                                        totals[2]+=data[ro][item]['remaining']
                            return totals
                        
                        final={}
                        for k in inventory.keys():
                            final[k]={"Total":0,"Damaged":0,"Fit To Ship":0,"Allocated to ROs":0,"Shipped":0,
                                     "Remaining in Warehouse":0,"Remaining on ROs":0,"Remaining After ROs":0}
                            vector=np.zeros(3)
                            final[k]["Total"]=inventory[k][0]
                            final[k]["Damaged"]=inventory[k][1]
                            final[k]["Fit To Ship"]=final[k]["Total"]-final[k]["Damaged"]   
                        
                            if k in bols:
                                if len(bols[k])>0:
                                    for ro in set(bols[k]):
                                        a,b,c=extract_qt(raw_ro,ro,k)[0],extract_qt(raw_ro,ro,k)[1],extract_qt(raw_ro,ro,k)[2]
                                        final[k]["Allocated to ROs"]+=a
                                        #final[k]["Shipped"]=inv_bill_of_ladings.groupby("ocean_bill_of_lading")[['quantity']].sum().loc[k,'quantity']
                                        final[k]["Shipped"]+=b
                        
                                        final[k]["Remaining in Warehouse"]=final[k]["Fit To Ship"]-final[k]["Shipped"]
                                        final[k]["Remaining on ROs"]=final[k]["Allocated to ROs"]-final[k]["Shipped"]
                                        final[k]["Remaining After ROs"]=final[k]["Fit To Ship"]-final[k]["Allocated to ROs"]
                                else:
                                    final[k]["Remaining in Warehouse"]=final[k]["Fit To Ship"]
                                    final[k]["Remaining on ROs"]=0
                                    final[k]["Remaining After ROs"]=final[k]["Fit To Ship"]-final[k]["Allocated to ROs"]
                            else:
                                pass
                        temp=pd.DataFrame(final).T
                        temp.loc["TOTAL"]=temp.sum(axis=0)
                        
                        
                        tempo=temp*2

                        inv_col1,inv_col2=st.columns([7,3])
                        with inv_col1:
                            st.subheader("By Ocean BOL,UNITS")
                            st.dataframe(temp)
                            st.subheader("By Ocean BOL,TONS")
                            st.dataframe(tempo)        
                        with inv_col2:
                            edit_inventory_checkbox=st.checkbox("CHECK TO EDIT INVENTORY")
                            if edit_inventory_checkbox:
                                
                                bol_to_edit=st.selectbox("EDIT INVENTORY",[i for i in temp.index])
                                dinv1,dinv2,_=st.columns([3,3,4])
                                with dinv1:
                                    st.write(f"DAMAGED")
                                    st.write(f"TOTAL")
                                with dinv2:
                                    damaged_edit=st.number_input("lala",value=map['bol_mapping'][bol_to_edit]['damaged'],label_visibility='collapsed')
                                    total_edit=st.number_input("tata",value=map['bol_mapping'][bol_to_edit]['total'],label_visibility='collapsed',key="dsd")
                                if st.button("SUBMIT CHANGE",key="t2ds"):
                                    map['bol_mapping'][bol_to_edit]['total']=total_edit
                                    map['bol_mapping'][bol_to_edit]['damaged']=damaged_edit
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"map.json")
                                    blob.upload_from_string(json.dumps(map))
                                    st.success(f"Updated Inventory Database",icon="✅")
                                    
                                    user="admin"
                                    log_type="Warehouse Adjustment"
                                    inventory_log=log_inventory_change(bol_to_edit, total_edit, damaged_edit, user, log_type)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"inventory_log.json")
                                    blob.upload_from_string(json.dumps(inventory_log))
                                    st.success(f"Logged Inventory Change",icon="✅")
                                    
                                see_inventory_change_checkbox=st.checkbox("SEE INVENTORY EDIT LOGS")
                                if see_inventory_change_checkbox:
                                    try:
                                        inventory_log=gcp_download(target_bucket,rf"inventory_log.json")
                                        inventory_log=json.loads(inventory_log)
                                    except :
                                        inventory_log = []
                                       
                                    st.write(inventory_log[bol_to_edit])


                            
                    with unregistered:
                        alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                        alien_vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308","LAGUNA-3142","LYSEFJORD-2308","FRONTIER-55VC"])
                        alien_list=pd.DataFrame(alien_units[alien_vessel]).T
                        alien_list.reset_index(inplace=True)
                        alien_list.index=alien_list.index+1
                        st.markdown(f"**{len(alien_list)} units that are not on the shipping file found on {alien_vessel}**")
                        st.dataframe(alien_list)
                        
                      
            with mill_progress:
                pass
                # inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                # dfb=pd.read_json(inv_bill_of_ladings).T
                # # dfb=pd.DataFrame(inv_bill_of_ladings).T
                # # st.write(dfb)
                # dfb["St_Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in dfb["issued"]]
                # schedule=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
                # schedule=json.loads(schedule)
                # flattened_data = []
                # for date, locations in schedule.items():
                #     for location, location_data in locations.items():
                #         for order, carriers in location_data.items():
                #             for carrier, shipments in carriers.items():
                #                 for shipment in shipments:
                #                     dfb=dfb[dfb["Date Shipped"]==selected_date_datetime]
                #                     status="NONE"
                #                     if shipment in dfb.index:
                #                         status_="SHIPPED"
                #                     else:
                #                         status_="Scheduled"
                #                     shipment_parts = shipment.split("|") if "|" in shipment else [shipment]
                #                     carrier_=carrier.split("-")[1]
                #                     flattened_data.append({
                #                         "Date": date,
                #                         "Location": location,
                #                         "Order": order,
                #                         "Carrier": carrier_,
                #                         "EDI Bill Of Lading":shipment,
                #                         "MF Number": shipment_parts[0] if len(shipment_parts) > 1 else None,
                #                         "Shipment ID": shipment_parts[1] if len(shipment_parts) > 1 else shipment_parts[0]
                #                     })

                
                # flat_df=pd.DataFrame(flattened_data)
                # flat_df["Date"] = pd.to_datetime(flat_df["Date"])#.dt.date
                # flat_df.insert(1,"Day",flat_df["Date"].dt.day_name())
                # flat_df["Status"]="None"
                # flat_df['Status'] = flat_df['EDI Bill Of Lading'].apply(lambda x: 'SHIPPED' if x in dfb else 'Scheduled')
                # flat_df.reset_index(drop=True,inplace=True)
                # flat_df.index+=1
                # styled_df =flat_df.style.apply(style_row, axis=1)
                # st.write(styled_df.to_html(), unsafe_allow_html=True)
                # pass
                # # mf_numbers=json.loads(gcp_download(target_bucket,rf"release_orders/mf_numbers.json"))
                
                # # bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                # # bill_of_ladings=json.loads(bill_of_ladings)
                # # bill=pd.DataFrame(bill_of_ladings).T
                # # values=[]
                # # mfs=[]
                # # for i in bill.index:
                # #     if len(i.split("|"))>1:
                # #         values.append(i.split("|")[1])
                # #         mfs.append(i.split("|")[0])
                # #     else:
                # #         values.append(i)
                # #         mfs.append(i)
                # # bill.insert(0,"Shipment",values)
                # # bill.insert(1,"MF",mfs)
                
                # # suzano_shipment_=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
                # # suzano_shipment=json.loads(suzano_shipment_)
                # # suzano_shipment=pd.DataFrame(suzano_shipment).T

                # # suzano_shipment["PK"]=suzano_shipment["PK"].astype("str")
                # # bill["Shipment"]=bill["Shipment"].astype("str")
                # # suzano_shipment["Pickup"]=pd.to_datetime(suzano_shipment["Pickup"])
                # # suzano_shipment=suzano_shipment[suzano_shipment["Pickup"]>datetime.datetime(2024, 9, 24)]
                # # suzano_shipment.reset_index(drop=True,inplace=True)
                # # for i in suzano_shipment.index:
                # #     sh=suzano_shipment.loc[i,"Shipment ID"]
                # #     #print(sh)
                # #     if sh in bill[~bill["Shipment"].isna()]["Shipment"].to_list():
                # #         vehicle=bill.loc[bill["Shipment"]==sh,'vehicle'].values[0]
                # #         bol=str(bill.loc[bill["Shipment"]==sh].index.values[0])
                # #         suzano_shipment.loc[i,"Transit Status"]="COMPLETED"
                # #         suzano_shipment.loc[i,"BOL"]=bol
                # #         suzano_shipment.loc[i,"Vehicle ID"]=vehicle
                # # for rel,value in mf_numbers.items():
                # #     for date in value:
                # #         for carrier,liste in value[date].items():
                # #             if len(liste)>0:
                # #                 try:
                # #                     for k in liste:
                # #                         try:
                # #                             suzano_shipment.loc[suzano_shipment["Shipment ID"]==k.split("|")[1],"Transit Status"]="SCHEDULED"
                # #                         except:
                # #                             suzano_shipment.loc[suzano_shipment["Shipment ID"]==k,"Transit Status"]="SCHEDULED"
                # #                 except:
                # #                     pass
                # # st.subheader("SUZANO OTM LIST")
                # # st.write(suzano_shipment)
                # # maintenance=False
                # # if maintenance:
                # #     st.title("CURRENTLY IN MAINTENANCE, CHECK BACK LATER")
                # # else:
                # #     st.subheader("WEEKLY SHIPMENTS BY MILL (IN TONS)")
                # #     zf=inv_bill_of_ladings.copy()
                # #     zf['WEEK'] = pd.to_datetime(zf['issued'])
                # #     zf.set_index('WEEK', inplace=True)
                    
                # #     def sum_quantity(x):
                # #         return x.resample('W')['quantity'].sum()*2
                # #     resampled_quantity = zf.groupby('destination').apply(sum_quantity).unstack(level=0)
                # #     resampled_quantity=resampled_quantity.fillna(0)
                # #     resampled_quantity.loc["TOTAL"]=resampled_quantity.sum(axis=0)
                # #     resampled_quantity["TOTAL"]=resampled_quantity.sum(axis=1)
                # #     resampled_quantity=resampled_quantity.reset_index()
                # #     resampled_quantity["WEEK"][:-1]=[i.strftime("%Y-%m-%d") for i in resampled_quantity["WEEK"][:-1]]
                # #     resampled_quantity.set_index("WEEK",drop=True,inplace=True)
                # #     st.dataframe(resampled_quantity)
                    
                # #     st.download_button(
                # #     label="DOWNLOAD WEEKLY REPORT AS CSV",
                # #     data=convert_df(resampled_quantity),
                # #     file_name=f'WEEKLY SHIPMENT REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                # #     mime='text/csv')
    
    
                   
                # #     zf['issued'] = pd.to_datetime(zf['issued'])                   
                   
                # #     weekly_tonnage = zf.groupby(['destination', pd.Grouper(key='issued', freq='W')])['quantity'].sum() * 2  # Assuming 2 tons per quantity
                # #     weekly_tonnage = weekly_tonnage.reset_index()                   
                  
                # #     weekly_tonnage = weekly_tonnage.rename(columns={'issued': 'WEEK', 'quantity': 'Tonnage'})
                  
                # #     fig = px.bar(weekly_tonnage, x='WEEK', y='Tonnage', color='destination',
                # #                  title='Weekly Shipments Tonnage per Location',
                # #                  labels={'Tonnage': 'Tonnage (in Tons)', 'WEEK': 'Week'})
                 
                # #     fig.update_layout(width=1000, height=700)  # You can adjust the width and height values as needed
                    
                # #     st.plotly_chart(fig)


            with status:
                
                status_dict={}
                sales_group=["001","002","003","004","005"]
                for ro in raw_ro:
                    for sale in [i for i in raw_ro[ro] if i in sales_group]:
                        status_dict[f"{ro}-{sale}"]={"Release Order #":ro,"Sales Order #":sale,
                                            "Destination":raw_ro[ro]['destination'],
                                            "Ocean BOL":raw_ro[ro][sale]['ocean_bill_of_lading'],
                                            "Total":raw_ro[ro][sale]['total'],
                                            "Shipped":raw_ro[ro][sale]['shipped'],
                                            "Remaining":raw_ro[ro][sale]['remaining']}
                status_frame=pd.DataFrame(status_dict).T.set_index("Release Order #",drop=True)
                active_frame_=status_frame[status_frame["Remaining"]>0]
                status_frame.loc["Total"]=status_frame[["Total","Shipped","Remaining"]].sum()
                active_frame=active_frame_.copy()
                active_frame.loc["Total"]=active_frame[["Total","Shipped","Remaining"]].sum()
                
                st.markdown(active_frame.to_html(render_links=True),unsafe_allow_html=True)

                
                release_orders = status_frame.index[:-1]
                release_orders = pd.Categorical(release_orders)
                active_order_names = [f"{i} to {raw_ro[i]['destination']}" for i in active_frame_.index]
                destinations=[raw_ro[i]['destination'] for i in active_frame_.index]
                active_orders=[str(i) for i in active_frame.index]
               
                # fig = go.Figure()
                # fig.add_trace(go.Bar(x=active_orders, y=active_frame["Total"], name='Total', marker_color='lightgray'))
                # fig.add_trace(go.Bar(x=active_orders, y=active_frame["Shipped"], name='Shipped', marker_color='blue', opacity=0.7))
                # remaining_data = [remaining if remaining > 0 else None for remaining in active_frame_["Remaining"]]
                # fig.add_trace(go.Scatt
                
                #annotations = [dict(x=release_order, y=total_quantity, text=destination, showarrow=True, arrowhead=4, ax=0, ay=-30) for release_order, total_quantity, destination in zip(active_orders, active_frame["Total"], destinations)]
                #fig.update_layout(annotations=annotations)

                # fig.add_annotation(x="3172296", y=800, text="destination",
                #                        showarrow=True, arrowhead=4, ax=0, ay=-30)
                
                # fig.update_layout(title='ACTIVE RELEASE ORDERS',
                #                   xaxis_title='Release Orders',
                #                   yaxis_title='Quantities',
                #                   barmode='overlay',
                #                   width=1300,
                #                   height=700,
                #                   xaxis=dict(tickangle=-90, type='category'))
                
                # st.plotly_chart(fig)
                
                
                duration=st.toggle("Duration Report")
                if duration:
                    
                    temp_dict={}
                        
                    for rel_ord in raw_ro:
                        for sales in [i for i in raw_ro[rel_ord] if i in ["001","002","003","004","005"]]:
                            temp_dict[rel_ord,sales]={}
                            dest=raw_ro[rel_ord]['destination']
                            vessel=raw_ro[rel_ord][sales]['vessel']
                            total=raw_ro[rel_ord][sales]['total']
                            remaining=raw_ro[rel_ord][sales]['remaining']
                            temp_dict[rel_ord,sales]={'destination': dest,'vessel': vessel,'total':total,'remaining':remaining}
                    temp_df=pd.DataFrame(temp_dict).T
                  
                    temp_df= temp_df.rename_axis(['release_order','sales_order'], axis=0)
                
                    temp_df['First Shipment'] = temp_df.index.map(inv_bill_of_ladings.groupby(['release_order','sales_order'])['issued'].first())
                    
                    for i in temp_df.index:
                        if temp_df.loc[i,'remaining']<=2:
                            try:
                                temp_df.loc[i,"Last Shipment"]=inv_bill_of_ladings.groupby(['release_order','sales_order']).issued.last().loc[i]
                            except:
                                temp_df.loc[i,"Last Shipment"]=datetime.datetime.now()
                            temp_df.loc[i,"Duration"]=(pd.to_datetime(temp_df.loc[i,"Last Shipment"])-pd.to_datetime(temp_df.loc[i,"First Shipment"])).days+1
                    
                    temp_df['First Shipment'] = temp_df['First Shipment'].fillna(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
                    temp_df['Last Shipment'] = temp_df['Last Shipment'].fillna(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
                    
                    ####
                    
                    def business_days(start_date, end_date):
                        return pd.date_range(start=start_date, end=end_date, freq=BDay())
                    temp_df['# of Shipment Days'] = temp_df.apply(lambda row: len(business_days(row['First Shipment'], row['Last Shipment'])), axis=1)
                    df_temp=inv_bill_of_ladings.copy()
                    df_temp["issued"]=[pd.to_datetime(i).date() for i in df_temp["issued"]]
                    for i in temp_df.index:
                        try:
                            temp_df.loc[i,"Utilized Shipment Days"]=df_temp.groupby(["release_order",'sales_order'])[["issued"]].nunique().loc[i,'issued']
                        except:
                            temp_df.loc[i,"Utilized Shipment Days"]=0
                    
                    temp_df['First Shipment'] = temp_df['First Shipment'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),'%d-%b,%Y'))
                    temp_df['Last Shipment'] = temp_df['Last Shipment'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),'%d-%b,%Y') if type(x)==str else None)
                    liste=['Duration','# of Shipment Days',"Utilized Shipment Days"]
                    for col in liste:
                        temp_df[col] = temp_df[col].apply(lambda x: f" {int(x)} days" if not pd.isna(x) else np.nan)
                    temp_df['remaining'] = temp_df['remaining'].apply(lambda x: int(x))
                    temp_df.columns=['Destination', 'Vessel', 'Total Units', 'Remaining Units', 'First Shipment',
                           'Last Shipment', 'Duration', '# of Calendar Shipment Days',
                           'Utilized Calendar Shipment Days']
                    st.dataframe(temp_df)
                   
                



    ########################                                WAREHOUSE                            ####################
    
    elif username == 'warehouse':
        map=gcp_download(target_bucket,rf"map.json")
        map=json.loads(map)
        release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
        release_order_database=json.loads(release_order_database)
        dispatched=gcp_download(target_bucket,rf"dispatched.json")
        dispatched=json.loads(dispatched)
        carrier_list=map['carriers']
        mill_info=map["mill_info"]
        
        bill_mapping=gcp_download(target_bucket,"bill_mapping.json")   ###### DOWNLOADS
        bill_mapping=json.loads(bill_mapping)
        mf_numbers_for_load=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")  ###### DOWNLOADS
        mf_numbers_for_load=json.loads(mf_numbers_for_load)

        bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")   ###### DOWNLOADS
        bill_of_ladings=json.loads(bill_of_ladings)
        
        no_dispatch=0
       
        number=None  
        if number not in st.session_state:
            st.session_state.number=number
                       
      
        double_load=False
        
        if len(dispatched.keys())>0 and not no_dispatch:
            loadout,schedule=st.tabs(["LOADOUT","SCHEDULE"])
            
            with schedule:
                st.subheader("TODAYS ACTION/SCHEDULE")
                bill_for_schedule=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                bill_for_schedule=json.loads(bill_for_schedule)
                schedule=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
                schedule=json.loads(schedule)
                dfb=pd.DataFrame.from_dict(bill_for_schedule).T[1:]
                #dfb=bill.copy()
                dfb["St_Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in dfb["issued"]]
                #dfb=dfb[dfb["St_Date"]==(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date()]
                scheduled=[]
                today=str((datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date())
                selected_date_datetime=st.date_input("SELECT DATE",(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).date())
                selected_date=str(selected_date_datetime)
                
                if selected_date in schedule:
                    dfb=dfb[dfb["St_Date"]==selected_date_datetime]
                    for dest in schedule[selected_date]:
                        for rel in schedule[selected_date][dest]:
                            for carrier in schedule[selected_date][dest][rel]:
                                scheduled.append({"Destination":dest,
                                      "Release Order":rel,"Sales Item":"001",
                                      "ISP":release_order_database[rel]["001"]['grade'],
                                      "Prep":release_order_database[rel]["001"]['unitized'],
                                      "Carrier":carrier.split("-")[1],
                                      "Scheduled":int(len(schedule[selected_date][dest][rel][carrier])),
                                      "Loaded":int(dfb[(dfb["release_order"]==rel)&(dfb["sales_order"]=="001")&
                                       (dfb["carrier_id"]==str(carrier.split("-")[0]))].vehicle.count()),
                                      "Remaining":0})
                    scheduled=pd.DataFrame(scheduled)
                    scheduled["Scheduled"] = scheduled["Scheduled"].astype(int)
                    scheduled["Loaded"] = scheduled["Loaded"].astype(int)
                    scheduled["Remaining"] = scheduled["Remaining"].astype(int)

                    if len(scheduled)>0:
                        
                        scheduled["Remaining"]=[int(i) for i in scheduled["Scheduled"]-scheduled["Loaded"]]
                        scheduled.loc["Total",["Scheduled","Loaded","Remaining"]]=scheduled[["Scheduled","Loaded","Remaining"]].sum()
                        #scheduled.set_index('Destination',drop=True,inplace=True)
                        scheduled["Scheduled"] = scheduled["Scheduled"].astype(int)
                        scheduled["Loaded"] = scheduled["Loaded"].astype(int)
                        scheduled["Remaining"] = scheduled["Remaining"].astype(int)
                        scheduled.loc["Total",["Scheduled","Loaded","Remaining"]].astype(int)
                        scheduled.fillna("",inplace=True)


                        def style_row(row,code=1):
                            
                            if code==2:
                                shipment_status = row["Status"]
                                location = row["Location"]
                            else:
                                location = row['Destination']
                            # Define colors for different locations
                            colors = {
                                "CLATSKANIE": "background-color: #d1e7dd;",  # light green
                                "LEWISTON": "background-color: #ffebcd;",    # light coral
                                "HALSEY": "background-color: #add8e6;",      # light blue
                            }
                            
                            # Base style for the entire row based on location
                            base_style = colors.get(location, "")
                            if code==2:
                                if shipment_status == "SHIPPED":
                                    base_style += "font-weight: lighter; font-style: italic; text-decoration: line-through;"  # Less bold, italic, and strikethrough
                                else:
                                    base_style += "font-weight: bold;"  # Slightly bolder for other statuses
                    
                            return [base_style] * len(row)
                        
                        list_view=st.checkbox("LIST VIEW")
                        if list_view:
                            flattened_data = []
                            for date, locations in schedule.items():
                                for location, location_data in locations.items():
                                    for order, carriers in location_data.items():
                                        for carrier, shipments in carriers.items():
                                            for shipment in shipments:
                                                dfb_=dfb[dfb["St_Date"]==selected_date_datetime]
                                                status="NONE"
                                                # Split the shipment data if needed (separate IDs if joined by "|")
                                                if shipment in dfb_.index:
                                                    status_="SHIPPED"
                                                else:
                                                    status_="Scheduled"
                                                shipment_parts = shipment.split("|") if "|" in shipment else [shipment]
                                                carrier_=carrier.split("-")[1]
                                                flattened_data.append({
                                                    "Date": date,
                                                    "Location": location,
                                                    "Order": order,
                                                    "Carrier": carrier_,
                                                    "EDI Bill Of Lading":shipment,
                                                    "Shipment ID": shipment_parts[0],
                                                    "MF Number": shipment_parts[1] if len(shipment_parts) > 1 else None,
                                                    "Status":status_
                                                })

                        # Convert to DataFrame
                            flat_df = pd.DataFrame(flattened_data)
                            #flat_df["Status"]=["Scheduled"]*len(flat_df)
                            flat_df=flat_df[flat_df.Date==selected_date]
                            flat_df.reset_index(drop=True,inplace=True)
                            flat_df.index+=1

                        #styled_schedule =scheduled.style.apply(style_row, axis=1)
                        if list_view:
                            styled_schedule = flat_df.style.apply(lambda row: style_row(row, code=2), axis=1)
                        else:
                            styled_schedule = scheduled.style.apply(lambda row: style_row(row, code=1), axis=1)

                     
                        st.write(styled_schedule.to_html(), unsafe_allow_html=True)
        
                else:
                    st.write("Nothing Scheduled")
                    
            with loadout:
                
                menu_destinations={}
                    
                for rel_ord in dispatched.keys():
                    for sales in dispatched[rel_ord]:
                        try:
                            menu_destinations[f"{rel_ord} -{sales}"]=dispatched[rel_ord][sales]["destination"]
                        except:
                            pass
                if 'work_order_' not in st.session_state:
                    st.session_state.work_order_ = None
                    
                liste=[f"{i} to {menu_destinations[i]}" for i in menu_destinations.keys()]
            
                work_order_=st.selectbox("**SELECT RELEASE ORDER/SALES ORDER TO WORK**",liste,index=0 if st.session_state.work_order_ else 0) 
                st.session_state.work_order_=work_order_
                if work_order_:
                    
                    work_order=work_order_.split(" ")[0]
                    order=["001","002","003","004","005","006"]
                    
                    current_release_order=work_order
                    current_sales_order=work_order_.split(" ")[1][1:]
                    vessel=release_order_database[current_release_order][current_sales_order]["vessel"]
                    destination=release_order_database[current_release_order]['destination']
                    
                    
                    
          
                    try:
                        next_release_order=dispatched['002']['release_order']    #############################  CHECK HERE ######################## FOR MIXED LOAD
                        next_sales_order=dispatched['002']['sales_order']
                        
                    except:
                        
                        pass
                    
    
                    
                        
                    
                    st.markdown(rf'**:blue[CURRENTLY WORKING] :**')
                    load_col1,load_col2=st.columns([9,1])
                    
                    with load_col1:
                        wrap_dict={"ISU":"UNWRAPPED","ISP":"WRAPPED","AEP":"WRAPPED"}
                        wrap=release_order_database[current_release_order][current_sales_order]["grade"]
                        ocean_bill_of_=release_order_database[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                        unitized=release_order_database[current_release_order][current_sales_order]["unitized"]
                        quant_=release_order_database[current_release_order][current_sales_order]["total"]
                        real_quant=int(math.floor(quant_))
                        ship_=release_order_database[current_release_order][current_sales_order]["shipped"]
                        ship_bale=(ship_-math.floor(ship_))*8
                        remaining=release_order_database[current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                        temp={f"<b>Release Order #":current_release_order,"<b>Destination":destination,"<b>VESSEL":vessel}
                        temp2={"<b>Ocean B/L":ocean_bill_of_,"<b>Type":wrap_dict[wrap],"<b>Prep":unitized}
                        temp3={"<b>Total Units":quant_,"<b>Shipped Units":ship_,"<b>Remaining Units":remaining}
                        #temp4={"<b>Total Bales":0,"<b>Shipped Bales":int(8*(ship_-math.floor(ship_))),"<b>Remaining Bales":int(8*(remaining-math.floor(remaining)))}
                        temp5={"<b>Total Tonnage":quant_*2,"<b>Shipped Tonnage":ship_*2,"<b>Remaining Tonnage":quant_*2-(ship_*2)}
    
    
                        
                        sub_load_col1,sub_load_col2,sub_load_col3,sub_load_col4=st.columns([3,3,3,3])
                        
                        with sub_load_col1:   
                            st.markdown(rf'**Release Order-{current_release_order}**')
                            st.markdown(rf'**Destination : {destination}**')
                            st.markdown(rf'**VESSEL-{vessel}**')
                            #st.write (pd.DataFrame(temp.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                     
                        with sub_load_col2:
                            st.markdown(rf'**Ocean B/L: {ocean_bill_of_}**')
                            st.markdown(rf'**Type : {wrap_dict[wrap]}**')
                            st.markdown(rf'**Prep : {unitized}**')
                            #st.write (pd.DataFrame(temp2.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                       
                            
                        with sub_load_col3:
        
                            st.markdown(rf'**Total Units : {quant_}**')
                            st.markdown(rf'**Shipped Units : {ship_}**')
                            if remaining<=10:
                                st.markdown(rf'**:red[CAUTION : Remaining : {remaining} Units]**')
                            else:
                                st.markdown(rf'**Remaining Units : {remaining}**')
                        with sub_load_col4:
                            
                        
                            #st.write (pd.DataFrame(temp5.items(),columns=["TONNAGE","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                            
                            st.markdown(rf'**Total Tonnage : {quant_*2}**')
                            st.markdown(rf'**Shipped Tonnage : {ship_*2}**')
                            st.markdown(rf'**Remaining Tonnage : {remaining*2}**')
                    
                    with load_col2:
                        if double_load:
                            
                            try:
                                st.markdown(rf'**NEXT ITEM : Release Order-{next_release_order}**')
                                st.markdown(rf'**Sales Order Item-{next_sales_order}**')
                                st.markdown(f'**Ocean Bill Of Lading : {info[vessel][next_release_order][next_sales_order]["ocean_bill_of_lading"]}**')
                                st.markdown(rf'**Total Quantity : {info[vessel][next_release_order][next_sales_order]["quantity"]}**')
                            except:
                                pass
    
                    st.divider()
                    ###############    LOADOUT DATA ENTRY    #########
                    
                    col1, col2,col3,col4= st.columns([2,2,2,2])
                    yes=False
                  
                    release_order_number=current_release_order
                    medium="TRUCK"
                    
                    with col1:
                    ######################  LETS GET RID OF INPUT BOXES TO SIMPLIFY LONGSHORE SCREEN
               
                        terminal_code="OLYM"
                        
                        file_date=datetime.datetime.today()-datetime.timedelta(hours=utc_difference)
                        if file_date not in st.session_state:
                            st.session_state.file_date=file_date
                        file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False)
                      
                        delivery_date=datetime.datetime.today()-datetime.timedelta(hours=utc_difference)
                        eta_date=delivery_date
                        
                        carrier_code=release_order_database[current_release_order][current_sales_order]["carrier_code"]
                        vessel=release_order_database[current_release_order][current_sales_order]["vessel"]
                        transport_sequential_number="TRUCK"
                        transport_type="TRUCK"
                        placeholder = st.empty()
                        with placeholder.container():
                            vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=7)
                            manual_bill=st.toggle("Toggle for Manual BOL")
                            if manual_bill:
                                manual_bill_of_lading_number=st.text_input("ENTER BOL",key="eirufs")
                            mf=True
                            load_mf_number_issued=False
                            if destination=="CLEARWATER-Lewiston,ID":
                                carrier_code=st.selectbox("Carrier Code",[carrier_code,"310897-Ashley"],disabled=False,key=29)
                            else:
                                carrier_code=st.text_input("Carrier Code",carrier_code,disabled=True,key=9)
                               
                            today_str=str(st.date_input("Shipment Date",datetime.datetime.today(),disabled=False,key="popdo3"))
                            mf_date_str=datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d")
                            dest=destination.split("-")[1].split(",")[0].upper()
                            if 'load_mf_number' not in st.session_state:
                                st.session_state.load_mf_number = None
                                
                            if today_str in mf_numbers_for_load.keys():
                                try:
                                    mf_liste=[i for i in mf_numbers_for_load[today_str][dest][release_order_number][f"{carrier_code.split('-')[0]}-{carrier_code.split('-')[1].upper()}"]]
                                except:
                                    mf_liste=[]
                                if len(mf_liste)>0:
                                    try:
                                        load_mf_number = st.selectbox("MF NUMBER", mf_liste, disabled=False, key=14551, index=mf_liste.index(st.session_state.load_mf_number) if st.session_state.load_mf_number else 0)
                                    except:
                                        load_mf_number = st.selectbox("MF NUMBER", mf_liste, disabled=False, key=14551)
                                    mf=True
                                    load_mf_number_issued=True
                                    yes=True
                                    st.session_state.load_mf_number = load_mf_number
                                   
                                else:
                                    st.write(f"**:red[ASK ADMIN TO PUT SHIPMENT NUMBERS]**")
                                    mf=False
                                    yes=False
                                    load_mf_number_issued=False  
                            else:
                                st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
                                mf=False
                                yes=False
                                load_mf_number_issued=False  
                               
                            foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=8)
                            foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=123)
                        click_clear1 = st.button('CLEAR VEHICLE-QUANTITY INPUTS', key=34)
                        if click_clear1:
                             with placeholder.container():
                                 vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=17)
                        
                                 mf=True
                                 load_mf_number_issued=False
                                 carrier_code=st.text_input("Carrier Code",info[current_release_order][current_sales_order]["carrier_code"],disabled=True,key=19)
                                
                                 if carrier_code=="123456-KBX":
                                   if release_order_number in mf_numbers_for_load.keys():
                                       mf_liste=[i for i in mf_numbers_for_load[release_order_number]]
                                       if len(mf_liste)>0:
                                           load_mf_number=st.selectbox("SHIPMENT NUMBER",mf_liste,disabled=False,key=14551)
                                           mf=True
                                           load_mf_number_issued=True
                                           yes=True
                                       else:
                                           st.write(f"**:red[ASK ADMIN TO PUT SHIPMENT NUMBERS]**")
                                           mf=False
                                           yes=False
                                           load_mf_number_issued=False  
                                   else:
                                       st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
                                       mf=False
                                       yes=False
                                       load_mf_number_issued=False  
                                 foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=18)
                                 foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=1123)
                                 
                               
                        
                                
                        
                    with col2:
                        ocean_bol_to_batch = {"GSSWKIR6013D": 45302855,"GSSWKIR6013E": 45305548}
                        if double_load:
                            #release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                            release_order_number=current_release_order
                            #sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                            sales_order_item=current_sales_order
                            #ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                            ocean_bill_of_lading=info[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                            current_ocean_bill_of_lading=ocean_bill_of_lading
                            next_ocean_bill_of_lading=info[next_release_order][next_sales_order]["ocean_bill_of_lading"]
                            #batch=st.text_input("Batch",info[current_release_order][current_sales_order]["batch"],disabled=True)
                            batch=info[current_release_order][current_sales_order]["batch"]
                            #grade=st.text_input("Grade",info[current_release_order][current_sales_order]["grade"],disabled=True)
                            grade=info[current_release_order][current_sales_order]["grade"]
                            
                            #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                            pass
                        else:
                            release_order_number=current_release_order
                            sales_order_item=current_sales_order
                            ocean_bill_of_lading=release_order_database[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                            
                            batch=release_order_database[current_release_order][current_sales_order]["batch"]
                     
                            grade=release_order_database[current_release_order][current_sales_order]["grade"]
                            
                   
                        updated_quantity=0
                        live_quantity=0
                        if updated_quantity not in st.session_state:
                            st.session_state.updated_quantity=updated_quantity
                        load_digit=-2 if vessel=="KIRKENES-2304" else -3
                        def audit_unit(x):
                            if vessel=="LAGUNA-3142":
                                return True
                            if vessel=="KIRKENES-2304":
                                if len(x)>=10:
                                    if bill_mapping[vessel][x[:-2]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-2]]["Batch"]!=batch:
                                        st.write("**:red[WRONG B/L, DO NOT LOAD BELOW!]**")
                                        return False
                                    else:
                                        return True
                            else:
                                if bill_mapping[vessel][x[:-3]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-3]]["Batch"]!=batch:
                                    return False
                                else:
                                    return True
                        def audit_split(release,sales):
                            if vessel=="KIRKENES-2304":
                                if len(x)>=10:
                                    if bill_mapping[vessel][x[:-2]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-2]]["Batch"]!=batch:
                                        st.write("**:red[WRONG B/L, DO NOT LOAD BELOW!]**")
                                        return False
                                    else:
                                        return True
                            else:
                                if bill_mapping[vessel][x[:-3]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[vessel][x[:-3]]["Batch"]!=batch:
                                        return False
                                else:
                                    return True
                        
                        flip=False 
                        first_load_input=None
                        second_load_input=None
                        load_input=None
                        bale_load_input=None
                        if double_load:
                            
                            try:
                                next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                                
                                first_load_input=st.text_area("**FIRST SKU LOADS**",height=300)
                                first_quantity=0
                                second_quantity=0
                                if first_load_input is not None:
                                    first_textsplit = first_load_input.splitlines()
                                    first_textsplit=[i for i in first_textsplit if len(i)>8]
                                    first_quantity=len(first_textsplit)
                                second_load_input=st.text_area("**SECOND SKU LOADS**",height=300)
                                if second_load_input is not None:
                                    second_textsplit = second_load_input.splitlines()
                                    second_textsplit = [i for i in second_textsplit if len(i)>8]
                                    second_quantity=len(second_textsplit)
                                updated_quantity=first_quantity+second_quantity
                                st.session_state.updated_quantity=updated_quantity
                            except Exception as e: 
                                st.write(e)
                                #st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                                pass
                            
                        
                        else:
                            
            
            
                            placeholder1 = st.empty()
                            
                            
                            
                            load_input=placeholder1.text_area("**UNITS**",value="",height=300,key=1)#[:-2]
                        
                            
                    with col3:
                        placeholder2 = st.empty()
                        bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1111)#[:-2]
                        
                            
                    with col2:
                        click_clear = st.button('CLEAR SCANNED INPUTS', key=3)
                        if click_clear:
                            load_input = placeholder1.text_area("**UNITS**",value="",height=300,key=2)#
                            bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1121)#
                        if load_input is not None :
                            textsplit = load_input.splitlines()
                            textsplit=[i for i in textsplit if len(i)>8]
                            updated_quantity=len(textsplit)
                            st.session_state.updated_quantity=updated_quantity
                        if bale_load_input is not None:
                            bale_textsplit = bale_load_input.splitlines()
                            bale_textsplit=[i for i in bale_textsplit if len(i)>8]
                            bale_updated_quantity=len(bale_textsplit)
                            st.session_state.updated_quantity=updated_quantity+bale_updated_quantity*0.125
                       
                    with col4:
                        
                        if double_load:
                            first_faults=[]
                            if first_load_input is not None:
                                first_textsplit = first_load_input.splitlines()
                                first_textsplit=[i for i in first_textsplit if len(i)>8]
                                #st.write(textsplit)
                                for j,x in enumerate(first_textsplit):
                                    if audit_split(current_release_order,current_sales_order):
                                        st.text_input(f"Unit No : {j+1}",x)
                                        first_faults.append(0)
                                    else:
                                        st.text_input(f"Unit No : {j+1}",x)
                                        first_faults.append(1)
                            second_faults=[]
                            if second_load_input is not None:
                                second_textsplit = second_load_input.splitlines()
                                second_textsplit = [i for i in second_textsplit if len(i)>8]
                                #st.write(textsplit)
                                for i,x in enumerate(second_textsplit):
                                    if audit_split(next_release_order,next_sales_order):
                                        st.text_input(f"Unit No : {len(first_textsplit)+1+i}",x)
                                        second_faults.append(0)
                                    else:
                                        st.text_input(f"Unit No : {j+1+i+1}",x)
                                        second_faults.append(1)
            
                            loads=[]
                            
                            for k in first_textsplit:
                                loads.append(k)
                            for l in second_textsplit:
                                loads.append(l)
                                
                        ####   IF NOT double load
                        else:
                            
                            past_loads=[]
                            for row in bill_of_ladings.keys():
                                if row!="11502400":
                                    past_loads+=[load for load in bill_of_ladings[row]['loads'].keys()]
                            faults=[]
                            bale_faults=[]
                            fault_messaging={}
                            bale_fault_messaging={}
                            textsplit={}
                            bale_textsplit={}
                            if load_input is not None:
                                textsplit = load_input.splitlines()
                                
                                    
                                textsplit=[i for i in textsplit if len(i)>8]
                           
                                seen=set()
                                alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                for i,x in enumerate(textsplit):
                                    alternate_vessels=[ship for ship in bill_mapping if ship!=vessel]
                                    if x[:load_digit] in ["ds"]:
                                        st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                        faults.append(1)
                                        st.markdown("**:red[THIS LOT# IS FROM THE OTHER VESSEL!]**")
                                    else:
                                            
                                        if x[:load_digit] in bill_mapping[vessel] or vessel=="LAGUNA-3142" :
                                            if audit_unit(x):
                                                if x in seen:
                                                    st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                    faults.append(1)
                                                    st.markdown("**:red[This unit has been scanned TWICE!]**")
                                                if x in past_loads:
                                                    st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                    faults.append(1)
                                                    st.markdown("**:red[This unit has been SHIPPED!]**")   
                                                else:
                                                    st.write(f"**Unit No : {i+1}-{x}**")
                                                    faults.append(0)
                                            else:
                                                st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                st.write(f"**:red[WRONG B/L, DO NOT LOAD UNIT {x}]**")
                                                faults.append(1)
                                            seen.add(x)
                                        else:
                                            #st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                            faults.append(1)
                                            #st.markdown("**:red[This LOT# NOT IN INVENTORY!]**")
                                            #st.info(f"VERIFY THIS UNIT CAME FROM {vessel} - {'Unwrapped' if grade=='ISU' else 'wrapped'} piles")
                                            with st.expander(f"**:red[Unit No : {i+1}-{x} This LOT# NOT IN INVENTORY!---VERIFY UNIT {x} CAME FROM {vessel} - {'Unwrapped' if grade=='ISU' else 'wrapped'} piles]**"):
                                                st.write("Verify that the unit came from the pile that has the units for this release order and click to inventory")
                                                if st.button("ADD UNIT TO INVENTORY",key=f"{x}"):
                                                    updated_bill=bill_mapping.copy()
                                                    updated_bill[vessel][x[:load_digit]]={"Batch":batch,"Ocean_bl":ocean_bill_of_lading}
                                                    updated_bill=json.dumps(updated_bill)
                                                    #storage_client = storage.Client()
                                                    storage_client = get_storage_client()
                                                    bucket = storage_client.bucket(target_bucket)
                                                    blob = bucket.blob(rf"bill_mapping.json")
                                                    blob.upload_from_string(updated_bill)
    
                                                    alien_units=gcp_download(target_bucket,rf"alien_units.json")
                                                    alien_units=json.loads(alien_units)
                                                    if vessel not in alien_units:
                                                        alien_units[vessel]={}
                                                    alien_units[vessel][x]={}
                                                    alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                            "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                                    alien_units=json.dumps(alien_units)
                                                    #storage_client = storage.Client()
                                                    storage_client = get_storage_client()
                                                    bucket = storage_client.bucket(target_bucket)
                                                    blob = bucket.blob(rf"alien_units.json")
                                                    blob.upload_from_string(alien_units)
                                                    
                                                    
                                                    subject=f"FOUND UNIT {x} NOT IN INVENTORY"
                                                    body=f"Clerk identified an uninventoried {'Unwrapped' if grade=='ISU' else 'wrapped'} unit {x}, and after verifying the physical pile, inventoried it into Ocean Bill Of Lading : {ocean_bill_of_lading} for vessel {vessel}. Unit has been put into alien unit list."
                                                    sender = "warehouseoly@gmail.com"
                                                    #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                                    recipients = ["afsiny@portolympia.com"]
                                                    password = "xjvxkmzbpotzeuuv"
                                                    send_email(subject, body, sender, recipients, password)
                                                    time.sleep(0.1)
                                                    st.success(f"Added Unit {x} to Inventory!",icon="✅")
                                                    st.rerun()
                                    
                            if bale_load_input is not None:
                            
                                bale_textsplit = bale_load_input.splitlines()                       
                                bale_textsplit=[i for i in bale_textsplit if len(i)>8]                           
                                seen=set()
                                alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                for i,x in enumerate(bale_textsplit):
                                    alternate_vessel=[ship for ship in bill_mapping if ship!=vessel][0]
                                    if x[:load_digit] in bill_mapping[alternate_vessel]:
                                        st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                        faults.append(1)
                                        st.markdown("**:red[THIS BALE LOT# IS FROM THE OTHER VESSEL!]**")
                                    else:
                                        if x[:load_digit] in bill_mapping[vessel]:
                                            if audit_unit(x):
                                                st.write(f"**Bale No : {i+1}-{x}**")
                                                faults.append(0)
                                            else:
                                                st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                                st.write(f"**:red[WRONG B/L, DO NOT LOAD BALE {x}]**")
                                                faults.append(1)
                                            seen.add(x)
                                        else:
                                            faults.append(1)
                                            with st.expander(f"**:red[Bale No : {i+1}-{x} This LOT# NOT IN INVENTORY!---VERIFY BALE {x} CAME FROM {vessel} - {'Unwrapped' if grade=='ISU' else 'wrapped'} piles]**"):
                                                st.write("Verify that the bale came from the pile that has the units for this release order and click to inventory")
                                                if st.button("ADD BALE TO INVENTORY",key=f"{x}"):
                                                    updated_bill=bill_mapping.copy()
                                                    updated_bill[vessel][x[:load_digit]]={"Batch":batch,"Ocean_bl":ocean_bill_of_lading}
                                                    updated_bill=json.dumps(updated_bill)
                                                    #storage_client = storage.Client()
                                                    storage_client = get_storage_client()
                                                    bucket = storage_client.bucket(target_bucket)
                                                    blob = bucket.blob(rf"bill_mapping.json")
                                                    blob.upload_from_string(updated_bill)
    
                                                    alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                                    alien_units[vessel][x]={}
                                                    alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                            "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                                    alien_units=json.dumps(alien_units)
                                                   # storage_client = storage.Client()
                                                    storage_client = get_storage_client()
                                                    bucket = storage_client.bucket(target_bucket)
                                                    blob = bucket.blob(rf"alien_units.json")
                                                    blob.upload_from_string(alien_units)
                                                    
                                                    
                                                    subject=f"FOUND UNIT {x} NOT IN INVENTORY"
                                                    body=f"Clerk identified an uninventoried {'Unwrapped' if grade=='ISU' else 'wrapped'} unit {x}, and after verifying the physical pile, inventoried it into Ocean Bill Of Lading : {ocean_bill_of_lading} for vessel {vessel}. Unit has been put into alien unit list."
                                                    sender = "warehouseoly@gmail.com"
                                                    #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                                    recipients = ["afsiny@portolympia.com"]
                                                    password = "xjvxkmzbpotzeuuv"
                                                    send_email(subject, body, sender, recipients, password)
                                                    time.sleep(0.1)
                                                    st.success(f"Added Unit {x} to Inventory!",icon="✅")
                                                   # st.rerun()
                           
                               
                            loads={}
                            pure_loads={}
                            yes=True
                            if 1 in faults or 1 in bale_faults:
                                yes=False
                            
                            if yes:
                                pure_loads={**{k:0 for k in textsplit},**{k:0 for k in bale_textsplit}}
                                loads={**{k[:load_digit]:0 for k in textsplit},**{k[:load_digit]:0 for k in bale_textsplit}}
                                for k in textsplit:
                                    loads[k[:load_digit]]+=1
                                    pure_loads[k]+=1
                                for k in bale_textsplit:
                                    loads[k[:load_digit]]+=0.125
                                    pure_loads[k]+=0.125
                    with col3:
                        quantity=st.number_input("**Scanned Quantity of Units**",st.session_state.updated_quantity, key=None, help=None, on_change=None, disabled=True, label_visibility="visible")
                        st.markdown(f"**{quantity*2} TONS - {round(quantity*2*2204.62,1)} Pounds**")
                        
                        admt=round(float(release_order_database[current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
                        st.markdown(f"**ADMT= {admt} TONS**")
                    manual_time=False   
                    #st.write(faults)                  
                    if st.checkbox("Check for Manual Entry for Date/Time"):
                        manual_time=True
                        file_date=st.date_input("File Date",datetime.datetime.today(),disabled=False,key="popo3")
                        a=datetime.datetime.strftime(file_date,"%Y%m%d")
                        a_=datetime.datetime.strftime(file_date,"%Y-%m-%d")
                        file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False,key="popop")
                        b=file_time.strftime("%H%M%S")
                        b_=file_time.strftime("%H:%M:%S")
                        c=datetime.datetime.strftime(eta_date,"%Y%m%d")
                    else:     
                        
                        a=datetime.datetime.strftime(file_date,"%Y%m%d")
                        a_=datetime.datetime.strftime(file_date,"%Y-%m-%d")
                        b=file_time.strftime("%H%M%S")
                        b=(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).strftime("%H%M%S")
                        b_=(datetime.datetime.now()-datetime.timedelta(hours=utc_difference)).strftime("%H:%M:%S")
                        c=datetime.datetime.strftime(eta_date,"%Y%m%d")
                        
                        
                    
                    liste=[(i,datetime.datetime.strptime(bill_of_ladings[i]['issued'],"%Y-%m-%d %H:%M:%S")) for i in bill_of_ladings if i!="11502400"]
                    liste.sort(key=lambda a: a[1])
                    last_submitted=[i[0] for i in liste[-3:]]
                    last_submitted.reverse()
                    st.markdown(f"**Last Submitted Bill Of Ladings (From most recent) : {last_submitted}**")
                    
                    
                    
                    if yes and mf and quantity>0:
                        
                        if st.button('**:blue[SUBMIT EDI]**'):
                         
                            proceed=True
                            if not yes:
                                proceed=False
                                         
                            if fault_messaging.keys():
                                for i in fault_messaging.keys():
                                    error=f"**:red[Unit {fault_messaging[i]}]**"
                                    proceed=False
                            if remaining<0:
                                proceed=False
                                error="**:red[No more Items to ship on this Sales Order]"
                                st.write(error)
                            if not vehicle_id: 
                                proceed=False
                                error="**:red[Please check Vehicle ID]**"
                                st.write(error)
                            
                            if quantity!=foreman_quantity+int(foreman_bale_quantity)/8:
                                proceed=False
                                error=f"**:red[{updated_quantity} units and {bale_updated_quantity} bales on this truck. Please check. You planned for {foreman_quantity} units and {foreman_bale_quantity} bales!]** "
                                st.write(error)
                            if proceed:
                                carrier_code_mf=f"{carrier_code.split('-')[0]}-{carrier_code.split('-')[1]}"
                                carrier_code=carrier_code.split("-")[0]
                                
                                suzano_report=gcp_download(target_bucket,rf"suzano_report.json")
                                suzano_report=json.loads(suzano_report)
    
                                consignee=destination.split("-")[0]
                                consignee_city=mill_info[destination]["city"]
                                consignee_state=mill_info[destination]["state"]
                                vessel_suzano,voyage_suzano=vessel.split("-")
                                if manual_time:
                                    eta=datetime.datetime.strftime(file_date+datetime.timedelta(hours=mill_info[destination]['Driving_Hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['Driving_Minutes']+30),"%Y-%m-%d  %H:%M:%S")
                                else:
                                    eta=datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=mill_info[destination]['Driving_Hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['Driving_Minutes']+30),"%Y-%m-%d  %H:%M:%S")
        
                                if double_load:
                                    bill_of_lading_number,bill_of_ladings=gen_bill_of_lading()
                                    edi_name= f'{bill_of_lading_number}.txt'
                                    bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                                 "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                                 "quantity":len(first_textsplit),"issued":f"{a_} {b_}","edi_no":edi_name} 
                                    bill_of_ladings[str(bill_of_lading_number+1)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":next_sales_order,
                                                                                 "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                                 "quantity":len(second_textsplit),"issued":f"{a_} {b_}","edi_no":edi_name} 
                                
                                else:
                                    bill_of_lading_number,bill_of_ladings=gen_bill_of_lading()
                                    if load_mf_number_issued:
                                        bill_of_lading_number=st.session_state.load_mf_number
                                    if manual_bill:
                                        bill_of_lading_number=manual_bill_of_lading_number
                                    edi_name= f'{bill_of_lading_number}.txt'
                                    bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                                 "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                                 "quantity":st.session_state.updated_quantity,"issued":f"{a_} {b_}","edi_no":edi_name,"loads":pure_loads} 
                                                    
                                bill_of_ladings=json.dumps(bill_of_ladings)
                                #storage_client = storage.Client()
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                                blob.upload_from_string(bill_of_ladings)
                                
                                
                                
                                terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",bill_of_lading_number,disabled=True)
                                success_container=st.empty()
                                success_container.info("Uploading Bill of Lading")
                                time.sleep(0.1) 
                                success_container.success("Uploaded Bill of Lading...",icon="✅")
                                process()
                                #st.toast("Creating EDI...")
                                try:
                                    suzano_report_keys=[int(i) for i in suzano_report.keys()]
                                    next_report_no=max(suzano_report_keys)+1
                                except:
                                    next_report_no=1
                                if double_load:
                                    
                                    suzano_report.update({next_report_no:{"Date Shipped":f"{a_} {b_}","Vehicle":vehicle_id, "Shipment ID #": bill_of_lading_number, "Consignee":consignee,"Consignee City":consignee_city,
                                                         "Consignee State":consignee_state,"Release #":release_order_number,"Carrier":carrier_code,
                                                         "ETA":eta,"Ocean BOL#":ocean_bill_of_lading,"Warehouse":"OLYM","Vessel":vessel_suzano,"Voyage #":voyage_suzano,"Grade":wrap,"Quantity":quantity,
                                                         "Metric Ton": quantity*2, "ADMT":admt,"Mode of Transportation":transport_type}})
                                else:
                                   
                                    suzano_report.update({next_report_no:{"Date Shipped":f"{a_} {b_}","Vehicle":vehicle_id, "Shipment ID #": bill_of_lading_number, "Consignee":consignee,
                                                                          "Consignee City":consignee_city,
                                                         "Consignee State":consignee_state,"Release #":release_order_number,"Carrier":carrier_code,
                                                         "ETA":eta,"Ocean BOL#":ocean_bill_of_lading,"Batch#":batch,
                                                         "Warehouse":"OLYM","Vessel":vessel_suzano,"Voyage #":voyage_suzano,"Grade":wrap,"Quantity":quantity,
                                                         "Metric Ton": quantity*2, "ADMT":admt,"Mode of Transportation":transport_type}})
                                    suzano_report=json.dumps(suzano_report)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"suzano_report.json")
                                    blob.upload_from_string(suzano_report)
                                    success_container1=st.empty()
                                    time.sleep(0.1)                            
                                    success_container1.success(f"Updated Suzano Report",icon="✅")
        
                                  
                                    
                                if double_load:
                                    info[current_release_order][current_sales_order]["shipped"]=info[current_release_order][current_sales_order]["shipped"]+len(first_textsplit)
                                    info[current_release_order][current_sales_order]["remaining"]=info[current_release_order][current_sales_order]["remaining"]-len(first_textsplit)
                                    info[next_release_order][next_sales_order]["shipped"]=info[next_release_order][next_sales_order]["shipped"]+len(second_textsplit)
                                    info[next_release_order][next_sales_order]["remaining"]=info[next_release_order][next_sales_order]["remaining"]-len(second_textsplit)
                                else:
                                    release_order_database[current_release_order][current_sales_order]["shipped"]=release_order_database[current_release_order][current_sales_order]["shipped"]+quantity
                                    release_order_database[current_release_order][current_sales_order]["remaining"]=release_order_database[current_release_order][current_sales_order]["remaining"]-quantity
                                if release_order_database[current_release_order][current_sales_order]["remaining"]<=0:
                                    to_delete=[]
                                    for release in dispatched.keys():
                                        if release==current_release_order:
                                            for sales in dispatched[release].keys():
                                                if sales==current_sales_order:
                                                    to_delete.append((release,sales))
                                    for victim in to_delete:
                                        del dispatched[victim[0]][victim[1]]
                                        if len(dispatched[victim[0]].keys())==0:
                                            del dispatched[victim[0]]
                                    
                                    json_data = json.dumps(dispatched)
                                    #storage_client = storage.Client()
                                    storage_client = get_storage_client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"dispatched.json")
                                    blob.upload_from_string(json_data)       
                                def check_complete(data,ro):
                                    complete=True
                                    for i in data[ro]:
                                        if i in ["001","002","003","004","005"]:
                                            if data[ro][i]['remaining']>0:
                                                complete=False
                                    return complete
                                if check_complete(release_order_database,current_release_order):
                                    release_order_database[current_release_order]['complete']=True
                                
                                json_data = json.dumps(release_order_database)
                                #storage_client = storage.Client()
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                                blob.upload_from_string(json_data)
                                                          
                                success_container3=st.empty()
                                time.sleep(0.1)                            
                                success_container3.success(f"Updated Release Order Database",icon="✅")
                                with open('placeholder.txt', 'r') as f:
                                    output_text = f.read()
                                
                                #st.markdown("**EDI TEXT**")
                                #st.text_area('', value=output_text, height=600)
                                with open('placeholder.txt', 'r') as f:
                                    file_content = f.read()
                                newline="\n"
                                filename = f'{bill_of_lading_number}'
                                file_name= f'{bill_of_lading_number}.txt'
                                
                                
                                subject = f'Suzano_EDI_{a}_ R.O:{release_order_number}-Terminal BOL :{bill_of_lading_number}-Destination : {destination}'
                                body = f"EDI for Below attached.{newline}Release Order Number : {current_release_order} - Sales Order Number:{current_sales_order}{newline} Destination : {destination} Ocean Bill Of Lading : {ocean_bill_of_lading}{newline}Terminal Bill of Lading: {terminal_bill_of_lading} - Grade : {wrap} {newline}{2*quantity} tons {unitized} cargo were loaded to vehicle : {vehicle_id} with Carried ID : {carrier_code} {newline}Truck loading completed at {a_} {b_}"
                                #st.write(body)           
                                sender = "warehouseoly@gmail.com"
                                recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                #recipients = ["afsiny@portolympia.com"]
                                password = "xjvxkmzbpotzeuuv"
                        
                      
                        
                        
                                with open('temp_file.txt', 'w') as f:
                                    f.write(file_content)
                        
                                file_path = 'temp_file.txt'  # Use the path of the temporary file
                        
                                
                                upload_cs_file(target_bucket, 'temp_file.txt',rf"EDIS/{file_name}")
                                success_container5=st.empty()
                                time.sleep(0.1)                            
                                success_container5.success(f"Uploaded EDI File",icon="✅")
                                
                                
                                
                                try:
                                        mf_numbers_for_load[a_][dest][release_order_number][carrier_code_mf.upper()].remove(str(bill_of_lading_number))
                                except:
                                        mf_numbers_for_load[a_][dest][release_order_number][carrier_code_mf.upper()].remove(int(bill_of_lading_number))
                                mf_numbers_for_load=json.dumps(mf_numbers_for_load)
                                #storage_client = storage.Client()
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"release_orders/mf_numbers.json")
                                blob.upload_from_string(mf_numbers_for_load)
                                st.write("Updated MF numbers...")
                                  
                                send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                                success_container6=st.empty()
                                time.sleep(0.1)                            
                                success_container6.success(f"Sent EDI Email",icon="✅")
                                st.markdown("**SUCCESS! EDI FOR THIS LOAD HAS BEEN SUBMITTED,THANK YOU**")
                                st.write(filename,current_release_order,current_sales_order,destination,ocean_bill_of_lading,terminal_bill_of_lading,wrap)
                                this_shipment_aliens=[]
                                for i in pure_loads:
                                    try:
                                            
                                        if i in alien_units[vessel]:       
                                            alien_units[vessel][i]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                    "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S"),
                                                    "Destination":destination,"Release_Order":current_release_order,"Terminal_Bill_of Lading":terminal_bill_of_lading,"Truck":vehicle_id}
                                            this_shipment_aliens.append(i)
                                        if i not in alien_units[vessel]:
                                            if i[:-2] in [u[:-2] for u in alien_units[vessel]]:
                                                alien_units[vessel][i]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                    "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S"),
                                                    "Destination":destination,"Release_Order":current_release_order,"Terminal_Bill_of Lading":terminal_bill_of_lading,"Truck":vehicle_id}
                                                this_shipment_aliens.append(i)
                                    except:
                                        pass
                                alien_units=json.dumps(alien_units)
                                storage_client = get_storage_client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"alien_units.json")
                                blob.upload_from_string(alien_units)   
                                
                                if len(this_shipment_aliens)>0:
                                    subject=f"UNREGISTERED UNITS SHIPPED TO {destination} on RELEASE ORDER {current_release_order}"
                                    body=f"{len([i for i in this_shipment_aliens])} unregistered units were shipped on {vehicle_id} to {destination} on {current_release_order}.<br>{[i for i in this_shipment_aliens]}"
                                    sender = "warehouseoly@gmail.com"
                                    recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                    #recipients = ["afsiny@portolympia.com"]
                                    password = "xjvxkmzbpotzeuuv"
                                    send_email(subject, body, sender, recipients, password)
                                
                            else:   ###cancel bill of lading
                                pass
        
                    

    
                
        else:
            st.subheader("**Nothing dispatched!**")
   
        
                    ###########################       SUZANO INVENTORY BOARD    ###########################
 
    elif username == 'olysuzanodash':
        #map=gcp_download_new(target_bucket,rf"map.json")
        map=gcp_download(target_bucket,rf"map.json")
        map=json.loads(map)
        mill_info=map["mill_info"]
        inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
        inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
     
        raw_ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
        raw_ro = json.loads(raw_ro)
        bol_mapping = map["bol_mapping"]
        
        suzano,edi_bank,main_inventory,status,mill_progress=st.tabs(["SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","RELEASE ORDER STATUS","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS"])
        
        with suzano:
            
            
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')
            
            now=datetime.datetime.now()-datetime.timedelta(hours=utc_difference)
            suzano_report_=gcp_download(target_bucket,rf"suzano_report.json")
            suzano_report=json.loads(suzano_report_)
            suzano_report=pd.DataFrame(suzano_report).T
            suzano_report=suzano_report[["Date Shipped","Vehicle", "Shipment ID #", "Consignee","Consignee City","Consignee State","Release #","Carrier","ETA","Ocean BOL#","Batch#","Warehouse","Vessel","Voyage #","Grade","Quantity","Metric Ton", "ADMT","Mode of Transportation"]]
            suzano_report["Shipment ID #"]=[str(i) for i in suzano_report["Shipment ID #"]]
            suzano_report["Batch#"]=[str(i) for i in suzano_report["Batch#"]]
            daily_suzano=suzano_report.copy()
            daily_suzano["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in suzano_report["Date Shipped"]]
            daily_suzano_=daily_suzano[daily_suzano["Date"]==now.date()]
            
            choose = st.radio(
                            "Select Daily or Accumulative Report",
                            ["DAILY", "ACCUMULATIVE", "FIND BY DATE","FIND DATE RANGE","BY RELEASE ORDER","BY BATCH"])
            if choose=="DAILY":
                daily_suzano_=daily_suzano_.reset_index(drop=True)
                daily_suzano_.index=[i+1 for i in daily_suzano_.index]
                daily_suzano_.loc["TOTAL"]=daily_suzano_[["Quantity","Metric Ton","ADMT"]].sum()
                st.dataframe(daily_suzano_)
                csv=convert_df(daily_suzano_)
                file_name=f'OLYMPIA_DAILY_REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
            elif choose=="FIND BY DATE":
                required_date=st.date_input("CHOOSE DATE",key="dssar")
                filtered_suzano=daily_suzano[daily_suzano["Date"]==required_date]
                filtered_suzano=filtered_suzano.reset_index(drop=True)
                filtered_suzano.index=[i+1 for i in filtered_suzano.index]
                filtered_suzano.loc["TOTAL"]=filtered_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                st.dataframe(filtered_suzano)
                csv=convert_df(filtered_suzano)
                file_name=f'OLYMPIA_SHIPMENT_REPORT-{datetime.datetime.strftime(required_date,"%m-%d,%Y")}.csv'

            elif choose=="BY RELEASE ORDER":
                report_ro=st.selectbox("SELECT RELEASE ORDER",([i for i in raw_ro]),key="sdgerw")
                ro_suzano=daily_suzano[daily_suzano["Release #"]==report_ro]
                ro_suzano=ro_suzano.reset_index(drop=True)
                ro_suzano.index=[i+1 for i in ro_suzano.index]
                ro_suzano.loc["TOTAL"]=ro_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                st.dataframe(ro_suzano)
                csv=convert_df(ro_suzano)
                file_name=f'OLYMPIA_SHIPMENT_REPORT-ReleaseOrder-{report_ro}.csv'

            elif choose=="BY BATCH":
                report_batch=st.selectbox("SELECT BATCH",(daily_suzano["Batch#"].unique()),key="sddgerw")
                batch_suzano=daily_suzano[daily_suzano["Batch#"]==report_batch]
                batch_suzano=batch_suzano.reset_index(drop=True)
                batch_suzano.index=[i+1 for i in batch_suzano.index]
                batch_suzano.loc["TOTAL"]=batch_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                st.dataframe(batch_suzano)
                csv=convert_df(batch_suzano)
                file_name=f'OLYMPIA_SHIPMENT_REPORT-Batch-{report_batch}.csv'

            elif choose=="FIND DATE RANGE":
                datecol1,datecol2,datecol3=st.columns([3,3,4])
                with datecol1:
                    tarih1=st.date_input("FROM",key="dsssaar")
                with datecol2:
                    tarih2=st.date_input("TO",key="dssdar")
                    
                range_suzano=daily_suzano[(daily_suzano["Date"]>=tarih1)&(daily_suzano["Date"]<=tarih2)]
                range_suzano=range_suzano.reset_index(drop=True)
                range_suzano.index=[i+1 for i in range_suzano.index]
                range_suzano.loc["TOTAL"]=range_suzano[["Quantity","Metric Ton","ADMT"]].sum()
                st.dataframe(range_suzano)
                csv=convert_df(range_suzano)
                file_name=f'OLYMPIA_SHIPMENT_REPORT-daterange.csv'
            
            else:
                st.dataframe(suzano_report)
                csv=convert_df(suzano_report)
                file_name=f'OLYMPIA_ALL_SHIPMENTS to {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
            
            
            
           
            
        
            st.download_button(
                label="DOWNLOAD REPORT AS CSV",
                data=csv,
                file_name=file_name,
                mime='text/csv')
            

        with edi_bank:
            edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/")
            requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
            
            display_edi=st.toggle("DISPLAY EDI")
            if display_edi:
                data=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}")
                st.text_area("EDI",data,height=400)                                
           
            st.download_button(
                label="DOWNLOAD EDI",
                data=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}"),
                file_name=f'{requested_edi_file}',
                mime='text/csv')
            
            
            
        with main_inventory:
            
            maintenance=False
                            
            if maintenance:
                st.title("CURRENTLY UNDER MAINTENANCE, CHECK BACK LATER")
                           
            else:
                inventory,daily,unregistered=st.tabs(["INVENTORY","DAILY SHIPMENT REPORT","UNREGISTERED LOTS FOUND"])
                
                with daily:
                    
                    amount_dict={"KIRKENES-2304":9200,"JUVENTAS-2308":10000,"LYSEFJORD-2308":10000,"LAGUNA-3142":453,"FRONTIER-55VC":9811,"BEIJA_FLOR-88VC":11335}
                    inv_vessel=st.selectbox("Select Vessel",[i for i in map['batch_mapping']])
                    kf=inv_bill_of_ladings.iloc[1:].copy()
                    kf['issued'] = pd.to_datetime(kf['issued'])
                    kf=kf[kf["vessel"]==inv_vessel]
                    
                    kf['Date'] = kf['issued'].dt.date
                    kf['Date'] = pd.to_datetime(kf['Date'])
                    # Create a date range from the minimum to maximum date in the 'issued' column
                    date_range = pd.date_range(start=kf['Date'].min(), end=kf['Date'].max(), freq='D')
                    
                    # Create a DataFrame with the date range
                    date_df = pd.DataFrame({'Date': date_range})
                    # Merge the date range DataFrame with the original DataFrame based on the 'Date' column
                    merged_df = pd.merge(date_df, kf, how='left', on='Date')
                    merged_df['quantity'].fillna(0, inplace=True)
                    merged_df['Shipped Tonnage']=merged_df['quantity']*2
                    merged_df_grouped=merged_df.groupby('Date')[['quantity','Shipped Tonnage']].sum()
                    merged_df_grouped['Accumulated_Quantity'] = merged_df_grouped['quantity'].cumsum()
                    merged_df_grouped["Accumulated_Tonnage"]=merged_df_grouped['Accumulated_Quantity']*2
                    merged_df_grouped["Remaining_Units"]=[amount_dict[inv_vessel]-i for i in merged_df_grouped['Accumulated_Quantity']]
                    merged_df_grouped["Remaining_Tonnage"]=merged_df_grouped["Remaining_Units"]*2
                    merged_df_grouped.rename(columns={'quantity':"Shipped Quantity", 'Accumulated_Quantity':"Shipped Qty To_Date",
                                                      'Accumulated_Tonnage':"Shipped Tonnage To_Date"},inplace=True)
                    merged_df_grouped=merged_df_grouped.reset_index()
                    merged_df_grouped["Date"]=merged_df_grouped['Date'].dt.strftime('%m-%d-%Y, %A')
                    #merged_df_grouped=merged_df_grouped.set_index("Date",drop=True)
                  
                    st.dataframe(merged_df_grouped)
                    csv_inventory=convert_df(merged_df_grouped)
                    st.download_button(
                        label="DOWNLOAD INVENTORY REPORT AS CSV",
                        data=csv_inventory,
                        file_name=f'INVENTORY REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                        mime='text/csv')   
                    
                        
                with inventory:
                    bols = {}
                    for key, value in raw_ro.items():
                        for item_key, item_value in value.items():
                            if isinstance(item_value, dict):
                                ocean_bill_of_lading = item_value.get('ocean_bill_of_lading')
                                if ocean_bill_of_lading:
                                    if ocean_bill_of_lading not in bols:
                                        bols[ocean_bill_of_lading] = []
                                    bols[ocean_bill_of_lading].append(key)
                    for i in map['bol_mapping']:
                        if i not in bols:
                            bols[i]=[]

                    inventory={}
                    a=[i for i in map['bol_mapping']]
                    for bill in a:
                        inventory[bill]=[map['bol_mapping'][bill]['total'],map['bol_mapping'][bill]['damaged']]
                        
                    def extract_qt(data,ro,bol):
                        totals=[0,0,0]
                        sales_group=["001","002","003","004","005"]
                        for item in data[ro]:
                            if item in sales_group:
                                if data[ro][item]['ocean_bill_of_lading']==bol:
                                    totals[0]+=data[ro][item]['total']
                                    totals[1]+=data[ro][item]['shipped']
                                    totals[2]+=data[ro][item]['remaining']
                        return totals
                    
                    final={}
                    for k in inventory.keys():
                        final[k]={"Total":0,"Damaged":0,"Fit To Ship":0,"Allocated to ROs":0,"Shipped":0,
                                 "Remaining in Warehouse":0,"Remaining on ROs":0,"Remaining After ROs":0}
                        vector=np.zeros(3)
                        final[k]["Total"]=inventory[k][0]
                        final[k]["Damaged"]=inventory[k][1]
                        final[k]["Fit To Ship"]=final[k]["Total"]-final[k]["Damaged"]   
                    
                        if k in bols:
                            if len(bols[k])>0:
                                for ro in set(bols[k]):
                                    a,b,c=extract_qt(raw_ro,ro,k)[0],extract_qt(raw_ro,ro,k)[1],extract_qt(raw_ro,ro,k)[2]
                                    final[k]["Allocated to ROs"]+=a
                                    #final[k]["Shipped"]=inv_bill_of_ladings.groupby("ocean_bill_of_lading")[['quantity']].sum().loc[k,'quantity']
                                    final[k]["Shipped"]+=b
                    
                                    final[k]["Remaining in Warehouse"]=final[k]["Fit To Ship"]-final[k]["Shipped"]
                                    final[k]["Remaining on ROs"]=final[k]["Allocated to ROs"]-final[k]["Shipped"]
                                    final[k]["Remaining After ROs"]=final[k]["Fit To Ship"]-final[k]["Allocated to ROs"]
                            else:
                                final[k]["Remaining in Warehouse"]=final[k]["Fit To Ship"]
                                final[k]["Remaining on ROs"]=0
                                final[k]["Remaining After ROs"]=final[k]["Fit To Ship"]-final[k]["Allocated to ROs"]
                        else:
                            pass
                    temp=pd.DataFrame(final).T
                    temp.loc["TOTAL"]=temp.sum(axis=0)
                    
                    
                    tempo=temp*2

                    inv_col1,inv_col2=st.columns([7,3])
                    with inv_col1:
                        st.subheader("By Ocean BOL,UNITS")
                        st.dataframe(temp)
                        st.subheader("By Ocean BOL,TONS")
                        st.dataframe(tempo)
        
                with status:
                
                    status_dict={}
                    sales_group=["001","002","003","004","005"]
                    for ro in raw_ro:
                        for sale in [i for i in raw_ro[ro] if i in sales_group]:
                            status_dict[f"{ro}-{sale}"]={"Release Order #":ro,"Sales Order #":sale,
                                                "Destination":raw_ro[ro]['destination'],
                                                "Ocean BOL":raw_ro[ro][sale]['ocean_bill_of_lading'],
                                                "Total":raw_ro[ro][sale]['total'],
                                                "Shipped":raw_ro[ro][sale]['shipped'],
                                                "Remaining":raw_ro[ro][sale]['remaining']}
                    status_frame=pd.DataFrame(status_dict).T.set_index("Release Order #",drop=True)
                    active_frame_=status_frame[status_frame["Remaining"]>0]
                    status_frame.loc["Total"]=status_frame[["Total","Shipped","Remaining"]].sum()
                    active_frame=active_frame_.copy()
                    active_frame.loc["Total"]=active_frame[["Total","Shipped","Remaining"]].sum()
                    
                    st.markdown(active_frame.to_html(render_links=True),unsafe_allow_html=True)
    
                    
                    release_orders = status_frame.index[:-1]
                    release_orders = pd.Categorical(release_orders)
                    active_order_names = [f"{i} to {raw_ro[i]['destination']}" for i in active_frame_.index]
                    destinations=[raw_ro[i]['destination'] for i in active_frame_.index]
                    active_orders=[str(i) for i in active_frame.index]
                   
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=active_orders, y=active_frame["Total"], name='Total', marker_color='lightgray'))
                    fig.add_trace(go.Bar(x=active_orders, y=active_frame["Shipped"], name='Shipped', marker_color='blue', opacity=0.7))
                    remaining_data = [remaining if remaining > 0 else None for remaining in active_frame_["Remaining"]]
                    fig.add_trace(go.Scatter(x=active_orders, y=remaining_data, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
                    
                    #annotations = [dict(x=release_order, y=total_quantity, text=destination, showarrow=True, arrowhead=4, ax=0, ay=-30) for release_order, total_quantity, destination in zip(active_orders, active_frame["Total"], destinations)]
                    #fig.update_layout(annotations=annotations)
    
                    # fig.add_annotation(x="3172296", y=800, text="destination",
                    #                        showarrow=True, arrowhead=4, ax=0, ay=-30)
                    
                    fig.update_layout(title='ACTIVE RELEASE ORDERS',
                                      xaxis_title='Release Orders',
                                      yaxis_title='Quantities',
                                      barmode='overlay',
                                      width=1300,
                                      height=700,
                                      xaxis=dict(tickangle=-90, type='category'))
                    
                    st.plotly_chart(fig)
                    
                    
                    duration=st.toggle("Duration Report")
                    if duration:
                        
                        temp_dict={}
                            
                        for rel_ord in raw_ro:
                            for sales in [i for i in raw_ro[rel_ord] if i in ["001","002","003","004","005"]]:
                                temp_dict[rel_ord,sales]={}
                                dest=raw_ro[rel_ord]['destination']
                                vessel=raw_ro[rel_ord][sales]['vessel']
                                total=raw_ro[rel_ord][sales]['total']
                                remaining=raw_ro[rel_ord][sales]['remaining']
                                temp_dict[rel_ord,sales]={'destination': dest,'vessel': vessel,'total':total,'remaining':remaining}
                        temp_df=pd.DataFrame(temp_dict).T
                      
                        temp_df= temp_df.rename_axis(['release_order','sales_order'], axis=0)
                    
                        temp_df['First Shipment'] = temp_df.index.map(inv_bill_of_ladings.groupby(['release_order','sales_order'])['issued'].first())
                        
                        for i in temp_df.index:
                            if temp_df.loc[i,'remaining']<=2:
                                try:
                                    temp_df.loc[i,"Last Shipment"]=inv_bill_of_ladings.groupby(['release_order','sales_order']).issued.last().loc[i]
                                except:
                                    temp_df.loc[i,"Last Shipment"]=datetime.datetime.now()
                                temp_df.loc[i,"Duration"]=(pd.to_datetime(temp_df.loc[i,"Last Shipment"])-pd.to_datetime(temp_df.loc[i,"First Shipment"])).days+1
                        
                        temp_df['First Shipment'] = temp_df['First Shipment'].fillna(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
                        temp_df['Last Shipment'] = temp_df['Last Shipment'].fillna(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
                        
                        ####
                        
                        def business_days(start_date, end_date):
                            return pd.date_range(start=start_date, end=end_date, freq=BDay())
                        temp_df['# of Shipment Days'] = temp_df.apply(lambda row: len(business_days(row['First Shipment'], row['Last Shipment'])), axis=1)
                        df_temp=inv_bill_of_ladings.copy()
                        df_temp["issued"]=[pd.to_datetime(i).date() for i in df_temp["issued"]]
                        for i in temp_df.index:
                            try:
                                temp_df.loc[i,"Utilized Shipment Days"]=df_temp.groupby(["release_order",'sales_order'])[["issued"]].nunique().loc[i,'issued']
                            except:
                                temp_df.loc[i,"Utilized Shipment Days"]=0
                        
                        temp_df['First Shipment'] = temp_df['First Shipment'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),'%d-%b,%Y'))
                        temp_df['Last Shipment'] = temp_df['Last Shipment'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),'%d-%b,%Y') if type(x)==str else None)
                        liste=['Duration','# of Shipment Days',"Utilized Shipment Days"]
                        for col in liste:
                            temp_df[col] = temp_df[col].apply(lambda x: f" {int(x)} days" if not pd.isna(x) else np.nan)
                        temp_df['remaining'] = temp_df['remaining'].apply(lambda x: int(x))
                        temp_df.columns=['Destination', 'Vessel', 'Total Units', 'Remaining Units', 'First Shipment',
                               'Last Shipment', 'Duration', '# of Calendar Shipment Days',
                               'Utilized Calendar Shipment Days']
                        st.dataframe(temp_df)
        with mill_progress:
            
            mf_numbers=json.loads(gcp_download(target_bucket,rf"release_orders/mf_numbers.json"))
                
            bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            bill_of_ladings=json.loads(bill_of_ladings)
            bill=pd.DataFrame(bill_of_ladings).T
            values=[]
            mfs=[]
            for i in bill.index:
                if len(i.split("|"))>1:
                    values.append(i.split("|")[1])
                    mfs.append(i.split("|")[0])
                else:
                    values.append(None)
                    mfs.append(i)
            bill.insert(0,"Shipment",values)
            bill.insert(1,"MF",mfs)
            
            suzano_shipment_=gcp_download(target_bucket,rf"release_orders/suzano_shipments.json")
            suzano_shipment=json.loads(suzano_shipment_)
            suzano_shipment=pd.DataFrame(suzano_shipment).T

            suzano_shipment["Shipment ID"]=suzano_shipment["Shipment ID"].astype("str")
            bill["Shipment"]=bill["Shipment"].astype("str")
            suzano_shipment["Pickup"]=pd.to_datetime(suzano_shipment["Pickup"])
            suzano_shipment=suzano_shipment[suzano_shipment["Pickup"]>datetime.datetime(2024, 9, 24)]
            suzano_shipment.reset_index(drop=True,inplace=True)
            for i in suzano_shipment.index:
                sh=suzano_shipment.loc[i,"Shipment ID"]
                #print(sh)
                if sh in bill[~bill["Shipment"].isna()]["Shipment"].to_list():
                    vehicle=bill.loc[bill["Shipment"]==sh,'vehicle'].values[0]
                    bol=str(bill.loc[bill["Shipment"]==sh].index.values[0])
                    suzano_shipment.loc[i,"Transit Status"]="COMPLETED"
                    suzano_shipment.loc[i,"BOL"]=bol
                    suzano_shipment.loc[i,"Vehicle ID"]=vehicle
            for rel,value in mf_numbers.items():
                for date in value:
                    for carrier,liste in value[date].items():
                        if len(liste)>0:
                            try:
                                for k in liste:
                                    suzano_shipment.loc[suzano_shipment["Shipment ID"]==k.split("|")[1],"Transit Status"]="SCHEDULED"
                            except:
                                pass
            st.subheader("SUZANO OTM LIST")
            st.write(suzano_shipment)

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
    
    
    
    
        
     





















