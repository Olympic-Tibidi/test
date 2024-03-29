import streamlit as st
import streamlit.components.v1 as components
#import cv2
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
import seaborn as sns
import datetime as dt
#from pyzbar import pyzbar
import pickle
import yaml
from yaml.loader import SafeLoader
#from streamlit_extras.dataframe_explorer import dataframe_explorer
import math
import plotly.express as px               #to create interactive charts
import plotly.graph_objects as go         #to create interactive charts
import zipfile
import requests
from bs4 import BeautifulSoup
from PIL import Image
import plotly.graph_objects as go
import re
import tempfile
import plotly.graph_objects as go
import pydeck as pdk
from pandas.tseries.offsets import BDay
import calendar
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
#from IPython.display import IFrame
from reportlab.platypus import Table, TableStyle
import pypdfium2 as pdfium
#import streamlit_option_menu
#from streamlit_modal import Modal

#import streamlit_option_menu
#from streamlit_modal import Modal

st.set_page_config(layout="wide")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"

target_bucket="suzano2"
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
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_text()
    return data
    
def gcp_download_x(bucket_name, source_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_bytes()
    return data

def gcp_csv_to_df(bucket_name, source_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data),index_col=None)
    print(f'Pulled down file from bucket {bucket_name}, file name: {source_file_name}')
    return df
def upload_cs_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)
    return True
def upload_json_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name,content_type="application/json")
    return True
def upload_xl_file(bucket_name, uploaded_file, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    uploaded_file.seek(0)

    # Upload the file from the file object provided by st.file_uploader
    blob.upload_from_file(uploaded_file)
# define function that list files in the bucket
def list_cs_files(bucket_name): 
    storage_client = storage.Client()

    file_list = storage_client.list_blobs(bucket_name)
    file_list = [file.name for file in file_list]

    return file_list
def list_cs_files_f(bucket_name, folder_name):
    storage_client = storage.Client()

    # List all blobs in the bucket
    blobs = storage_client.list_blobs(bucket_name)

    # Filter blobs that are within the specified folder
    folder_files = [blob.name for blob in blobs if blob.name.startswith(folder_name)]

    return folder_files
def list_files_in_folder(bucket_name, folder_name):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)

    # Extract only the filenames without the folder path
    filenames = [blob.name.split("/")[-1] for blob in blobs if "/" in blob.name]

    return filenames
def list_files_in_subfolder(bucket_name, folder_name):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name, delimiter='/')

    # Extract only the filenames without the folder path
    filenames = [blob.name.split('/')[-1] for blob in blobs]

    return filenames
def store_release_order_data(release_order_number,destination,po_number,sales_order_item,vessel,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code):
       
    # Create a dictionary to store the release order data
    release_order_data = { release_order_number:{
        
        
        'destination':destination,
        "po_number":po_number,
        sales_order_item: {
        "vessel":vessel,
        "batch": batch,
        "ocean_bill_of_lading": ocean_bill_of_lading,
        "grade": wrap,
        "dryness":dryness,
        "transport_type": transport_type,
        "carrier_code": carrier_code,
        "unitized":unitized,
        "quantity":quantity,
        "tonnage":tonnage,
        "shipped":0,
        "remaining":quantity       
        }}              
    }
    
                         

    # Convert the dictionary to JSON format
    json_data = json.dumps(release_order_data)
    return json_data

def add_release_order_data(file,release_order_number,destination,po_number,sales_order_item,vessel,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code):
       
    # Edit the loaded current dictionary.
    file[release_order_number]["destination"]= destination
    file[release_order_number]["po_number"]= po_number
    if sales_order_item not in file[release_order_number]:
        file[release_order_number][sales_order_item]={}
    file[release_order_number][sales_order_item]["vessel"]= vessel
    file[release_order_number][sales_order_item]["batch"]= batch
    file[release_order_number][sales_order_item]["ocean_bill_of_lading"]= ocean_bill_of_lading
    file[release_order_number][sales_order_item]["grade"]= wrap
    file[release_order_number][sales_order_item]["dryness"]= dryness
    file[release_order_number][sales_order_item]["transport_type"]= transport_type
    file[release_order_number][sales_order_item]["carrier_code"]= carrier_code
    file[release_order_number][sales_order_item]["unitized"]= unitized
    file[release_order_number][sales_order_item]["quantity"]= quantity
    file[release_order_number][sales_order_item]["tonnage"]= tonnage
    file[release_order_number][sales_order_item]["shipped"]= 0
    file[release_order_number][sales_order_item]["remaining"]= quantity
    
    
       

    # Convert the dictionary to JSON format
    json_data = json.dumps(file)
    return json_data

def edit_release_order_data(file,sales_order_item,quantity,tonnage,shipped,remaining,carrier_code):
       
    # Edit the loaded current dictionary.
    
    file[release_order_number][sales_order_item]["quantity"]= quantity
    file[release_order_number][sales_order_item]["tonnage"]= tonnage
    file[release_order_number][sales_order_item]["shipped"]= shipped
    file[release_order_number][sales_order_item]["remaining"]= remaining
    file[release_order_number][sales_order_item]["carrier_code"]= carrier_code
    
    
       

    # Convert the dictionary to JSON format
    json_data = json.dumps(file)
    return json_data

def process():
           
    line1="1HDR:"+a+b+terminal_code
    tsn="01" if medium=="TRUCK" else "02"
    
    tt="0001" if medium=="TRUCK" else "0002"
    if double_load:
        line21="2DTD:"+current_release_order+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(first_quantity*2000)+" "*(16-len(str(first_quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
        line22="2DTD:"+next_release_order+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(second_quantity*2000)+" "*(16-len(str(second_quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
    line2="2DTD:"+release_order_number+" "*(10-len(release_order_number))+"000"+sales_order_item+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(int(quantity*2000))+" "*(16-len(str(int(quantity*2000))))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
               
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
        
        forecast=get_weather()
        
        
        select=st.sidebar.radio("SELECT FUNCTION",
            ('ADMIN', 'LOADOUT', 'INVENTORY','LABOR','WEATHER','TIDES','FINANCE'))
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
        with st.sidebar.container(border=True):
            st.markdown(f"Temperature: {forecast['current']['temp_f']}")
            st.markdown(f"Conditions: {forecast['current']['condition']['text']}")
            st.markdown(f"Wind: {forecast['current']['wind_dir']}-{forecast['current']['wind_mph']}")
            st.markdown(f"Cloud Cover: %{forecast['current']['cloud']}")
            #st.markdown(f"Chance of Rain Today: %{forecast['forecast']['forecastday'][0]['day']['daily_chance_of_rain']} ")
            st.markdown(f"Current Precipitation: {forecast['current']['precip_in']} Inches")
            #st.markdown(f"Day's Total Expected Rain: {forecast['forecast']['forecastday'][0]['day']['totalprecip_in']} Inches")
            events=[]
            for i,j in enumerate(forecast['forecast']['forecastday'][:3]):
                if forecast['forecast']['forecastday'][i]['astro']['moon_phase'] in ['New Moon','Full Moon']:
                    events.append(f"{forecast['forecast']['forecastday'][i]['astro']['moon_phase']} Coming up in {i} {'days' if i>1 else 'day'}. Check for King Tides for the following days.")
            if len(events)>0:
                st.warning(events[0])
        if select=="LABOR":
            labor_issue=False
            secondary=True
            
            if secondary:
                pma_rates=gcp_download(target_bucket,rf"pma_dues.json")
                pma_rates=json.loads(pma_rates)
                assessment_rates=gcp_download(target_bucket,rf"occ_codes2023.json")
                assessment_rates=json.loads(assessment_rates)
                lab_tab1,lab_tab2,lab_tab3,lab_tab4=st.tabs(["LABOR TEMPLATE", "JOBS","RATES","LOOKUP"])

                with lab_tab4:
                    itsreadytab4=True
                    if itsreadytab4:
                        
                        def dfs_sum(dictionary, key):
                            total_sum = 0
                        
                            for k, v in dictionary.items():
                                if k == key:
                                    total_sum += v
                                elif isinstance(v, dict):
                                    total_sum += dfs_sum(v, key)
                        
                            return total_sum
                        mt_jobs_=gcp_download(target_bucket,rf"mt_jobs.json")
                        mt_jobs=json.loads(mt_jobs_)
                        c1,c2,c3=st.columns([2,2,6])
                        with c1:
                            with st.container(border=True):
                                by_year=st.selectbox("SELECT YEAR",mt_jobs.keys())
                                by_job=st.selectbox("SELECT JOB",mt_jobs[by_year].keys())
                                by_date=st.selectbox("SELECT DATE",mt_jobs[by_year][by_job]["RECORDS"].keys())
                                by_shift=st.selectbox("SELECT SHIFT",mt_jobs[by_year][by_job]["RECORDS"][by_date].keys())
                                
                        with c2:
                            info=mt_jobs[by_year][by_job]["INFO"]
                            st.dataframe(info)
                        with c3:
                            with st.container(border=True):
                                
                                d1,d2,d3=st.columns([4,4,2])
                                with d1:
                                    by_choice=st.radio("SELECT INVOICE",["LABOR","EQUIPMENT","MAINTENANCE"])
                                with d2:
                                    by_location=st.radio("SELECT INVOICE",["DOCK","WAREHOUSE","LINES"])
                            with st.container(border=True):
                                e1,e2,e3=st.columns([4,4,2])
                                with e1:
                                    st.write(f"TOTAL {by_location}-{by_choice} COST for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],'TOTAL COST'),2)}")
                                    st.write(f"TOTAL {by_location}-{by_choice} MARKUP for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],'Mark UP'),2)}")
                                    if by_choice=="LABOR":
                                        st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],'INVOICE'),2)}")
                                    else:
                                        st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this SHIFT :  ${round(dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][by_date][by_shift][by_choice][by_location],f'{by_choice} INVOICE'),2)}")
                                with e2:
                                    st.write(f"TOTAL {by_location}-{by_choice} COST for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],'TOTAL COST') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
                                    st.write(f"TOTAL {by_location}-{by_choice} MARKUP for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],'Mark UP') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
                                    if by_choice=="LABOR":
                                        st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],'INVOICE') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
                                    else:
                                        st.write(f"TOTAL {by_location}-{by_choice} INVOICE for this JOB :  ${round(sum([sum([dfs_sum(mt_jobs[by_year][by_job]['RECORDS'][date][shift][by_choice][by_location],f'{by_choice} INVOICE') for shift in mt_jobs[by_year][by_job]['RECORDS'][date]]) for date in mt_jobs[by_year][by_job]['RECORDS']]),2) }")
                            #st.write(mt_jobs[by_year][by_job]["RECORDS"][by_date][by_shift][by_choice][by_location])
                        a=pd.DataFrame(mt_jobs[by_year][by_job]["RECORDS"][by_date][by_shift][by_choice][by_location]).T
                        if by_choice=="LABOR":
                            try:
                                a.loc["TOTAL FOR SHIFT"]=a[["Quantity","Hours","OT","Hour Cost","OT Cost","Total Wage","Benefits","PMA Assessments","TOTAL COST","SIU","Mark UP","INVOICE"]].sum()
                            except:
                                pass
                        else:
                            try:
                                a.loc["TOTAL FOR SHIFT"]=a[[ "Quantity","Hours","TOTAL COST","Mark UP",f"{by_choice} INVOICE"]].sum()
                            except:
                                pass
                        st.write(a)
                            
                           
                        
                with lab_tab3:
                    with st.container(border=True):
                        
                        tinker,tailor=st.columns([5,5])
                        with tinker:
                            select_year=st.selectbox("SELECT ILWU PERIOD",["JUL 2023","JUL 2022","JUL 2021"],key="ot1221")
                        with tailor:
                            select_pmayear=st.selectbox("SELECT PMA PERIOD",["JUL 2023","JUL 2022","JUL 2021"],key="ot12w21")
                    
                    year=select_year.split(' ')[1]
                    month=select_year.split(' ')[0]
                    pma_year=select_pmayear.split(' ')[1]
                    pma_rates_=pd.DataFrame(pma_rates).T
                    occ_codes=pd.DataFrame(assessment_rates).T
                    occ_codes=occ_codes.rename_axis('Occ_Code')
                    shortened_occ_codes=occ_codes.loc[["0036","0037","0055","0092","0101","0103","0115","0129","0213","0215"]]
                    shortened_occ_codes=shortened_occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
                    occ_codes=occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
                    rates=st.checkbox("SELECT TO DISPLAY RATE TABLE FOR THE YEAR",key="iueis")
                    if rates:
                        
                        lan1,lan2=st.columns([2,2])
                        with lan1:
                            st.write(occ_codes)
                        with lan2:
                            st.write(pma_rates[pma_year])
                with lab_tab2:
                    lab_col1,lab_col2,lab_col3=st.columns([2,2,2])
                    with lab_col1:
                        with st.container(border=True):
                            
                            job_vessel=st.text_input("VESSEL",disabled=False)
                            vessel_length=st.number_input("VESSEL LENGTH",step=1,disabled=False)
                            job_number=st.text_input("MT JOB NO",disabled=False)
                            shipper=st.text_input("SHIPPER",disabled=False)
                            cargo=st.text_input("CARGO",disabled=False)
                            agent=st.selectbox("AGENT",["TALON","ACGI","NORTON LILLY"],disabled=False)
                            stevedore=st.selectbox("STEVEDORE",["SSA","JONES"],disabled=False)
                            
                            alongside_date=st.date_input("ALONGSIDE DATE",disabled=False,key="arr")
                            alongside_date=datetime.datetime.strftime(alongside_date,"%Y-%m-%d")
                            
                            alongside_time=st.time_input("ALONGSIDE TIME",disabled=False,key="arrt")
                            alongside_time=alongside_time.strftime("%H:%M")
                            
                            departure_date=st.date_input("DEPARTURE DATE",disabled=False,key="dep")
                            departure_date=datetime.datetime.strftime(departure_date,"%Y-%m-%d")
                           
                            departure_time=st.time_input("DEPARTURE TIME",disabled=False,key="dept")
                            departure_time=departure_time.strftime("%H:%M")
                            
                            
                        if st.button("RECORD JOB"):
                            year="2023"
                            mt_jobs_=gcp_download(target_bucket,rf"mt_jobs.json")
                            mt_jobs=json.loads(mt_jobs_)
                            if year not in mt_jobs:
                                mt_jobs[year]={}
                            if job_number not in mt_jobs[year]:
                                mt_jobs[year][job_number]={"INFO":{},"RECORDS":{}}
                            mt_jobs[year][job_number]["INFO"]={"Vessel":job_vessel,"Vessel Length":vessel_length,"Cargo":cargo,
                                                "Shipper":shipper,"Agent":agent,"Stevedore":stevedore,"Alongside Date":alongside_date,
                                                "Alongside Time":alongside_time,"Departure Date":departure_date,"Departure Time":departure_time}
                            mt_jobs=json.dumps(mt_jobs)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"mt_jobs.json")
                            blob.upload_from_string(mt_jobs)
                            st.success(f"RECORDED JOB NO {job_number} ! ")
                with lab_tab1:
                    equipment_tariff={"CRANE":850,"FORKLIFT":65,"TRACTOR":65,"KOMATSU":160,"GENIE MANLIFT":50,"Z135 MANLIFT":95}
                    foreman=False
                    with st.container(border=True):
                        
                        tinker,tailor=st.columns([5,5])
                        with tinker:
                            select_year=st.selectbox("SELECT ILWU PERIOD",["JUL 2023","JUL 2022","JUL 2021"])
                        with tailor:
                            select_pmayear=st.selectbox("SELECT PMA PERIOD",["JUL 2023","JUL 2022","JUL 2021"])
                    
                    year=select_year.split(' ')[1]
                    month=select_year.split(' ')[0]
                    pma_year=select_pmayear.split(' ')[1]
                    pma_rates_=pd.DataFrame(pma_rates).T
                    occ_codes=pd.DataFrame(assessment_rates).T
                    occ_codes=occ_codes.rename_axis('Occ_Code')
                    shortened_occ_codes=occ_codes.loc[["0036","0037","0055","0092","0101","0103","0115","0129","0213","0215"]]
                    shortened_occ_codes=shortened_occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
                    occ_codes=occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
                    
                    
                    
                    if "scores" not in st.session_state:
                        st.session_state.scores = pd.DataFrame(
                            {"Code": [], "Shift":[],"Quantity": [], "Hours": [], "OT": [],"Hour Cost":[],"OT Cost":[],"Total Wage":[],"Benefits":[],"PMA Assessments":[],"SIU":[],"TOTAL COST":[],"Mark UP":[],"INVOICE":[]})
                    if "eq_scores" not in st.session_state:
                        st.session_state.eq_scores = pd.DataFrame(
                            {"Equipment": [], "Quantity":[],"Hours": [], "TOTAL COST":[],"Mark UP":[],"EQUIPMENT INVOICE":[]})
                    if "maint_scores" not in st.session_state:
                        st.session_state.maint_scores = pd.DataFrame(
                            {"Quantity":[2],"Hours": [8], "TOTAL COST":[1272],"Mark UP":[381.6],"MAINTENANCE INVOICE":[1653.6]})
                    if "maint" not in st.session_state:
                        st.session_state.maint=False
                    
                    ref={"DAY":["1ST","1OT"],"NIGHT":["2ST","2OT"],"WEEKEND":["2OT","2OT"],"HOOT":["3ST","3OT"]}
                    
                    def equip_scores():
                        equipment=st.session_state.equipment
                        equipment_qty=st.session_state.eqqty
                        equipment_hrs=st.session_state.eqhrs
                        equipment_cost=equipment_qty*equipment_hrs*equipment_tariff[equipment]
                        equipment_markup=equipment_cost*st.session_state.markup/100
                        eq_score=pd.DataFrame({ "Equipment": [equipment],
                                "Quantity": [equipment_qty],
                                "Hours": [equipment_hrs*equipment_qty],
                                "TOTAL COST":equipment_cost,
                                "Mark UP":[round(equipment_markup,2)],
                                "EQUIPMENT INVOICE":[round(equipment_cost+equipment_markup,2)]})
                        st.session_state.eq_scores = pd.concat([st.session_state.eq_scores, eq_score], ignore_index=True)
                   
                    def new_scores():
                        
                        if num_code=='0129':
                            foreman=True
                        else:
                            foreman=False
                        
                        pension=pma_rates[pma_year]["LS_401k"]
                        if foreman:
                            pension=pma_rates[pma_year]["Foreman_401k"]
                                     
                        
                        qty=st.session_state.qty
                        total_hours=st.session_state.hours+st.session_state.ot
                        hour_cost=st.session_state.hours*occ_codes.loc[st.session_state.code,ref[st.session_state.shift][0]]
                        ot_cost=st.session_state.ot*occ_codes.loc[st.session_state.code,ref[st.session_state.shift][1]]
                        wage_cost=hour_cost+ot_cost
                        benefits=wage_cost*0.062+wage_cost*0.0145+wage_cost*0.0021792                 #+wage_cost*st.session_state.siu/100
                        
                        assessments=total_hours*pma_rates[pma_year]["Cargo_Dues"]+total_hours*pma_rates[pma_year]["Electronic_Input"]+total_hours*pma_rates[pma_year]["Benefits"]+total_hours*pension
                        total_cost=wage_cost+benefits+assessments
                        siu_choice=wage_cost*st.session_state.siu/100
                        
                        with_siu=total_cost+siu_choice
                        markup=with_siu*st.session_state.markup/100   ##+benefits*st.session_state.markup/100+assessments*st.session_state.markup/100
                        if foreman:
                            markup=with_siu*st.session_state.f_markup/100  ###+benefits*st.session_state.f_markup/100+assessments*st.session_state.f_markup/100
                        invoice=total_cost+markup
                        
                        
                        new_score = pd.DataFrame(
                            {
                                "Code": [st.session_state.code],
                                "Shift": [st.session_state.shift],
                                "Quantity": [st.session_state.qty],
                                "Hours": [st.session_state.hours*qty],
                                "OT": [st.session_state.ot*qty],
                                "Hour Cost": [hour_cost*qty],
                                "OT Cost": [ot_cost*qty],
                                "Total Wage": [round(wage_cost*qty,2)],
                                "Benefits":[round(benefits*qty,2)],
                                "PMA Assessments":[round(assessments*qty,2)],
                                "TOTAL COST":[round(total_cost*qty,2)],
                                "SIU":[round(siu_choice*qty,2)],
                                "Mark UP":[round(markup*qty,2)],
                                "INVOICE":[round(invoice*qty,2)]
                                
                            }
                        )
                        st.session_state.scores = pd.concat([st.session_state.scores, new_score], ignore_index=True)
                        
                        
                 
                    
                        
                    # Form for adding a new score
                    
                    with st.form("new_score_form"):
                        st.write("##### LABOR")
                        form_col1,form_col2,form_col3=st.columns([3,3,4])
                        with form_col1:
                            
                            st.session_state.siu=st.number_input("ENTER SIU (UNEMPLOYMENT) PERCENTAGE",step=1,key="kdsha")
                            st.session_state.markup=st.number_input("ENTER MARKUP",step=1,key="wer")
                            st.session_state.f_markup=st.number_input("ENTER FOREMAN MARKUP",step=1,key="wfder")
                            
                        with form_col2:
                            st.session_state.shift=st.selectbox("SELECT SHIFT",["DAY","NIGHT","WEEKEND DAY","WEEKEND NIGHT","HOOT"])
                            st.session_state.shift_record=st.session_state.shift
                            st.session_state.shift="WEEKEND" if st.session_state.shift in ["WEEKEND DAY","WEEKEND NIGHT"] else st.session_state.shift
                        
                            # Dropdown for selecting Code
                            st.session_state.code = st.selectbox(
                                "Occupation Code", options=list(shortened_occ_codes.index)
                            )
                            # Number input for Quantity
                            st.session_state.qty = st.number_input(
                                "Quantity", step=1, value=0, min_value=0
                        )
                        with form_col3:
                            
                            # Number input for Hours
                            st.session_state.hours = st.number_input(
                                "Hours", step=0.5, value=0.0, min_value=0.0
                            )
                        
                            # Number input for OT
                            st.session_state.ot = st.number_input(
                                "OT", step=1, value=0, min_value=0
                            )
                            
                            # Form submit button
                            submitted = st.form_submit_button("Submit")
                        # If form is submitted, add the new score
                    
                    if submitted:
                        num_code=st.session_state.code[1].strip()
                        new_scores()
                        st.success("Rank added successfully!")
                    
                    
                    with st.form("equipment_form"):
                        st.write("##### EQUIPMENT")
                        eqform_col1,eqform_col2,eqform_col3=st.columns([3,3,4])
                        with eqform_col1: 
                            st.session_state.equipment = st.selectbox(
                                "Equipment", options=["CRANE","FORKLIFT","TRACTOR","KOMATSU","GENIE MANLIFT","Z135 MANLIFT"],key="sds11")
                        with eqform_col2:
                            # Number input for Equipment Quantity
                            st.session_state.eqqty = st.number_input(
                                "Equipment Quantity", key="sds",step=1, value=0, min_value=0)
                        with eqform_col3:
                            st.session_state.eqhrs = st.number_input(
                                "Equipment Hours",key="sdsss", step=1, value=0, min_value=0)
                            eq_submitted = st.form_submit_button("Submit Equipment")
                    if eq_submitted:
                        equip_scores()
                        st.success("Equipment added successfully!")
                    
                        
                    with st.container(border=True):
                        
                        sub_col1,sub_col2,sub_col3=st.columns([3,3,4])
                        with sub_col1:
                            pass
                        with sub_col2:
                            template_check=st.checkbox("LOAD FROM TEMPLATE")
                            if template_check:
                                with sub_col3:
                                    template_choice_valid=False
                                    template_choice=st.selectbox("Select Recorded Template",["Pick From List"]+[i for i in list_files_in_subfolder(target_bucket, rf"labor_templates/")],
                                                                  label_visibility="collapsed")
                                    if template_choice!="Pick From List":
                                        template_choice_valid=True 
                                    if template_choice_valid:
                                        loaded_template=gcp_csv_to_df(target_bucket,rf"labor_templates/{template_choice}")
                                
                                
                       
                   
                        display=pd.DataFrame(st.session_state.scores)
                        display.loc["TOTAL FOR SHIFT"]=display[["Quantity","Hours","OT","Hour Cost","OT Cost","Total Wage","Benefits","PMA Assessments","TOTAL COST","SIU","Mark UP","INVOICE"]].sum()
                        display=display[["Code","Shift","Quantity","Hours","OT","Hour Cost","OT Cost","Total Wage","Benefits","PMA Assessments","TOTAL COST","SIU","Mark UP","INVOICE"]]
                        display.rename(columns={"SIU":f"%{st.session_state.siu} SIU"},inplace=True)
                        eq_display=pd.DataFrame(st.session_state.eq_scores)
                        eq_display.loc["TOTAL FOR SHIFT"]=eq_display[[ "Quantity","Hours","TOTAL COST","Mark UP","EQUIPMENT INVOICE"]].sum()
                        if template_check and template_choice_valid:
                            st.dataframe(loaded_template)
                        else:
                            st.write("##### LABOR")
                            st.dataframe(display)
                            part1,part2=st.columns([5,5])
                            with part1:
                                st.write("##### EQUIPMENT")
                                st.dataframe(eq_display)
                            maint1,maint2,maint3=st.columns([2,2,6])
                            st.session_state.maint=False
                            with maint1:
                                st.write("##### MAINTENANCE (IF NIGHT/WEEKEND SHIFT)")
                            with maint2:
                                maint=st.checkbox("Check to add maint crew")
                            if maint:
                                st.session_state.maint=True
                                st.dataframe(st.session_state.maint_scores)
                            else:
                                st.session_state.maint=False
                            with part2:
                                subpart1,subpart2,subpart3=st.columns([3,3,4])
                                with subpart1:
                                    with st.container(border=True):
                                        st.write(f"###### COSTS")
                                        st.write(f"###### LABOR: {round(display.loc['TOTAL FOR SHIFT','TOTAL COST'],2)}")
                                        st.write(f"###### EQUIPMENT: {round(eq_display.loc['TOTAL FOR SHIFT','TOTAL COST'],2)}")
                                        if st.session_state.maint:
                                            st.write(f"###### MAINTENANCE: {round(st.session_state.maint_scores['TOTAL COST'].values[0],2)}")
                                            st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','TOTAL COST']+eq_display.loc['TOTAL FOR SHIFT','TOTAL COST']+st.session_state.maint_scores['TOTAL COST'].values[0],2)}")
                                        else:
                                            st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','TOTAL COST']+eq_display.loc['TOTAL FOR SHIFT','TOTAL COST'],2)}")

                                with subpart2:
                                    with st.container(border=True):
                                        st.write(f"###### MARKUPS")
                                        st.write(f"###### LABOR: {round(display.loc['TOTAL FOR SHIFT','Mark UP'],2)}")
                                        st.write(f"###### EQUIPMENT: {round(eq_display.loc['TOTAL FOR SHIFT','Mark UP'],2)}")
                                        if st.session_state.maint:
                                            st.write(f"###### MAINTENANCE: {round(st.session_state.maint_scores['Mark UP'].values[0],2)}")
                                            st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','Mark UP']+eq_display.loc['TOTAL FOR SHIFT','Mark UP']+st.session_state.maint_scores['Mark UP'].values[0],2)}")
                                        else:
                                            st.write(f"##### TOTAL: {round(display.loc['TOTAL FOR SHIFT','Mark UP']+eq_display.loc['TOTAL FOR SHIFT','Mark UP'],2)}")

                                with subpart3:
                                    with st.container(border=True):
                                        st.write(f"###### TOTALS")
                                        st.write(f"###### TOTAL LABOR: {round(display.loc['TOTAL FOR SHIFT','INVOICE'],2)}")
                                        st.write(f"###### TOTAL EQUIPMENT: {round(eq_display.loc['TOTAL FOR SHIFT','EQUIPMENT INVOICE'],2)}")
                                        if st.session_state.maint:
                                            st.write(f"###### TOTAL MAINTENANCE: {round(st.session_state.maint_scores['MAINTENANCE INVOICE'].values[0],2)}")
                                            st.write(f"##### TOTAL INVOICE: {round(display.loc['TOTAL FOR SHIFT','INVOICE']+eq_display.loc['TOTAL FOR SHIFT','EQUIPMENT INVOICE']+st.session_state.maint_scores['MAINTENANCE INVOICE'].values[0],2)}")
                                        else:
                                            st.write(f"##### TOTAL INVOICE: {round(display.loc['TOTAL FOR SHIFT','INVOICE']+eq_display.loc['TOTAL FOR SHIFT','EQUIPMENT INVOICE'],2)}")
                                                
                            
                                
                           
                    clear1,clear2,clear3=st.columns([2,2,4])
                    with clear1:
                        if st.button("CLEAR LABOR TABLE"):
                            try:
                                st.session_state.scores = pd.DataFrame(
                                {"Code": [], "Shift":[],"Quantity": [], "Hours": [], "OT": [],"Hour Cost":[],"OT Cost":[],
                                 "Total Wage":[],"Benefits":[],"PMA Assessments":[],"SIU":[],"TOTAL COST":[],"Mark UP":[],"INVOICE":[]})
                                st.rerun()
                            except:
                                pass
                    with clear2:
                        if st.button("CLEAR EQUIPMENT TABLE",key="54332dca"):
                            try:
                               st.session_state.eq_scores = pd.DataFrame({"Equipment": [], "Quantity":[],"Hours": [], "TOTAL COST":[],"Mark UP":[],"EQUIPMENT INVOICE":[]})
                               st.rerun()
                            except:
                                pass
                    
                    csv=convert_df(display)
                    file_name=f'Gang_Cost_Report-{datetime.datetime.strftime(datetime.datetime.now(),"%m-%d,%Y")}.csv'
                    down_col1,down_col2,down_col3,down_col4=st.columns([2,2,2,4])
                    with down_col1:
                        #st.write(" ")
                        filename=st.text_input("Name the Template",key="7dr3")
                        template=st.button("SAVE AS TEMPLATE",key="srfqw")
                        if template:
                            temp=display.to_csv(index=False)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            
                            # Upload CSV string to GCS
                            blob = bucket.blob(rf"labor_templates/{filename}.csv")
                            blob.upload_from_string(temp, content_type="text/csv")
                    with down_col2:
                        mt_jobs_=gcp_download(target_bucket,rf"mt_jobs.json")
                        mt_jobs=json.loads(mt_jobs_)
                        #st.write(st.session_state.scores.T.to_dict())
                        job_no=st.selectbox("SELECT JOB NO",[i for i in mt_jobs["2023"]])
                        year="2023"
                        work_type=st.selectbox("SELECT JOB NO",["DOCK","WAREHOUSE","LINES"])
                        work_date=st.date_input("Work Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="work_date")
                        record=st.button("RECORD TO JOB",key="srfqwdsd")
                        if record:
                            
                            if year not in mt_jobs:
                                mt_jobs[year]={}
                            if job_no not in mt_jobs[year]:
                                mt_jobs[year][job_no]={}
                            if "RECORDS" not in mt_jobs[year][job_no]:
                                mt_jobs[year][job_no]["RECORDS"]={}
                            if str(work_date) not in mt_jobs[year][job_no]["RECORDS"]:
                                mt_jobs[year][job_no]["RECORDS"][str(work_date)]={}
                            if st.session_state.shift_record not in mt_jobs[year][job_no]["RECORDS"][str(work_date)]:
                                mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]={}
                            if "LABOR" not in mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]:
                                mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]["LABOR"]={"DOCK":{},"LINES":{},"WAREHOUSE":{}}
                            if 'EQUIPMENT' not in mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]:
                                mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]["EQUIPMENT"]={"DOCK":{},"LINES":{},"WAREHOUSE":{}}
                            if 'MAINTENANCE' not in mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]:
                                mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]["MAINTENANCE"]={"DOCK":{},"LINES":{},"WAREHOUSE":{}}
                            mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]['LABOR'][work_type]=st.session_state.scores.T.to_dict()
                            mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]['EQUIPMENT'][work_type]=st.session_state.eq_scores.T.to_dict()
                            if st.session_state.maint:
                                mt_jobs[year][job_no]["RECORDS"][str(work_date)][st.session_state.shift_record]['MAINTENANCE'][work_type]=st.session_state.maint_scores.T.to_dict()
                            mt_jobs_=json.dumps(mt_jobs)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"mt_jobs.json")
                            blob.upload_from_string(mt_jobs_)
                            st.success(f"RECORDED JOB NO {job_no} ! ")
                        
                       
                        
                                               
                    
                    index=st.number_input("Enter Index To Delete",step=1,key="1224aa")
                    if st.button("DELETE BY INDEX"):
                        try:
                            st.session_state.scores=st.session_state.scores.drop(index)
                            st.session_state.scores.reset_index(drop=True,inplace=True)
                        except:
                            pass
        if select=='WEATHER':
            st.caption("Live Data for Olympia From Weather.gov API")
            gov_forecast=get_weather()
            data=dict()
            for day in forecast['forecast']['forecastday']:
                data[day['date']]={}
                data[day['date']]['DAY']={}
                data[day['date']]['ASTRO']={}
                for item in day['day']:
                    data[day['date']]['DAY'][item]=day['day'][item]
                for astro in day['astro']:
                    data[day['date']]['ASTRO'][astro]=day['astro'][astro]
                for dic in day['hour']:
                    data[day['date']][dic['time']]={}
                    for measure in dic:
                        if measure=='condition':
                            data[day['date']][dic['time']][measure]=dic[measure]['text']
                            data[day['date']][dic['time']]['condition_png']=dic[measure]['icon']
                        else:
                            data[day['date']][dic['time']][measure]=dic[measure]
            index=[]
            temperatures=[]
            condition=[]
            wind=[]
            wind_dir=[]
            pressure=[]
            cloud=[]
            rain=[]
            dew_point=[]
            will_rain=[]
            chance_rain=[]
            for day in data:
                for hour in data[day]:
                    if hour not in ['DAY','ASTRO']:
                        #print(hour)
                        index.append(hour)
                        temperatures.append(data[day][hour]['temp_f'])
                        condition.append(data[day][hour]['condition'])
                        wind.append(data[day][hour]['wind_mph'])
                        wind_dir.append(data[day][hour]['wind_dir'])
                        pressure.append(data[day][hour]['pressure_mb'])
                        cloud.append(data[day][hour]['cloud'])
                        rain.append(data[day][hour]['precip_in'])
                        will_rain.append(True if data[day][hour]['will_it_rain']==1 else False)
                        chance_rain.append(data[day][hour]['chance_of_rain'])
                        dew_point.append(data[day][hour]['dewpoint_f'])    
            weather_tab1,weather_tab2=st.tabs(["TABULAR","GRAPH"])
            with weather_tab1:
                temperatures_=["{:.1f}".format(number)for number in temperatures]
                rain_=["{:.2f}".format(number)for number in rain]
                weather_df=pd.DataFrame({'Temperature':temperatures_,"Sky":condition,'Rain Amount':rain_,
                'Chance of Rain':chance_rain,"Wind":[f"{i}-{j}" for i,j in zip(wind_dir,wind)]
                },index=index)
                
                st.table(weather_df)
            with weather_tab2:
                st.markdown('Powered by <a href="https://www.weatherapi.com/" title="Free Weather API">WeatherAPI.com</a>',unsafe_allow_html=True)
                
                
                fig = go.Figure()

                # Add traces for each weather parameter
                fig.add_trace(go.Bar(x=index, y=temperatures, name='Temperature'))
                fig.add_trace(go.Scatter(x=index, y=wind, mode='markers', name='Wind Speed'))
                fig.add_trace(go.Scatter(x=index, y=pressure, mode='lines', name='Pressure'))
                fig.add_trace(go.Bar(x=index, y=rain, name='Rain Amount'))
                fig.add_trace(go.Bar(x=index, y=chance_rain,name='Chance of Rain'))
                fig.add_trace(go.Scatter(x=index, y=cloud, mode='lines', name='Cloud Cover'))
                # Add traces for other weather parameters...
                rain_threshold = 0.02
                fig.add_shape(
                    dict(
                        type='line',
                        x0=min(index),
                        x1=max(index),
                        y0=rain_threshold,
                        y1=rain_threshold,
                        line=dict(color='red', width=2),
                        visible='Rain Amount' in [trace.name for trace in fig.data],  # Show only when Rain graph is selected
                    )
                )
                fig.add_annotation(
                        go.layout.Annotation(
                            x=0.5,  # Set x to the middle of the x-axis (adjust as needed)
                            y=rain_threshold+0.005,
                            xref='paper',
                            yref='y',
                            text=f'{rain_threshold} inches/h',
                            showarrow=False,
                            font=dict(color='red', size=14),
                        )
                    )
                # Create a dropdown menu for selecting weather parameters
                button_list = []
                for parameter in ['Temperature', 'Wind Speed','Rain Amount','Chance of Rain','Cloud Cover', 'Pressure']:
                    button_list.append(dict(label=parameter, method='update', args=[{'visible': [trace.name == parameter for trace in fig.data]}]))
                
                # Add a "Show All" button
                button_list.append(dict(label='Show All', method='update', args=[{'visible': [True] * len(fig.data)}]))
                
                # Update layout to include the dropdown menu
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type='dropdown',
                            showactive=True,
                            buttons=button_list,
                            x=0.5,
                            xanchor='left',
                            y=1.15,
                            yanchor='top',
                        ),
                    ]
                )
                
                # Update layout for better readability
                fig.update_layout(title='PORT OF OLYMPIA Weather Forecast',
                                  xaxis_title='Hour',
                                  yaxis_title='Value',
                                  template='plotly_white',
                                  width=900, 
                                  height=600)
                st.plotly_chart(fig)
               
        
        if select=="TIDES":
            dada=True
            if dada:
                
                st.header("OLYMPIA TIDES")
                st.subheader("LIVE FROM NOAA API  - STATION 9446969")
                st.write("Don't go further ahead than 1 year")
                
                a1,a2,a3=st.columns([2,2,6])
                with a1:
                    begin_date=st.date_input("FROM")
                with a2:
                    end_date=st.date_input("TO",key="erresa")
               
                
           
                begin_date=dt.datetime.strftime(begin_date,"%Y%m%d")
                end_date=dt.datetime.strftime(end_date,"%Y%m%d")
                
                url = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={begin_date}&end_date={end_date}&station=9446969&product=predictions&datum=MLLW&time_zone=lst_ldt&interval=hilo&units=english&application=DataAPI_Sample&format=xml'
                headers = { 
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
                                'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
                                'Accept-Language' : 'en-US,en;q=0.5', 
                                'Accept-Encoding' : 'gzip', 
                                'DNT' : '1', # Do Not Track Request Header 
                                'Connection' : 'close' }
                # Send a GET request to the URL and retrieve the response
                response = requests.get(url,headers=headers)
                soup = BeautifulSoup(response.content, 'html5lib')
            
                # Find all the <a> tags in the HTML content
                tides=defaultdict()
            
                link_tags = soup.find_all('pr')
                #print(link_tags)
                tides["Time"]=[]
                tides["Tide"]=[]
                tides["Height"]=[]
                for i in link_tags:
                    tides["Time"].append(i.get("t"))
                    tides["Tide"].append(i.get("type"))
                    tides["Height"].append(i.get("v"))
                tides=pd.DataFrame(tides)
                tides["Tide"]=["Low" if i=="L" else "High" for i in tides["Tide"]]
                #print(tides["Time"].dtype)
                tides["Time"]=[dt.datetime.strftime(dt.datetime.strptime(i,"%Y-%m-%d %H:%M"),"%B-%d,%a--%H:%M") for i in tides["Time"]]
                tides.set_index("Time",drop=True,inplace=True)
                #st.table(tides)
                html_data=tides.to_html()
                st.markdown(html_data, unsafe_allow_html=True)    
                st.download_button(
                                    label="Download as HTML",
                                    data=html_data,
                                    file_name="tides_data.html",
                                    mime="text/html"
                                )

            
        if select=="FINANCE":
            hadi=False
            fin_password=st.sidebar.text_input("Enter Password",key="sas")
            if fin_password=="marineterm98501!":
                hadi=True
            if hadi:
                ttab1,ttab2=st.tabs(["MT LEDGERS","UPLOAD CSV LEDGER UPDATES"])
                
                with ttab2:
                    
                    if st.checkbox("UPLOAD LEDGER CSV",key="fsdsw"):
                        led_col1,led_col2,led_col3,led_col4=st.columns([3,2,2,2])
                        with led_col1:
                            
                            m30 = st.file_uploader("**Upload Ledger 030 csv**", type=["csv"],key="34wss")
                            m32= st.file_uploader("**Upload Ledger 032 csv**", type=["csv"],key="34ws2ss")
                            m36 = st.file_uploader("**Upload Ledger 036 csv**", type=["csv"],key="34wsas")
                            m40 = st.file_uploader("**Upload Ledger 040 csv**", type=["csv"],key="34wsss")
                            ledgers=[m30,m32,m36,m40]
                            file_names=["030-2023","032-2023","036-2023","040-2023"]
                            if m30 and m32 and m36 and m40:
                                                        
                                for k,file in zip(ledgers,file_names):
                                    df=pd.read_csv(k,header=None) 
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
                            
                                    with led_col2:
                                        
                                        st.markdown(f"**Processing ledger {file}...**")
                                        st.write("Total Credit :${:,}".format(round(df.Credit.sum(),2)))
                                        st.write("Total Debit  :${:,}".format(round(df.Debit.sum(),2)))
                                        st.write("Net          :${:,}".format(round(df.Credit.sum()-df.Debit.sum(),2)))
                                    feather_data = BytesIO()
                                    df.reset_index().to_feather(feather_data)
                                    # Create a temporary local file to store Feather data
                                    temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
                                    df.reset_index().to_feather(temp_file_path)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"FIN/NEW/{file}.ftr")
                                    blob.upload_from_filename(temp_file_path)
                                
                                    set=pd.read_feather(feather_data).set_index("index",drop=True).reset_index(drop=True)
                                    if k==m30:
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
                                        temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
                                        depreciation.reset_index().to_feather(temp_file_path)
                                        storage_client = storage.Client()
                                        bucket = storage_client.bucket(target_bucket)
                                        blob = bucket.blob(rf"FIN/main2023-30.ftr")
                                        blob.upload_from_filename(temp_file_path)
                                    if k==m32:
                                        
                                        set["Net"]=set["Credit"]-set["Debit"]
                                        
                                        first=set.copy()
                                        with led_col4:
                                            st.markdown(f"**Re-processing Ledger 032**")
                                            st.write("Total Credit :${:,}".format(round(first.Credit.sum(),2)))
                                            st.write("Total Debit  :${:,}".format(round(first.Debit.sum(),2)))
                                            st.write("Net          :${:,}".format(round(first.Credit.sum()-first.Debit.sum(),2)))
                                   
                                    if k==m36:
                                        
                                        set["Net"]=set["Credit"]-set["Debit"]
                                        third=set.copy()
                                        with led_col4:
                                            st.markdown(f"**Re-processing Ledger 036**")
                                            st.write("Total Credit :${:,}".format(round(third.Credit.sum(),2)))
                                            st.write("Total Debit  :${:,}".format(round(third.Debit.sum(),2)))
                                            st.write("Net          :${:,}".format(round(third.Credit.sum()-third.Debit.sum(),2)))
                                    if k==m40:
                                        
                                        set["Net"]=set["Credit"]-set["Debit"]
                                        fourth=set.copy()
                                        with led_col4:
                                            st.markdown(f"**Re-processing Ledger 040**")
                                            st.write("Total Credit :${:,}".format(round(fourth.Credit.sum(),2)))
                                            st.write("Total Debit  :${:,}".format(round(fourth.Debit.sum(),2)))
                                            st.write("Net          :${:,}".format(round(fourth.Credit.sum()-fourth.Debit.sum(),2)))
                                store=pd.concat([first,main30,overhead,third,fourth])
                                    
                                    
                                temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
                                store.reset_index().to_feather(temp_file_path)
                                storage_client = storage.Client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"FIN/main2023.ftr")
                                blob.upload_from_filename(temp_file_path)
                                with led_col2:
                                    st.success("**SUCCESS. 2023 Ledger has been updated!", icon="✅")     
                
                    
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
                    tt=f"MARINE TERMINAL FINANCIALS"
                    original_title = f'<p style="font-family:Arial;font-weight: bold; color:Black; font-size: 20px;">{tt}</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                
                    fintab1,fintab2,fintab3,fintab4,fintab5=st.tabs(["LEDGERS-MONTHLY","YEAR NET","BUDGET",
                                                                     "SEARCH BY CUSTOMER/VENDOR STRING","DEPRECIATION"])
                
                    #### LOAD BUDGET CODES, has information on 2022 and 2023 budget by account number
                
                    budget_codes=gcp_download_x(target_bucket,rf"FIN/budget_codes.feather")
                    budget_codes=pd.read_feather(io.BytesIO(budget_codes))
                    budget_codes.set_index("index",drop=True,inplace=True)
                    budget=gcp_download_x(target_bucket,rf"FIN/budget.pkl")
                    budget = pickle.load(io.BytesIO(budget))
                    keys={}
                    revenues_codes=list(get_all_keys(budget["Revenues"],keys).keys())
                    keys={}
                    operations_codes=list(get_all_keys(budget["Operating Expenses"],keys).keys())
                    keys={}
                    maintenance_codes=list(get_all_keys(budget["Maintenance Expenses"],keys).keys())
                    keys={}
                    depreciation_codes=list(get_all_keys(budget["Depreciation"],keys).keys())
                    keys={}
                    overhead_codes=list(get_all_keys(budget["G & A Overhead"],keys).keys())
                
                    expenses=operations_codes+maintenance_codes+overhead_codes
                    expenses_dep=expenses+depreciation_codes
                
                    accounts_classes=gcp_download_x(target_bucket,rf"FIN/accounts_classes.pkl")
                    accounts_classes = pickle.load(io.BytesIO(accounts_classes))
                
                     
                
                    with fintab1: 
                        
                        year=st.selectbox("Select Year",["2023","2022","2021","2020","2019","2018", "2017","2016"])
                        
                        ### LETS PUT YEAR in st.session state to use later.
                        if year not in st.session_state:
                            st.session_state.year=year
                            
                        ### LOAD LEDGERS by year
                        ledgers=gcp_download_x(target_bucket,rf"FIN/main{year}.ftr")
                        ledgers=pd.read_feather(io.BytesIO(ledgers))
                        ledgers["Account"]=ledgers["Account"].astype("str")
                        ledgers.set_index("index",drop=True,inplace=True)
                        
                        ### MAKE A COPY OF LEDGERS to change Account column to our structure : 6311000-32
                        ledgers_b=ledgers.copy()
                        ledgers_b.Account=[str(i)+"-"+str(j) for i,j in zip(ledgers_b.Account,ledgers_b.Sub_Cat)]
                        
                        
                        ##### LOAD THE MARINE ACCOUNT STRUCTURE dictionary from pickle file - AFSIN budget structure
                       
                            
                        st.session_state.category=None
                        st.session_state.sub_category=None
                        st.session_state.sub_item=None
                        ###START LEVELS
                        ###CHOOSE AND RECORD CAT in st session state
                        category=st.selectbox("Select Ledger",["Revenues","Operating Expenses","Maintenance Expenses","G & A Overhead","Depreciation"])
                                   
                        if category not in st.session_state:
                            st.session_state.category=category
                
                        # Lets check if Deep categories or shallow (Revenue versus Depreciation)
                        deep=True if category in ["Revenues","Operating Expenses","Maintenance Expenses"] else False
                        structure=budget.copy()
                        # LOAD a list of sub_cats to display. If not deep the keys becomes the names of subcats due to shallow depth.(last nodes)
                        liste=[f"ALL {category.upper()}"]+list(structure[category].keys()) if deep else [f"ALL {category.upper()}"]+list(structure[category].values())
                            
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
                                all_values = []
                                ###  WE CHOOSE DICTIONARY Items so we can go deeper
                                for key, inner_dict in structure[category].items():
                                    for inner_key, value in inner_dict.items():
                                        all_values.append(inner_key)
                                final=ledgers_b[ledgers_b["Account"].isin([i for i in all_values])]
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
                                sub_item=st.selectbox("Select Item",[f"ALL {sub_category.upper()}"]+list(structure[category][sub_category].values()))
                                monthly_label=sub_item
                                
                                display_allsubitem=True if sub_item==f"ALL {sub_category.upper()}" else False
                                                    
                                if display_allsubitem:
                                    final=ledgers_b[ledgers_b["Account"].isin([i for i in structure[category][sub_category].keys()])]
                                else:
                                    level=3
                                    account=[i for i in list(structure[category][sub_category]) if structure[category][sub_category][i]==sub_item][0]
                                    final=ledgers_b[ledgers_b["Account"]==account]
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
                                            monthly=ledgers_b[ledgers_b["Account"]==key]
                                            accounts=[key]         
                                    
                            elif level==2:
                                if deep:
                                    #st.write(structure[a][b])
                                    monthly=ledgers_b[ledgers_b["Account"].isin(structure[a][b].keys())]
                                    accounts=structure[a][b].keys()
                                else:
                                    for key, value in structure[a].items():
                                        if value == b:
                                            #st.write(key,":",value)
                                            accounts=[key]
                            elif level==1:
                                #st.write(structure[a])
                                keys={}
                                accounts=get_all_keys(structure[a],keys).keys()
                                
                            
                                           
                            #st.write(accounts)
                            st.subheader(f'Monthly {monthly_label} {st.session_state.year}')
                            
                           
                            #accounts=structure[a][b].keys()
                            
                    #                 for i in accounts:
                    #                     st.write(budget_codes.loc[budget_codes["Account"]==i]["2023 Adopted"])
                            #st.write(accounts)
                             
                            #st.write(ledgers_b)
                            monthly=ledgers_b[ledgers_b["Account"].isin(accounts)]
                            #st.write(monthly)
                            monthly.set_index("Date",drop=True, inplace=True)
                            #st.write(monthly)
                            monthly_=monthly.resample("M")["Debit","Credit","Net"].sum()
                            monthly_.index=[i.month_name() for i in monthly_.index]
                            avg=round(monthly_.Net.sum()/monthly_.shape[0],1)
                            total=round(monthly_.Net.sum(),1)
                            if year=="2023":
                               
                                annual_budget=budget_codes[budget_codes["Account"].isin(accounts)]["2023 Adopted"].values.sum()
                                #st.write(annual_budget)
                            elif year=="2022":
                                annual_budget=budget_codes[budget_codes["Account"].isin(accounts)]["2022 Adopted"].values.sum()
                                annual_budget1=budget_codes[budget_codes["Account"].isin(accounts)]["2023 Adopted"].values.sum()
                            
                            else:
                                
                                annual_budget=budget_codes[budget_codes["Account"].isin(accounts)]["2022 Adopted"].values.sum()
                                
                               
                            budgeted_monthly=annual_budget/12
                            monthly_=monthly_.applymap(dollar_format)
                            #st.write(annual_budget)
                            #st.write(accounts)
                            col1, col2,col3= st.columns([2,2,5])
                            
                            with col1:
                                st.write(monthly_)
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
                                            'axis': {'range': [None, 1.5*abs(budgeted)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                            'bar': {'color': "darkblue"},
                                            'bgcolor': "white",
                                            'borderwidth': 2,
                                            'bordercolor': "gray",
                                            'steps': [
                                                {'range': [0, abs(budgeted)], 'color': 'cyan'},
                                                {'range': [abs(budgeted),
                                                          abs(1.5*budgeted)], 'color': 'royalblue'}],
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
                                yillar=["2023","2022","2021","2020","2019","2018", "2017","2016"]
                                results=[]
                                for k in yillar:
                                    temp=gcp_download_x(target_bucket,rf"FIN/main{k}.ftr")
                                    temp=pd.read_feather(io.BytesIO(temp))
                                    
                                
                                    temp.set_index("index",drop=True,inplace=True)
                
                                    ### MAKE A COPY OF LEDGERS to change Account column to our structure : 6311000-32
                                    temp1=temp.copy()
                                    temp1.Account=[str(i)+"-"+str(j) for i,j in zip(temp1.Account,temp1.Sub_Cat)]
                                    result=temp1[temp1["Account"].isin(accounts)]["Net"].sum()
                                    results.append(result)
                                fig2 = go.Figure(data=[go.Bar(x=yillar, y=results)])
                                st.plotly_chart(fig2)
                            #st.write(monthly)
                            
                    with fintab2:
                        year=st.selectbox("Select Year",["2023","2022","2021","2020","2019","2018", "2017","2016"],key="second")
                        
                        ### LETS PUT YEAR in st.session state to use later.
                        
                            
                        ### LOAD LEDGERS by year
                        ledger_b=gcp_download_x(target_bucket,rf"FIN/main{year}.ftr")
                        ledger_b=pd.read_feather(io.BytesIO(ledger_b))
                        ledger_b["Account"]=ledger_b["Account"].astype("str")
                        ledger_b.set_index("index",drop=True,inplace=True)
                        
                        ### MAKE A COPY OF LEDGERS to change Account column to our structure : 6311000-32
                        
                        ledger_b.Account=[str(i)+"-"+str(j) for i,j in zip(ledger_b.Account,ledger_b.Sub_Cat)]
                        
                        
                        ins=ledger_b[ledger_b["Account"].isin(revenues_codes)].Net.sum()
                        outs=ledger_b[ledger_b["Account"].isin(expenses)].Net.sum()
                        outs_dep=ledger_b[ledger_b["Account"].isin(expenses_dep)].Net.sum()
                        dep=ledger_b[ledger_b["Account"].isin(depreciation_codes)].Net.sum()
                        
                        a1, a2,= st.columns([2,5])
                        with a1:
                            
                            st.write(f"**REVENUES     :  {'${:,.1f}**'.format(ins)}")
                            st.write(f"**EXPENSES      :  {'${:,.1f}**'.format(outs)}")
                            if ins+outs<0:
                                tt=f"NET BEFORE DEPRECIATION:  {'${:,.1f}'.format(ins+outs)}"
                                original_title = f'<p style="font-family:Arial;font-weight: bold; color:Red; font-size: 15px;">{tt}</p>'
                                st.markdown(original_title, unsafe_allow_html=True)
                            else:
                                st.write(f"**NET BEFORE DEPRECIATION:  {'${:,.1f}**'.format(ins+outs)}")
                            st.write(f"**DEPRECIATION:  {'${:,.1f}**'.format(dep)}")
                            if ins+outs+dep<0:
                                tt=f"NET AFTER DEPRECIATION:  {'${:,.1f}'.format(ins+outs+dep)}"
                                original_title = f'<p style="font-family:Arial;font-weight: bold; color:Red; font-size: 15px;">{tt}</p>'
                                st.markdown(original_title, unsafe_allow_html=True)
                            else:
                                st.write(f"**NET AFTER DEPRECIATION:  {'${:,.1f}**'.format(ins+outs+dep)}")
                        
                        with a2:
                            
                        # Define the list of values and labels for the waterfall chart
                            values = [ins,outs,ins+outs, dep,ins+outs+dep]
                            labels = ['Revenues', 'Expenses', 'Net Before Depreciation', 'Depreciation', 'Net After Depreciation']
                            
                            # Define the colors for each bar in the waterfall chart
                            end=ins+outs+dep
                            #st.write(end)
                            if end<0:
                                #colors = ['#4CAF50', '#FFC107', '#2196F3', '#F44336', '#F44336']
                                totals={"marker":{"color":"maroon", "line":{"color":"rgb(63, 63, 63)", "width":1}}}
                            else:
                                #colors=['#4CAF50', '#FFC107', '#2196F3', '#F44336', '#EF9A9A']
                                totals={"marker":{"color":f"#2196F3", "line":{"color":"rgb(63, 63, 63)", "width":1}}}
                           
                
                            # Define the text for each bar in the waterfall chart
                            text = ['<b>${:,.1f}<b>'.format(value) for value in values]
                            text_font = {'size': 14,'color':['black','red','black','red','black']}
                            this_year=f"So Far This Year" if year=="2023" else ""
                            # Create the trace for the waterfall chart
                            trace = go.Waterfall(
                                name = "Net Result",
                                orientation = "v",
                                measure = ['absolute', 'relative', 'total', 'relative', 'total'],
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
                                        title = f'MARINE TERMINAL FINANCIALS-WATERFALL-{year}<br>{this_year}',
                                        titlefont=dict(size=20, family='Arial', color='black',),
                                        xaxis = { 'titlefont': dict(size=18, family='Arial', color='black'),
                                                 'tickfont': dict(size=16, family='Arial', color='black',)},
                                        yaxis = {'title': 'Amount ($)', 'titlefont': dict(size=18, family='Arial', color='black'),
                                                 'tickfont': dict(size=16, family='Arial', color='black',)},
                                        shapes = [
                                            {'type': 'line', 'x0': -0.5, 'y0': -0, 'x1': len(labels)-0.5, 'y1': 0, 'line': {'color': 'red', 'width': 2}}
                                        ],height=600,
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
                        labels=['Vessel Operations','Labor','Tenants','Stormwater Revenue','Other Revenue',
                               'TOTAL REVENUE',
                               'OPERATING EXPENSES',
                               'Operating Overhead','Terminal Operating Expense','Outside Professional Services','Labor','Vessel Operational Expenses',
                                     'Stormwater Operating Expenses','Utilities',
                                "MAINTENANCE EXPENSES",
                                'Maintenance Overhead','Property Maintenance','Equipment Maintenance','Stormwater Maintenance Expenses',
                                      'Other Maintenance Expenses',
                                "DEPRECIATION",
                                'Depreciation Terminal','Depreciation Grants','Depreciation Stormwater',
                                 "G&A OVERHEAD",
                                'Executive G&A Overhead','Marketing G&A Overhead',
                                'Finance G&A Overhead','Engineering G&A Overhead',
                                     'I/S G&A Overhead','Administrative G&A Overhead',
                                "Excess Revenue",
                                "Loss After Depreciation"]
                
                        keys={}
                        
                        revs=[abs(ledger_b[ledger_b["Account"].isin(get_all_keys(budget["Revenues"][i],keys).keys())]["Net"].sum()) for i in labels[:5]]
                
                        ops=[abs(ledger_b[ledger_b["Account"].isin(get_all_keys(budget["Operating Expenses"][i],keys).keys())]["Net"].sum()) for i in labels[7:14]]
                
                        maint=[abs(ledger_b[ledger_b["Account"].isin(get_all_keys(budget["Maintenance Expenses"][i],keys).keys())]["Net"].sum())for i in labels[15:20]]
                
                        dep=[abs(ledger_b[ledger_b["Account"]==key_from_value(budget["Depreciation"],i)]["Net"].sum()) for i in labels[21:24]]
                
                        overhead=[abs(ledger_b[ledger_b["Account"]==key_from_value(budget["G & A Overhead"],i)]["Net"].sum()) for i in labels[25:31]]
                        overall=sum(revs)-sum(ops)-sum(maint)-sum(dep)-sum(overhead)
                        valerians=[]
                        for i in revs:
                            valerians.append(i)
                        valerians.append(sum(revs))
                        valerians.append(sum(ops))
                        for i in ops:
                            valerians.append(i)
                        valerians.append(sum(maint))
                        for i in maint:
                            valerians.append(i)
                        valerians.append(sum(dep))
                        for i in dep:
                            valerians.append(i)
                
                        valerians.append(sum(overhead))
                        for i in overhead:
                            valerians.append(i)
                
                        valerians.append(overall)
                
                        valerians.append(overall)
                
                        valerians=['<b>${:,.1f}<b>'.format(round(i,1)) for i in valerians]
                        labels=[f'<b>{i}<b>' for i in labels]
                        if overall>0:
                            
                          
                            source=[0,1,2,3,4]+[5,5, 5 ,5,5]+ [6,6,6, 6, 6, 6, 6]+[14,14,14,14,14]+[20,20,20]+[24,24,24,24,24,24]
                            target=[5,5,5,5,5]+[6,14,20,24,31]+[7,8,9,10,11,12,13]+[15,16,17,18,19]+[21,22,23]+[25,26,27,28,29,30,]
                            values=revs+[sum(ops)]+[sum(maint)]+[sum(dep)]+[sum(overhead)]+[overall]+ops+maint+dep+overhead
                            linkcolor=['#66CD00']*5+\
                                  ['#FFB90F','#BF3EFF', '#A6E3D7', '#EC7063','#FFC0CB',]+\
                                    ['#FFB90F']*7+\
                                        ['#BF3EFF']*5+\
                                    ['#A6E3D7']*3+\
                                    ['#EC7063']*6
                        else:
                            
                           
                            source=[0,1,2,3,4,32]+[5,5, 5 ,5,]+ [6,6,6, 6, 6, 6, 6]+[14,14,14,14,14]+[20,20,20]+[24,24,24,24,24,24]
                            target=[5,5,5,5,5,5]+[6,14,20,24,]+[7,8,9,10,11,12,13]+[15,16,17,18,19]+[21,22,23]+[25,26,27,28,29,30,]
                            values=revs+[abs(overall)]+[sum(ops)]+[sum(maint)]+[sum(dep)]+[sum(overhead)]+ops+maint+dep+overhead
                            linkcolor=['#66CD00']*5+['#FFC0CB']+\
                                  ['#FFB90F','#BF3EFF', '#A6E3D7', '#EC7063',]+\
                                    ['#FFB90F']*7+\
                                        ['#BF3EFF']*5+\
                                    ['#A6E3D7']*3+\
                                    ['#EC7063']*6
                        #'#104E8B'
                        title_=f'{year}-YTD' if year=="2023" else year  
                        fig = go.Figure(data=[go.Sankey(
                            node = dict(
                            thickness = 10,
                            #label = [f'<b>{i}<b>'+"-"+str(valerians[labels.index(i)]) for i in labels],
                           # label = [str(valerians[labels.index(i)]) for i in labels],
                            #label = [i+" - "+str(valerians[labels.index(i)]) for i in labels],
                            label=[f'<b>{i} - {str(valerians[labels.index(i)])}</b>' for i in labels],
                            color = [
                                    '#808B96', 
                                    '#EC7063', '#F7DC6F', '#48C9B0', '#AF7AC5',
                                    '#EC7063', '#EC7063',
                                    '#F7DC6F', '#F7DC6F',
                                    '#48C9B0', '#48C9B0', '#48C9B0', '#48C9B0', '#48C9B0', '#48C9B0',
                                    '#AF7AC5', '#AF7AC5', '#AF7AC5'] #"cyan"
                                        ),
                            link = dict(
                
                            # indices correspond to labels
                            source = source,
                            target = target,
                            value = values,
                            color=linkcolor
                            )
                        )])
                        # fig.add_annotation(
                        #                                         x=0.5,
                        #                                         y=-0.05,
                        #                                         text="From Budgeted Monthly",
                        #                                         showarrow=False,
                        #                                         font=dict(
                        #                                             size=16,
                        #                                             color="darkblue",
                        #                                             family="Arial"
                        #                                         )
                        #                                     )
                        fig.update_layout(width=1200, height=800,
                            title=title_,hovermode = 'x',
                                          
                            font=dict(size = 12, color = 'black'),paper_bgcolor='#FCE6C9',margin=dict(
                                l=50,  # Set the left margin to 50 pixels
                                r=350,  # Set the right margin to 150 pixels
                                t=50,  # Set the top margin to 50 pixels
                                b=50,  # Set the bottom margin to 50 pixels
                            ),
                                          
                
                
                        )
                        st.plotly_chart(fig)
                    #fig.write_html(fr'c:\Users\{loc}\Desktop\OldBudget.html')
                    #fig.show()
                
                
                    with fintab3:
                        resim=gcp_download_x(target_bucket,rf"FIN/2023Adopted.png")
                        resim=Image.open(io.BytesIO(resim))
                        
                        agree = st.checkbox('CHECK BOX TO SEE 2023 BUDGET STRUCTURE')
                
                        if agree:
                            st.image(resim)
                        
                        temp=gcp_download_x(target_bucket,rf"FIN/main2022.ftr")
                        temp=pd.read_feather(io.BytesIO(temp)) 
                        temp2023=gcp_download_x(target_bucket,rf"FIN/main2023.ftr")
                        temp2023=pd.read_feather(io.BytesIO(temp2023))
                        temp["Account"]=temp["Account"].astype("str")
                        temp.set_index("index",drop=True,inplace=True)
                        temp.Account=[str(i)+"-"+str(j) for i,j in zip(temp.Account,temp.Sub_Cat)]
                        temp2023["Account"]=temp2023["Account"].astype("str")
                        temp2023.set_index("index",drop=True,inplace=True)
                        temp2023.Account=[str(i)+"-"+str(j) for i,j in zip(temp2023.Account,temp2023.Sub_Cat)]
                        
                        temp1=budget_codes.copy()
                        temp1.drop(columns=["2021 Final"],inplace=True)
                        temp1.insert(5,"2022 Actual",[temp[temp["Account"]==i]["Net"].sum() for i in temp1.Account])
                        temp1.insert(6,"2022 Monthly",[round(temp[temp["Account"]==i]["Net"].sum()/12,1) for i in temp1.Account])
                        months=int(temp2023.Date.max().month)-1
                        st.write(months)
                        
                        #months=5
                        temp1["2023 Monthly Budgeted"]=[round(i/12,2) for i in temp1["2023 Adopted"]]
                        temp1["2023 Monthly"]=[round(temp2023[temp2023["Account"]==i]["Net"].sum()/months,1) for i in temp1.Account]
                        x=gcp_download_x(target_bucket,rf"FIN/2024annual-try.pkl")
                        x=io.BytesIO(x)
                        temp1["2024 PROPOSED"]= pd.read_pickle(x)["2024 PROPOSED"]
                        temp1["2024 Monthly"]=[round(i/12,1) for i in temp1["2024 PROPOSED"]]
                        temp1=st.experimental_data_editor(temp1)
                        if st.button("SAVE 2024 BUDGET EDITS"):
                            
                            temp1["2024 PROPOSED"].to_pickle(fr'c:\Users\afsiny\Desktop\Dashboard\2024annual.pkl')
                            temp1.to_pickle(fr'c:\Users\afsiny\Desktop\Dashboard\2024annual-try.pkl')
                            temp1.to_excel(fr'c:\Users\afsiny\Desktop\Dashboard\2024annual-try.xlsx')
                            temp1 = pd.read_pickle(fr'c:\Users\afsiny\Desktop\Dashboard\2024annual-try.pkl')
                        b1,b2,b3,b4= st.columns([2,2,2,6])
                        
                        with b1:
                            sankey=temp1.groupby(["Group"])[["2022 Adopted"]].sum()
                            
                            st.write(sankey)
                        with b2:
                            sankey=temp1.groupby(["Group"])[["2022 Actual"]].sum()
                            
                            st.write(sankey)
                        with b3:
                            sankey=temp1.groupby(["Group"])[["2023 Adopted"]].sum()
                            
                            st.write(sankey)
                    #             revs=[temp.loc[("Revenue",i)].values[0] for i in labels[:5]]
                    # 
                    #             ops=[abs(temp.loc[("Operating Expenses",i)].values[0]) for i in labels[7:13]]
                    #             maint=[abs(temp.loc[("Maintenance Expenses",i)].values[0]) for i in labels[14:19]]
                    #             dep=[abs(temp.loc[("Depreciation",i)].values[0]) for i in labels[20:22]]
                    #             overhead=[abs(temp.loc[("Overhead",i)].values[0]) for i in labels[22:23]]
                    #             overall=sum(revs)-sum(ops)-sum(maint)-sum(dep)-sum(overhead)
                    #             #print(overall)
                    #             revs
                    with fintab4:
                        ear=st.selectbox("Select Year",["2023","2022","2021"],key="yeartab2")
                
                        ledgers=gcp_download_x(target_bucket,rf"FIN/main{ear}.ftr")
                        ledgers=pd.read_feather(io.BytesIO(ledgers))
                        ledgers["Account"]=ledgers["Account"].astype("str")
                        ledgers.set_index("index",drop=True,inplace=True)
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
                            
                            
                            
                            match = re.match(pattern, tata[1500])
                       
                
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
                            
                    #                 jobs=[]
                    #                 pattern = r"\b\d+\b"
                    #                 # Loop over the strings and print the vendor codes and names
                    #                 for s in ledgers["Job_No"].values.tolist():
                    #                     
                    #                     try:
                    #                         match = re.match(pattern, s)
                    #                     except:
                    #                         pass
                    #                     if match:
                    #                         jobs.append(s)
                    #                 
                            
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
                            #st.write(jobs)
                                    #print(f'{vendor_code} {vendor_name}')
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
                                
                        #st.write(final)
                    with fintab5:
                        year=st.selectbox("SELECT YEAR",["2023","2022","2021","2020","2019","2018","2017"],key="depreciation")
                        terminal_depreciation=gcp_download_x(target_bucket,rf"FIN/main{year}-30.ftr")
                        terminal_depreciation=pd.read_feather(io.BytesIO(terminal_depreciation)).set_index("index",drop=True).reset_index(drop=True)
                        
                        a=terminal_depreciation[terminal_depreciation["Account"].isin( [i for i in terminal_depreciation.Account.unique().tolist() if i>1700000 and  i<2000000])]
                        a=a.groupby(["Account"])[["Credit"]].sum()
                        a.insert(0,"Name",[terminal_depreciation.loc[terminal_depreciation["Account"]==i,"Name"].values[0] for i in a.index])
                        
                        divisor=3 if year=="2023" else 12
                        
                        
                        
                        th_props = [
                              ('font-size', '16px'),
                              ('text-align', 'center'),
                              ('font-weight', 'bold'),
                              ('color', '#6d6d6d'),
                              ('background-color', '#f7ffff')
                              ]
                                                           
                        td_props = [
                          ('font-size', '15px'),
                          ('background-color', '#r9f9ff')
                          ]
                        def highlight_total(val):
                            return 'font-weight: bold; font-size: 30px;'
                           
                        styles = [
                                  dict(selector="th", props=th_props),
                                  dict(selector="td", props=td_props)
                                  ]
                        
                    #             sns.set_style("darkgrid", {"axes.facecolor": ".9"})
                    #                      
                    #             colors = ['#FFD700', '#32CD32', '#FF69B4', '#ADD8E6', '#FFA07A']
                    #             
                    #             width = st.sidebar.slider("plot width", 1, 25, 3)
                    #             height = st.sidebar.slider("plot height", 1, 25, 1)
                    #             selection=st.sidebar.checkbox("I agree")
                    #             fig, ax = plt.subplots(figsize=(6,4))
                    #             
                    #             ax.axis('off')
                    #             labels = [f'{name[5:]}\n${credit:,.2f}' if credit>50000 else "*" for name, credit in zip(a.index, a['Credit'])]
                    #             
                    #             squarify.plot(sizes=a['Credit'], label=labels, color=colors, alpha=0.8, text_kwargs={'fontsize':6, 'fontweight':'bold'})
                    #             plt.title(f'TERMINAL DEPRECIATION - {year}', fontweight='bold', fontsize=14, y=1.08)
                    #             small_items = a[a['Credit'] <= 50000][['Credit']]
                    #             small_ax = fig.add_axes([0.905, 0.3, 0.1, 0.5])
                    #             small_ax.barh(y=[i[5:] for i in small_items.index], width=small_items['Credit'], color='#808080')
                    #             small_ax.set_xlabel('Amount')
                    #             small_ax.yaxis.set_label_position("right")
                    #             small_ax.yaxis.tick_right()
                    #             #fig.set_size_inches(4, 6)
                        labels = [f'{name[5:]}\n${credit:,.2f}' if credit>50000 else "*" for name, credit in zip(a["Name"], a['Credit'])]
                        fig = px.treemap(a, 
                             path=["Name"], 
                             values='Credit',
                                         labels={"Name":labels},
                             color='Credit',
                             color_continuous_scale='Blues')
                
                    # Update the layout
                        fig.update_layout(
                            margin=dict(t=50, l=0, r=0, b=0),
                            font=dict(size=16),
                            title='TERMINAL DEPRECIATION',
                            title_font_size=24,
                            title_font_family='Arial')
                
                    # Show the plot
                
                        
                        st.plotly_chart(fig)
                        a.set_index("Name",drop=True,inplace=True)
                        a.loc["TOTAL"]=a.sum()
                        a["Monthly"]=['${:,.1f}'.format(i/divisor) for i in a["Credit"]]          
                        a["Credit"]=['${:,.1f}'.format(i) for i in a["Credit"]]
                        
                        a=a.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles).applymap(highlight_total, subset=pd.IndexSlice["TOTAL", ["Credit","Monthly"]])
                        st.table(a)    
            
            
              
        if select=="ADMIN" :
            admin_tab1,admin_tab2,admin_tab3,admin_tab4=st.tabs(["RELEASE ORDERS","BILL OF LADINGS","EDI'S","STORAGE"])
            with admin_tab4:
                maintenance=False
                if not maintenance:
                    def calculate_balance(start_tons, daily_rate, storage_rate):
                        balances={}
                        tons_remaining = start_tons
                        accumulated=0
                        day=1
                        while tons_remaining>daily_rate:
                            #print(day)
                            balances[day]={"Remaining":tons_remaining,"Charge":0,"Accumulated":0}
                            if day % 7 < 5:  # Consider only weekdays
                                tons_remaining-=daily_rate
                                #print(tons_remaining)
                                
                                balances[day]={"Remaining":tons_remaining,"Charge":0,"Accumulated":0}
                    
                                # If storage free days are over, start applying storage charges
                            elif day % 7 in ([5,6]):
                                balances[day]={"Remaining":tons_remaining,"Charge":0,"Accumulated":accumulated}
                            if day >free_days_till:
                                charge = round(tons_remaining*storage_rate,2)  # You can adjust the storage charge after the free days
                                accumulated+=charge
                                accumulated=round(accumulated,2)
                                balances[day]={"Remaining":tons_remaining,"Charge":charge,"Accumulated":accumulated}
                            
                            day+=1
                        return balances
                        
                    here1,here2,here3=st.columns([2,5,3])
                    
                    with here1:
                        with st.container(border=True):
                            initial_tons =st.number_input("START TONNAGE", min_value=1000, help=None, on_change=None,step=50, disabled=False, label_visibility="visible",key="fas2aedseq")
                            daily_rate=st.slider("DAILY SHIPMENT TONNAGE",min_value=248, max_value=544, step=10,key="fdee2a")
                            storage_rate = st.number_input("STORAGE RATE DAILY ($)",value=0.15, help="dsds", on_change=None, disabled=False, label_visibility="visible",key="fdee2dsdseq")
                            free_days_till = st.selectbox("FREE DAYS",[15,30,45,60])
                    
                    with here3:
                        with st.container(border=True):    
                            balances = calculate_balance(initial_tons, daily_rate, storage_rate)
                            d=pd.DataFrame(balances).T
                            start_date = pd.to_datetime('today').date()
                            end_date = start_date + pd.DateOffset(days=120)  # Adjust as needed
                            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                            
                            d.columns=["Remaining Tonnage","Daily Charge","Accumulated Charge"]
                            d.rename_axis("Days",inplace=True)
                            total=round(d.loc[len(d),'Accumulated Charge'],1)
                            st.dataframe(d)

                    with here2:
                        with st.container(border=True):     
                            st.write(f"######  Cargo: {initial_tons} - Loadout Rate/Day: {daily_rate} Tons - Free Days : {free_days_till}" )
                            st.write(f"##### TOTAL CHARGES:  ${total}" )
                            st.write(f"##### DURATION OF LOADOUT:  {len(d)} Days")
                            st.write(f"##### MONTHLY REVENUE: ${round(total/len(d)*30,1)} ")
                            fig = px.bar(d, x=d.index, y="Accumulated Charge", title="Accumulated Charges Over Days")
        
                            # Add a horizontal line for the monthly average charge
                            average_charge = round(total/len(d)*30,1)
                            fig.add_shape(
                                dict(
                                    type="line",
                                    x0=d.index.min(),
                                    x1=d.index.max(),
                                    y0=average_charge,
                                    y1=average_charge,
                                    line=dict(color="red", dash="dash"),
                                )
                            )
                            
                            # Add annotation with the average charge value
                            fig.add_annotation(
                                x=d.index.max()//3,
                                y=average_charge,
                                text=f'Monthly Average Income: <b><i>${average_charge:.2f}</b></i> ',
                                showarrow=True,
                                arrowhead=4,
                                ax=-50,
                                ay=-30,
                                font=dict(size=16),
                                bgcolor='rgba(255, 255, 255, 0.6)',
                            )
                            
                            # Set layout options
                            fig.update_layout(
                                xaxis_title="Days",
                                yaxis_title="Accumulated Charge",
                                sliders=[
                                    {
                                        "steps": [
                                            {"args": [[{"type": "scatter", "x": d.index, "y": d["Accumulated Charge"]}], "layout"], "label": "All", "method": "animate"},
                                        ],
                                    }
                                ],
                            )
                            st.plotly_chart(fig)
                    
            with admin_tab2:
                bill_data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                admin_bill_of_ladings=json.loads(bill_data)
                admin_bill_of_ladings=pd.DataFrame.from_dict(admin_bill_of_ladings).T[1:]
                
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')
                use=True
                if use:
                    now=datetime.datetime.now()-datetime.timedelta(hours=utc_difference)
                    
                    daily_admin_bill_of_ladings=admin_bill_of_ladings.copy()
                    
                    daily_admin_bill_of_ladings["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in admin_bill_of_ladings["issued"]]
                    daily_admin_bill_of_ladings_=daily_admin_bill_of_ladings[daily_admin_bill_of_ladings["Date"]==now.date()]
                    choose = st.radio(
                                    "Select Today's Bill of Ladings or choose by Date or choose ALL",
                                    ["DAILY", "ACCUMULATIVE", "FIND BY DATE"],key="wewas")
                    if choose=="DAILY":
                        st.dataframe(daily_admin_bill_of_ladings_)
                        csv=convert_df(daily_admin_bill_of_ladings_)
                        file_name=f'OLYMPIA_DAILY_BILL_OF_LADINGS-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                    elif choose=="FIND BY DATE":
                        required_date=st.date_input("CHOOSE DATE",key="dssar")
                        filtered_daily_admin_bill_of_ladings=daily_admin_bill_of_ladings[daily_admin_bill_of_ladings["Date"]==required_date]
                        
                        st.dataframe(filtered_daily_admin_bill_of_ladings)
                        csv=convert_df(filtered_daily_admin_bill_of_ladings)
                        file_name=f'OLYMPIA_BILL_OF_LADINGS_FOR-{datetime.datetime.strftime(required_date,"%m-%d,%Y")}.csv'
                    else:
                        st.dataframe(admin_bill_of_ladings)
                        csv=convert_df(admin_bill_of_ladings)
                        file_name=f'OLYMPIA_ALL_BILL_OF_LADINGS to {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'

                    st.download_button(
                        label="DOWNLOAD BILL OF LADINGS",
                        data=csv,
                        file_name=file_name,
                        mime='text/csv')
            with admin_tab3:
                edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                try:
                    requested_edi=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}")
                    st.text_area("EDI",requested_edi,height=400)                                
                   
                    st.download_button(
                        label="DOWNLOAD EDI",
                        data=requested_edi,
                        file_name=f'{requested_edi_file}',
                        mime='text/csv')

                except:
                    st.write("NO EDI FILES IN DIRECTORY")
                                                                                 
            
                            

            
                          
            with admin_tab1:
                carrier_list_=gcp_download(target_bucket,rf"carrier.json")
                carrier_list=json.loads(carrier_list_)
                junk=gcp_download(target_bucket,rf"junk_release.json")
                junk=json.loads(junk)
                mill_shipments=gcp_download(target_bucket,rf"mill_shipments.json")
                mill_shipments=json.loads(mill_shipments)
                mill_df=pd.DataFrame.from_dict(mill_shipments).T
                mill_df["Terminal Code"]=mill_df["Terminal Code"].astype(str)
                mill_df["New Product"]=mill_df["New Product"].astype(str)
                try:
                    release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                    release_order_database=json.loads(release_order_database)
                except:
                    release_order_database={}
                
              
                release_order_tab1,release_order_tab2,release_order_tab3=st.tabs(["CREATE RELEASE ORDER","RELEASE ORDER DATABASE","RELEASE ORDER STATUS"])
                with release_order_tab3:
                    maintenance=False
                    if not maintenance:
                        inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                        inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
                        ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                        raw_ro = json.loads(ro)
                        grouped_df = inv_bill_of_ladings.groupby(['release_order','sales_order','destination'])[['quantity']].agg(sum)
                        info=grouped_df.T.to_dict()
                        for rel_ord in raw_ro:
                            for sales in raw_ro[rel_ord]:
                                try:
                                    found_key = next((key for key in info.keys() if rel_ord in key and sales in key), None)
                                    qt=info[found_key]['quantity']
                                except:
                                    qt=0
                                info[rel_ord,sales,raw_ro[rel_ord][sales]['destination']]={'total':raw_ro[rel_ord][sales]['total'],
                                                        'shipped':qt,'remaining':raw_ro[rel_ord][sales]['remaining']}
                                                    
                        new=pd.DataFrame(info).T
                        new=new.reset_index()
                        #new.groupby('level_1')['remaining'].sum()
                        new.columns=["Release Order #","Sales Order #","Destination","Total","Shipped","Remaining"]
                        new.index=[i for i in new.index]
                        #new.columns=["Release Order #","Sales Order #","Destination","Total","Shipped","Remaining"]
                        new.index=[i+1 for i in new.index]
                        new.loc["Total"]=new[["Total","Shipped","Remaining"]].sum()
                        release_orders = [str(key[0]) for key in info.keys()]
                        release_orders=[str(i) for i in release_orders]
                        release_orders = pd.Categorical(release_orders)
                        
                        total_quantities = [item['total'] for item in info.values()]
                        shipped_quantities = [item['shipped'] for item in info.values()]
                        remaining_quantities = [item['remaining'] for item in info.values()]
                        destinations = [key[2] for key in info.keys()]
                        # Calculate the percentage of shipped quantities
                        #percentage_shipped = [shipped / total * 100 for shipped, total in zip(shipped_quantities, total_quantities)]
                        
                        # Create a Plotly bar chart
                        fig = go.Figure()
                        
                        # Add bars for total quantities
                        fig.add_trace(go.Bar(x=release_orders, y=total_quantities, name='Total', marker_color='lightgray'))
                        
                        # Add filled bars for shipped quantities
                        fig.add_trace(go.Bar(x=release_orders, y=shipped_quantities, name='Shipped', marker_color='blue', opacity=0.7))
                        
                        # Add remaining quantities as separate scatter points
                        #fig.add_trace(go.Scatter(x=release_orders, y=remaining_quantities, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
                        
                        remaining_data = [remaining if remaining > 0 else None for remaining in remaining_quantities]
                        fig.add_trace(go.Scatter(x=release_orders, y=remaining_data, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
                        
                        # Add destinations as annotations
                        annotations = [dict(x=release_order, y=total_quantity, text=destination, showarrow=True, arrowhead=4, ax=0, ay=-30) for release_order, total_quantity, destination in zip(release_orders, total_quantities, destinations)]
                        #fig.update_layout(annotations=annotations)
                        
                        # Update layout
                        fig.update_layout(title='Shipment Status',
                                          xaxis_title='Release Orders',
                                          yaxis_title='Quantities',
                                          barmode='overlay',
                                          xaxis=dict(tickangle=-90, type='category'))
                        relcol1,relcol2=st.columns([5,5])
                        with relcol1:
                            st.dataframe(new)
                        with relcol2:
                            st.plotly_chart(fig)
                        
                        temp_dict={}
                        for rel_ord in raw_ro:
                            
                            
                            for sales in raw_ro[rel_ord]:
                                temp_dict[rel_ord,sales]={}
                                dest=raw_ro[rel_ord][sales]['destination']
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
                        
                with release_order_tab1:   ###CREATE RELEASE ORDER
                    #vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308"])  ###-###
                   
                    add=st.checkbox("CHECK TO ADD TO EXISTING RELEASE ORDER",disabled=False)
                    edit=st.checkbox("CHECK TO EDIT EXISTING RELEASE ORDER")
                    batch_mapping=gcp_download(target_bucket,rf"batch_mapping.json")
                    batch_mapping=json.loads(batch_mapping)
                    if edit:
                        
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in [i.replace(".json","") for i in list_files_in_subfolder(target_bucket, rf"release_orders/ORDERS/")[1:]] if i not in junk]))
                        to_edit=gcp_download(target_bucket,rf"release_orders/ORDERS/{release_order_number}.json")
                        to_edit=json.loads(to_edit)
                        po_number_edit=st.text_input("PO No",to_edit[release_order_number]["po_number"],disabled=False)
                        destination_edit=st.text_input("Destination",to_edit[release_order_number]["destination"],disabled=False)
                        sales_order_item_edit=st.selectbox("Sales Order Item",list(to_edit[release_order_number].keys())[2:],disabled=False)
                        vessel_edit=vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308","LAGUNA-3142","LYSEFJORD-2308"],key="poFpoa")
                        ocean_bill_of_lading_edit=st.selectbox("Ocean Bill Of Lading",batch_mapping[vessel_edit].keys(),key="trdfeerw") 
                        wrap_edit=st.text_input("Grade",to_edit[release_order_number][sales_order_item_edit]["grade"],disabled=False)
                        batch_edit=st.text_input("Batch No",to_edit[release_order_number][sales_order_item_edit]["batch"],disabled=False)
                        dryness_edit=st.text_input("Dryness",to_edit[release_order_number][sales_order_item_edit]["dryness"],disabled=False)
                        admt_edit=st.text_input("ADMT PER UNIT",round(float(batch_mapping[vessel_edit][ocean_bill_of_lading_edit]["dryness"])/90,6),disabled=False)
                        unitized_edit=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity_edit=st.number_input("Quantity of Units", 0, disabled=False, label_visibility="visible")
                        tonnage_edit=2*quantity_edit
                        shipped_edit=st.number_input("Shipped # of Units",to_edit[release_order_number][sales_order_item_edit]["shipped"],disabled=True)
                        remaining_edit=st.number_input("Remaining # of Units",
                                                       quantity_edit-to_edit[release_order_number][sales_order_item_edit]["shipped"],disabled=True)
                        carrier_code_edit=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()],key="dsfdssa")          
                        
                    elif add:
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in [i.replace(".json","") for i in list_files_in_subfolder(target_bucket, rf"release_orders/ORDERS/")][1:] if i not in junk]))
                        to_add=gcp_download(target_bucket,rf"release_orders/ORDERS/{release_order_number}.json")
                        to_add=json.loads(to_add)
                        po_number_add=st.text_input("PO No",to_add[release_order_number]["po_number"],disabled=False)
                        destination_add=st.text_input("Destination",to_add[release_order_number]["destination"],disabled=False)
                        sales_order_item_add=st.text_input("Sales Order Item",disabled=False)
                        vessel_add=vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308","LAGUNA-3142","LYSEFJORD-2308"],key="popoa")
                        ocean_bill_of_lading_add=st.selectbox("Ocean Bill Of Lading",batch_mapping[vessel_add].keys(),key="treerw")  
                        wrap_add=st.text_input("Grade",batch_mapping[vessel_add][ocean_bill_of_lading_add]["grade"],disabled=True) 
                        batch_add=st.text_input("Batch No",batch_mapping[vessel_add][ocean_bill_of_lading_add]["batch"],disabled=False)
                        dryness_add=st.text_input("Dryness",batch_mapping[vessel_add][ocean_bill_of_lading_add]["dryness"],disabled=False)
                        admt_add=st.text_input("ADMT PER UNIT",round(float(batch_mapping[vessel_add][ocean_bill_of_lading_add]["dryness"])/90,6),disabled=False)
                        unitized_add=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity_add=st.number_input("Quantity of Units", 0, disabled=False, label_visibility="visible")
                        tonnage_add=2*quantity_add
                        shipped_add=0
                        remaining_add=st.number_input("Remaining # of Units", quantity_add,disabled=True)
                        transport_type_add=st.radio("Select Transport Type",("TRUCK","RAIL"))
                        carrier_code_add=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()])            
                    else:  ### If creating new release order
                        
                        release_order_number=st.text_input("Release Order Number")
                        po_number=st.text_input("PO No")
                        destination_list=list(set([f"{i}-{j}" for i,j in zip(mill_df["Group"].tolist(),mill_df["Final Destination"].tolist())]))
                        #st.write(destination_list)
                        destination=st.selectbox("SELECT DESTINATION",destination_list)
                        sales_order_item=st.text_input("Sales Order Item")  
                        vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308","LAGUNA-3142","LYSEFJORD-2308"],key="tre")
                        ocean_bill_of_lading=st.selectbox("Ocean Bill Of Lading",batch_mapping[vessel].keys())   #######
                        wrap=st.text_input("Grade",batch_mapping[vessel][ocean_bill_of_lading]["grade"],disabled=True)   ##### batch mapping injection
                        batch=st.text_input("Batch No",batch_mapping[vessel][ocean_bill_of_lading]["batch"],disabled=True)   #####
                        dryness=st.text_input("Dryness",batch_mapping[vessel][ocean_bill_of_lading]["dryness"],disabled=True)   #####
                        admt=st.text_input("ADMT PER UNIT",round(float(batch_mapping[vessel][ocean_bill_of_lading]["dryness"])/90,6),disabled=True)  #####
                        unitized=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity=st.number_input("Quantity of Units", min_value=1, max_value=5000, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                        tonnage=2*quantity
                        #queue=st.number_input("Place in Queue", min_value=1, max_value=20, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                        transport_type=st.radio("Select Transport Type",("TRUCK","RAIL"))
                        carrier_code=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()])            
                    
        
                    create_release_order=st.button("SUBMIT")
                    if create_release_order:
                        
                        if add: 
                            data=gcp_download(target_bucket,rf"release_orders/ORDERS/{release_order_number}.json")
                            to_edit=json.loads(data)
                            temp=add_release_order_data(to_add,release_order_number,destination_add,po_number_add,sales_order_item_add,vessel_add,batch_add,ocean_bill_of_lading_add,wrap_add,dryness_add,unitized_add,quantity_add,tonnage_add,transport_type_add,carrier_code_add)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/ORDERS/{release_order_number}.json")
                            blob.upload_from_string(temp)
                            st.success(f"ADDED sales order item {sales_order_item_add} to release order {release_order_number}!")
                        elif edit:
                            data=gcp_download(target_bucket,rf"release_orders/ORDERS/{release_order_number}.json")
                            to_edit=json.loads(data)
                            temp=edit_release_order_data(to_edit,sales_order_item_edit,quantity_edit,tonnage_edit,shipped_edit,remaining_edit,carrier_code_edit)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/ORDERS/{release_order_number}.json")
                            blob.upload_from_string(temp)
                            st.success(f"Edited release order {release_order_number} successfully!")
                            
                        else:
                            
                            temp=store_release_order_data(release_order_number,destination,po_number,sales_order_item,vessel,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/ORDERS/{release_order_number}.json")
                            blob.upload_from_string(temp)
                            st.success(f"Created release order {release_order_number} successfully!")
                        
                        try:
                            junk=gcp_download(target_bucket,rf"release_orders/{vessel}/junk_release.json")
                        except:
                            junk=gcp_download(target_bucket,rf"junk_release.json")
                        junk=json.loads(junk)
                        try:
                            del junk[release_order_number]
                            jason_data=json.dumps(junk)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"junk_release.json")
                            blob.upload_from_string(jason_data)
                        except:
                            pass
                        

                        

                        if edit:
                            release_order_database[release_order_number][sales_order_item_edit]={"destination":destination_edit,"vessel":vessel_edit,"total":quantity_edit,"remaining":remaining_edit}
                        elif add:
                            if sales_order_item_add not in release_order_database[release_order_number]:
                                release_order_database[release_order_number][sales_order_item_add]={}
                                release_order_database[release_order_number][sales_order_item_add]={"destination":destination_add,"vessel":vessel_add,"total":quantity_add,"remaining":quantity_add}
                        else:
                            release_order_database[release_order_number]={}
                            release_order_database[release_order_number][sales_order_item]={}
                            
                            release_order_database[release_order_number][sales_order_item]={"destination":destination,"vessel":vessel,"total":quantity,"remaining":quantity}
                            st.write(f"Updated Release Order Database")
                        
                        release_orders_json=json.dumps(release_order_database)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                        blob.upload_from_string(release_orders_json)
                        
                with release_order_tab2:  ##   RELEASE ORDER DATABASE ##
                    
                    #vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308"],key="other")
                    rls_tab1,rls_tab2,rls_tab3=st.tabs(["ACTIVE RELEASE ORDERS","COMPLETED RELEASE ORDERS","ENTER MF NUMBERS"])
                    data=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")  #################
                    try:
                        release_order_database=json.loads(data)
                    except: 
                        release_order_dictionary={}
                    
                    with rls_tab1:
                        
                        completed_release_orders=[]   #### Check if any of them are completed.
                     
                        
                        for key in release_order_database:
                            not_yet=0
                            for sales in release_order_database[key]:
                                if release_order_database[key][sales]["remaining"]>0:
                                    not_yet=1
                                else:
                                    pass
                            if not_yet==0:
                                completed_release_orders.append(key)
                    
                        files_in_folder_ = [i.replace(".json","") for i in list_files_in_subfolder(target_bucket, rf"release_orders/ORDERS/")]   ### REMOVE json extension from name
                        
                        junk=gcp_download(target_bucket,rf"junk_release.json")
                        junk=json.loads(junk)
                        files_in_folder=[i for i in files_in_folder_ if i not in completed_release_orders]        ###  CHECK IF COMPLETED
                        files_in_folder=[i for i in files_in_folder if i not in junk.keys()]        ###  CHECK IF COMPLETED
                        
                        ###       Make Release order destinaiton map for dropdown menu
                        
                        release_order_dest_map={} 
                        for i in release_order_database:
                            for sales in release_order_database[i]:
                                release_order_dest_map[i]=release_order_database[i][sales]["destination"]
                        destinations_of_release_orders=[f"{i} to {release_order_dest_map[i]}" for i in files_in_folder if i!=""]

                        ###       Dropdown menu
                        nofile=0
                        requested_file_=st.selectbox("ACTIVE RELEASE ORDERS",destinations_of_release_orders)
                        requested_file=requested_file_.split(" ")[0]
                 
                        
                       
                        data=gcp_download(target_bucket,rf"release_orders/ORDERS/{requested_file}.json")
                        release_order_json = json.loads(data)
                            
                            
                        target=release_order_json[requested_file]
                        destination=target['destination']
                        po_number=target["po_number"]
                        if len(target.keys())==0:
                            nofile=1
                           
                        number_of_sales_orders=len([i for i in target if i not in ["destination","po_number"]])   ##### WRONG CAUSE THERE IS NOW DESTINATION KEYS
                    
                        
                        
                        rel_col1,rel_col2,rel_col3,rel_col4=st.columns([2,2,2,2])
                        #### DISPATCHED CLEANUP  #######
                        
                        try:
                            dispatched=gcp_download(target_bucket,rf"dispatched.json")
                            dispatched=json.loads(dispatched)
                            #st.write(dispatched)
                        except:
                            pass
                        to_delete=[]            
                        try:
                            for i in dispatched.keys():
                                if not dispatched[i].keys():
                                    del dispatched[i]
                                
                            for k in to_delete:
                                dispatched.pop(k)
                                #st.write("deleted k")
                           
                            json_data = json.dumps(dispatched)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"dispatched.json")
                            blob.upload_from_string(json_data)
                        except:
                            pass
                        
                            
                        
                        
                        
                        ### END CLEAN DISPATCH
        
                        
                                              
                        if nofile!=1 :         
                                        
                            targets=[i for i in target if i not in ["destination","po_number"]] ####doing this cause we set jason path {downloadedfile[releaseorder] as target. i have to use one of the keys (release order number) that is in target list
                            sales_orders_completed=[k for k in targets if target[k]['remaining']<=0]
                            
                            with rel_col1:
                                
                                st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                st.markdown(f"**:blue[PO Number] : {target['po_number']}**")
                                if targets[0] in sales_orders_completed:
                                    st.markdown(f"**:orange[Sales Order Item : {targets[0]} - COMPLETED]**")
                                    target0_done=True
                                    
                                else:
                                    st.markdown(f"**:blue[Sales Order Item] : {targets[0]}**")
                                st.markdown(f"**:blue[Destination] : {target['destination']}**")
                                st.write(f"        Total Quantity-Tonnage : {target[targets[0]]['quantity']} Units - {target[targets[0]]['tonnage']} Metric Tons")
                                st.write(f"        Ocean Bill Of Lading : {target[targets[0]]['ocean_bill_of_lading']}")
                                st.write(f"        Batch : {target[targets[0]]['batch']} WIRES : {target[targets[0]]['unitized']}")
                                st.write(f"        Units Shipped : {target[targets[0]]['shipped']} Units - {2*target[targets[0]]['shipped']} Metric Tons")
                                if 0<target[targets[0]]['remaining']<=10:
                                    st.markdown(f"**:red[Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Metric Tons]**")
                                elif target[targets[0]]['remaining']<=0:
                                    st.markdown(f":orange[Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Metric Tons]")                                                                        
                                else:
                                    st.write(f"       Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Metric Tons")
                            with rel_col2:
                                try:
                                
                                    st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                    if targets[1] in sales_orders_completed:
                                        st.markdown(f"**:orange[Sales Order Item : {targets[1]} - COMPLETED]**")                                    
                                    else:
                                        st.markdown(f"**:blue[Sales Order Item] : {targets[1]}**")
                                    st.markdown(f"**:blue[Destination : {target['destination']}]**")
                                    st.write(f"        Total Quantity-Tonnage : {target[targets[1]]['quantity']} Units - {target[targets[1]]['tonnage']} Metric Tons")                        
                                    st.write(f"        Ocean Bill Of Lading : {target[targets[1]]['ocean_bill_of_lading']}")
                                    st.write(f"        Batch : {target[targets[1]]['batch']} WIRES : {target[targets[1]]['unitized']}")
                                    st.write(f"        Units Shipped : {target[targets[1]]['shipped']} Units - {2*target[targets[1]]['shipped']} Metric Tons")
                                    if 0<target[targets[1]]['remaining']<=10:
                                        st.markdown(f"**:red[Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Metric Tons]**")
                                    elif target[targets[1]]['remaining']<=0:
                                        st.markdown(f":orange[Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Metric Tons]")
                                    else:
                                        st.write(f"       Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Metric Tons")
                                        
                                except:
                                    pass
                
                            with rel_col3:
                                try:
                                
                                    st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                    if targets[2] in sales_orders_completed:
                                        st.markdown(f"**:orange[Sales Order Item : {targets[2]} - COMPLETED]**")
                                    else:
                                        st.markdown(f"**:blue[Sales Order Item] : {targets[2]}**")
                                    st.markdown(f"**:blue[Destination : {target['destination']}]**")
                                    st.write(f"        Total Quantity-Tonnage : {target[targets[2]]['quantity']} Units - {target[targets[2]]['tonnage']} Metric Tons")
                                    st.write(f"        Ocean Bill Of Lading : {target[targets[2]]['ocean_bill_of_lading']}")
                                    st.write(f"        Batch : {target[targets[2]]['batch']} WIRES : {target[targets[2]]['unitized']}")
                                    st.write(f"        Units Shipped : {target[targets[2]]['shipped']} Units - {2*target[targets[2]]['shipped']} Metric Tons")
                                    if 0<target[targets[2]]['remaining']<=10:
                                        st.markdown(f"**:red[Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Metric Tons]**")
                                    elif target[targets[2]]['remaining']<=0:
                                        st.markdown(f":orange[Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Metric Tons]")
                                    else:
                                        st.write(f"       Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Metric Tons")
                                    
                                    
                                except:
                                    pass
            
                            with rel_col4:
                                try:
                                
                                    st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                    if targets[3] in sales_orders_completed:
                                        st.markdown(f"**:orange[Sales Order Item : {targets[3]} - COMPLETED]**")
                                    else:
                                        st.markdown(f"**:blue[Sales Order Item] : {targets[3]}**")
                                    st.markdown(f"**:blue[Destination : {target['destination']}]**")
                                    st.write(f"        Total Quantity-Tonnage : {target[targets[3]]['quantity']} Units - {target[targets[3]]['tonnage']} Metric Tons")
                                    st.write(f"        Ocean Bill Of Lading : {target[targets[3]]['ocean_bill_of_lading']}")
                                    st.write(f"        Batch : {target[targets[3]]['batch']} WIRES : {target[targets[3]]['unitized']}")
                                    st.write(f"        Units Shipped : {target[targets[3]]['shipped']} Units - {2*target[targets[3]]['shipped']} Metric Tons")
                                    if 0<target[targets[3]]['remaining']<=10:
                                        st.markdown(f"**:red[Units Remaining : {target[targets[3]]['remaining']} Units - {2*target[targets[3]]['remaining']} Metric Tons]**")
                                    elif target[targets[3]]['remaining']<=0:
                                        st.markdown(f":orange[Units Remaining : {target[targets[3]]['remaining']} Units - {2*target[targets[3]]['remaining']} Metric Tons]")
                                    else:
                                        st.write(f"       Units Remaining : {target[targets[3]]['remaining']} Units - {2*target[targets[3]]['remaining']} Metric Tons")
                                    
                                    
                                except:
                                    pass
                            
                            # dispatched={"vessel":vessel,"date":datetime.datetime.strftime(datetime.datetime.today()-datetime.timedelta(hours=7),"%b-%d-%Y"),
                                     #               "time":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%H:%M:%S"),
                                       #                 "release_order":requested_file,"sales_order":hangisi,"ocean_bill_of_lading":ocean_bill_of_lading,"batch":batch}
                            
                            hangisi=st.selectbox("**:green[SELECT SALES ORDER ITEM TO DISPATCH]**",([i for i in target if i not in sales_orders_completed and i not in ["destination","po_number"]]))
                            dol1,dol2,dol3,dol4=st.columns([2,2,2,2])
                            with dol1:
                               
                                if st.button("DISPATCH TO WAREHOUSE",key="lala"):
                                   
                                    
                                    
                                    dispatch=dispatched.copy()
                                    try:
                                        last=list(dispatch[requested_file].keys())[-1]
                                        #dispatch[requested_file]={}
                                        dispatch[requested_file][hangisi]={"vessel":vessel,"date":datetime.datetime.strftime(datetime.datetime.today()-datetime.timedelta(hours=7),"%b-%d-%Y"),
                                                    "time":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%H:%M:%S"),
                                                     "release_order":requested_file,"sales_order":hangisi,"destination":destination,"ocean_bill_of_lading":target[hangisi]["ocean_bill_of_lading"],"batch":target[hangisi]["batch"]}
                                    except:
                                        dispatch[requested_file]={}
                                        dispatch[requested_file][hangisi]={"vessel":vessel,"date":datetime.datetime.strftime(datetime.datetime.today()-datetime.timedelta(hours=7),"%b-%d-%Y"),
                                                    "time":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%H:%M:%S"),
                                                     "release_order":requested_file,"sales_order":hangisi,"destination":destination,"ocean_bill_of_lading":target[hangisi]["ocean_bill_of_lading"],"batch":target[hangisi]["batch"]}
            
                                    
                                    json_data = json.dumps(dispatch)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"dispatched.json")
                                    blob.upload_from_string(json_data)
                                    st.markdown(f"**DISPATCHED Release Order Number {requested_file} Item No : {hangisi} to Warehouse**")
                            with dol4:
                                
                                if st.button("DELETE SALES ORDER ITEM",key="lalag"):
                                    
                                    data_d=gcp_download("olym_suzano",rf"release_orders/{vessel}/{requested_file}.json")
                                    to_edit_d=json.loads(data_d)
                                    to_edit_d[vessel][requested_file].pop(hangisi)
                                    #st.write(to_edit_d)
                                    
                                    json_data = json.dumps(to_edit_d)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"release_orders/{vessel}/{requested_file}.json")
                                    blob.upload_from_string(json_data)
                                if st.button("DELETE RELEASE ORDER ITEM!",key="laladg"):
                                    junk=gcp_download(target_bucket,rf"junk_release.json")
                                    junk=json.loads(junk)
                                   
                                    junk[requested_file]=1
                                    json_data = json.dumps(junk)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"junk_release.json")
                                    blob.upload_from_string(json_data)
                                           
                            with dol2:  
                                if st.button("CLEAR DISPATCH QUEUE!"):
                                    dispatch={}
                                    json_data = json.dumps(dispatch)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"dispatched.json")
                                    blob.upload_from_string(json_data)
                                    st.markdown(f"**CLEARED ALL DISPATCHES**")   
                            with dol3:
                                dispatch=gcp_download(target_bucket,rf"dispatched.json")
                                dispatch=json.loads(dispatch)
                                try:
                                    item=st.selectbox("CHOOSE ITEM",dispatch.keys())
                                    if st.button("CLEAR DISPATCH ITEM"):                                       
                                        del dispatch[item]
                                        json_data = json.dumps(dispatch)
                                        storage_client = storage.Client()
                                        bucket = storage_client.bucket(target_bucket)
                                        blob = bucket.blob(rf"dispatched.json")
                                        blob.upload_from_string(json_data)
                                        st.markdown(f"**CLEARED DISPATCH ITEM {item}**")   
                                except:
                                    pass
                            st.markdown("**CURRENT DISPATCH QUEUE**")
                            try:
                                dispatch=gcp_download(target_bucket,rf"dispatched.json")
                                dispatch=json.loads(dispatch)
                                try:
                                    for dispatched_release in dispatch.keys():
                                        #st.write(dispatched_release)
                                        for sales in dispatch[dispatched_release].keys():
                                            #st.write(sales)
                                            st.markdown(f'**Release Order = {dispatched_release}, Sales Item : {sales}, Destination : {dispatch[dispatched_release][sales]["destination"]} .**')
                                except:
                                    pass
                            except:
                                st.write("NO DISPATCH ITEMS")
                        
                        else:
                            st.write("NO RELEASE ORDERS IN DATABASE")
                    with rls_tab2:
                        data=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                        completed_release_orders=[]
                        for key in release_order_database:
                            not_yet=0
                            for sales in release_order_database[key]:
                                if release_order_database[key][sales]["remaining"]>0:
                                    not_yet=1
                                else:
                                    pass
                            if not_yet==0:
                                completed_release_orders.append(key)
                        
                        #for completed in completed_release_orders:
                            #st.write(completed)
                            #data=gcp_download(target_bucket,rf"release_orders/ORDERS/{completed}.json")
                            #comp_rel_order=json.loads(data)
                        
                        completed_release_order_dest_map={}
                        for i in release_order_database:
                            if i in completed_release_orders:
                                completed_release_order_dest_map[i]=release_order_database[i]["001"]#["destination"]
                        if len(pd.DataFrame(completed_release_order_dest_map).T)>=1:
                            st.write(pd.DataFrame(completed_release_order_dest_map).T)
                            
                            #destinations_of_completed_release_orders=[f"{i} to {completed_release_order_dest_map[i]}" for i in completed_release_orders]
                            #requested_file_=st.selectbox("COMPLETED RELEASE ORDERS",destinations_of_completed_release_orders,key=16)
                            #requested_file=requested_file_.split(" ")[0]
                            #nofile=0
                    with rls_tab3:
                        mf_numbers_=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")
                        mf_numbers=json.loads(mf_numbers_)
                        goahead=True
                        if goahead:
                            
                            gp_release_orders=[i for i in release_order_database if release_order_database[i]["001"]["destination"] in ["GP-Clatskanie,OR","GP-Halsey,OR"] ]   # and (release_order_database[i]["001"]["remaining"]>0|release_order_database[i]["001"]["remaining"]>0)
                            
                            destinations_of_release_orders=[f"{i} to {release_order_dest_map[i]}" for i in release_order_database if release_order_database[i]["001"]["destination"] in ["GP-Clatskanie,OR","GP-Halsey,OR"]]
                            if len(destinations_of_release_orders)==0:
                                st.warning("NO GP RELEASE ORDERS FOR THIS VESSEL")
                            else:
                                
                                release_order_number_mf=st.selectbox("SELECT RELEASE ORDER FOR MF",destinations_of_release_orders,key="tatata")
                                release_order_number_mf=release_order_number_mf.split(" ")[0]
                                input_mf_numbers=st.text_area("**ENTER MF NUMBERS**",height=100,key="juy")
                                if input_mf_numbers is not None:
                                    input_mf_numbers = input_mf_numbers.splitlines()
                                    input_mf_numbers=[i for i in input_mf_numbers if len(i)==10]####### CAREFUL THIS ASSUMES SAME DIGIT MF EACH TIME
                               
                                if st.button("SUBMIT MF NUMBERS",key="ioeru" ):
                                    if release_order_number_mf not in mf_numbers.keys():   
                                        mf_numbers[release_order_number_mf]=[]
                                    mf_numbers[release_order_number_mf]+=input_mf_numbers
                                    mf_numbers[release_order_number_mf]=list(set(mf_numbers[release_order_number_mf]))
                                    mf_data=json.dumps(mf_numbers)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"release_orders/mf_numbers.json")
                                    blob.upload_from_string(mf_data)
                                if st.button("REMOVE MF NUMBERS",key="ioerssu" ):
                                    for i in input_mf_numbers:
                                        if i in mf_numbers[release_order_number_mf]:
                                            mf_numbers[release_order_number_mf].remove(i)
                                    mf_data=json.dumps(mf_numbers)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket(target_bucket)
                                    blob = bucket.blob(rf"release_orders/mf_numbers.json")
                                    blob.upload_from_string(mf_data)
                                st.write(mf_numbers)
                        else:
                            #st.write("NO RELEASE ORDER FOR THIS VESSEL IN DATABASE")
                            pass
                        
                                
        
                        
        
        ##########  LOAD OUT  ##############
        
        
        
        if select=="LOADOUT" :
        
            
            bill_mapping=gcp_download(target_bucket,"bill_mapping.json")
            bill_mapping=json.loads(bill_mapping)
            mill_info_=gcp_download(target_bucket,rf"mill_info.json")
            mill_info=json.loads(mill_info_)
            mf_numbers_for_load=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")
            mf_numbers_for_load=json.loads(mf_numbers_for_load)
            no_dispatch=0
            number=None
            if number not in st.session_state:
                st.session_state.number=number
            try:
                dispatched=gcp_download(target_bucket,"dispatched.json")
                dispatched=json.loads(dispatched)
            except:
                no_dispatch=1
                pass
           
            
            double_load=False
            
            if len(dispatched.keys())>0 and not no_dispatch:
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
                #st.write(liste)
                work_order_=st.selectbox("**SELECT RELEASE ORDER/SALES ORDER TO WORK**",liste,index=0 if st.session_state.work_order_ else 0) 
                st.session_state.work_order_=work_order_
                work_order=work_order_.split(" ")[0]
                order=["001","002","003","004","005","006"]
                
                current_release_order=work_order
                current_sales_order=work_order_.split(" ")[1][1:]
                vessel=dispatched[work_order][current_sales_order]["vessel"]
                destination=dispatched[work_order][current_sales_order]['destination']
                
                
                
               #for i in order:                   ##############HERE we check all the sales orders in dispatched[releaseordernumber] dictionary. it breaks after it finds the first sales order
               #     if i in dispatched[work_order].keys():
               #         current_release_order=work_order
                #        current_sales_order=i
                 #       vessel=dispatched[work_order][i]["vessel"]
                  #      destination=dispatched[work_order][i]['destination']
                  #      break
                  #  else:
                  #      pass
                try:
                    next_release_order=dispatched['002']['release_order']    #############################  CHECK HERE ######################## FOR MIXED LOAD
                    next_sales_order=dispatched['002']['sales_order']
                    
                except:
                    
                    pass
                info=gcp_download(target_bucket,rf"release_orders/ORDERS/{work_order}.json")
                info=json.loads(info)
                
                vessel=info[current_release_order][current_sales_order]["vessel"]
                #if st.checkbox("CLICK TO LOAD MIXED SKU"):
                #    try:
                  #      next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                  #      double_load=True
                 #   except:
                   #     st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                    
                
                st.markdown(rf'**:blue[CURRENTLY WORKING] :**')
                load_col1,load_col2=st.columns([9,1])
                
                with load_col1:
                    wrap_dict={"ISU":"UNWRAPPED","ISP":"WRAPPED","AEP":"WRAPPED"}
                    wrap=info[current_release_order][current_sales_order]["grade"]
                    ocean_bill_of_=info[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                    unitized=info[current_release_order][current_sales_order]["unitized"]
                    quant_=info[current_release_order][current_sales_order]["quantity"]
                    real_quant=int(math.floor(quant_))
                    ship_=info[current_release_order][current_sales_order]["shipped"]
                    ship_bale=(ship_-math.floor(ship_))*8
                    remaining=info[current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                    temp={f"<b>Release Order #":current_release_order,"<b>Destination":destination,"<b>VESSEL":vessel}
                    temp2={"<b>Ocean B/L":ocean_bill_of_,"<b>Type":wrap_dict[wrap],"<b>Prep":unitized}
                    temp3={"<b>Total Units":quant_,"<b>Shipped Units":ship_,"<b>Remaining Units":remaining}
                    #temp4={"<b>Total Bales":0,"<b>Shipped Bales":int(8*(ship_-math.floor(ship_))),"<b>Remaining Bales":int(8*(remaining-math.floor(remaining)))}
                    temp5={"<b>Total Tonnage":quant_*2,"<b>Shipped Tonnage":ship_*2,"<b>Remaining Tonnage":quant_*2-(ship_*2)}


                    
                    sub_load_col1,sub_load_col2,sub_load_col3,sub_load_col4=st.columns([3,3,3,3])
                    
                    with sub_load_col1:   
                        #st.markdown(rf'**Release Order-{current_release_order}**')
                        #st.markdown(rf'**Destination : {destination}**')
                        #st.markdown(rf'**VESSEL-{vessel}**')
                        st.write (pd.DataFrame(temp.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                    with sub_load_col2:
                        st.write (pd.DataFrame(temp2.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                        
                    with sub_load_col3:
                        
                        #st.markdown(rf'**Total Quantity : {quant_} Units - {quant_*2} Tons**')
                        #st.markdown(rf'**Shipped : {ship_} Units - {ship_*2} Tons**')
                        
                        if remaining<=10:
                            st.markdown(rf'**:red[CAUTION : Remaining : {remaining} Units]**')

                        a=pd.DataFrame(temp3.items(),columns=["UNITS","Data"])
                        a["Data"]=a["Data"].astype("float")
                        st.write (a.to_html (escape=False, index=False), unsafe_allow_html=True)
                   
                    with sub_load_col4:
                    
                        st.write (pd.DataFrame(temp5.items(),columns=["TONNAGE","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                
                
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
                if info[current_release_order][current_sales_order]["transport_type"]=="TRUCK":
                    medium="TRUCK"
                else:
                    medium="RAIL"
                
                with col1:
                ######################  LETS GET RID OF INPUT BOXES TO SIMPLIFY LONGSHORE SCREEN
                    #terminal_code=st.text_input("Terminal Code","OLYM",disabled=True)
                    terminal_code="OLYM"
                    #file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="file_dates",disabled=True)
                    file_date=datetime.datetime.today()-datetime.timedelta(hours=utc_difference)
                    if file_date not in st.session_state:
                        st.session_state.file_date=file_date
                    file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False)
                   #delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="delivery_date",disabled=True,visible=False)
                    delivery_date=datetime.datetime.today()-datetime.timedelta(hours=utc_difference)
                   #eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                    eta_date=delivery_date
                    carrier_code=info[current_release_order][current_sales_order]["carrier_code"]
                    vessel=info[current_release_order][current_sales_order]["vessel"]
                    transport_sequential_number="TRUCK"
                    transport_type="TRUCK"
                    placeholder = st.empty()
                    with placeholder.container():
                        vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=7)
                    
                        mf=True
                        load_mf_number_issued=False
                        if destination=="CLEARWATER-Lewiston,ID":
                            carrier_code=st.selectbox("Carrier Code",[info[current_release_order][current_sales_order]["carrier_code"],"310897-Ashley"],disabled=False,key=29)
                        else:
                            carrier_code=st.text_input("Carrier Code",info[current_release_order][current_sales_order]["carrier_code"],disabled=True,key=9)
                        if carrier_code=="123456-KBX":
                            if 'load_mf_number' not in st.session_state:
                                st.session_state.load_mf_number = None
                            if release_order_number in mf_numbers_for_load.keys():
                                mf_liste=[i for i in mf_numbers_for_load[release_order_number]]
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
                                    st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
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
                                       load_mf_number=st.selectbox("MF NUMBER",mf_liste,disabled=False,key=14551)
                                       mf=True
                                       load_mf_number_issued=True
                                       yes=True
                                   else:
                                       st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
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
                        #release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                        release_order_number=current_release_order
                        #sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                        sales_order_item=current_sales_order
                        #ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                        ocean_bill_of_lading=info[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                        
                         #batch=st.text_input("Batch",info[current_release_order][current_sales_order]["batch"],disabled=True)
                        batch=info[current_release_order][current_sales_order]["batch"]
                        #grade=st.text_input("Grade",info[current_release_order][current_sales_order]["grade"],disabled=True)
                        grade=info[current_release_order][current_sales_order]["grade"]
                        
               
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
                        units_shipped=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                        units_shipped=pd.read_json(units_shipped).T
                        load_dict={}
                        for row in units_shipped.index[1:]:
                            for unit in units_shipped.loc[row,'loads'].keys():
                                load_dict[unit]={"BOL":row,"RO":units_shipped.loc[row,'release_order'],"destination":units_shipped.loc[row,'destination'],
                                                 "OBOL":units_shipped.loc[row,'ocean_bill_of_lading'],
                                                 "grade":units_shipped.loc[row,'grade'],"carrier_Id":units_shipped.loc[row,'carrier_id'],
                                                 "vehicle":units_shipped.loc[row,'vehicle'],"date":units_shipped.loc[row,'issued'] }
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
                                alternate_vessel=[ship for ship in bill_mapping if ship!=vessel][0]
                                if x[:load_digit] in bill_mapping[alternate_vessel]:
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
                                            if x in load_dict.keys():
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
                                                storage_client = storage.Client()
                                                bucket = storage_client.bucket(target_bucket)
                                                blob = bucket.blob(rf"bill_mapping.json")
                                                blob.upload_from_string(updated_bill)

                                                alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                                alien_units[vessel][x]={}
                                                alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                        "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                                alien_units=json.dumps(alien_units)
                                                storage_client = storage.Client()
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
                                                storage_client = storage.Client()
                                                bucket = storage_client.bucket(target_bucket)
                                                blob = bucket.blob(rf"bill_mapping.json")
                                                blob.upload_from_string(updated_bill)

                                                alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                                alien_units[vessel][x]={}
                                                alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                        "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                                alien_units=json.dumps(alien_units)
                                                storage_client = storage.Client()
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
                    
                    admt=round(float(info[current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
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
                    
                    
                
                load_bill_data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                load_admin_bill_of_ladings=json.loads(load_bill_data)
                load_admin_bill_of_ladings=pd.DataFrame.from_dict(load_admin_bill_of_ladings).T[1:]
                load_admin_bill_of_ladings=load_admin_bill_of_ladings.sort_values(by="issued")
                last_submitted=load_admin_bill_of_ladings.index[-3:].to_list()
                last_submitted.reverse()
                st.markdown(f"**Last Submitted Bill Of Ladings (From most recent) : {last_submitted}**")

                ####  BOL CREATION
                
                bol,bill_of_ladings=gen_bill_of_lading()
                if load_mf_number_issued:
                    bol=st.session_state.load_mf_number
                bol_obl=ocean_bill_of_lading
                bol_ro=release_order_number
                bol_carrier=carrier_code
                if bol_carrier=="432602-NOLAN TRANSPORTATION GROUP":
                    bol_carrier="432602-NTG"
                bol_customer_po=f"{info[current_release_order]['po_number']} OLYM"
                bol_date=datetime.datetime.strftime(datetime.datetime.now(),"%b %d,%Y %H:%M")
                
                bol_admt=admt=round(float(info[current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
                bol_weight=round(quantity*2*2204.62,1)
                bol_vehicle=vehicle_id
                bol_batch=batch
                bol_grade=wrap
                bol_dryness=f"{info[current_release_order][current_sales_order]['dryness']}%"
                bol_mt=quantity*2
                bol_bales=quantity*8
                bol_destination=info[current_release_order]["destination"]
               
                
                def create_pdf():
                    mill_data=json.loads(gcp_download(target_bucket,rf"new_mill_database.json"))
                    
                    # Create a PDF document
                    buffer = io.BytesIO()
                    c = canvas.Canvas("example.pdf", pagesize=letter)
                    
                    # Set title and draw text
                    c.setTitle("Sample PDF")
                    c.setFont("Helvetica", 16)  # Set the default font family and size
                    c.drawString(30, 750, "STRAIGHT BILL OF LADING")
                    c.setFont("Times-Bold", 16)
                    
                    c.drawString(420, 750, f"BOL #: {bol}")
                    
                    
                    
                    data1 = [
                        ['VESSEL', vessel,'LOAD DATE',f'{bol_date}'],
                        [ 'OBL', f'{bol_obl}','GROSS WEIGHT',f'{bol_weight}'],
                        [ 'RELEASE ORDER',f'{bol_ro}','VEHICLE ID',f'{bol_vehicle}'],
                        [ 'CARRIER', f'{bol_carrier}','ADMT',f'{bol_admt}'],
                    ]
                
                    # Create a table
                    col_widths = [100, 200, 100, 135]
                    table1 = Table(data1,colWidths=col_widths,rowHeights=25)
                
                    # Add style to the table
                    style = TableStyle([
                                        ('BACKGROUND', (0, 0), (0, 3), (0.8, 0.7, 0.6)),  # Background color
                                        ('BACKGROUND', (2, 0), (-2, -1), (0.8, 0.7, 0.6)),  # Background color
                                        ('INNERGRID', (0, 0), (-1, -1), 0.25, (0.2, 0.2, 0.2)),  # Inner grid style
                                        ('BOX', (0, 0), (-1, -1), 0.25, (0.2, 0.2, 0.2)),  # Box style
                                        ('WORDWRAP', (3, 1), (3, -1), True),  # Enable text wrapping
                                    ])
                    table1.setStyle(style)
                    # Draw the table on the canvas
                    table1.wrapOn(c, 0, 0)
                    table1.drawOn(c, 30, 630)  # Position the table at (100, 500)
                    
                    # Draw a rectangle
                    c.rect(30, 525 ,270, 100)
                    c.rect(300, 525 ,270, 100)
                    
                    c.setFont("Times-Bold", 13)
                    c.drawString(35, 610, "SHIP FROM :")
                    c.drawString(315, 610, "SHIP TO :")
                    c.setFont("Helvetica", 13)
                    c.drawString(120, 590, "Port Of Olympia")
                    c.drawString(120, 570, "915 Washington St NE")
                    c.drawString(120, 550, "Olympia,WA 98501")
                    c.drawString(120, 530, "United States")
                    c.drawString(340, 590, mill_data[destination]["BOL_Name"])
                    c.drawString(340, 570, mill_data[destination]["BOL_Name_2"])
                    c.drawString(340, 550, mill_data[destination]["BOL_Addr_1"])
                    c.drawString(340, 530, mill_data[destination]["BOL_Addr_2"])
                    
                    data2 = [
                        ['Batch #', 'Grade','Dryness','MT','Bales'],
                        [ bol_batch, bol_grade,bol_dryness,bol_mt,bol_bales]
                        
                    ]
                
                    table2= Table(data2,colWidths=108,rowHeights=25)
                    table2.setStyle(TableStyle([('BACKGROUND', (0, 0), (4, 0), (0.8, 0.7, 0.6))]))
                    table2.setStyle(style)
                    table2.wrapOn(c, 0, 0)
                    table2.drawOn(c, 30, 470)  # Position the table at (100, 500)
                    
                    c.drawString(150, 430, "FSC Certified Products. FSC Mix Credit. SCS-COC-009938")
                    
                    c.setFont("Helvetica-Bold", 13)
                    c.rect(30,300,537,100)
                    c.drawString(200, 380, "Truck Inspection Record")
                    c.setFont("Helvetica", 13)
                    c.drawString(110, 360, "Truck is clean and deemed suitable for transporting product.")
                    c.line(50,325,200,325)
                    c.line(360,325,500,325)
                    c.drawString(115, 310, "Sign")
                    c.drawString(400, 310, "Print Name")
                    
                    
                    c.rect(30, 250, 250, 50)
                    c.rect(280, 250, 287, 50)
                    c.drawString(350, 270, "Corrective action if answer “NO”")
                    c.drawString(35, 280, "If trailer is in poor condition, do not load!")
                    c.drawString(35, 260,"The Port will contact the customer.")
                    
                    c.rect(30, 220, 250, 30)
                    c.rect(280, 220, 30, 30)
                    c.rect(310, 220, 30, 30)
                    c.rect(340, 220, 227, 30)
                    c.drawString(290, 232, "Y")
                    c.drawString(320, 232, "N")
                    c.drawString(32, 230, "Is the Trailer clean?")
                    
                    c.rect(30, 190, 250, 30)
                    c.rect(280, 190, 30, 30)
                    c.rect(310, 190, 30, 30)
                    c.rect(340, 190, 227, 30)
                    c.drawString(290, 202, "Y")
                    c.drawString(320, 202, "N")
                    c.drawString(32, 200, "Is the trailer free of debris?")
                    
                    c.rect(30, 160, 250, 30)
                    c.rect(280, 160, 30, 30)
                    c.rect(310, 160, 30, 30)
                    c.rect(340, 160, 227, 30)
                    c.drawString(290, 172, "Y")
                    c.drawString(320, 172, "N")
                    c.drawString(32, 170, "Is the trailer free of pest/rodent activity?")
                    
                    c.rect(30, 130, 250, 30)
                    c.rect(280, 130, 30, 30)
                    c.rect(310, 130, 30, 30)
                    c.rect(340, 130, 227, 30)
                    c.drawString(290, 142, "Y")
                    c.drawString(320, 142, "N")
                    c.drawString(32, 140, "Is the trailer free of odors?")
                    
                    c.setFont("Helvetica-Bold", 6)
                    c.drawString(32, 120,"ATTN Trucker: Your signature will acknowledge receipt of the correct description, (size and grade) AND total amount of ______ pieces. THIS WAREHOUSE will NOT be responsible if shipment")
                    c.drawString(32, 110,"of incorrect product is made for the loading and securing of product. Trucker is responsible for providing or approving the load plan and for securing the product for over-the-road transportation.")
                    
                    c.setFont("Helvetica-Bold", 11)
                    data3 = [
                        ['', f'                                                                 {bol_date}'],
                        ['', f'                                                                 {bol_date}']   ]
                    
                    table3= Table(data3,colWidths=280,rowHeights=30)
                    table3.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, 1), (0.8, 0.7, 0.6))]))
                    table3.setStyle(style)
                    table3.wrapOn(c, 0, 0)
                    table3.drawOn(c, 20, 45)  # Position the table at (100, 500)
                    c.drawString(32, 82,"Carrier Signature/ Pick up date")
                    c.drawString(32, 52,"Warehouse Signature/ Pick up date")
                    
                    
                    
                    # Save the PDF document
                    c.save()
                    # pdf_bytes = buffer.getvalue()
                    # buffer.close()
                    with open("example.pdf", "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                    return PDFbyte
                   # return "example.pdf"
                
                #pdf_file = create_pdf()
                
                
                st.download_button(label="CREATE/DOWNLOAD BOL", 
                        data=create_pdf(),
                        file_name=f"{calendar.month_name[datetime.datetime.now().month][:3]}-{datetime.datetime.now().day}-{destination}-{bol}.pdf",
                        mime='application/octet-stream')
                
                
                
                if yes and mf:
                    
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
                            carrier_code=carrier_code.split("-")[0]
                            try:
                                suzano_report_=gcp_download(target_bucket,rf"suzano_report.json")
                                suzano_report=json.loads(suzano_report_)
                            except:
                                suzano_report={}
                            consignee=destination.split("-")[0]
                            consignee_city=mill_info[destination]["city"]
                            consignee_state=mill_info[destination]["state"]
                            vessel_suzano,voyage_suzano=vessel.split("-")
                            if manual_time:
                                eta=datetime.datetime.strftime(file_date+datetime.timedelta(hours=mill_info[destination]['hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")
                            else:
                                eta=datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=mill_info[destination]['hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")
    
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
                                edi_name= f'{bill_of_lading_number}.txt'
                                bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                             "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                             "quantity":st.session_state.updated_quantity,"issued":f"{a_} {b_}","edi_no":edi_name,"loads":pure_loads} 
                                                
                            bill_of_ladings=json.dumps(bill_of_ladings)
                            storage_client = storage.Client()
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
                                storage_client = storage.Client()
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
                                info[current_release_order][current_sales_order]["shipped"]=info[current_release_order][current_sales_order]["shipped"]+quantity
                                info[current_release_order][current_sales_order]["remaining"]=info[current_release_order][current_sales_order]["remaining"]-quantity
                            if info[current_release_order][current_sales_order]["remaining"]<=0:
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
                                storage_client = storage.Client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"dispatched.json")
                                blob.upload_from_string(json_data)       
                            
                            json_data = json.dumps(info)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/ORDERS/{current_release_order}.json")
                            blob.upload_from_string(json_data)
                            success_container2=st.empty()
                            time.sleep(0.1)                            
                            success_container2.success(f"Updated Release Order {current_release_order}",icon="✅")
    
                            try:
                                release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                                release_order_database=json.loads(release_order_database)
                            except:
                                release_order_database={}
                           
                            release_order_database[current_release_order][current_sales_order]["remaining"]=release_order_database[current_release_order][current_sales_order]["remaining"]-quantity
                            release_order_database=json.dumps(release_order_database)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                            blob.upload_from_string(release_order_database)
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
                            #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                            recipients = ["afsiny@portolympia.com"]
                            password = "xjvxkmzbpotzeuuv"
                    
                  
                    
                    
                            with open('temp_file.txt', 'w') as f:
                                f.write(file_content)
                    
                            file_path = 'temp_file.txt'  # Use the path of the temporary file
                    
                            
                            upload_cs_file(target_bucket, 'temp_file.txt',rf"EDIS/{file_name}")
                            success_container5=st.empty()
                            time.sleep(0.1)                            
                            success_container5.success(f"Uploaded EDI File",icon="✅")
                            if load_mf_number_issued:
                                mf_numbers_for_load[release_order_number].remove(load_mf_number)
                                mf_numbers_for_load=json.dumps(mf_numbers_for_load)
                                storage_client = storage.Client()
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
                                        
                            alien_units=json.dumps(alien_units)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"alien_units.json")
                            blob.upload_from_string(alien_units)   
                            
                            if len(this_shipment_aliens)>0:
                                subject=f"UNREGISTERED UNITS SHIPPED TO {destination} on RELEASE ORDER {current_release_order}"
                                body=f"{len([i for i in this_shipment_aliens])} unregistered units were shipped on {vehicle_id} to {destination} on {current_release_order}.<br>{[i for i in this_shipment_aliens]}"
                                sender = "warehouseoly@gmail.com"
                                #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                recipients = ["afsiny@portolympia.com"]
                                password = "xjvxkmzbpotzeuuv"
                                send_email(subject, body, sender, recipients, password)
                            
                        else:   ###cancel bill of lading
                            pass
                
                            
        
            
                        
            else:
                st.subheader("**Nothing dispatched!**")
                    
            
                    
                        
        ##########################################################################
        
                
                        
        if select=="INVENTORY" :
            Inventory=gcp_csv_to_df(target_bucket, "kirkenes_with_ghosts_found.csv")
            data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            bill_of_ladings=json.loads(data)
            mill_info=json.loads(gcp_download(target_bucket,rf"mill_info.json"))
            inv1,inv2,inv3,inv4,inv5,inv6=st.tabs(["DAILY ACTION","SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS","RELEASE ORDER STATUS"])
            with inv1:
                
                daily1,daily2,daily3=st.tabs(["TODAY'SHIPMENTS","TRUCKS ENROUTE","TRUCKS AT DESTINATION"])
                with daily1:
                    now=datetime.datetime.now()-datetime.timedelta(hours=7)
                    text=f"SHIPPED TODAY ON {datetime.datetime.strftime(now.date(),'%b %d, %Y')} - Indexed By Terminal Bill Of Lading"
                    st.markdown(f"<p style='font-family:arial,monospace; color: #0099ff;text-shadow: 2px 2px 4px #99ccff;'>{text}</p>",unsafe_allow_html=True)     
                    df_bill=pd.DataFrame(bill_of_ladings).T
                    df_bill=df_bill[["vessel","release_order","destination","sales_order","ocean_bill_of_lading","grade","carrier_id","vehicle","quantity","issued"]]
                    df_bill.columns=["VESSEL","RELEASE ORDER","DESTINATION","SALES ORDER","OCEAN BILL OF LADING","GRADE","CARRIER ID","VEHICLE NO","QUANTITY (UNITS)","ISSUED"]
                    df_bill["Date"]=[None]+[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in df_bill["ISSUED"].values[1:]]
                    
                    df_today=df_bill[df_bill["Date"]==now.date()]
                    df_today.insert(9,"TONNAGE",[i*2 for i in df_today["QUANTITY (UNITS)"]])
                    df_today.loc["TOTAL","QUANTITY (UNITS)"]=df_today["QUANTITY (UNITS)"].sum()
                    df_today.loc["TOTAL","TONNAGE"]=df_today["TONNAGE"].sum()
                       
                    st.dataframe(df_today)

            
                with daily2:
                    enroute_vehicles={}
                    arrived_vehicles={}
                    today_arrived_vehicles={}
                    for i in bill_of_ladings:
                        if i!="11502400":
                            date_strings=bill_of_ladings[i]["issued"].split(" ")
                            
                            ship_date=datetime.datetime.strptime(date_strings[0],"%Y-%m-%d")
                            ship_time=datetime.datetime.strptime(date_strings[1],"%H:%M:%S").time()
                            
                            #st.write(bill_of_ladings[i]["issued"])
                            destination=bill_of_ladings[i]['destination']
                            truck=bill_of_ladings[i]['vehicle']
                            distance=mill_info[bill_of_ladings[i]['destination']]["distance"]
                            hours_togo=mill_info[bill_of_ladings[i]['destination']]["hours"]
                            minutes_togo=mill_info[bill_of_ladings[i]['destination']]["minutes"]
                            combined_departure=datetime.datetime.combine(ship_date,ship_time)
                           
                            estimated_arrival=combined_departure+datetime.timedelta(minutes=60*hours_togo+minutes_togo)
                            estimated_arrival_string=datetime.datetime.strftime(estimated_arrival,"%B %d,%Y -- %H:%M")
                            now=datetime.datetime.now()-datetime.timedelta(hours=7)
                            if estimated_arrival>now:
                                
                                enroute_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                                 "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}","ETA":estimated_arrival_string}
                            elif estimated_arrival.date()==now.date() and estimated_arrival<now:
                                today_arrived_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                                 "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}",
                                                                 "ARRIVAL TIME":estimated_arrival_string}
                            else:
                                
                                arrived_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                                 "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}",
                                                                 "ARRIVAL TIME":estimated_arrival_string,"ARRIVAL":estimated_arrival}
                                        
                    arrived_vehicles=pd.DataFrame(arrived_vehicles)
                    arrived_vehicles=arrived_vehicles.rename_axis('TRUCK NO')
                    arrived_vehicles=arrived_vehicles.T
                    arrived_vehicles=arrived_vehicles.sort_values(by="ARRIVAL")
                    today_arrived_vehicles=pd.DataFrame(today_arrived_vehicles)
                    today_arrived_vehicles=today_arrived_vehicles.rename_axis('TRUCK NO')
                    enroute_vehicles=pd.DataFrame(enroute_vehicles)
                    enroute_vehicles=enroute_vehicles.rename_axis('TRUCK NO')
                    st.dataframe(enroute_vehicles.T)                      
                    for i in enroute_vehicles:
                        st.write(f"Truck No : {i} is Enroute to {enroute_vehicles[i]['DESTINATION']} at {enroute_vehicles[i]['ETA']}")
                with daily3:
                    select = st.radio(
                                    "Select Today's Arrived Vehicles or All Delivered Vehicles",
                                    ["TODAY'S ARRIVALS", "ALL ARRIVALS"])
                    if select=="TODAY'S ARRIVALS":
                        st.table(today_arrived_vehicles.T)
                    if select=="ALL ARRIVALS":
                        
                        st.table(arrived_vehicles.drop(columns=['ARRIVAL']))
            
            with inv2:
                
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')
                try:
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
                                    ["DAILY", "ACCUMULATIVE", "FIND BY DATE"])
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
                    else:
                        st.dataframe(suzano_report)
                        csv=convert_df(suzano_report)
                        file_name=f'OLYMPIA_ALL_SHIPMENTS to {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                    
                    
                    
                   
                    
                
                    st.download_button(
                        label="DOWNLOAD REPORT AS CSV",
                        data=csv,
                        file_name=file_name,
                        mime='text/csv')
                except:
                    st.write("NO REPORTS RECORDED")
                

            with inv3:
                edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                try:
                    requested_edi=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}")
                    st.text_area("EDI",requested_edi,height=400)
                    st.download_button(
                        label="DOWNLOAD EDI",
                        data=requested_edi,
                        file_name=f'{requested_edi_file}',
                        mime='text/csv')

                except:
                    st.write("NO EDI FILES IN DIRECTORY")
                

                
            with inv4:
                inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
                ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                
                
                maintenance=False
                                
                if maintenance:
                    st.title("CURRENTLY UNDER MAINTENANCE, CHECK BACK LATER")

                               
                else:
                    inv4tab1,inv4tab2,inv4tab3=st.tabs(["DAILY SHIPMENT REPORT","INVENTORY","UNREGISTERED LOTS FOUND"])
                    with inv4tab1:
                        
                        amount_dict={"KIRKENES-2304":9200,"JUVENTAS-2308":10000}
                        inv_vessel=st.selectbox("Select Vessel",["KIRKENES-2304","JUVENTAS-2308","LAGUNA-3142"])
                        kf=inv_bill_of_ladings.iloc[1:].copy()
                        kf['issued'] = pd.to_datetime(kf['issued'])
                        if inv_vessel in kf['vessel']:
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
                        else:
                            st.markdown("No Shipments yet from this vessel!")
                            
                    with inv4tab2:
                        combined=gcp_csv_to_df(target_bucket,rf"combined.csv")
                        combined["Batch"]=combined["Batch"].astype(str)
                        
                        inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                        inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
                        ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                        raw_ro = json.loads(ro)
                        grouped_df = inv_bill_of_ladings.groupby('ocean_bill_of_lading')['release_order'].agg(set)
                        bols=grouped_df.T.to_dict()
                        
                        grouped_df = inv_bill_of_ladings.groupby(['release_order','ocean_bill_of_lading','destination'])[['quantity']].agg(sum)
                        info=grouped_df.T.to_dict()
                        info_=info.copy()
                        for bol in bols: #### for each bill of lading
                            for rel_ord in bols[bol]:##   (for each release order on that bill of lading)
                                found_keys = [key for key in info.keys() if rel_ord in key]
                                for key in found_keys:
                                    #print(key)
                                    qt=info[key]['quantity']
                                    info_.update({key:{'total':sum([raw_ro[rel_ord][sales]['total'] for sales in raw_ro[rel_ord]]) if rel_ord in ro else 0,
                                                      'shipped':qt,'remaining':sum([raw_ro[rel_ord][sales]['remaining'] for sales in raw_ro[rel_ord]])}})
                        new=pd.DataFrame(info_).T
                        new=new.reset_index()
                        new.groupby('level_1')['remaining'].sum()
                        
                        temp1=new.groupby("level_1")[["total","shipped","remaining"]].sum()
                        temp2=combined.groupby("Ocean B/L")[["Bales","Shipped","Remaining"]].sum()/8
                        temp=temp2.copy()
                        temp["Shipped"]=temp.index.map(lambda x: temp1.loc[x,"shipped"] if x in temp1.index else 0)
                        temp.columns=["Total","Shipped","Remaining"]
                        temp["Remaining"]=temp.Total-temp.Shipped
                        temp.loc["TOTAL"]=temp.sum(axis=0)
                        tempo=temp*2

                        inv_col1,inv_col2=st.columns([2,2])
                        with inv_col1:
                            st.subheader("By Ocean BOL,UNITS")
                            st.dataframe(temp)
                        with inv_col2:
                            st.subheader("By Ocean BOL,TONS")
                            st.dataframe(tempo)
                
                    with inv4tab3:
                        alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                        alien_vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308","LAGUNA-3142"])
                        alien_list=pd.DataFrame(alien_units[alien_vessel]).T
                        alien_list.reset_index(inplace=True)
                        alien_list.index=alien_list.index+1
                        st.markdown(f"**{len(alien_list)} units that are not on the shipping file found on {alien_vessel}**")
                        st.dataframe(alien_list)
                        
                      
            with inv5:
                inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
                maintenance=False
                if maintenance:
                    st.title("CURRENTLY IN MAINTENANCE, CHECK BACK LATER")
                else:
                    st.subheader("WEEKLY SHIPMENTS BY MILL (IN TONS)")
                    zf=inv_bill_of_ladings.copy()
                    zf['WEEK'] = pd.to_datetime(zf['issued'])
                    zf.set_index('WEEK', inplace=True)
                    
                    def sum_quantity(x):
                        return x.resample('W')['quantity'].sum()*2
                    resampled_quantity = zf.groupby('destination').apply(sum_quantity).unstack(level=0)
                    resampled_quantity=resampled_quantity.fillna(0)
                    resampled_quantity.loc["TOTAL"]=resampled_quantity.sum(axis=0)
                    resampled_quantity["TOTAL"]=resampled_quantity.sum(axis=1)
                    resampled_quantity=resampled_quantity.reset_index()
                    resampled_quantity["WEEK"][:-1]=[i.strftime("%Y-%m-%d") for i in resampled_quantity["WEEK"][:-1]]
                    resampled_quantity.set_index("WEEK",drop=True,inplace=True)
                    st.dataframe(resampled_quantity)
                    csv_weekly=convert_df(resampled_quantity)
                    st.download_button(
                    label="DOWNLOAD WEEKLY REPORT AS CSV",
                    data=csv_weekly,
                    file_name=f'WEEKLY SHIPMENT REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                    mime='text/csv')


                   
                    zf['issued'] = pd.to_datetime(zf['issued'])                   
                   
                    weekly_tonnage = zf.groupby(['destination', pd.Grouper(key='issued', freq='W')])['quantity'].sum() * 2  # Assuming 2 tons per quantity
                    weekly_tonnage = weekly_tonnage.reset_index()                   
                  
                    weekly_tonnage = weekly_tonnage.rename(columns={'issued': 'WEEK', 'quantity': 'Tonnage'})
                  
                    fig = px.bar(weekly_tonnage, x='WEEK', y='Tonnage', color='destination',
                                 title='Weekly Shipments Tonnage per Location',
                                 labels={'Tonnage': 'Tonnage (in Tons)', 'WEEK': 'Week'})
                 
                    fig.update_layout(width=1000, height=700)  # You can adjust the width and height values as needed
                    
                    st.plotly_chart(fig)


            with inv6:
                inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
                ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                raw_ro = json.loads(ro)
                grouped_df = inv_bill_of_ladings.groupby(['release_order','sales_order','destination'])[['quantity']].agg(sum)
                info=grouped_df.T.to_dict()
                for rel_ord in raw_ro:
                    for sales in raw_ro[rel_ord]:
                        try:
                            found_key = next((key for key in info.keys() if rel_ord in key and sales in key), None)
                            qt=info[found_key]['quantity']
                        except:
                            qt=0
                        info[rel_ord,sales,raw_ro[rel_ord][sales]['destination']]={'total':raw_ro[rel_ord][sales]['total'],
                                                'shipped':qt,'remaining':raw_ro[rel_ord][sales]['remaining']}
                                            
                new=pd.DataFrame(info).T
                new=new.reset_index()
                #new.groupby('level_1')['remaining'].sum()
                new.columns=["Release Order #","Sales Order #","Destination","Total","Shipped","Remaining"]
                new.index=[i+1 for i in new.index]
                #new.columns=["Release Order #","Sales Order #","Destination","Total","Shipped","Remaining"]
                new.index=[i+1 for i in new.index]
                new.loc["Total"]=new[["Total","Shipped","Remaining"]].sum()
                release_orders = [str(key[0]) for key in info.keys()]
                release_orders=[str(i) for i in release_orders]
                release_orders = pd.Categorical(release_orders)
                
                total_quantities = [item['total'] for item in info.values()]
                shipped_quantities = [item['shipped'] for item in info.values()]
                remaining_quantities = [item['remaining'] for item in info.values()]
                destinations = [key[2] for key in info.keys()]
                # Calculate the percentage of shipped quantities
                #percentage_shipped = [shipped / total * 100 for shipped, total in zip(shipped_quantities, total_quantities)]
                
                # Create a Plotly bar chart
                fig = go.Figure()
                
                # Add bars for total quantities
                fig.add_trace(go.Bar(x=release_orders, y=total_quantities, name='Total', marker_color='lightgray'))
                
                # Add filled bars for shipped quantities
                fig.add_trace(go.Bar(x=release_orders, y=shipped_quantities, name='Shipped', marker_color='blue', opacity=0.7))
                
                # Add remaining quantities as separate scatter points
                #fig.add_trace(go.Scatter(x=release_orders, y=remaining_quantities, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
                
                remaining_data = [remaining if remaining > 0 else None for remaining in remaining_quantities]
                fig.add_trace(go.Scatter(x=release_orders, y=remaining_data, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
                
                # Add destinations as annotations
                annotations = [dict(x=release_order, y=total_quantity, text=destination, showarrow=True, arrowhead=4, ax=0, ay=-30) for release_order, total_quantity, destination in zip(release_orders, total_quantities, destinations)]
                #fig.update_layout(annotations=annotations)
                
                # Update layout
                fig.update_layout(title='Shipment Status',
                                  xaxis_title='Release Orders',
                                  yaxis_title='Quantities',
                                  barmode='overlay',
                                  xaxis=dict(tickangle=-90, type='category'))
                relcol1,relcol2=st.columns([5,5])
                with relcol1:
                    st.dataframe(new)
                with relcol2:
                    st.plotly_chart(fig)
                
                temp_dict={}
                for rel_ord in raw_ro:
                    
                    
                    for sales in raw_ro[rel_ord]:
                        temp_dict[rel_ord,sales]={}
                        dest=raw_ro[rel_ord][sales]['destination']
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
        bill_mapping=gcp_download(target_bucket,"bill_mapping.json")
        bill_mapping=json.loads(bill_mapping)
        mill_info_=gcp_download(target_bucket,rf"mill_info.json")
        mill_info=json.loads(mill_info_)
        mf_numbers_for_load=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")
        mf_numbers_for_load=json.loads(mf_numbers_for_load)
        no_dispatch=0
        number=None
        if number not in st.session_state:
            st.session_state.number=number
        try:
            dispatched=gcp_download(target_bucket,"dispatched.json")
            dispatched=json.loads(dispatched)
        except:
            no_dispatch=1
            pass
       
        
        double_load=False
        
        if len(dispatched.keys())>0 and not no_dispatch:
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
            #st.write(liste)
            work_order_=st.selectbox("**SELECT RELEASE ORDER/SALES ORDER TO WORK**",liste,index=0 if st.session_state.work_order_ else 0) 
            st.session_state.work_order_=work_order_
            work_order=work_order_.split(" ")[0]
            order=["001","002","003","004","005","006"]
            
            current_release_order=work_order
            current_sales_order=work_order_.split(" ")[1][1:]
            vessel=dispatched[work_order][current_sales_order]["vessel"]
            destination=dispatched[work_order][current_sales_order]['destination']
            
            
            
           #for i in order:                   ##############HERE we check all the sales orders in dispatched[releaseordernumber] dictionary. it breaks after it finds the first sales order
           #     if i in dispatched[work_order].keys():
           #         current_release_order=work_order
            #        current_sales_order=i
             #       vessel=dispatched[work_order][i]["vessel"]
              #      destination=dispatched[work_order][i]['destination']
              #      break
              #  else:
              #      pass
            try:
                next_release_order=dispatched['002']['release_order']    #############################  CHECK HERE ######################## FOR MIXED LOAD
                next_sales_order=dispatched['002']['sales_order']
                
            except:
                
                pass
            info=gcp_download(target_bucket,rf"release_orders/ORDERS/{work_order}.json")
            info=json.loads(info)
            
            vessel=info[current_release_order][current_sales_order]["vessel"]
            #if st.checkbox("CLICK TO LOAD MIXED SKU"):
            #    try:
              #      next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
              #      double_load=True
             #   except:
               #     st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                
            
            st.markdown(rf'**:blue[CURRENTLY WORKING] :**')
            load_col1,load_col2=st.columns([9,1])
            
            with load_col1:
                wrap_dict={"ISU":"UNWRAPPED","ISP":"WRAPPED","AEP":"WRAPPED"}
                wrap=info[current_release_order][current_sales_order]["grade"]
                ocean_bill_of_=info[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                unitized=info[current_release_order][current_sales_order]["unitized"]
                quant_=info[current_release_order][current_sales_order]["quantity"]
                real_quant=int(math.floor(quant_))
                ship_=info[current_release_order][current_sales_order]["shipped"]
                ship_bale=(ship_-math.floor(ship_))*8
                remaining=info[current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                temp={f"<b>Release Order #":current_release_order,"<b>Destination":destination,"<b>VESSEL":vessel}
                temp2={"<b>Ocean B/L":ocean_bill_of_,"<b>Type":wrap_dict[wrap],"<b>Prep":unitized}
                temp3={"<b>Total Units":quant_,"<b>Shipped Units":ship_,"<b>Remaining Units":remaining}
                #temp4={"<b>Total Bales":0,"<b>Shipped Bales":int(8*(ship_-math.floor(ship_))),"<b>Remaining Bales":int(8*(remaining-math.floor(remaining)))}
                temp5={"<b>Total Tonnage":quant_*2,"<b>Shipped Tonnage":ship_*2,"<b>Remaining Tonnage":quant_*2-(ship_*2)}


                
                sub_load_col1,sub_load_col2,sub_load_col3,sub_load_col4=st.columns([3,3,3,3])
                
                with sub_load_col1:   
                    #st.markdown(rf'**Release Order-{current_release_order}**')
                    #st.markdown(rf'**Destination : {destination}**')
                    #st.markdown(rf'**VESSEL-{vessel}**')
                    st.write (pd.DataFrame(temp.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                with sub_load_col2:
                    st.write (pd.DataFrame(temp2.items(),columns=["Inquiry","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
                    
                with sub_load_col3:
                    
                    #st.markdown(rf'**Total Quantity : {quant_} Units - {quant_*2} Tons**')
                    #st.markdown(rf'**Shipped : {ship_} Units - {ship_*2} Tons**')
                    
                    if remaining<=10:
                        st.markdown(rf'**:red[CAUTION : Remaining : {remaining} Units]**')

                    a=pd.DataFrame(temp3.items(),columns=["UNITS","Data"])
                    a["Data"]=a["Data"].astype("float")
                    st.write (a.to_html (escape=False, index=False), unsafe_allow_html=True)
               
                with sub_load_col4:
                
                    st.write (pd.DataFrame(temp5.items(),columns=["TONNAGE","Data"]).to_html (escape=False, index=False), unsafe_allow_html=True)
            
            
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
            if info[current_release_order][current_sales_order]["transport_type"]=="TRUCK":
                medium="TRUCK"
            else:
                medium="RAIL"
            
            with col1:
            ######################  LETS GET RID OF INPUT BOXES TO SIMPLIFY LONGSHORE SCREEN
                #terminal_code=st.text_input("Terminal Code","OLYM",disabled=True)
                terminal_code="OLYM"
                #file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="file_dates",disabled=True)
                file_date=datetime.datetime.today()-datetime.timedelta(hours=utc_difference)
                if file_date not in st.session_state:
                    st.session_state.file_date=file_date
                file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False)
               #delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="delivery_date",disabled=True,visible=False)
                delivery_date=datetime.datetime.today()-datetime.timedelta(hours=utc_difference)
               #eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                eta_date=delivery_date
                carrier_code=info[current_release_order][current_sales_order]["carrier_code"]
                vessel=info[current_release_order][current_sales_order]["vessel"]
                transport_sequential_number="TRUCK"
                transport_type="TRUCK"
                placeholder = st.empty()
                with placeholder.container():
                    vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=7)
                
                    mf=True
                    load_mf_number_issued=False
                    if destination=="CLEARWATER-Lewiston,ID":
                        carrier_code=st.selectbox("Carrier Code",[info[current_release_order][current_sales_order]["carrier_code"],"310897-Ashley"],disabled=False,key=29)
                    else:
                        carrier_code=st.text_input("Carrier Code",info[current_release_order][current_sales_order]["carrier_code"],disabled=True,key=9)
                    if carrier_code=="123456-KBX":
                        if 'load_mf_number' not in st.session_state:
                            st.session_state.load_mf_number = None
                        if release_order_number in mf_numbers_for_load.keys():
                            mf_liste=[i for i in mf_numbers_for_load[release_order_number]]
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
                                st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
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
                                   load_mf_number=st.selectbox("MF NUMBER",mf_liste,disabled=False,key=14551)
                                   mf=True
                                   load_mf_number_issued=True
                                   yes=True
                               else:
                                   st.write(f"**:red[ASK ADMIN TO PUT MF NUMBERS]**")
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
                    #release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                    release_order_number=current_release_order
                    #sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                    sales_order_item=current_sales_order
                    #ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                    ocean_bill_of_lading=info[current_release_order][current_sales_order]["ocean_bill_of_lading"]
                    
                     #batch=st.text_input("Batch",info[current_release_order][current_sales_order]["batch"],disabled=True)
                    batch=info[current_release_order][current_sales_order]["batch"]
                    #grade=st.text_input("Grade",info[current_release_order][current_sales_order]["grade"],disabled=True)
                    grade=info[current_release_order][current_sales_order]["grade"]
                    
           
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
                    units_shipped=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                    units_shipped=pd.read_json(units_shipped).T
                    load_dict={}
                    for row in units_shipped.index[1:]:
                        for unit in units_shipped.loc[row,'loads'].keys():
                            load_dict[unit]={"BOL":row,"RO":units_shipped.loc[row,'release_order'],"destination":units_shipped.loc[row,'destination'],
                                             "OBOL":units_shipped.loc[row,'ocean_bill_of_lading'],
                                             "grade":units_shipped.loc[row,'grade'],"carrier_Id":units_shipped.loc[row,'carrier_id'],
                                             "vehicle":units_shipped.loc[row,'vehicle'],"date":units_shipped.loc[row,'issued'] }
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
                            alternate_vessel=[ship for ship in bill_mapping if ship!=vessel][0]
                            if x[:load_digit] in bill_mapping[alternate_vessel]:
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
                                        if x in load_dict.keys():
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
                                            storage_client = storage.Client()
                                            bucket = storage_client.bucket(target_bucket)
                                            blob = bucket.blob(rf"bill_mapping.json")
                                            blob.upload_from_string(updated_bill)

                                            alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                            alien_units[vessel][x]={}
                                            alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                    "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                            alien_units=json.dumps(alien_units)
                                            storage_client = storage.Client()
                                            bucket = storage_client.bucket(target_bucket)
                                            blob = bucket.blob(rf"alien_units.json")
                                            blob.upload_from_string(alien_units)
                                            
                                            
                                            subject=f"FOUND UNIT {x} NOT IN INVENTORY"
                                            body=f"Clerk identified an uninventoried {'Unwrapped' if grade=='ISU' else 'wrapped'} unit {x}, and after verifying the physical pile, inventoried it into Ocean Bill Of Lading : {ocean_bill_of_lading} for vessel {vessel}. Unit has been put into alien unit list."
                                            sender = "warehouseoly@gmail.com"
                                            recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                                            #recipients = ["afsiny@portolympia.com"]
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
                                            storage_client = storage.Client()
                                            bucket = storage_client.bucket(target_bucket)
                                            blob = bucket.blob(rf"bill_mapping.json")
                                            blob.upload_from_string(updated_bill)

                                            alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                                            alien_units[vessel][x]={}
                                            alien_units[vessel][x]={"Ocean_Bill_Of_Lading":ocean_bill_of_lading,"Batch":batch,"Grade":grade,
                                                                    "Date_Found":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y,%m-%d %H:%M:%S")}
                                            alien_units=json.dumps(alien_units)
                                            storage_client = storage.Client()
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
                
                admt=round(float(info[current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
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
                
                
            
            load_bill_data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            load_admin_bill_of_ladings=json.loads(load_bill_data)
            load_admin_bill_of_ladings=pd.DataFrame.from_dict(load_admin_bill_of_ladings).T[1:]
            load_admin_bill_of_ladings=load_admin_bill_of_ladings.sort_values(by="issued")
            last_submitted=load_admin_bill_of_ladings.index[-3:].to_list()
            last_submitted.reverse()
            st.markdown(f"**Last Submitted Bill Of Ladings (From most recent) : {last_submitted}**")

            ####  BOL CREATION
            
            bol,bill_of_ladings=gen_bill_of_lading()
            if load_mf_number_issued:
                bol=st.session_state.load_mf_number
            bol_obl=ocean_bill_of_lading
            bol_ro=release_order_number
            bol_carrier=carrier_code
            if bol_carrier=="432602-NOLAN TRANSPORTATION GROUP":
                bol_carrier="432602-NTG"
            bol_customer_po=f"{info[current_release_order]['po_number']} OLYM"
            bol_date=datetime.datetime.strftime(datetime.datetime.now(),"%b %d,%Y %H:%M")
            
            bol_admt=admt=round(float(info[current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
            bol_weight=round(quantity*2*2204.62,1)
            bol_vehicle=vehicle_id
            bol_batch=batch
            bol_grade=wrap
            bol_dryness=f"{info[current_release_order][current_sales_order]['dryness']}%"
            bol_mt=quantity*2
            bol_bales=quantity*8
            bol_destination=info[current_release_order]["destination"]
           
            
            def create_pdf():
                mill_data=json.loads(gcp_download(target_bucket,rf"new_mill_database.json"))
                
                # Create a PDF document
                buffer = io.BytesIO()
                c = canvas.Canvas("example.pdf", pagesize=letter)
                
                # Set title and draw text
                c.setTitle("Sample PDF")
                c.setFont("Helvetica", 16)  # Set the default font family and size
                c.drawString(30, 750, "STRAIGHT BILL OF LADING")
                c.setFont("Times-Bold", 16)
                
                c.drawString(420, 750, f"BOL #: {bol}")
                
                
                
                data1 = [
                    ['VESSEL', vessel,'LOAD DATE',f'{bol_date}'],
                    [ 'OBL', f'{bol_obl}','GROSS WEIGHT',f'{bol_weight}'],
                    [ 'RELEASE ORDER',f'{bol_ro}','VEHICLE ID',f'{bol_vehicle}'],
                    [ 'CARRIER', f'{bol_carrier}','ADMT',f'{bol_admt}'],
                ]
            
                # Create a table
                col_widths = [100, 200, 100, 135]
                table1 = Table(data1,colWidths=col_widths,rowHeights=25)
            
                # Add style to the table
                style = TableStyle([
                                    ('BACKGROUND', (0, 0), (0, 3), (0.8, 0.7, 0.6)),  # Background color
                                    ('BACKGROUND', (2, 0), (-2, -1), (0.8, 0.7, 0.6)),  # Background color
                                    ('INNERGRID', (0, 0), (-1, -1), 0.25, (0.2, 0.2, 0.2)),  # Inner grid style
                                    ('BOX', (0, 0), (-1, -1), 0.25, (0.2, 0.2, 0.2)),  # Box style
                                    ('WORDWRAP', (3, 1), (3, -1), True),  # Enable text wrapping
                                ])
                table1.setStyle(style)
                # Draw the table on the canvas
                table1.wrapOn(c, 0, 0)
                table1.drawOn(c, 30, 630)  # Position the table at (100, 500)
                
                # Draw a rectangle
                c.rect(30, 525 ,270, 100)
                c.rect(300, 525 ,270, 100)
                
                c.setFont("Times-Bold", 13)
                c.drawString(35, 610, "SHIP FROM :")
                c.drawString(315, 610, "SHIP TO :")
                c.setFont("Helvetica", 13)
                c.drawString(120, 590, "Port Of Olympia")
                c.drawString(120, 570, "915 Washington St NE")
                c.drawString(120, 550, "Olympia,WA 98501")
                c.drawString(120, 530, "United States")
                c.drawString(340, 590, mill_data[destination]["BOL_Name"])
                c.drawString(340, 570, mill_data[destination]["BOL_Name_2"])
                c.drawString(340, 550, mill_data[destination]["BOL_Addr_1"])
                c.drawString(340, 530, mill_data[destination]["BOL_Addr_2"])
                
                data2 = [
                    ['Batch #', 'Grade','Dryness','MT','Bales'],
                    [ bol_batch, bol_grade,bol_dryness,bol_mt,bol_bales]
                    
                ]
            
                table2= Table(data2,colWidths=108,rowHeights=25)
                table2.setStyle(TableStyle([('BACKGROUND', (0, 0), (4, 0), (0.8, 0.7, 0.6))]))
                table2.setStyle(style)
                table2.wrapOn(c, 0, 0)
                table2.drawOn(c, 30, 470)  # Position the table at (100, 500)
                
                c.drawString(150, 430, "FSC Certified Products. FSC Mix Credit. SCS-COC-009938")
                
                c.setFont("Helvetica-Bold", 13)
                c.rect(30,300,537,100)
                c.drawString(200, 380, "Truck Inspection Record")
                c.setFont("Helvetica", 13)
                c.drawString(110, 360, "Truck is clean and deemed suitable for transporting product.")
                c.line(50,325,200,325)
                c.line(360,325,500,325)
                c.drawString(115, 310, "Sign")
                c.drawString(400, 310, "Print Name")
                
                
                c.rect(30, 250, 250, 50)
                c.rect(280, 250, 287, 50)
                c.drawString(350, 270, "Corrective action if answer “NO”")
                c.drawString(35, 280, "If trailer is in poor condition, do not load!")
                c.drawString(35, 260,"The Port will contact the customer.")
                
                c.rect(30, 220, 250, 30)
                c.rect(280, 220, 30, 30)
                c.rect(310, 220, 30, 30)
                c.rect(340, 220, 227, 30)
                c.drawString(290, 232, "Y")
                c.drawString(320, 232, "N")
                c.drawString(32, 230, "Is the Trailer clean?")
                
                c.rect(30, 190, 250, 30)
                c.rect(280, 190, 30, 30)
                c.rect(310, 190, 30, 30)
                c.rect(340, 190, 227, 30)
                c.drawString(290, 202, "Y")
                c.drawString(320, 202, "N")
                c.drawString(32, 200, "Is the trailer free of debris?")
                
                c.rect(30, 160, 250, 30)
                c.rect(280, 160, 30, 30)
                c.rect(310, 160, 30, 30)
                c.rect(340, 160, 227, 30)
                c.drawString(290, 172, "Y")
                c.drawString(320, 172, "N")
                c.drawString(32, 170, "Is the trailer free of pest/rodent activity?")
                
                c.rect(30, 130, 250, 30)
                c.rect(280, 130, 30, 30)
                c.rect(310, 130, 30, 30)
                c.rect(340, 130, 227, 30)
                c.drawString(290, 142, "Y")
                c.drawString(320, 142, "N")
                c.drawString(32, 140, "Is the trailer free of odors?")
                
                c.setFont("Helvetica-Bold", 6)
                c.drawString(32, 120,"ATTN Trucker: Your signature will acknowledge receipt of the correct description, (size and grade) AND total amount of ______ pieces. THIS WAREHOUSE will NOT be responsible if shipment")
                c.drawString(32, 110,"of incorrect product is made for the loading and securing of product. Trucker is responsible for providing or approving the load plan and for securing the product for over-the-road transportation.")
                
                c.setFont("Helvetica-Bold", 11)
                data3 = [
                    ['', f'                                                                 {bol_date}'],
                    ['', f'                                                                 {bol_date}']   ]
                
                table3= Table(data3,colWidths=280,rowHeights=30)
                table3.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, 1), (0.8, 0.7, 0.6))]))
                table3.setStyle(style)
                table3.wrapOn(c, 0, 0)
                table3.drawOn(c, 20, 45)  # Position the table at (100, 500)
                c.drawString(32, 82,"Carrier Signature/ Pick up date")
                c.drawString(32, 52,"Warehouse Signature/ Pick up date")
                
                
                
                # Save the PDF document
                c.save()
                # pdf_bytes = buffer.getvalue()
                # buffer.close()
                with open("example.pdf", "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                return PDFbyte
               # return "example.pdf"
            
            #pdf_file = create_pdf()
            
            
            st.download_button(label="CREATE/DOWNLOAD BOL", 
                    data=create_pdf(),
                    file_name=f"{calendar.month_name[datetime.datetime.now().month][:3]}-{datetime.datetime.now().day}-{destination}-{bol}.pdf",
                    mime='application/octet-stream')
            
            if yes and mf:
                
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
                        carrier_code=carrier_code.split("-")[0]
                        try:
                            suzano_report_=gcp_download(target_bucket,rf"suzano_report.json")
                            suzano_report=json.loads(suzano_report_)
                        except:
                            suzano_report={}
                        consignee=destination.split("-")[0]
                        consignee_city=mill_info[destination]["city"]
                        consignee_state=mill_info[destination]["state"]
                        vessel_suzano,voyage_suzano=vessel.split("-")
                        if manual_time:
                            eta=datetime.datetime.strftime(file_date+datetime.timedelta(hours=mill_info[destination]['hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")
                        else:
                            eta=datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=mill_info[destination]['hours']-utc_difference)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")

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
                            edi_name= f'{bill_of_lading_number}.txt'
                            bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                         "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                         "quantity":st.session_state.updated_quantity,"issued":f"{a_} {b_}","edi_no":edi_name,"loads":pure_loads} 
                                            
                        bill_of_ladings=json.dumps(bill_of_ladings)
                        storage_client = storage.Client()
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
                            storage_client = storage.Client()
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
                            info[current_release_order][current_sales_order]["shipped"]=info[current_release_order][current_sales_order]["shipped"]+quantity
                            info[current_release_order][current_sales_order]["remaining"]=info[current_release_order][current_sales_order]["remaining"]-quantity
                        if info[current_release_order][current_sales_order]["remaining"]<=0:
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
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"dispatched.json")
                            blob.upload_from_string(json_data)       
                        
                        json_data = json.dumps(info)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"release_orders/ORDERS/{current_release_order}.json")
                        blob.upload_from_string(json_data)
                        success_container2=st.empty()
                        time.sleep(0.1)                            
                        success_container2.success(f"Updated Release Order {current_release_order}",icon="✅")

                        try:
                            release_order_database=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
                            release_order_database=json.loads(release_order_database)
                        except:
                            release_order_database={}
                       
                        release_order_database[current_release_order][current_sales_order]["remaining"]=release_order_database[current_release_order][current_sales_order]["remaining"]-quantity
                        release_order_database=json.dumps(release_order_database)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                        blob.upload_from_string(release_order_database)
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
                        if load_mf_number_issued:
                            mf_numbers_for_load[release_order_number].remove(load_mf_number)
                            mf_numbers_for_load=json.dumps(mf_numbers_for_load)
                            storage_client = storage.Client()
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
                                    
                        alien_units=json.dumps(alien_units)
                        storage_client = storage.Client()
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
        
        Inventory=gcp_csv_to_df(target_bucket, "kirkenes_with_ghosts_found.csv")
        data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
        bill_of_ladings=json.loads(data)
        mill_info=json.loads(gcp_download(target_bucket,rf"mill_info.json"))
        inv1,inv2,inv3,inv4,inv5,inv6=st.tabs(["DAILY ACTION","SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS","RELEASE ORDER STATUS"])
        with inv1:
            
            daily1,daily2,daily3=st.tabs(["TODAY'SHIPMENTS","TRUCKS ENROUTE","TRUCKS AT DESTINATION"])
            with daily1:
                now=datetime.datetime.now()-datetime.timedelta(hours=7)
                text=f"SHIPPED TODAY ON {datetime.datetime.strftime(now.date(),'%b %d, %Y')} - Indexed By Terminal Bill Of Lading"
                st.markdown(f"<p style='font-family:arial,monospace; color: #0099ff;text-shadow: 2px 2px 4px #99ccff;'>{text}</p>",unsafe_allow_html=True)     
                df_bill=pd.DataFrame(bill_of_ladings).T
                df_bill=df_bill[["vessel","release_order","destination","sales_order","ocean_bill_of_lading","grade","carrier_id","vehicle","quantity","issued"]]
                df_bill.columns=["VESSEL","RELEASE ORDER","DESTINATION","SALES ORDER","OCEAN BILL OF LADING","GRADE","CARRIER ID","VEHICLE NO","QUANTITY (UNITS)","ISSUED"]
                df_bill["Date"]=[None]+[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in df_bill["ISSUED"].values[1:]]
                
                df_today=df_bill[df_bill["Date"]==now.date()]
                df_today.insert(9,"TONNAGE",[i*2 for i in df_today["QUANTITY (UNITS)"]])
                df_today.loc["TOTAL","QUANTITY (UNITS)"]=df_today["QUANTITY (UNITS)"].sum()
                df_today.loc["TOTAL","TONNAGE"]=df_today["TONNAGE"].sum()
                   
                st.dataframe(df_today)

        
            with daily2:
                enroute_vehicles={}
                arrived_vehicles={}
                today_arrived_vehicles={}
                for i in bill_of_ladings:
                    if i!="11502400":
                        date_strings=bill_of_ladings[i]["issued"].split(" ")
                        
                        ship_date=datetime.datetime.strptime(date_strings[0],"%Y-%m-%d")
                        ship_time=datetime.datetime.strptime(date_strings[1],"%H:%M:%S").time()
                        
                        #st.write(bill_of_ladings[i]["issued"])
                        destination=bill_of_ladings[i]['destination']
                        truck=bill_of_ladings[i]['vehicle']
                        distance=mill_info[bill_of_ladings[i]['destination']]["distance"]
                        hours_togo=mill_info[bill_of_ladings[i]['destination']]["hours"]
                        minutes_togo=mill_info[bill_of_ladings[i]['destination']]["minutes"]
                        combined_departure=datetime.datetime.combine(ship_date,ship_time)
                       
                        estimated_arrival=combined_departure+datetime.timedelta(minutes=60*hours_togo+minutes_togo)
                        estimated_arrival_string=datetime.datetime.strftime(estimated_arrival,"%B %d,%Y -- %H:%M")
                        now=datetime.datetime.now()-datetime.timedelta(hours=7)
                        if estimated_arrival>now:
                            
                            enroute_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                             "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}","ETA":estimated_arrival_string}
                        elif estimated_arrival.date()==now.date() and estimated_arrival<now:
                            today_arrived_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                             "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}",
                                                             "ARRIVAL TIME":estimated_arrival_string}
                        else:
                            
                            arrived_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                             "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}",
                                                             "ARRIVAL TIME":estimated_arrival_string,"ARRIVAL":estimated_arrival}
                                    
                arrived_vehicles=pd.DataFrame(arrived_vehicles)
                arrived_vehicles=arrived_vehicles.rename_axis('TRUCK NO')
                arrived_vehicles=arrived_vehicles.T
                arrived_vehicles=arrived_vehicles.sort_values(by="ARRIVAL")
                today_arrived_vehicles=pd.DataFrame(today_arrived_vehicles)
                today_arrived_vehicles=today_arrived_vehicles.rename_axis('TRUCK NO')
                enroute_vehicles=pd.DataFrame(enroute_vehicles)
                enroute_vehicles=enroute_vehicles.rename_axis('TRUCK NO')
                st.dataframe(enroute_vehicles.T)                      
                for i in enroute_vehicles:
                    st.write(f"Truck No : {i} is Enroute to {enroute_vehicles[i]['DESTINATION']} at {enroute_vehicles[i]['ETA']}")
            with daily3:
                select = st.radio(
                                "Select Today's Arrived Vehicles or All Delivered Vehicles",
                                ["TODAY'S ARRIVALS", "ALL ARRIVALS"])
                if select=="TODAY'S ARRIVALS":
                    st.table(today_arrived_vehicles.T)
                if select=="ALL ARRIVALS":
                    
                    st.table(arrived_vehicles.drop(columns=['ARRIVAL']))
        
        with inv2:
            
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')
            try:
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
                                ["DAILY", "ACCUMULATIVE", "FIND BY DATE"])
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
                else:
                    st.dataframe(suzano_report)
                    csv=convert_df(suzano_report)
                    file_name=f'OLYMPIA_ALL_SHIPMENTS to {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%m-%d,%Y")}.csv'
                
                
                
               
                
            
                st.download_button(
                    label="DOWNLOAD REPORT AS CSV",
                    data=csv,
                    file_name=file_name,
                    mime='text/csv')
            except:
                st.write("NO REPORTS RECORDED")
            

        with inv3:
            edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/")
            requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
            try:
                requested_edi=gcp_download(target_bucket, rf"EDIS/{requested_edi_file}")
                st.text_area("EDI",requested_edi,height=400)
                st.download_button(
                    label="DOWNLOAD EDI",
                    data=requested_edi,
                    file_name=f'{requested_edi_file}',
                    mime='text/csv')

            except:
                st.write("NO EDI FILES IN DIRECTORY")
            

            
        with inv4:
            inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
            
            maintenance=False
                            
            if maintenance:
                st.title("CURRENTLY UNDER MAINTENANCE, CHECK BACK LATER")

                           
            else:
                inv4tab1,inv4tab2,inv4tab3=st.tabs(["DAILY SHIPMENT REPORT","INVENTORY","UNREGISTERED LOTS FOUND"])
                with inv4tab1:
                    
                    amount_dict={"KIRKENES-2304":9200,"JUVENTAS-2308":10000}
                    inv_vessel=st.selectbox("Select Vessel",["KIRKENES-2304","JUVENTAS-2308"])
                    kf=inv_bill_of_ladings.iloc[1:].copy()
                    kf['issued'] = pd.to_datetime(kf['issued'])
                    if inv_vessel in kf['vessel']:
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
                    else:
                        st.markdown("No Shipments yet from this vessel!")
                with inv4tab2:
                    combined=gcp_csv_to_df(target_bucket,rf"combined.csv")
                    combined["Batch"]=combined["Batch"].astype(str)
                    
                    #no_of_unaccounted=Inventory[Inventory["Accounted"]==False]["Bales"].sum()/8
                    #st.write(f'**Unaccounted Units Registered : {no_of_unaccounted} Units/{no_of_unaccounted*2} Tons**')
                    
                    temp=inv_bill_of_ladings.groupby("ocean_bill_of_lading")[["quantity"]].sum()
                    temp1=combined.groupby("Ocean B/L")[["Bales","Shipped","Remaining"]].sum()/8
                    temp=pd.merge(temp, temp1, left_index=True, right_index=True, how='outer', suffixes=('_df1', '_df2'))
                    temp=temp[["Bales","quantity","Remaining"]]
                    temp.columns=["Total","Shipped","Remaining"]
                    temp["Remaining"]=temp.Total-temp.Shipped
                    temp.loc["TOTAL"]=temp.sum(axis=0)
                    tempo=temp*2
                    inv_col1,inv_col2=st.columns([2,2])
                    with inv_col1:
                        st.subheader("By Ocean BOL,UNITS")
                        st.dataframe(temp)
                    with inv_col2:
                        st.subheader("By Ocean BOL,TONS")
                        st.dataframe(tempo)
                with inv4tab3:
                    alien_units=json.loads(gcp_download(target_bucket,rf"alien_units.json"))
                    alien_vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304","JUVENTAS-2308"])
                    alien_list=pd.DataFrame(alien_units[alien_vessel]).T
                    alien_list.reset_index(inplace=True)
                    alien_list.index=alien_list.index+1
                    st.dataframe(alien_list)
                    
                    
        with inv5:
            inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
            maintenance=False
            if maintenance:
                st.title("CURRENTLY IN MAINTENANCE, CHECK BACK LATER")
            else:
                st.subheader("WEEKLY SHIPMENTS BY MILL (IN TONS)")
                zf=inv_bill_of_ladings.copy()
                zf['WEEK'] = pd.to_datetime(zf['issued'])
                zf.set_index('WEEK', inplace=True)
                
                def sum_quantity(x):
                    return x.resample('W')['quantity'].sum()*2
                resampled_quantity = zf.groupby('destination').apply(sum_quantity).unstack(level=0)
                resampled_quantity=resampled_quantity.fillna(0)
                resampled_quantity.loc["TOTAL"]=resampled_quantity.sum(axis=0)
                resampled_quantity["TOTAL"]=resampled_quantity.sum(axis=1)
                resampled_quantity=resampled_quantity.reset_index()
                resampled_quantity["WEEK"][:-1]=[i.strftime("%Y-%m-%d") for i in resampled_quantity["WEEK"][:-1]]
                resampled_quantity.set_index("WEEK",drop=True,inplace=True)
                st.dataframe(resampled_quantity)
                csv_weekly=convert_df(resampled_quantity)
                st.download_button(
                label="DOWNLOAD WEEKLY REPORT AS CSV",
                data=csv_weekly,
                file_name=f'WEEKLY SHIPMENT REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                mime='text/csv')


               
                zf['issued'] = pd.to_datetime(zf['issued'])                   
               
                weekly_tonnage = zf.groupby(['destination', pd.Grouper(key='issued', freq='W')])['quantity'].sum() * 2  # Assuming 2 tons per quantity
                weekly_tonnage = weekly_tonnage.reset_index()                   
              
                weekly_tonnage = weekly_tonnage.rename(columns={'issued': 'WEEK', 'quantity': 'Tonnage'})
              
                fig = px.bar(weekly_tonnage, x='WEEK', y='Tonnage', color='destination',
                             title='Weekly Shipments Tonnage per Location',
                             labels={'Tonnage': 'Tonnage (in Tons)', 'WEEK': 'Week'})
             
                fig.update_layout(width=1000, height=700)  # You can adjust the width and height values as needed
                
                st.plotly_chart(fig)

        with inv6:
            inv_bill_of_ladings=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
            inv_bill_of_ladings=pd.read_json(inv_bill_of_ladings).T
            ro=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")
            raw_ro = json.loads(ro)
            grouped_df = inv_bill_of_ladings.groupby(['release_order','sales_order','destination'])[['quantity']].agg(sum)
            info=grouped_df.T.to_dict()
            for rel_ord in raw_ro:
                for sales in raw_ro[rel_ord]:
                    try:
                        found_key = next((key for key in info.keys() if rel_ord in key and sales in key), None)
                        qt=info[found_key]['quantity']
                    except:
                        qt=0
                    info[rel_ord,sales,raw_ro[rel_ord][sales]['destination']]={'total':raw_ro[rel_ord][sales]['total'],
                                            'shipped':qt,'remaining':raw_ro[rel_ord][sales]['remaining']}
                                        
            new=pd.DataFrame(info).T
            new=new.reset_index()
            #new.groupby('level_1')['remaining'].sum()
            new.columns=["Release Order #","Sales Order #","Destination","Total","Shipped","Remaining"]
            new.index=[i+1 for i in new.index]
            #new.columns=["Release Order #","Sales Order #","Destination","Total","Shipped","Remaining"]
            new.index=[i+1 for i in new.index]
            new.loc["Total"]=new[["Total","Shipped","Remaining"]].sum()
            
            release_orders = [str(key[0]) for key in info.keys()]
            release_orders=[str(i) for i in release_orders]
            release_orders = pd.Categorical(release_orders)
            
            total_quantities = [item['total'] for item in info.values()]
            shipped_quantities = [item['shipped'] for item in info.values()]
            remaining_quantities = [item['remaining'] for item in info.values()]
            destinations = [key[2] for key in info.keys()]
            # Calculate the percentage of shipped quantities
            #percentage_shipped = [shipped / total * 100 for shipped, total in zip(shipped_quantities, total_quantities)]
            
            # Create a Plotly bar chart
            fig = go.Figure()
            
            # Add bars for total quantities
            fig.add_trace(go.Bar(x=release_orders, y=total_quantities, name='Total', marker_color='lightgray'))
            
            # Add filled bars for shipped quantities
            fig.add_trace(go.Bar(x=release_orders, y=shipped_quantities, name='Shipped', marker_color='blue', opacity=0.7))
            
            # Add remaining quantities as separate scatter points
            #fig.add_trace(go.Scatter(x=release_orders, y=remaining_quantities, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
            
            remaining_data = [remaining if remaining > 0 else None for remaining in remaining_quantities]
            fig.add_trace(go.Scatter(x=release_orders, y=remaining_data, mode='markers', name='Remaining', marker=dict(color='red', size=10)))
            
            # Add destinations as annotations
            annotations = [dict(x=release_order, y=total_quantity, text=destination, showarrow=True, arrowhead=4, ax=0, ay=-30) for release_order, total_quantity, destination in zip(release_orders, total_quantities, destinations)]
            #fig.update_layout(annotations=annotations)
            
            # Update layout
            fig.update_layout(title='Shipment Status',
                              xaxis_title='Release Orders',
                              yaxis_title='Quantities',
                              barmode='overlay',
                              xaxis=dict(tickangle=-90, type='category'))
            relcol1,relcol2=st.columns([5,5])
            with relcol1:
                st.dataframe(new)
            with relcol2:
                st.plotly_chart(fig)
            
            temp_dict={}
            for rel_ord in raw_ro:
                
                
                for sales in raw_ro[rel_ord]:
                    temp_dict[rel_ord,sales]={}
                    dest=raw_ro[rel_ord][sales]['destination']
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


elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')




    
 
