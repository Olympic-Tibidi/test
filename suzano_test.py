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


import plotly.graph_objects as go
st.set_page_config(layout="wide")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"

target_bucket="olym_suzano_test"
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
    
def gcp_download_x(bucket_name, source_file_name,dest):
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
    df = pd.read_csv(io.BytesIO(data))
    print(f'Pulled down file from bucket {bucket_name}, file name: {source_file_name}')
    return df
def upload_cs_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)
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
def store_release_order_data(vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code):
       
    # Create a dictionary to store the release order data
    release_order_data = { vessel: {
        
        release_order_number:{
        'destination':destination,"po_number":po_number,
        sales_order_item: {
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
    }
                         

    # Convert the dictionary to JSON format
    json_data = json.dumps(release_order_data)
    return json_data

def add_release_order_data(file,vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code):
       
    # Edit the loaded current dictionary.
    file[vessel][release_order_number]["destination"]= destination
    file[vessel][release_order_number]["po_number"]= po_number
    if sales_order_item not in file[vessel][release_order_number]:
        file[vessel][release_order_number][sales_order_item]={}
    file[vessel][release_order_number][sales_order_item]["batch"]= batch
    file[vessel][release_order_number][sales_order_item]["ocean_bill_of_lading"]= ocean_bill_of_lading
    file[vessel][release_order_number][sales_order_item]["grade"]= wrap
    file[vessel][release_order_number][sales_order_item]["dryness"]= dryness
    file[vessel][release_order_number][sales_order_item]["transport_type"]= transport_type
    file[vessel][release_order_number][sales_order_item]["carrier_code"]= carrier_code
    file[vessel][release_order_number][sales_order_item]["unitized"]= unitized
    file[vessel][release_order_number][sales_order_item]["quantity"]= quantity
    file[vessel][release_order_number][sales_order_item]["tonnage"]= tonnage
    file[vessel][release_order_number][sales_order_item]["shipped"]= 0
    file[vessel][release_order_number][sales_order_item]["remaining"]= quantity
    
    
       

    # Convert the dictionary to JSON format
    json_data = json.dumps(file)
    return json_data

def edit_release_order_data(file,sales_order_item,quantity,tonnage,shipped,remaining):
       
    # Edit the loaded current dictionary.
    
    file[vessel][release_order_number][sales_order_item]["quantity"]= quantity
    file[vessel][release_order_number][sales_order_item]["tonnage"]= tonnage
    file[vessel][release_order_number][sales_order_item]["shipped"]= shipped
    file[vessel][release_order_number][sales_order_item]["remaining"]= remaining
    
    
       

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
            loadls.append("2DEV:"+current_release_order+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+i[:-2]+" "*(10-len(i[:-2]))+"0"*16+str(2000))
        for k in second_textsplit:
            loadls.append("2DEV:"+next_release_order+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+k[:-3]+" "*(10-len(k[:-2]))+"0"*16+str(2000))
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

name, authentication_status, username = authenticator.login('PORT OF OLYMPIA TOS LOGIN', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    if username == 'ayilmaz' or username=='gatehouse':
        st.subheader("PORT OF OLYMPIA TOS")
        st.write(f'Welcome *{name}*')
        select=st.sidebar.radio("SELECT FUNCTION",
            ('ADMIN', 'LOADOUT', 'INVENTORY','DATA BACKUP'))
        
            #tab1,tab2,tab3,tab4= st.tabs(["UPLOAD SHIPMENT FILE","ENTER LOADOUT DATA","INVENTORY","CAPTURE"])
            
        if select=="DATA BACKUP" :
            st.write(datetime.datetime.now()-datetime.timedelta(hours=utc_difference))
            try_lan=True
            if try_lan:
                assessment_rates=gcp_download(target_bucket,rf"occ_codes.json")
                assessment_rates=json.loads(assessment_rates)
                occ_codes=pd.DataFrame(assessment_rates).T
                occ_codes=occ_codes.rename_axis('Occ_Code')
                occ_codes=occ_codes.reset_index().set_index(["DESCRIPTION","Occ_Code"],drop=True)
                st.write(occ_codes)
               
                
                
                if "scores" not in st.session_state:
                    st.session_state.scores = pd.DataFrame(
                        {"Code": [], "Shift":[],"Quantity": [], "Hours": [], "OT": []}
                    )
                ref={"DAY":["1ST","1OT"],"NIGHT":["2ST","2OT"],"WEEKEND":["2OT","2OT"]}
                # Function to add a new score to the DataFrame
                def new_scores():
                    total_hours=st.session_state.hours+st.session_state.ot
                    hour_cost=st.session_state.hours*occ_codes.loc[st.session_state.code,ref[shift][0]]
                    ot_cost=st.session_state.ot*occ_codes.loc[st.session_state.code,ref[shift][1]]
                    pension=2.81 if st.session_state.code=="FOREMAN - DOCK" else 0.75
                    wage_cost=hour_cost+ot_cost
                    benefits=wage_cost*0.062+wage_cost*0.0145+wage_cost*0.0021792+wage_cost*st.session_state.siu/100+total_hours*1.58+total_hours*0.14+total_hours*29.15+total_hours*pension
                    new_score = pd.DataFrame(
                        {
                            "Code": [st.session_state.code],
                            "Shift": [st.session_state.shift],
                            "Quantity": [st.session_state.qty],
                            "Hours": [st.session_state.hours],
                            "OT": [st.session_state.ot],
                            "Hour Cost": hour_cost,
                            "OT Cost": ot_cost,
                            "Total Wage": wage_cost,
                            "Benefits&PMA":benefits,
                            "TOTAL COST":wage_cost+benefits,
                            "Mark UP":(wage_cost+benefits)*st.session_state.markup/100,
                            "INVOICE":wage_cost+benefits+((wage_cost+benefits)*st.session_state.markup/100)
                            
                        }
                    )
                    st.session_state.scores = pd.concat(
                        [st.session_state.scores, new_score], ignore_index=True
                    )
                
               
                
                # Form for adding a new score
                st.write("# Add a New Rank")
                with st.form("new_score_form"):

                    st.session_state.siu=st.number_input("ENTER SIU PERCENTAGE",step=1,key="kdsha")
                    st.session_state.markup=st.number_input("ENTER MARKUP",step=1,key="wer")
                    st.session_state.shift=st.selectbox("SELECT SHIFT",["DAY","NIGHT","WEEKEND"])

                    
                    # Dropdown for selecting Code
                    st.session_state.code = st.selectbox(
                        "Occupation Code", options=list(occ_codes.index)
                    )
                
                    # Number input for Quantity
                    st.session_state.qty = st.number_input(
                        "Quantity", step=1, value=0, min_value=0
                    )
                
                    # Number input for Hours
                    st.session_state.hours = st.number_input(
                        "Hours", step=1, value=0, min_value=0
                    )
                
                    # Number input for OT
                    st.session_state.ot = st.number_input(
                        "OT", step=1, value=0, min_value=0
                    )
                
                    # Form submit button
                    submitted = st.form_submit_button("Submit")
                
                # If form is submitted, add the new score
                if submitted:
                    new_scores()
                    st.success("Rank added successfully!")
                
                # Display the updated DataFrame
                st.write("# Updated Cost Table")
                st.dataframe(pd.DataFrame(st.session_state.scores))
                
                index=st.number_input("Enter Index To Delete",step=1,key="1224aa")
                if st.button("DELETE BY INDEX"):
                    st.session_state.scores=st.session_state.scores.drop(index)
                    st.session_state.scores.reset_index(drop=True,inplace=True)
            def download_files_in_folder(bucket, folder_name, output_directory):
                blob_iterator = bucket.list_blobs(prefix=folder_name)
            
                for blob in blob_iterator:
                    # Skip folders (objects ending with '/')
                    if blob.name.endswith('/'):
                        continue
            
                    # Download the file to the specified output directory
                    output_path = os.path.join(output_directory, os.path.basename(blob.name))
                    blob.download_to_filename(output_path)

            if st.button("BACKUP DATA"):
                st.write("OK")
                client = storage.Client()
                bucket = client.bucket(target_bucket)
            
                list_files_to_download = ['dispatched.json','terminal_bill_of_ladings.json','truck_schedule.xlsx','suzano_report.json',
                                          'mill_progress.json', 'Inventory.csv']
                
                # Create a temporary directory to store the downloaded files
                with st.spinner("Downloading files..."):
                    for file_to_download in list_files_to_download:
                        blob = bucket.blob(file_to_download)
                        blob.download_to_filename(f'./{blob.name}')
                
                
            
            
            
              
        if select=="ADMIN" :
            admin_tab1,admin_tab2,admin_tab3,admin_tab4,admin_tab5=st.tabs(["RELEASE ORDERS","BILL OF LADINGS","EDI'S","VESSEL SHIPMENT FILES","LABOR"])
            with admin_tab5:
                labor_issue=True
                if labor_issue:
                    ranks={"FOREMAN":{"code":"129","DAY":{"hr":74.04,"ot":111.06},"NIGHT":{"hr":98.72,"ot":111.06},"WEEKEND":{"hr":111.06,"ot":111.06},
                                      "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                           "COMPUTER CLERK":{"code":"115","DAY":{"hr":57.52,"ot":86.28},"NIGHT":{"hr":76.69,"ot":86.28},"WEEKEND":{"hr":86.28,"ot":86.28},
                                    "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                           "TRACTOR-SEMI-DOCK":{"code":"036","DAY":{"hr":55.25,"ot":82.88},"NIGHT":{"hr":73.67,"ot":82.88},"WEEKEND":{"hr":82.88,"ot":82.88},
                                                "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                           "UTILITY LIFT DRIVER":{"code":"037","DAY":{"hr":55.25,"ot":82.88},"NIGHT":{"hr":73.67,"ot":82.88},"WEEKEND":{"hr":82.88,"ot":82.88},
                                                  "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                           "LIFT TRUCK HEAVY":{"code":"055","DAY":{"hr":57.52,"ot":86.28},"NIGHT":{"hr":76.69,"ot":86.28},"WEEKEND":{"hr":86.28,"ot":86.28},
                                               "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                           "BASIC CLERK-DOCK":{"code":"101","DAY":{"hr":52.85,"ot":79.28},"NIGHT":{"hr":70.47,"ot":79.28},"WEEKEND":{"hr":79.28,"ot":79.28},
                                               "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                           "LINESMAN":{"code":"213","DAY":{"hr":52.85,"ot":79.28},"NIGHT":{"hr":70.47,"ot":79.28},"WEEKEND":{"hr":79.28,"ot":79.28},
                                       "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0},
                          "CLERK":{"code":"103","DAY":{"hr":55.25,"ot":82.88},"NIGHT":{"hr":73.67,"ot":82.88},"WEEKEND":{"hr":82.88,"ot":82.88},
                                                "qt":0,"shift":None,"hours":0,"ot":0,"standard_wage":0,"ot_wage":0,"cost":0,"state":0,"pma":0}}
                    assessment_rates=gcp_download(target_bucket,rf"assessment_rates.json")
                    assessment_rates=json.loads(assessment_rates)
                    #st.write(assessment_rates)
                    siu=st.number_input("ENTER SIU PERCENTAGE",step=1,key="32324")
                    markup=st.number_input("ENTER MARKUP PERCENTAGE",step=1,key="3e2324")
                    lab_col7,lab_col8,lab_col9,lab_col10,lab_col11,lab_col12=st.columns([3,2,1,1,1,1])
                    with lab_col10:
                        st.markdown("**BENEFITS**")
                    with lab_col7:
                        shift=st.radio("**SELECT SHIFT**",["DAY","NIGHT","WEEKEND"],horizontal=True)
                        for i in ranks:
                            ranks[i]["shift"]=shift
                    with lab_col8:
                        st.markdown("**HOURS/OT**")
                    with lab_col9:
                        st.markdown("**WAGES**")
                    
                    with lab_col11:
                        st.markdown("**TOTAL**")

                    with lab_col12:
                        st.markdown("**WITH MARKUP**")
                    
                    lab_col1,lab_col2,lab_col3,lab_col4,lab_col5,lab_col6,lab_colend=st.columns([1,1,2,2,1,1,1])
                    with lab_col4:
                            
                            st.write(" ")
                            st.write(" ")
                                          
                    with lab_col1:
                        st.write(" ")
                        st.write(" ")
                        
                        foreman=st.checkbox("FOREMAN")
                        
                        if foreman:
                            with lab_col2:
                                number_of_foreman=st.number_input("How Many?",step=1,key="1150368")
                                ranks["FOREMAN"]["qt"]=number_of_foreman
                            with lab_col3:
                                a1,a2=st.columns([2,2])
                                with a1:
                                    hours=st.number_input("HOURS",step=1,key="1150369")
                                    ranks["FOREMAN"]["hours"]=hours
                                    ranks["FOREMAN"]["standard_wage"]=hours*ranks["FOREMAN"][shift]["hr"]*ranks["FOREMAN"]["qt"]
                                with a2:
                                    ot=st.number_input("OT",step=1,key="1150370")
                                    ranks["FOREMAN"]["ot"]=ot
                                    ranks["FOREMAN"]["ot_wage"]=ot*ranks["FOREMAN"][shift]["ot"]*ranks["FOREMAN"]["qt"]
                                ranks["FOREMAN"]["cost"]=ranks["FOREMAN"]["standard_wage"]+ranks["FOREMAN"]["ot_wage"]
                                ranks["FOREMAN"]["state"]=ranks["FOREMAN"]["cost"]*0.062+ranks["FOREMAN"]["cost"]*0.0145+ranks["FOREMAN"]["cost"]*0.0021792+ranks["FOREMAN"]["cost"]*siu/100
                                ranks["FOREMAN"]["pma"]=(hours+ot)*1.58*ranks["FOREMAN"]["qt"]+(hours+ot)*0.14*ranks["FOREMAN"]["qt"]+(hours+ot)*29.15*ranks["FOREMAN"]["qt"]+(hours+ot)*2.81*ranks["FOREMAN"]["qt"]
                            with lab_col4:
                                st.markdown(f"Standard:${ranks['FOREMAN']['standard_wage']}, OT:${ranks['FOREMAN']['ot_wage']}")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col5:
                                st.write(" ")
                                st.write(" ")
                                st.write(f'${round(ranks["FOREMAN"]["state"]+ranks["FOREMAN"]["pma"],2)}')
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col6:
                                st.write(" ")
                                st.write(" ")
                                st.write(f"$ {round(ranks['FOREMAN']['cost']+ranks['FOREMAN']['state']+ranks['FOREMAN']['pma'],2)}")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_colend:
                                st.write(" ")
                                st.write(" ")
                                st.write(f"$ {round((markup+100)/100*(ranks['FOREMAN']['cost']+ranks['FOREMAN']['state']+ranks['FOREMAN']['pma']),2)}")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        elif not foreman:
                            with lab_col2:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        
                            with lab_col3:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col4:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col5:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col6:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_colend:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        clerk=st.checkbox("CLERK")
                        if clerk:
                            with lab_col2:
                                number_of_clerk=st.number_input("How Many?",step=1,key="115071")
                                ranks["CLERK"]["qt"]=number_of_clerk
                            with lab_col3:
                                b1,b2=st.columns([2,2])
                                with b1:
                                    hours=st.number_input("HOURS",step=1,key="1150372")
                                    ranks["CLERK"]["hours"]=hours
                                    ranks["CLERK"]["standard_wage"]=hours*ranks["CLERK"][shift]["hr"]*ranks["CLERK"]["qt"]
                                with b2:
                                    ot=st.number_input("OT",step=1,key="1150373")
                                    ranks["CLERK"]["ot"]=ot
                                    ranks["CLERK"]["ot_wage"]=ot*ranks["CLERK"][shift]["ot"]*ranks["CLERK"]["qt"]
                                ranks["CLERK"]["cost"]=ranks["CLERK"]["standard_wage"]+ranks["CLERK"]["ot_wage"]
                                ranks["CLERK"]["state"]=ranks["CLERK"]["cost"]*0.062+ranks["CLERK"]["cost"]*0.0145+ranks["CLERK"]["cost"]*0.0021792+ranks["CLERK"]["cost"]*siu/100
                                ranks["CLERK"]["pma"]=(hours+ot)*1.58*ranks["CLERK"]["qt"]+(hours+ot)*0.14*ranks["CLERK"]["qt"]+(hours+ot)*29.15*ranks["CLERK"]["qt"]+(hours+ot)*0.75*ranks["CLERK"]["qt"]
                            with lab_col4:
                                st.write(f"Standard: $ {ranks['CLERK']['standard_wage']}, OT: $ {ranks['CLERK']['ot_wage']}")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col5:
                                st.write(f'${round(ranks["CLERK"]["state"]+ranks["CLERK"]["pma"],2)}')
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col6:
                                st.write(f"$ {round(ranks['CLERK']['cost']+ranks['CLERK']['state']+ranks['CLERK']['pma'],2)}")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_colend:
                                st.write(f"$ {round((markup+100)/100*(ranks['CLERK']['cost']+ranks['CLERK']['state']+ranks['CLERK']['pma']),2)}")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        elif not clerk:
                            with lab_col2:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        
                            with lab_col3:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col4:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col5:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col6:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_colend:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            
                            
                                
                        st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        heavy=st.checkbox("HEAVY")
                        if heavy:
                            with lab_col2:
                                number_of_heavy=st.number_input("How Many?",step=1,key="11503748")
                                ranks["LIFT TRUCK HEAVY"]["qt"]=number_of_heavy
                            with lab_col3:
                                b1,b2=st.columns([2,2])
                                with b1:
                                    hours=st.number_input("HOURS",step=1,key="1150375sa")
                                    ranks["LIFT TRUCK HEAVY"]["hours"]=hours
                                    ranks["LIFT TRUCK HEAVY"]["standard_wage"]=hours*ranks["LIFT TRUCK HEAVY"][shift]["hr"]*ranks["LIFT TRUCK HEAVY"]["qt"]
                                with b2:
                                    ot=st.number_input("OT",step=1,key="1150376")
                                    ranks["LIFT TRUCK HEAVY"]["ot"]=ot
                                    ranks["LIFT TRUCK HEAVY"]["ot_wage"]=ot*ranks["LIFT TRUCK HEAVY"][shift]["ot"]*ranks["LIFT TRUCK HEAVY"]["qt"]
                                ranks["LIFT TRUCK HEAVY"]["cost"]=ranks["LIFT TRUCK HEAVY"]["standard_wage"]+ranks["LIFT TRUCK HEAVY"]["ot_wage"]
                                ranks["LIFT TRUCK HEAVY"]["state"]=ranks["LIFT TRUCK HEAVY"]["cost"]*0.062+ranks["LIFT TRUCK HEAVY"]["cost"]*0.0145+ranks["LIFT TRUCK HEAVY"]["cost"]*0.0021792+ranks["LIFT TRUCK HEAVY"]["cost"]*siu/100
                                ranks["LIFT TRUCK HEAVY"]["pma"]=(hours+ot)*1.58*ranks["LIFT TRUCK HEAVY"]["qt"]+(hours+ot)*0.14*ranks["LIFT TRUCK HEAVY"]["qt"]+(hours+ot)*29.15*ranks["LIFT TRUCK HEAVY"]["qt"]+(hours+ot)*0.75*ranks["LIFT TRUCK HEAVY"]["qt"]
                                with lab_col4:
                                    st.write(f"Standard: $ {ranks['LIFT TRUCK HEAVY']['standard_wage']}, OT: $ {ranks['LIFT TRUCK HEAVY']['ot_wage']}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_col5:
                                    st.write(f'${round(ranks["LIFT TRUCK HEAVY"]["state"]+ranks["LIFT TRUCK HEAVY"]["pma"],2)}')
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_col6:
                                    st.write(f"$ {round(ranks['LIFT TRUCK HEAVY']['cost']+ranks['LIFT TRUCK HEAVY']['state']+ranks['LIFT TRUCK HEAVY']['pma'],2)}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_colend:
                                    st.write(f"$ {round((markup+100)/100*(ranks['LIFT TRUCK HEAVY']['cost']+ranks['LIFT TRUCK HEAVY']['state']+ranks['LIFT TRUCK HEAVY']['pma']),2)}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                        elif not heavy:
                            with lab_col2:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        
                            with lab_col3:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col4:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col5:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col6:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_colend:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        st.write(" ")
                        st.write(" ")  
                        st.write(" ")
                        ut=st.checkbox("UT")
                        if ut:
                            with lab_col2:
                                number_of_ut=st.number_input("How Many?",step=1,key="1150377")
                                ranks["UTILITY LIFT DRIVER"]["qt"]=number_of_ut
                            with lab_col3:
                                b1,b2=st.columns([2,2])
                                with b1:
                                    hours=st.number_input("HOURS",step=1,key="1150378")
                                    ranks["UTILITY LIFT DRIVER"]["hours"]=hours
                                    ranks["UTILITY LIFT DRIVER"]["standard_wage"]=hours*ranks["UTILITY LIFT DRIVER"][shift]["hr"]*ranks["UTILITY LIFT DRIVER"]["qt"]
                                with b2:
                                    ot=st.number_input("OT",step=1,key="1150379")
                                    ranks["UTILITY LIFT DRIVER"]["ot"]=ot
                                    ranks["UTILITY LIFT DRIVER"]["ot_wage"]=ot*ranks["UTILITY LIFT DRIVER"][shift]["ot"]*ranks["UTILITY LIFT DRIVER"]["qt"]
                                ranks["UTILITY LIFT DRIVER"]["cost"]=ranks["UTILITY LIFT DRIVER"]["standard_wage"]+ranks["UTILITY LIFT DRIVER"]["ot_wage"]
                                ranks["UTILITY LIFT DRIVER"]["state"]=ranks["UTILITY LIFT DRIVER"]["cost"]*0.062+ranks["UTILITY LIFT DRIVER"]["cost"]*0.0145+ranks["UTILITY LIFT DRIVER"]["cost"]*0.0021792+ranks["UTILITY LIFT DRIVER"]["cost"]*siu/100
                                ranks["UTILITY LIFT DRIVER"]["pma"]=(hours+ot)*1.58*ranks["UTILITY LIFT DRIVER"]["qt"]+(hours+ot)*0.14*ranks["UTILITY LIFT DRIVER"]["qt"]+(hours+ot)*29.15*ranks["UTILITY LIFT DRIVER"]["qt"]+(hours+ot)*0.75*ranks["UTILITY LIFT DRIVER"]["qt"]
                                with lab_col4:
                                    st.write(f"Standard: $ {ranks['UTILITY LIFT DRIVER']['standard_wage']}, OT: $ {ranks['UTILITY LIFT DRIVER']['ot_wage']}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_col5:
                                    st.write(f'${round(ranks["UTILITY LIFT DRIVER"]["state"]+ranks["UTILITY LIFT DRIVER"]["pma"],2)}')
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_col6:
                                    st.write(f"$ {round(ranks['UTILITY LIFT DRIVER']['cost']+ranks['UTILITY LIFT DRIVER']['state']+ranks['UTILITY LIFT DRIVER']['pma'],2)}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_colend:
                                    st.write(f"$ {round((markup+100)/100*(ranks['UTILITY LIFT DRIVER']['cost']+ranks['UTILITY LIFT DRIVER']['state']+ranks['UTILITY LIFT DRIVER']['pma']),2)}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                
                        elif not ut:
                            with lab_col2:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        
                            with lab_col3:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col4:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col5:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_col6:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                            with lab_colend:
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                                st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        tractor_semi=st.checkbox("TRACTOR-SEMI")
                        if tractor_semi:
                            with lab_col2:
                                number_of_tractor_semi=st.number_input("How Many?",step=1,key="isuada4")
                                ranks["TRACTOR-SEMI-DOCK"]["qt"]=number_of_tractor_semi

                            with lab_col3:
                                b1,b2=st.columns([2,2])
                                with b1:
                                    hours=st.number_input("HOURS",step=1,key="1150380")
                                    ranks["TRACTOR-SEMI-DOCK"]["hours"]=hours
                                    ranks["TRACTOR-SEMI-DOCK"]["standard_wage"]=hours*ranks["TRACTOR-SEMI-DOCK"][shift]["hr"]*ranks["TRACTOR-SEMI-DOCK"]["qt"]
                                with b2:
                                    ot=st.number_input("OT",step=1,key="1150381")
                                    ranks["TRACTOR-SEMI-DOCK"]["ot"]=ot
                                    ranks["TRACTOR-SEMI-DOCK"]["ot_wage"]=ot*ranks["TRACTOR-SEMI-DOCK"][shift]["ot"]*ranks["TRACTOR-SEMI-DOCK"]["qt"]
                                ranks["TRACTOR-SEMI-DOCK"]["cost"]=ranks["TRACTOR-SEMI-DOCK"]["standard_wage"]+ranks["TRACTOR-SEMI-DOCK"]["ot_wage"]
                                ranks["TRACTOR-SEMI-DOCK"]["state"]=ranks["TRACTOR-SEMI-DOCK"]["cost"]*0.062+ranks["TRACTOR-SEMI-DOCK"]["cost"]*0.0145+ranks["TRACTOR-SEMI-DOCK"]["cost"]*0.0021792+ranks["TRACTOR-SEMI-DOCK"]["cost"]*siu/100
                                ranks["TRACTOR-SEMI-DOCK"]["pma"]=(hours+ot)*1.58*ranks["TRACTOR-SEMI-DOCK"]["qt"]+(hours+ot)*0.14*ranks["TRACTOR-SEMI-DOCK"]["qt"]+(hours+ot)*29.15*ranks["TRACTOR-SEMI-DOCK"]["qt"]+(hours+ot)*0.75*ranks["TRACTOR-SEMI-DOCK"]["qt"]
                                with lab_col4:
                                    st.write(f"Standard: $ {ranks['TRACTOR-SEMI-DOCK']['standard_wage']}, OT: $ {ranks['TRACTOR-SEMI-DOCK']['ot_wage']}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_col5:
                                    st.write(f'${round(ranks["TRACTOR-SEMI-DOCK"]["state"]+ranks["TRACTOR-SEMI-DOCK"]["pma"],2)}')
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_col6:
                                    st.write(f"$ {round(ranks['TRACTOR-SEMI-DOCK']['cost']+ranks['TRACTOR-SEMI-DOCK']['state']+ranks['TRACTOR-SEMI-DOCK']['pma'],2)}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                with lab_colend:
                                    st.write(f"$ {round((markup+100)/100*(ranks['TRACTOR-SEMI-DOCK']['cost']+ranks['TRACTOR-SEMI-DOCK']['state']+ranks['TRACTOR-SEMI-DOCK']['pma']),2)}")
                                    st.write(" ")
                                    st.write(" ")
                                    st.write(" ")
                                
                        
                   
                    total_=round(sum([ranks[key]['cost'] for key in ranks])+sum([ranks[key]['state'] for key in ranks])+sum([ranks[key]['pma'] for key in ranks]),2)
                    st.subheader(f"     TOTAL COST FOR SHIFT: $ {round(total_,2)}")
                    st.subheader(f"     TOTAL CHARGE WITH MARKUP: $ {round((100+markup)/100*total_,2)}")
                            
                        
                        
                
            
            
            with admin_tab2:
                bill_data=gcp_download(target_bucket,rf"terminal_bill_of_ladings.json")
                admin_bill_of_ladings=json.loads(bill_data)
                admin_bill_of_ladings=pd.DataFrame.from_dict(admin_bill_of_ladings).T[1:]
                @st.cache
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
            with admin_tab3:
                edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/KIRKENES-2304/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                try:
                    requested_edi=gcp_download(target_bucket, rf"EDIS/KIRKENES-2304/{requested_edi_file}")
                    st.text_area("EDI",requested_edi,height=400)                                
                   
                    st.download_button(
                        label="DOWNLOAD EDI",
                        data=requested_edi,
                        file_name=f'{requested_edi_file}',
                        mime='text/csv')

                except:
                    st.write("NO EDI FILES IN DIRECTORY")
                                                                                 
            
                            

            with admin_tab4:
                st.markdown("SHIPMENT FILES")
                shipment_tab1,shipment_tab2=st.tabs(["UPLOAD/PROCESS SHIPMENT FILE","SHIPMENT FILE DATABASE"])
                with shipment_tab1:
                    
                    uploaded_file = st.file_uploader("Choose a file")
                    if uploaded_file is not None:
                        # To read file as bytes:
                        bytes_data = uploaded_file.getvalue()
                        #st.write(bytes_data)
                    
                        # To convert to a string based IO:
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        #st.write(stringio)
                    
                        # To read file as string:
                        string_data = stringio.read()
                        #st.write(string_data)
                    
                        # Can be used wherever a "file-like" object is accepted:
                        temp = pd.read_csv(uploaded_file,header=None)
                        temp=temp[1:-1]
                        gemi=temp[5].unique()[0]
                        voyage=int(temp[6].unique()[0])
                        df=pd.DataFrame(list(zip([i[5:] for i in temp[0]],[str(i)[13:15] for i in temp[7]],
                                  [str(i)[20:28] for i in temp[7]])),columns=["Lot","Lot Qty","Batch"])
                        df["Lot Qty"]=[int(int(i)/2) for i in df["Lot Qty"]]
                        df["Grade"]=[i[:3] for i in temp[1]]
                        df["Vessel"]=[i[-12:] for i in temp[7]]
                        df["DryWeight"]=[int(i) for i in temp[8]]
                        df["ADMT"]=[int(i)/0.9/100000 for i in temp[8]]
                        new_list=[]
                        lotq=[]
                        batch=[]
                        wrap=[]
                        vessel=[]
                        DryWeight=[]
                        ADMT=[]
                        for i in df.index:
                            #print(df.loc[i,"Lot"])
                            for j in range(1,df.loc[i,"Lot Qty"]+1):
                                #print(f"00{i}")
                                if j<10:
                                    new_list.append(f"{df.loc[i,'Lot']}00{j}")
                                else:
                                    new_list.append(f"{df.loc[i,'Lot']}0{j}")
                                lotq.append(df.loc[i,"Lot Qty"])
                                batch.append(str(df.loc[i,"Batch"]))
                                wrap.append(df.loc[i,"Grade"])
                                vessel.append(df.loc[i,"Vessel"])
                                DryWeight.append(df.loc[i,"DryWeight"])
                                ADMT.append(df.loc[i,"ADMT"])
                        new_df=pd.DataFrame(list(zip(new_list,lotq,batch,wrap,vessel,DryWeight,ADMT)),columns=df.columns.to_list())
                        new_df["Location"]="OLYM"
                        new_df["Warehouse_In"]="8/24/2023"
                        new_df["Warehouse_Out"]=""
                        new_df["Vehicle_Id"]=""
                        new_df["Release_Order_Number"]=""
                        new_df["Carrier_Code"]=""
                        new_df["BL"]=""
                        bls=new_df["Batch"].value_counts()
                        wraps=[new_df[new_df["Batch"]==k]["Grade"].unique()[0] for k in bls.keys()]
                        wrap_dict={"ISU":"Unwrapped","ISP":"Wrapped"}
                        col1, col2= st.columns([2,2])
                        with col1:
                            st.markdown(f"**VESSEL = {gemi} - VOYAGE = {voyage}**")
                            st.markdown(f"**TOTAL UNITS = {len(new_df)}**")
                        
                       
                            for i in range(len(bls)):
                                st.markdown(f"**{bls[i]} units of Bill of Lading {bls.keys()[i]} - -{wrap_dict[wraps[i]]}-{wraps[i]}**")
                        with col2:
                            if st.button("RECORD PARSED SHIPMENT TO DATABASE"):
                                temp=new_df.to_csv("temp.csv")
                                upload_cs_file(target_bucket, 'temp.csv',rf"shipping_files/{gemi}-{voyage}-shipping_file.csv") 
                                st.write(f"Uploaded {gemi}-{voyage}-shipping_file.csv to database")
                        st.dataframe(new_df)
                    with shipment_tab2:
                        folder_name = "olym_suzano/shipping_files"  # Replace this with the folder path you want to read
                        files_in_folder = list_files_in_folder(target_bucket, "shipping_files")
                        requested_file=st.selectbox("SHIPPING FILES IN DATABASE",files_in_folder[1:])
                        if st.button("LOAD SHIPPING FILE"):
                            requested_shipping_file=gcp_csv_to_df(target_bucket,requested_file)
                            filtered_df=requested_shipping_file[["Lot","Lot Qty","Batch","Grade","Ocean B/L","DryWeight","ADMT"]]
                            #st.data_editor(filtered_df, use_container_width=True)
                            st.data_editor(filtered_df)
                          
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
                
              
                release_order_tab1,release_order_tab2=st.tabs(["CREATE RELEASE ORDER","RELEASE ORDER DATABASE"])
                with release_order_tab1:
                    vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304"])
                    add=st.checkbox("CHECK TO ADD TO EXISTING RELEASE ORDER",disabled=True)
                    edit=st.checkbox("CHECK TO EDIT EXISTING RELEASE ORDER")
                    batch_mapping=gcp_download(target_bucket,rf"batch_mapping.json")
                    batch_mapping=json.loads(batch_mapping)
                    if edit:
                        
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in [i.replace(".json","") for i in list_files_in_subfolder(target_bucket, rf"release_orders/KIRKENES-2304/")[1:]] if i not in junk]))
                        to_edit=gcp_download(target_bucket,rf"release_orders/{vessel}/{release_order_number}.json")
                        to_edit=json.loads(to_edit)
                        po_number_edit=st.text_input("PO No",to_edit[vessel][release_order_number]["po_number"],disabled=True)
                        destination_edit=st.text_input("Destination",to_edit[vessel][release_order_number]["destination"],disabled=True)
                        sales_order_item_edit=st.text_input("Sales Order Item",list(to_edit[vessel][release_order_number].keys())[2],disabled=True)
                        ocean_bill_of_lading_edit=st.text_input("Ocean Bill Of Lading",to_edit[vessel][release_order_number][sales_order_item_edit]["ocean_bill_of_lading"],disabled=True)
                        wrap_edit=st.text_input("Grade",to_edit[vessel][release_order_number][sales_order_item_edit]["grade"],disabled=True)
                        batch_edit=st.text_input("Batch No",to_edit[vessel][release_order_number][sales_order_item_edit]["batch"],disabled=True)
                        dryness_edit=st.text_input("Dryness",to_edit[vessel][release_order_number][sales_order_item_edit]["dryness"],disabled=True)
                        admt_edit=st.text_input("ADMT PER UNIT",round(int(batch_mapping[ocean_bill_of_lading_edit]["dryness"])/90,6),disabled=True)
                        unitized_edit=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=True)
                        quantity_edit=st.number_input("Quantity of Units", 0, disabled=False, label_visibility="visible")
                        tonnage_edit=2*quantity_edit
                        shipped_edit=st.number_input("Shipped # of Units",to_edit[vessel][release_order_number][sales_order_item_edit]["shipped"],disabled=True)
                        remaining_edit=st.number_input("Remaining # of Units",
                                                       quantity_edit-to_edit[vessel][release_order_number][sales_order_item_edit]["shipped"],disabled=True)
                    elif add:
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in [i.replace(".json","") for i in list_files_in_subfolder(target_bucket, rf"release_orders/KIRKENES-2304/")] if i not in junk]))
                        
                        
                    else:
                        
                        release_order_number=st.text_input("Release Order Number")
                        po_number=st.text_input("PO No")
                        
                        destination_list=list(set([f"{i}-{j}" for i,j in zip(mill_df["Group"].tolist(),mill_df["Final Destination"].tolist())]))
                        #st.write(destination_list)
                        destination=st.selectbox("SELECT DESTINATION",destination_list)
                        sales_order_item=st.text_input("Sales Order Item")
                        ocean_bill_of_lading=st.selectbox("Ocean Bill Of Lading",batch_mapping.keys())
                        wrap=st.text_input("Grade",batch_mapping[ocean_bill_of_lading]["grade"],disabled=True)
                        batch=st.text_input("Batch No",batch_mapping[ocean_bill_of_lading]["batch"],disabled=True)
                        dryness=st.text_input("Dryness",batch_mapping[ocean_bill_of_lading]["dryness"],disabled=True)
                        admt=st.text_input("ADMT PER UNIT",round(int(batch_mapping[ocean_bill_of_lading]["dryness"])/90,6),disabled=True)
                        unitized=st.selectbox("UNITIZED/DE-UNITIZED",["UNITIZED","DE-UNITIZED"],disabled=False)
                        quantity=st.number_input("Quantity of Units", min_value=1, max_value=5000, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                        tonnage=2*quantity
                        #queue=st.number_input("Place in Queue", min_value=1, max_value=20, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                        transport_type=st.radio("Select Transport Type",("TRUCK","RAIL"))
                        carrier_code=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()])            
                    
        
                    create_release_order=st.button("SUBMIT")
                    if create_release_order:
                        
                        if add: 
                            data=gcp_download(target_bucket,rf"release_orders/{vessel}/{release_order_number}.json")
                            to_edit=json.loads(data)
                            temp=add_release_order_data(to_edit,vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code)
                            st.write(f"ADDED sales order item {sales_order_item} to release order {release_order_number}!")
                        elif edit:
                            data=gcp_download(target_bucket,rf"release_orders/{vessel}/{release_order_number}.json")
                            to_edit=json.loads(data)
                            temp=edit_release_order_data(to_edit,sales_order_item_edit,quantity_edit,tonnage_edit,shipped_edit,remaining_edit)
                            st.write(f"Edited release order {release_order_number} successfully!")
                            
                        else:
                            
                            temp=store_release_order_data(vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code)
                     
                        
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
                            blob = bucket.blob(rf"release_orders/{vessel}/junk_release.json")
                            blob.upload_from_string(jason_data)
                        except:
                            pass
                        

                        storage_client = storage.Client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"release_orders/{vessel}/{release_order_number}.json")
                        blob.upload_from_string(temp)

                        if edit:
                            release_order_database[release_order_number][sales_order_item_edit]={"destination":destination_edit,"total":quantity_edit,"remaining":remaining_edit}
                        else:
                                                    
                            try:
                                release_order_database[release_order_number][sales_order_item]={"destination":destination,"total":quantity,"remaining":quantity}
                                
                            except:
                                
                                release_order_database[release_order_number]={}
                                release_order_database[release_order_number][sales_order_item]={"destination":destination,"total":quantity,"remaining":quantity}
                            st.write(f"Recorded Release Order - {release_order_number} for Item No: {sales_order_item}")
                        release_orders_json=json.dumps(release_order_database)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                        blob.upload_from_string(release_orders_json)
                        
                with release_order_tab2:
                    
                    vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304"],key="other")
                    rls_tab1,rls_tab2,rls_tab3=st.tabs(["ACTIVE RELEASE ORDERS","COMPLETED RELEASE ORDERS","ENTER MF NUMBERS"])
                    data=gcp_download(target_bucket,rf"release_orders/RELEASE_ORDERS.json")  #################
                    try:
                        release_order_dictionary=json.loads(data)
                    except: 
                        release_order_dictionary={}
                    
                    with rls_tab1:
                        
                        completed_release_orders=[]
                        
                        for key in release_order_database:
                            not_yet=0
                            #st.write(key)
                            for sales in release_order_database[key]:
                                #st.write(sales)
                                if release_order_database[key][sales]["remaining"]>0:
                                    not_yet=1
                                else:
                                    pass#st.write(f"{key}{sales} seems to be finished")
                            if not_yet==0:
                                completed_release_orders.append(key)
                        
                        files_in_folder_ = [i.replace(".json","") for i in list_files_in_subfolder(target_bucket, rf"release_orders/{vessel}/")]   ### REMOVE json extension from name
                        
                        junk=gcp_download(target_bucket,rf"junk_release.json")
                        junk=json.loads(junk)
                        files_in_folder=[i for i in files_in_folder_ if i not in completed_release_orders]        ###  CHECK IF COMPLETED
                        files_in_folder=[i for i in files_in_folder if i not in junk.keys()]        ###  CHECK IF COMPLETED
                        release_order_dest_map={}
                        try:
                            
                            for i in release_order_dictionary:
                                for sales in release_order_dictionary[i]:
                                    release_order_dest_map[i]=release_order_dictionary[i][sales]["destination"]
                            
                            destinations_of_release_orders=[f"{i} to {release_order_dest_map[i]}" for i in files_in_folder]
                        
                                                                        
                            requested_file_=st.selectbox("ACTIVE RELEASE ORDERS",destinations_of_release_orders)
                            requested_file=requested_file_.split(" ")[0]
                            nofile=0
                        except:
                            st.write("NO RELEASE ORDERS YET")
                        try:
                            data=gcp_download(target_bucket,rf"release_orders/{vessel}/{requested_file}.json")
                            release_order_json = json.loads(data)
                            
                            
                            target=release_order_json[vessel][requested_file]
                            destination=target['destination']
                            po_number=target["po_number"]
                            if len(target.keys())==0:
                                nofile=1
                           
                            number_of_sales_orders=len(target)    ##### WRONG CAUSE THERE IS NOW DESTINATION KEYS
                    
                        
                        except:
                            nofile=1
                        
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
                                        
                            targets=[i for i in target if i not in ["destination","po_number"]] ####doing this cause we set jason path {downloadedfile[vessel][releaseorder] as target. i have to use one of the keys (release order number) that is in target list
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
                                    bucket = storage_client.bucket("olym_suzano")
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
                                        bucket = storage_client.bucket("olym_suzano")
                                        blob = bucket.blob(rf"dispatched.json")
                                        blob.upload_from_string(json_data)
                                        st.markdown(f"**CLEARED DISPATCH ITEM {item}**")   
                                except:
                                    pass
                            st.markdown("**CURRENT DISPATCH QUEUE**")
                            try:
                                dispatch=gcp_download("olym_suzano",rf"dispatched.json")
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
                            #st.write(key)
                            for sales in release_order_database[key]:
                                #st.write(sales)
                                if release_order_database[key][sales]["remaining"]>0:
                                    not_yet=1
                                else:
                                    pass#st.write(f"{key}{sales} seems to be finished")
                            if not_yet==0:
                                completed_release_orders.append(key)
                        
                        for completed in completed_release_orders:
                            #st.write(completed)
                            data=gcp_download(target_bucket,rf"release_orders/{vessel}/{completed}.json")
                            comp_rel_order=json.loads(data)
                        
                        completed_release_order_dest_map={}
                  
                        for i in release_order_dictionary:
                            if i in completed_release_orders:
                                completed_release_order_dest_map[i]=release_order_dictionary[i][sales]#["destination"]
                        st.write(pd.DataFrame(completed_release_order_dest_map).T)
                        destinations_of_completed_release_orders=[f"{i} to {completed_release_order_dest_map[i]}" for i in completed_release_orders]
                    
                                                                    
                        requested_file_=st.selectbox("COMPLETED RELEASE ORDERS",destinations_of_completed_release_orders,key=16)
                        requested_file=requested_file_.split(" ")[0]
                        nofile=0
                    with rls_tab3:
                        mf_numbers_=gcp_download(target_bucket,rf"release_orders/mf_numbers.json")
                        mf_numbers=json.loads(mf_numbers_)
                        gp_release_orders=[i for i in release_order_database if release_order_database[i]["001"]["destination"] in ["GP-Clatskanie,OR","GP-Halsey,OR"] and release_order_database[i]["001"]["remaining"]>2]
                        vessel_mf=st.selectbox("SELECT VESSEL",["KIRKENES-2304"],key="lalala")
                        destinations_of_release_orders=[f"{i} to {release_order_dest_map[i]}" for i in release_order_database if release_order_database[i]["001"]["destination"] in ["GP-Clatskanie,OR","GP-Halsey,OR"] and release_order_database[i]["001"]["remaining"]>2]
                        release_order_number_mf=st.selectbox("ACTIVE RELEASE ORDERS",destinations_of_release_orders,key="tatata")
                        input_mf_numbers=st.text_area("**ENTER MF NUMBERS**",height=100,key="juy")
                        if input_mf_numbers is not None:
                            input_mf_numbers = input_mf_numbers.splitlines()
                            input_mf_numbers=[i for i in input_mf_numbers if len(i)==10]
                        #st.write(input_mf_numbers)
                        if st.button("SUBMIT MF NUMBERS",key="ioeru" ):
                            if release_order_number_mf[:7] not in mf_numbers[vessel_mf].keys():
                                mf_numbers[vessel_mf][release_order_number_mf[:7]]=[]
                            mf_numbers[vessel_mf][release_order_number_mf[:7]]+=input_mf_numbers
                            mf_numbers[vessel_mf][release_order_number_mf[:7]]=list(set(mf_numbers[vessel_mf][release_order_number_mf[:7]]))
                            mf_data=json.dumps(mf_numbers)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/mf_numbers.json")
                            blob.upload_from_string(mf_data)
                        if st.button("REMOVE MF NUMBERS",key="ioerssu" ):
                            for i in input_mf_numbers:
                                if i in mf_numbers[vessel_mf][release_order_number_mf[:7]]:
                                    mf_numbers[vessel_mf][release_order_number_mf[:7]].remove(i)
                            mf_data=json.dumps(mf_numbers)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/mf_numbers.json")
                            blob.upload_from_string(mf_data)
                        st.table(pd.DataFrame(mf_numbers))
                        
                                
        
                        
        
        ##########  LOAD OUT  ##############
        
        
        
        if select=="LOADOUT" :
        
            
            bill_mapping=gcp_download("olym_suzano","bill_mapping.json")
            bill_mapping=json.loads(bill_mapping)
            mill_info_=gcp_download("olym_suzano",rf"mill_info.json")
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
                            menu_destinations[rel_ord]=dispatched[rel_ord][sales]["destination"]
                            break
                        except:
                            pass
                liste=[f"{i} to {menu_destinations[i]}" for i in dispatched.keys()]
                work_order_=st.selectbox("**SELECT RELEASE ORDER/SALES ORDER TO WORK**",liste)
                work_order=work_order_.split(" ")[0]
                order=["001","002","003","004","005","006"]
                
                for i in order:                   ##############HERE we check all the sales orders in dispatched[releaseordernumber] dictionary. it breaks after it finds the first sales order
                    if i in dispatched[work_order].keys():
                        current_release_order=work_order
                        current_sales_order=i
                        vessel=dispatched[work_order][i]["vessel"]
                        destination=dispatched[work_order][i]['destination']
                        break
                    else:
                        pass
                try:
                    next_release_order=dispatched['002']['release_order']    #############################  CHECK HERE ######################## FOR MIXED LOAD
                    next_sales_order=dispatched['002']['sales_order']
                    
                except:
                    
                    pass
                info=gcp_download(target_bucket,rf"release_orders/{vessel}/{work_order}.json")
                info=json.loads(info)
                
                
                #if st.checkbox("CLICK TO LOAD MIXED SKU"):
                #    try:
                  #      next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                  #      double_load=True
                 #   except:
                   #     st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                    
               
                st.markdown(rf'**:blue[CURRENTLY WORKING] :**')
                load_col1,load_col2,load_col3=st.columns([8,1,1])
                
                with load_col1:
                    wrap_dict={"ISU":"UNWRAPPED","ISP":"WRAPPED"}
                    wrap=info[vessel][current_release_order][current_sales_order]["grade"]
                    ocean_bill_of_=info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"]
                    #st.markdown(f'**Ocean Bill Of Lading : {ocean_bill_of_} - {wrap_dict[wrap]}**')
                    unitized=info[vessel][current_release_order][current_sales_order]["unitized"]
                    #st.markdown(rf'**{info[vessel][current_release_order][current_sales_order]["unitized"]}**')
                    quant_=info[vessel][current_release_order][current_sales_order]["quantity"]
                    real_quant=int(math.floor(quant_))
                    ship_=info[vessel][current_release_order][current_sales_order]["shipped"]
                    ship_bale=(ship_-math.floor(ship_))*8
                    remaining=info[vessel][current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                    temp={f"<b>Release Order #":current_release_order,"<b>Destination":destination,"<b>Sales Order Item":current_sales_order}
                    temp2={"<b>Ocean B/L":ocean_bill_of_,"<b>Type":wrap_dict[wrap],"<b>Prep":unitized}
                    temp3={"<b>Total Units":quant_,"<b>Shipped Units":ship_,"<b>Remaining Units":remaining}
                    #temp4={"<b>Total Bales":0,"<b>Shipped Bales":int(8*(ship_-math.floor(ship_))),"<b>Remaining Bales":int(8*(remaining-math.floor(remaining)))}
                    temp5={"<b>Total Tonnage":quant_*2,"<b>Shipped Tonnage":ship_*2,"<b>Remaining Tonnage":quant_*2-(ship_*2)}


                    
                    sub_load_col1,sub_load_col2,sub_load_col3,sub_load_col4=st.columns([3,3,2,2])
                    
                    with sub_load_col1:   
                        #st.markdown(rf'**Release Order-{current_release_order}**')
                        #st.markdown(rf'**Destination : {destination}**')
                        #st.markdown(rf'**Sales Order Item-{current_sales_order}**')
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


                ###############    LOADOUT DATA ENTRY    #########
                
                col1, col2,col3,col4,col5= st.columns([2,2,2,2,2])
                
              
               
                if info[vessel][current_release_order][current_sales_order]["transport_type"]=="TRUCK":
                    medium="TRUCK"
                else:
                    medium="RAIL"
                
                with col1:
                
                    terminal_code=st.text_input("Terminal Code","OLYM",disabled=True)
                    file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="file_dates",disabled=True)
                    if file_date not in st.session_state:
                        st.session_state.file_date=file_date
                    file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False)
                    delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="delivery_date",disabled=True)
                    eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                    
                with col2:
                    ocean_bol_to_batch = {"GSSWKIR6013D": 45302855,"GSSWKIR6013E": 45305548}
                    if double_load:
                        release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                        sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                        ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                        current_ocean_bill_of_lading=ocean_bill_of_lading
                        next_ocean_bill_of_lading=info[vessel][next_release_order][next_sales_order]["ocean_bill_of_lading"]
                        batch=st.text_input("Batch",info[vessel][current_release_order][current_sales_order]["batch"],disabled=True)
                        grade=st.text_input("Grade",info[vessel][current_release_order][current_sales_order]["grade"],disabled=True)
                        
                        #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                        pass
                    else:
                        release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                        sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                        ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                        
                        batch=st.text_input("Batch",info[vessel][current_release_order][current_sales_order]["batch"],disabled=True)
                        grade=st.text_input("Grade",info[vessel][current_release_order][current_sales_order]["grade"],disabled=True)
                        #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                   
                        
                    
                with col3: 
                    
                    placeholder = st.empty()
                    with placeholder.container():
                        
                        carrier_code=st.text_input("Carrier Code",info[vessel][current_release_order][current_sales_order]["carrier_code"],disabled=True,key=40)
                        transport_sequential_number=st.selectbox("Transport Sequential",["TRUCK","RAIL"],disabled=True,key=51)
                        transport_type=st.selectbox("Transport Type",["TRUCK","RAIL"],disabled=True,key=6)
                        vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=7)
                        mf=True
                        load_mf_number_issued=False
                        if carrier_code=="123456-KBX":
                           if release_order_number in mf_numbers_for_load[vessel].keys():
                               mf_liste=[i for i in mf_numbers_for_load[vessel][release_order_number]]
                               load_mf_number=st.selectbox("MF NUMBER",mf_liste,disabled=False,key=14551)
                               mf=True
                               load_mf_number_issued=True
                           else:
                               st.write("MF NUMBERS NOT IN!")
                               mf=False
                               load_mf_number_issued=False  
                        foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=8)
                        foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=123)

                        click_clear1 = st.button('CLEAR VEHICLE-QUANTITY INPUTS', key=34)
                    if click_clear1:
                         with placeholder.container():
                             
                           carrier_code=st.text_input("Carrier Code",info[vessel][current_release_order][current_sales_order]["carrier_code"],disabled=True,key=9)
                           transport_sequential_number=st.selectbox("Transport Sequential",["TRUCK","RAIL"],disabled=True,key=10)
                           transport_type=st.selectbox("Transport Type",["TRUCK","RAIL"],disabled=True,key=11)
                           vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=12)
                           
                           foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=193)
                           foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=13)


                
                with col4:
                    updated_quantity=0
                    live_quantity=0
                    if updated_quantity not in st.session_state:
                        st.session_state.updated_quantity=updated_quantity
                    def audit_unit(x):
                            if len(x)==10:
                              
                                if bill_mapping[x[:-2]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[x[:-2]]["Batch"]!=batch:
                                    
                                    return False
                                                                                
                                else:
                                    return True
                    def audit_split(release,sales):
                            if len(x)==10:
                                #st.write(bill_mapping[x[:-2]]["Batch"])
                                
                                if bill_mapping[x[:-2]]["Ocean_bl"]!=info[vessel][release][sales]["ocean_bill_of_lading"] and bill_mapping[x[:-2]]["Batch"]!=info[vessel][release][sales]["batch"]:
                                    st.write("**:red[WRONG B/L, DO NOT LOAD BELOW!]**")
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
                        placeholder2 = st.empty()
                        
                        
                        load_input=placeholder1.text_area("**UNITS**",value="",height=300,key=1)#[:-2]
                        bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1111)#[:-2]
                        
                        click_clear = st.button('CLEAR SCANNED INPUTS', key=3)
                        if click_clear:
                            load_input = placeholder1.text_area("**UNITS**",value="",height=300,key=2)#[:-2]
                            bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1121)#[:-2]
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
                        
                    quantity=st.number_input("**Scanned Quantity of Units**",st.session_state.updated_quantity, key=None, help=None, on_change=None, disabled=True, label_visibility="visible")
                    st.markdown(f"**{quantity*2} TONS - {round(quantity*2*2204.62,1)} Pounds**")
                    #ADMT=st.text_input("ADMT",round(info[vessel][current_release_order][current_sales_order]["dryness"]/90,4)*st.session_state.updated_quantity,disabled=True)
                    admt=round(float(info[vessel][current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
                    st.markdown(f"**ADMT= {admt} TONS**")
                    
       
                    
  
                 
                   
                with col5:
                    
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
                            for i,x in enumerate(textsplit):
                                
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
                        if bale_load_input is not None:
                        
                            bale_textsplit = bale_load_input.splitlines()                       
                            bale_textsplit=[i for i in bale_textsplit if len(i)>8]                           
                            seen=set()
                            for i,x in enumerate(bale_textsplit):
                                if audit_unit(x):
                                    if x in textsplit:
                                        st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                        st.write(f"**:red[This number is scanned as a whole UNIT!]**")
                                        bale_faults.append(1)
                                    else:
                                        st.markdown(f"**Bale No : {i+1}-{x}**",unsafe_allow_html=True)
                                        bale_faults.append(0)
                                else:
                                    st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                    st.write(f"**:red[WRONG B/L, DO NOT LOAD UNIT {x}]**")
                                    bale_faults.append(1)
                                seen.add(x)
                       
                           
                        loads={}
                        pure_loads={}
                        yes=True
                        if 1 in faults or 1 in bale_faults:
                            yes=False
                        
                        if yes:
                            pure_loads={**{k:0 for k in textsplit},**{k:0 for k in bale_textsplit}}
                            loads={**{k[:-2]:0 for k in textsplit},**{k[:-2]:0 for k in bale_textsplit}}
                            for k in textsplit:
                                loads[k[:-2]]+=1
                                pure_loads[k]+=1
                            for k in bale_textsplit:
                                loads[k[:-2]]+=0.125
                                pure_loads[k]+=0.125
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
                    
                    
                    
                if yes:
                    
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
                                    bill_of_lading_number=load_mf_number
                                edi_name= f'{bill_of_lading_number}.txt'
                                bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                             "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                             "quantity":st.session_state.updated_quantity,"issued":f"{a_} {b_}","edi_no":edi_name,"loads":pure_loads} 
                                                
                            bill_of_ladings=json.dumps(bill_of_ladings)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket("olym_suzano")
                            blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                            blob.upload_from_string(bill_of_ladings)
                            
                            
                            
                            terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",bill_of_lading_number,disabled=True)
                            process()
                           
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
                                bucket = storage_client.bucket("olym_suzano")
                                blob = bucket.blob(rf"suzano_report.json")
                                blob.upload_from_string(suzano_report)
    
                              
                                
                            if double_load:
                                info[vessel][current_release_order][current_sales_order]["shipped"]=info[vessel][current_release_order][current_sales_order]["shipped"]+len(first_textsplit)
                                info[vessel][current_release_order][current_sales_order]["remaining"]=info[vessel][current_release_order][current_sales_order]["remaining"]-len(first_textsplit)
                                info[vessel][next_release_order][next_sales_order]["shipped"]=info[vessel][next_release_order][next_sales_order]["shipped"]+len(second_textsplit)
                                info[vessel][next_release_order][next_sales_order]["remaining"]=info[vessel][next_release_order][next_sales_order]["remaining"]-len(second_textsplit)
                            else:
                                info[vessel][current_release_order][current_sales_order]["shipped"]=info[vessel][current_release_order][current_sales_order]["shipped"]+quantity
                                info[vessel][current_release_order][current_sales_order]["remaining"]=info[vessel][current_release_order][current_sales_order]["remaining"]-quantity
                            if info[vessel][current_release_order][current_sales_order]["remaining"]<=0:
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
                            blob = bucket.blob(rf"release_orders/{vessel}/{current_release_order}.json")
                            blob.upload_from_string(json_data)
    
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
                            with open('placeholder.txt', 'r') as f:
                                output_text = f.read()
                            st.markdown("**SUCCESS! EDI FOR THIS LOAD HAS BEEN SUBMITTED,THANK YOU**")
                            st.markdown("**EDI TEXT**")
                            st.text_area('', value=output_text, height=600)
                            with open('placeholder.txt', 'r') as f:
                                file_content = f.read()
                            newline="\n"
                            filename = f'{bill_of_lading_number}'
                            file_name= f'{bill_of_lading_number}.txt'
                            st.write(filename)
                            st.write(current_release_order,current_sales_order,destination,ocean_bill_of_lading,terminal_bill_of_lading,wrap)
                            subject = f'Suzano_EDI_{a}_ R.O:{release_order_number}-Terminal BOL :{bill_of_lading_number}-Destination : {destination}'
                            body = f"EDI for Below attached.{newline}Release Order Number : {current_release_order} - Sales Order Number:{current_sales_order}{newline} Destination : {destination} Ocean Bill Of Lading : {ocean_bill_of_lading}{newline}Terminal Bill of Lading: {terminal_bill_of_lading} - Grade : {wrap} {newline}{2*quantity} tons {unitized} cargo were loaded to vehicle : {vehicle_id} with Carried ID : {carrier_code} {newline}Truck loading completed at {a_} {b_}"
                            st.write(body)           
                            sender = "warehouseoly@gmail.com"
                            recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                            #recipients = ["afsiny@portolympia.com"]
                            password = "xjvxkmzbpotzeuuv"
                    
                  
                    
                    
                            with open('temp_file.txt', 'w') as f:
                                f.write(file_content)
                    
                            file_path = 'temp_file.txt'  # Use the path of the temporary file
                    
                            
                            upload_cs_file("olym_suzano", 'temp_file.txt',rf"EDIS/{vessel}/{file_name}")
                            if load_mf_number_issued:
                                mf_numbers_for_load[vessel][release_order_number].remove(load_mf_number)
                                mf_numbers_for_load=json.dumps(mf_numbers_for_load)
                                storage_client = storage.Client()
                                bucket = storage_client.bucket(target_bucket)
                                blob = bucket.blob(rf"release_orders/mf_numbers.json")
                                blob.upload_from_string(mf_numbers_for_load)
                            send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                            
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
            inv1,inv2,inv3,inv4,inv5=st.tabs(["DAILY ACTION","SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS"])
            with inv1:
                
                daily1,daily2,daily3=st.tabs(["TODAY'SHIPMENTS","TRUCKS ENROUTE","TRUCKS AT DESTINATION"])
                with daily1:
                    now=datetime.datetime.now()-datetime.timedelta(hours=7)
                    st.markdown(f"**SHIPPED TODAY ON {datetime.datetime.strftime(now.date(),'%b %d, %Y')} - Indexed By Terminal Bill Of Lading**")     
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
                @st.cache
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
                edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/KIRKENES-2304/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                try:
                    requested_edi=gcp_download(target_bucket, rf"EDIS/KIRKENES-2304/{requested_edi_file}")
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
                    inv4tab1,inv4tab2=st.tabs(["DAILY SHIPMENT REPORT","INVENTORY"])
                    with inv4tab1:
                        
                        kf=inv_bill_of_ladings.iloc[1:].copy()
                        kf['issued'] = pd.to_datetime(kf['issued'])
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
                        merged_df_grouped["Remaining_Units"]=[9200-i for i in merged_df_grouped['Accumulated_Quantity']]
                        merged_df_grouped["Remaining_Tonnage"]=merged_df_grouped["Remaining_Units"]*2
                        merged_df_grouped.rename(columns={'quantity':"Shipped Quantity", 'Accumulated_Quantity':"Shipped Qty To_Date",
                                                          'Accumulated_Tonnage':"Shipped Tonnage To_Date"},inplace=True)
                        merged_df_grouped=merged_df_grouped.reset_index()
                        merged_df_grouped["Date"]=merged_df_grouped['Date'].dt.strftime('%m-%d-%Y, %A')
                        merged_df_grouped=merged_df_grouped.set_index("Date",drop=True)
                        st.dataframe(merged_df_grouped)
                        csv_inventory=convert_df(merged_df_grouped)
                        st.download_button(
                            label="DOWNLOAD INVENTORY REPORT AS CSV",
                            data=csv_inventory,
                            file_name=f'INVENTORY REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                            mime='text/csv')            
                    with inv4tab2:
                        kirkenes_updated=gcp_csv_to_df(target_bucket,rf"kirkenes_updated.csv")
                        kirkenes_updated["Batch"]=kirkenes_updated["Batch"].astype(str)
                        st.write(kirkenes_updated)
                        if st.button("CLICK TO RE-RUN INVENTORY",key="tyuris"):
                            kirkenes_updated=gcp_csv_to_df(target_bucket,rf"kirkenes_with_ghosts_found.csv")
                            for line in inv_bill_of_ladings.loads[1:]:
                                for unit in line.keys():
                                    kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Shipped"]=kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Shipped"]+line[unit]*8
                                    kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Remaining"]=kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Remaining"]-line[unit]*8
                            
                            
                            temp=kirkenes_updated.to_csv("temp.csv",index=False)
                            upload_cs_file(target_bucket, 'temp.csv',rf"kirkenes_updated.csv") 
                        no_of_unaccounted=Inventory[Inventory["Accounted"]==False]["Bales"].sum()/8
                        st.write(f'**Unaccounted Units Registered : {no_of_unaccounted} Units/{no_of_unaccounted*2} Tons**')
                        temp1=kirkenes_updated.groupby("Ocean B/L")[["Bales","Shipped","Remaining"]].sum()/8
                        temp=inv_bill_of_ladings.groupby("ocean_bill_of_lading")[["quantity"]].sum()
                        temp.insert(0,"Total",temp1.Bales.values)
                        temp["Remaining"]=temp.Total-temp.quantity
                        temp.columns=["Total","Shipped","Remaining"]
                        temp.loc["TOTAL"]=temp.sum(axis=0)
                        tempo=temp*2
                        inv_col1,inv_col2=st.columns([2,2])
                        with inv_col1:
                            st.subheader("By Ocean BOL,UNITS")
                            st.dataframe(temp)
                        with inv_col2:
                            st.subheader("By Ocean BOL,TONS")
                            st.dataframe(tempo)
                   
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
                        menu_destinations[rel_ord]=dispatched[rel_ord][sales]["destination"]
                        break
                    except:
                        pass
            liste=[f"{i} to {menu_destinations[i]}" for i in dispatched.keys()]
            work_order_=st.selectbox("**SELECT RELEASE ORDER/SALES ORDER TO WORK**",liste)
            work_order=work_order_.split(" ")[0]
            order=["001","002","003","004","005","006"]
            
            for i in order:                   ##############HERE we check all the sales orders in dispatched[releaseordernumber] dictionary. it breaks after it finds the first sales order
                if i in dispatched[work_order].keys():
                    current_release_order=work_order
                    current_sales_order=i
                    vessel=dispatched[work_order][i]["vessel"]
                    destination=dispatched[work_order][i]['destination']
                    break
                else:
                    pass
            try:
                next_release_order=dispatched['002']['release_order']    #############################  CHECK HERE ######################## FOR MIXED LOAD
                next_sales_order=dispatched['002']['sales_order']
                
            except:
                
                pass
            info=gcp_download("olym_suzano",rf"release_orders/{vessel}/{work_order}.json")
            info=json.loads(info)
            
            
            #if st.checkbox("CLICK TO LOAD MIXED SKU"):
            #    try:
              #      next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
              #      double_load=True
             #   except:
               #     st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                
           
            st.markdown(rf'**:blue[CURRENTLY WORKING] :**')
            load_col1,load_col2,load_col3=st.columns([8,1,1])
            
            with load_col1:
                wrap_dict={"ISU":"UNWRAPPED","ISP":"WRAPPED"}
                wrap=info[vessel][current_release_order][current_sales_order]["grade"]
                ocean_bill_of_=info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"]
                #st.markdown(f'**Ocean Bill Of Lading : {ocean_bill_of_} - {wrap_dict[wrap]}**')
                unitized=info[vessel][current_release_order][current_sales_order]["unitized"]
                #st.markdown(rf'**{info[vessel][current_release_order][current_sales_order]["unitized"]}**')
                quant_=info[vessel][current_release_order][current_sales_order]["quantity"]
                real_quant=int(math.floor(quant_))
                ship_=info[vessel][current_release_order][current_sales_order]["shipped"]
                ship_bale=(ship_-math.floor(ship_))*8
                remaining=info[vessel][current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                temp={f"<b>Release Order #":current_release_order,"<b>Destination":destination,"<b>Sales Order Item":current_sales_order}
                temp2={"<b>Ocean B/L":ocean_bill_of_,"<b>Type":wrap_dict[wrap],"<b>Prep":unitized}
                temp3={"<b>Total Units":quant_,"<b>Shipped Units":ship_,"<b>Remaining Units":remaining}
                #temp4={"<b>Total Bales":0,"<b>Shipped Bales":int(8*(ship_-math.floor(ship_))),"<b>Remaining Bales":int(8*(remaining-math.floor(remaining)))}
                temp5={"<b>Total Tonnage":quant_*2,"<b>Shipped Tonnage":ship_*2,"<b>Remaining Tonnage":quant_*2-(ship_*2)}


                
                sub_load_col1,sub_load_col2,sub_load_col3,sub_load_col4=st.columns([3,3,2,2])
                
                with sub_load_col1:   
                    #st.markdown(rf'**Release Order-{current_release_order}**')
                    #st.markdown(rf'**Destination : {destination}**')
                    #st.markdown(rf'**Sales Order Item-{current_sales_order}**')
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


            ###############    LOADOUT DATA ENTRY    #########
            
            col1, col2,col3,col4,col5= st.columns([2,2,2,2,2])
            
          
           
            if info[vessel][current_release_order][current_sales_order]["transport_type"]=="TRUCK":
                medium="TRUCK"
            else:
                medium="RAIL"
            
            with col1:
            
                terminal_code=st.text_input("Terminal Code","OLYM",disabled=True)
                file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="file_dates",disabled=True)
                if file_date not in st.session_state:
                    st.session_state.file_date=file_date
                file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=utc_difference),step=60,disabled=False)
                delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=utc_difference),key="delivery_date",disabled=True)
                eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                
            with col2:
                ocean_bol_to_batch = {"GSSWKIR6013D": 45302855,"GSSWKIR6013E": 45305548}
                if double_load:
                    release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                    sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                    ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                    current_ocean_bill_of_lading=ocean_bill_of_lading
                    next_ocean_bill_of_lading=info[vessel][next_release_order][next_sales_order]["ocean_bill_of_lading"]
                    batch=st.text_input("Batch",info[vessel][current_release_order][current_sales_order]["batch"],disabled=True)
                    grade=st.text_input("Grade",info[vessel][current_release_order][current_sales_order]["grade"],disabled=True)
                    
                    #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                    pass
                else:
                    release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                    sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                    ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                    
                    batch=st.text_input("Batch",info[vessel][current_release_order][current_sales_order]["batch"],disabled=True)
                    grade=st.text_input("Grade",info[vessel][current_release_order][current_sales_order]["grade"],disabled=True)
                    #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
               
                    
                
            with col3: 
                
                placeholder = st.empty()
                with placeholder.container():
                    
                    carrier_code=st.text_input("Carrier Code",info[vessel][current_release_order][current_sales_order]["carrier_code"],disabled=True,key=40)
                    transport_sequential_number=st.selectbox("Transport Sequential",["TRUCK","RAIL"],disabled=True,key=51)
                    transport_type=st.selectbox("Transport Type",["TRUCK","RAIL"],disabled=True,key=6)
                    vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=7)
                    mf=True
                    load_mf_number_issued=False
                    if carrier_code=="123456-KBX":
                       if release_order_number in mf_numbers_for_load[vessel].keys():
                           mf_liste=[i for i in mf_numbers_for_load[vessel][release_order_number]]
                           load_mf_number=st.selectbox("MF NUMBER",mf_liste,disabled=False,key=14551)
                           mf=True
                           load_mf_number_issued=True
                       else:
                           st.write("MF NUMBERS NOT IN!")
                           mf=False
                           load_mf_number_issued=False  
                    foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=8)
                    foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=123)

                    click_clear1 = st.button('CLEAR VEHICLE-QUANTITY INPUTS', key=34)
                if click_clear1:
                     with placeholder.container():
                         
                       carrier_code=st.text_input("Carrier Code",info[vessel][current_release_order][current_sales_order]["carrier_code"],disabled=True,key=9)
                       transport_sequential_number=st.selectbox("Transport Sequential",["TRUCK","RAIL"],disabled=True,key=10)
                       transport_type=st.selectbox("Transport Type",["TRUCK","RAIL"],disabled=True,key=11)
                       vehicle_id=st.text_input("**:blue[Vehicle ID]**",value="",key=12)
                       
                       foreman_quantity=st.number_input("**:blue[ENTER Quantity of Units]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=193)
                       foreman_bale_quantity=st.number_input("**:blue[ENTER Quantity of Bales]**", min_value=0, max_value=30, value=0, step=1, help=None, on_change=None, disabled=False, label_visibility="visible",key=13)


            
            with col4:
                updated_quantity=0
                live_quantity=0
                if updated_quantity not in st.session_state:
                    st.session_state.updated_quantity=updated_quantity
                def audit_unit(x):
                        if len(x)==10:
                          
                            if bill_mapping[x[:-2]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[x[:-2]]["Batch"]!=batch:
                                
                                return False
                                                                            
                            else:
                                return True
                def audit_split(release,sales):
                        if len(x)==10:
                            #st.write(bill_mapping[x[:-2]]["Batch"])
                            
                            if bill_mapping[x[:-2]]["Ocean_bl"]!=info[vessel][release][sales]["ocean_bill_of_lading"] and bill_mapping[x[:-2]]["Batch"]!=info[vessel][release][sales]["batch"]:
                                st.write("**:red[WRONG B/L, DO NOT LOAD BELOW!]**")
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
                        next_item=gcp_download(target_bucket,rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                        
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
                    placeholder2 = st.empty()
                    
                    
                    load_input=placeholder1.text_area("**UNITS**",value="",height=300,key=1)#[:-2]
                    bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1111)#[:-2]
                    
                    click_clear = st.button('CLEAR SCANNED INPUTS', key=3)
                    if click_clear:
                        load_input = placeholder1.text_area("**UNITS**",value="",height=300,key=2)#[:-2]
                        bale_load_input=placeholder2.text_area("**INDIVIDUAL BALES**",value="",height=300,key=1121)#[:-2]
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
                    
                quantity=st.number_input("**Scanned Quantity of Units**",st.session_state.updated_quantity, key=None, help=None, on_change=None, disabled=True, label_visibility="visible")
                st.markdown(f"**{quantity*2} TONS - {round(quantity*2*2204.62,1)} Pounds**")
                #ADMT=st.text_input("ADMT",round(info[vessel][current_release_order][current_sales_order]["dryness"]/90,4)*st.session_state.updated_quantity,disabled=True)
                admt=round(float(info[vessel][current_release_order][current_sales_order]["dryness"])/90*st.session_state.updated_quantity*2,4)
                st.markdown(f"**ADMT= {admt} TONS**")
                
   
                

             
               
            with col5:
                
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
                        for i,x in enumerate(textsplit):
                            
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
                    if bale_load_input is not None:
                    
                        bale_textsplit = bale_load_input.splitlines()                       
                        bale_textsplit=[i for i in bale_textsplit if len(i)>8]                           
                        seen=set()
                        for i,x in enumerate(bale_textsplit):
                            if audit_unit(x):
                                if x in textsplit:
                                    st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                    st.write(f"**:red[This number is scanned as a whole UNIT!]**")
                                    bale_faults.append(1)
                                else:
                                    st.markdown(f"**Bale No : {i+1}-{x}**",unsafe_allow_html=True)
                                    bale_faults.append(0)
                            else:
                                st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                st.write(f"**:red[WRONG B/L, DO NOT LOAD UNIT {x}]**")
                                bale_faults.append(1)
                            seen.add(x)
                   
                       
                    loads={}
                    pure_loads={}
                    yes=True
                    if 1 in faults or 1 in bale_faults:
                        yes=False
                    
                    if yes:
                        pure_loads={**{k:0 for k in textsplit},**{k:0 for k in bale_textsplit}}
                        loads={**{k[:-2]:0 for k in textsplit},**{k[:-2]:0 for k in bale_textsplit}}
                        for k in textsplit:
                            loads[k[:-2]]+=1
                            pure_loads[k]+=1
                        for k in bale_textsplit:
                            loads[k[:-2]]+=0.125
                            pure_loads[k]+=0.125
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
                
                
                
            if yes:
                
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
                                bill_of_lading_number=load_mf_number
                            edi_name= f'{bill_of_lading_number}.txt'
                            bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"destination":destination,"sales_order":current_sales_order,
                                                                         "ocean_bill_of_lading":ocean_bill_of_lading,"grade":wrap,"carrier_id":carrier_code,"vehicle":vehicle_id,
                                                                         "quantity":st.session_state.updated_quantity,"issued":f"{a_} {b_}","edi_no":edi_name,"loads":pure_loads} 
                                            
                        bill_of_ladings=json.dumps(bill_of_ladings)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                        blob.upload_from_string(bill_of_ladings)
                        
                        
                        
                        terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",bill_of_lading_number,disabled=True)
                        process()
                       
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

                          
                            
                        if double_load:
                            info[vessel][current_release_order][current_sales_order]["shipped"]=info[vessel][current_release_order][current_sales_order]["shipped"]+len(first_textsplit)
                            info[vessel][current_release_order][current_sales_order]["remaining"]=info[vessel][current_release_order][current_sales_order]["remaining"]-len(first_textsplit)
                            info[vessel][next_release_order][next_sales_order]["shipped"]=info[vessel][next_release_order][next_sales_order]["shipped"]+len(second_textsplit)
                            info[vessel][next_release_order][next_sales_order]["remaining"]=info[vessel][next_release_order][next_sales_order]["remaining"]-len(second_textsplit)
                        else:
                            info[vessel][current_release_order][current_sales_order]["shipped"]=info[vessel][current_release_order][current_sales_order]["shipped"]+quantity
                            info[vessel][current_release_order][current_sales_order]["remaining"]=info[vessel][current_release_order][current_sales_order]["remaining"]-quantity
                        if info[vessel][current_release_order][current_sales_order]["remaining"]<=0:
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
                            bucket = storage_client.bucket("olym_suzano")
                            blob = bucket.blob(rf"dispatched.json")
                            blob.upload_from_string(json_data)       
                        
                        json_data = json.dumps(info)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(target_bucket)
                        blob = bucket.blob(rf"release_orders/{vessel}/{current_release_order}.json")
                        blob.upload_from_string(json_data)

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
                        with open('placeholder.txt', 'r') as f:
                            output_text = f.read()
                        st.markdown("**SUCCESS! EDI FOR THIS LOAD HAS BEEN SUBMITTED,THANK YOU**")
                        st.markdown("**EDI TEXT**")
                        st.text_area('', value=output_text, height=600)
                        with open('placeholder.txt', 'r') as f:
                            file_content = f.read()
                        newline="\n"
                        filename = f'{bill_of_lading_number}'
                        file_name= f'{bill_of_lading_number}.txt'
                        st.write(filename)
                        st.write(current_release_order,current_sales_order,destination,ocean_bill_of_lading,terminal_bill_of_lading,wrap)
                        subject = f'Suzano_EDI_{a}_ R.O:{release_order_number}-Terminal BOL :{bill_of_lading_number}-Destination : {destination}'
                        body = f"EDI for Below attached.{newline}Release Order Number : {current_release_order} - Sales Order Number:{current_sales_order}{newline} Destination : {destination} Ocean Bill Of Lading : {ocean_bill_of_lading}{newline}Terminal Bill of Lading: {terminal_bill_of_lading} - Grade : {wrap} {newline}{2*quantity} tons {unitized} cargo were loaded to vehicle : {vehicle_id} with Carried ID : {carrier_code} {newline}Truck loading completed at {a_} {b_}"
                        st.write(body)           
                        sender = "warehouseoly@gmail.com"
                        recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                        #recipients = ["afsiny@portolympia.com"]
                        password = "xjvxkmzbpotzeuuv"
                
              
                
                
                        with open('temp_file.txt', 'w') as f:
                            f.write(file_content)
                
                        file_path = 'temp_file.txt'  # Use the path of the temporary file
                
                        
                        upload_cs_file("olym_suzano", 'temp_file.txt',rf"EDIS/{vessel}/{file_name}")
                        if load_mf_number_issued:
                            mf_numbers_for_load[vessel][release_order_number].remove(load_mf_number)
                            mf_numbers_for_load=json.dumps(mf_numbers_for_load)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(target_bucket)
                            blob = bucket.blob(rf"release_orders/mf_numbers.json")
                            blob.upload_from_string(mf_numbers_for_load)
                        send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                        
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
        inv1,inv2,inv3,inv4,inv5=st.tabs(["DAILY ACTION","SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS"])
        with inv1:
            
            daily1,daily2,daily3=st.tabs(["TODAY'SHIPMENTS","TRUCKS ENROUTE","TRUCKS AT DESTINATION"])
            with daily1:
                now=datetime.datetime.now()-datetime.timedelta(hours=utc_difference)
                st.markdown(f"**SHIPPED TODAY ON {datetime.datetime.strftime(now.date(),'%b %d, %Y')} - Indexed By Terminal Bill Of Lading**")     
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
            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')
            try:
                now=datetime.datetime.now()-datetime.timedelta(hours=7)
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
            edi_files=list_files_in_subfolder(target_bucket, rf"EDIS/KIRKENES-2304/")
            requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
            try:
                requested_edi=gcp_download(target_bucket, rf"EDIS/KIRKENES-2304/{requested_edi_file}")
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
                inv4tab1,inv4tab2=st.tabs(["DAILY SHIPMENT REPORT","INVENTORY"])
                with inv4tab1:
                    
                    kf=inv_bill_of_ladings.iloc[1:].copy()
                    kf['issued'] = pd.to_datetime(kf['issued'])
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
                    merged_df_grouped["Remaining_Units"]=[9200-i for i in merged_df_grouped['Accumulated_Quantity']]
                    merged_df_grouped["Remaining_Tonnage"]=merged_df_grouped["Remaining_Units"]*2
                    merged_df_grouped.rename(columns={'quantity':"Shipped Quantity", 'Accumulated_Quantity':"Shipped Qty To_Date",
                                                      'Accumulated_Tonnage':"Shipped Tonnage To_Date"},inplace=True)
                    merged_df_grouped=merged_df_grouped.reset_index()
                    merged_df_grouped["Date"]=merged_df_grouped['Date'].dt.strftime('%m-%d-%Y, %A')
                    merged_df_grouped=merged_df_grouped.set_index("Date",drop=True)
                    st.dataframe(merged_df_grouped)
                    csv_inventory=convert_df(merged_df_grouped)
                    st.download_button(
                        label="DOWNLOAD INVENTORY REPORT AS CSV",
                        data=csv_inventory,
                        file_name=f'INVENTORY REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
                        mime='text/csv')            
                with inv4tab2:
                    kirkenes_updated=gcp_csv_to_df(target_bucket,rf"kirkenes_updated.csv")
                    kirkenes_updated["Batch"]=kirkenes_updated["Batch"].astype(str)
                    st.write(kirkenes_updated)
                   #if st.button("CLICK TO RE-RUN INVENTORY",key="tyuris"):
                   #     kirkenes_updated=gcp_csv_to_df(target_bucket,rf"kirkenes_with_ghosts_found.csv")
                    #    for line in inv_bill_of_ladings.loads[1:]:
                    #        for unit in line.keys():
                    #            kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Shipped"]=kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Shipped"]+line[unit]*8
                     #           kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Remaining"]=kirkenes_updated.loc[kirkenes_updated["Lot"]==unit[:-2],"Remaining"]-line[unit]*8
                        
                        
                     #   temp=kirkenes_updated.to_csv("temp.csv",index=False)
                     #   upload_cs_file(target_bucket, 'temp.csv',rf"kirkenes_updated.csv") 
                    no_of_unaccounted=Inventory[Inventory["Accounted"]==False]["Bales"].sum()/8
                    st.write(f'**Unaccounted Units Registered : {no_of_unaccounted} Units/{no_of_unaccounted*2} Tons**')
                    temp1=kirkenes_updated.groupby("Ocean B/L")[["Bales","Shipped","Remaining"]].sum()/8
                    temp=inv_bill_of_ladings.groupby("ocean_bill_of_lading")[["quantity"]].sum()
                    temp.insert(0,"Total",temp1.Bales.values)
                    temp["Remaining"]=temp.Total-temp.quantity
                    temp.columns=["Total","Shipped","Remaining"]
                    temp.loc["TOTAL"]=temp.sum(axis=0)
                    tempo=temp*2
                    inv_col1,inv_col2=st.columns([2,2])
                    with inv_col1:
                        st.subheader("By Ocean BOL,UNITS")
                        st.dataframe(temp)
                    with inv_col2:
                        st.subheader("By Ocean BOL,TONS")
                        st.dataframe(tempo)     
                       
                    
                    
               
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
                    file_name=f'OLYMPIA WEEKLY SHIPMENT REPORT-{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=utc_difference),"%Y_%m_%d")}.csv',
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




elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')




    
 
