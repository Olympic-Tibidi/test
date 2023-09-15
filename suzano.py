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

import zipfile


import plotly.graph_objects as go
st.set_page_config(layout="wide")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"

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

def edit_release_order_data(file,vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code):
       
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
    Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
    for i in loads:
        try:              
            Inventory.loc[Inventory["Lot"]==i,"Location"]="PARTIAL"
            Inventory.loc[Inventory["Lot"]==i,"Shipped"]=Inventory.loc[Inventory["Lot"]==i,"Shipped"].values[0]+loads[i]*8
            Inventory.loc[Inventory["Lot"]==i,"Remaining"]=Inventory.loc[Inventory["Lot"]==i,"Remaining"].values[0]-loads[i]*8
            Inventory.loc[Inventory["Lot"]==i,"Warehouse_Out"]=datetime.datetime.combine(file_date,file_time)
            Inventory.loc[Inventory["Lot"]==i,"Vehicle_Id"]=str(vehicle_id)
            Inventory.loc[Inventory["Lot"]==i,"Release_Order_Number"]=str(release_order_number)
            Inventory.loc[Inventory["Lot"]==i,"Carrier_Code"]=str(carrier_code)
            Inventory.loc[Inventory["Lot"]==i,"Terminal B/L"]=str(terminal_bill_of_lading)
        except:
            st.write("Check Unit Number,Unit Not In Inventory")         
        
        temp=Inventory.to_csv("temp.csv")
        upload_cs_file("olym_suzano", 'temp.csv',"Inventory.csv") 
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
    data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
    bill_of_ladings=json.loads(data)
    list_of_ladings=[]
    try:
        for key in bill_of_ladings:
            if int(key) % 2 == 0:
                list_of_ladings.append(int(key))
        bill_of_lading_number=max(list_of_ladings)+2
    except:
        bill_of_lading_number=11502400
    return bill_of_lading_number,bill_of_ladings
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
            pass
            if st.button("BACKUP DATA"):
                pass
              
        if select=="ADMIN" :
            admin_tab1,admin_tab2,admin_tab3,admin_tab4,admin_tab5=st.tabs(["RELEASE ORDERS","BILL OF LADINGS","EDI'S","VESSEL SHIPMENT FILES","MILL SHIPMENTS"])
            with admin_tab2:
                bill_data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
                admin_bill_of_ladings=json.loads(bill_data)
                st.dataframe(pd.DataFrame.from_dict(admin_bill_of_ladings).T[1:])
            with admin_tab3:
                edi_files=list_files_in_subfolder("olym_suzano", rf"EDIS/KIRKENES-2304/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                try:
                    requested_edi=gcp_download("olym_suzano", rf"EDIS/KIRKENES-2304/{requested_edi_file}")
                    st.text_area("EDI",requested_edi,height=400)                                
                   
                    st.download_button(
                        label="DOWNLOAD EDI",
                        data=requested_edi,
                        file_name=f'{requested_edi_file}',
                        mime='text/csv')

                except:
                    st.write("NO EDI FILES IN DIRECTORY")
                                                                                 
            with admin_tab5:
                current_schedule=gcp_csv_to_df("olym_suzano", "truck_schedule.csv")
                mill_shipments=gcp_download("olym_suzano",rf"mill_shipments.json")
                mill_shipments=json.loads(mill_shipments)
                mill_df=pd.DataFrame.from_dict(mill_shipments).T
                mill_df["Terminal Code"]=mill_df["Terminal Code"].astype(str)
                mill_df["New Product"]=mill_df["New Product"].astype(str)
                #st.table(mill_df)
                mill_tab1,mill_tab2=st.tabs(["CURRENT SCHEDULE","UPLOAD SCHEDULE"])
                with mill_tab1:
                    choice=st.radio("TRUCK LOADS OR TONS",["TRUCKS","TONS"])
                    current_schedule.rename(columns={"Unnamed: 0":"Date"},inplace=True)  
                    current_schedule.set_index("Date",drop=True,inplace=True)
                    current_schedule_str=current_schedule.copy()
                    if choice=="TRUCKS":
                        st.markdown("**TRUCKS**")                        
                        st.dataframe(pd.DataFrame(current_schedule_str))
                    else:
                        st.markdown("**TONS**")
                        totals=[0]*len(current_schedule)
                        for i in current_schedule_str.columns[:-1]:
                            
                            if i in ["Wauna, Oregon","Halsey, Oregon"]:
                                current_schedule_str[i]=current_schedule_str[i]*28
                                totals=[sum(x) for x in zip(totals, current_schedule_str[i])]
                            else:
                                current_schedule_str[i]=current_schedule_str[i]*20
                                totals=[sum(x) for x in zip(totals, current_schedule_str[i])]
                        current_schedule_str["Total"]=totals
                        st.dataframe(pd.DataFrame(current_schedule_str))
                                
                    
                    
                    #current_schedule_str.index = pd.to_datetime(current_schedule_str.index)
                    #dates=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in current_schedule_str.index]#datetime.datetime.strftime(i,"%b %d,%A")
                    #current_schedule_str.index=dates
                    
                    
                
                with mill_tab2:                    
                    uploaded_file = st.file_uploader("Choose a file",key="pdods")
                    if uploaded_file is not None:
                        
                        schedule=pd.read_excel(uploaded_file,header=0,index_col=None)
                        schedule=schedule.dropna(0, how="all")
                        schedule.reset_index(drop=True,inplace=True)
                        locations=[ i for i in schedule["SUZANO LOADOUT TRUCK SCHEDULE"].unique() if i!='Total' and str(i)!="nan"]
                        date_indexs=[]
                        locations=[ i for i in schedule["SUZANO LOADOUT TRUCK SCHEDULE"].unique() if i!='Total' and str(i)!="nan"]
                        date_indexs=[]
                        plan={}
                        for j in range(1,6):
                            
                            for i in schedule.index:
                                try:
                                    if schedule.loc[i,f"Unnamed: {j}"].date():
                                        #print(i)
                                        date_indexs.append(i)
                        
                                except:
                                    pass
                            for i in date_indexs[:-1]:
                                #print(i)
                                for k in range(i+1,date_indexs[date_indexs.index(i)+1]):
                                    #print(k)
                                    if schedule.loc[k,"SUZANO LOADOUT TRUCK SCHEDULE"] in locations:
                                        location=schedule.loc[k,"SUZANO LOADOUT TRUCK SCHEDULE"]
                                        #print(location)
                                        key=schedule.loc[i,f"Unnamed: {j}"]
                                        #print(key)            
                                        try:
                                            plan[key][location]=schedule.loc[k,f"Unnamed: {j}"]
                                        except:
                                            plan[key]={}
                                            plan[key][location]=schedule.loc[k,f"Unnamed: {j}"]
                        
                            for k in range(date_indexs[-1],len(schedule)):        
                                    if schedule.loc[k,"SUZANO LOADOUT TRUCK SCHEDULE"] in locations:
                                        location=schedule.loc[k,"SUZANO LOADOUT TRUCK SCHEDULE"]
                                        plan[schedule.loc[date_indexs[-1],f"Unnamed: {j}"]]={}
                                        plan[schedule.loc[date_indexs[-1],f"Unnamed: {j}"]]={location:schedule.loc[k,f"Unnamed: {j}"]}
                            df=pd.DataFrame(plan).T.sort_index().fillna("0")
                            #dates=[datetime.datetime.strftime(i,"%b %d,%A") for i in df.index]
                            #df.index=dates
                            df=df.astype(int)
                            df["Total"]=df.sum(axis=1)
                            df.loc["Total"]=df.sum(axis=0)
                            df_v=df.replace(0,"")
                        st.table(df_v)
                        if st.button("UPDATE DATABASE WITH NEW SCHEDULE",key="lolos"):
                            
                            temp=df.to_csv("temp.csv")
                            upload_cs_file("olym_suzano", 'temp.csv',"truck_schedule.csv") 
                            st.success('File Uploaded', icon="✅")
                            

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
                                #st.write(list_cs_files("olym_suzano"))
                                temp=new_df.to_csv("temp.csv")
                                upload_cs_file("olym_suzano", 'temp.csv',rf"shipping_files/{gemi}-{voyage}-shipping_file.csv") 
                                st.write(f"Uploaded {gemi}-{voyage}-shipping_file.csv to database")
                        st.dataframe(new_df)
                    with shipment_tab2:
                        folder_name = "olym_suzano/shipping_files"  # Replace this with the folder path you want to read
                        files_in_folder = list_files_in_folder("olym_suzano", "shipping_files")
                        requested_file=st.selectbox("SHIPPING FILES IN DATABASE",files_in_folder[1:])
                        if st.button("LOAD SHIPPING FILE"):
                            requested_shipping_file=gcp_csv_to_df("olym_suzano", requested_file)
                            filtered_df=requested_shipping_file[["Lot","Lot Qty","Batch","Grade","Ocean B/L","DryWeight","ADMT","Location",
                                                                                      "Warehouse_In","Warehouse_Out","Vehicle_Id","Release_Order_Number","Carrier_Code"]]
                            #st.data_editor(filtered_df, use_container_width=True)
                            st.data_editor(filtered_df)
                          
            with admin_tab1:
                carrier_list_=gcp_download("olym_suzano",rf"carrier.json")
                carrier_list=json.loads(carrier_list_)
                junk=gcp_download("olym_suzano",rf"junk_release.json")
                junk=json.loads(junk)
                try:
                    release_order_database=gcp_download("olym_suzano",rf"release_orders/RELEASE_ORDERS.json")
                    release_order_database=json.loads(release_order_database)
                except:
                    release_order_database={}
                
              
                release_order_tab1,release_order_tab2=st.tabs(["CREATE RELEASE ORDER","RELEASE ORDER DATABASE"])
                with release_order_tab1:
                    vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304"])
                    edit=st.checkbox("CHECK TO ADD TO EXISTING RELEASE ORDER")
                    batch_mapping=gcp_download("olym_suzano",rf"batch_mapping.json")
                    batch_mapping=json.loads(batch_mapping)
                    if edit:
                        #release_order_number=st.selectbox("SELECT RELEASE ORDER",(list_files_in_folder("olym_suzano", "release_orders/{vessel}")))
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i for i in [i.replace(".json","") for i in list_files_in_subfolder("olym_suzano", rf"release_orders/KIRKENES-2304/")] if i not in junk]))
                        po_number=st.text_input("PO No")
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
                    quantity=st.number_input("Quantity of Units", min_value=1, max_value=800, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                    tonnage=2*quantity
                    #queue=st.number_input("Place in Queue", min_value=1, max_value=20, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                    transport_type=st.radio("Select Transport Type",("TRUCK","RAIL"))
                    carrier_code=st.selectbox("Carrier Code",[f"{key}-{item}" for key,item in carrier_list.items()])            
                    
        
                    create_release_order=st.button("SUBMIT")
                    if create_release_order:
                        
                        if edit: 
                            data=gcp_download("olym_suzano",rf"release_orders/{vessel}/{release_order_number}.json")
                            to_edit=json.loads(data)
                            temp=edit_release_order_data(to_edit,vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code)
                            st.write(f"ADDED sales order item {sales_order_item} to release order {release_order_number}!")
                        else:
                            
                            temp=store_release_order_data(vessel,release_order_number,destination,po_number,sales_order_item,batch,ocean_bill_of_lading,wrap,dryness,unitized,quantity,tonnage,transport_type,carrier_code)
                     
                        try:
                            junk=gcp_download("olym_suzano",rf"release_orders/{vessel}/junk_release.json")
                        except:
                            junk=gcp_download("olym_suzano",rf"junk_release.json")
                        junk=json.loads(junk)
                        try:
                            del junk[release_order_number]
                            jason_data=json.dumps(junk)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket("olym_suzano")
                            blob = bucket.blob(rf"release_orders/{vessel}/junk_release.json")
                            blob.upload_from_string(jason_data)
                        except:
                            pass
                        

                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"release_orders/{vessel}/{release_order_number}.json")
                        blob.upload_from_string(temp)

                        
                        try:
                            release_order_database[release_order_number][sales_order_item]={"destination":destination,"total":quantity,"remaining":quantity}
                            
                        except:
                            
                            release_order_database[release_order_number]={}
                            release_order_database[release_order_number][sales_order_item]={"destination":destination,"total":quantity,"remaining":quantity}
                        release_orders_json=json.dumps(release_order_database)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"release_orders/RELEASE_ORDERS.json")
                        blob.upload_from_string(release_orders_json)
                        st.write(f"Recorded Release Order - {release_order_number} for Item No: {sales_order_item}")
                        
                with release_order_tab2:
                    
                    vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304"],key="other")
                    rls_tab1,rls_tab2=st.tabs(["ACTIVE RELEASE ORDERS","COMPLETED RELEASE ORDERS"])

                    data=gcp_download("olym_suzano",rf"release_orders/RELEASE_ORDERS.json")
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
                        
                        files_in_folder_ = [i.replace(".json","") for i in list_files_in_subfolder("olym_suzano", rf"release_orders/KIRKENES-2304/")]   ### REMOVE json extension from name
                        
                        junk=gcp_download("olym_suzano",rf"junk_release.json")
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
                            data=gcp_download("olym_suzano",rf"release_orders/{vessel}/{requested_file}.json")
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
                            dispatched=gcp_download("olym_suzano",rf"dispatched.json")
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
                            bucket = storage_client.bucket("olym_suzano")
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
                                    bucket = storage_client.bucket("olym_suzano")
                                    blob = bucket.blob(rf"release_orders/{vessel}/{requested_file}.json")
                                    blob.upload_from_string(json_data)
                                if st.button("DELETE RELEASE ORDER ITEM!",key="laladg"):
                                    junk=gcp_download("olym_suzano",rf"junk_release.json")
                                    junk=json.loads(junk)
                                   
                                    junk[requested_file]=1
                                    json_data = json.dumps(junk)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket("olym_suzano")
                                    blob = bucket.blob(rf"junk_release.json")
                                    blob.upload_from_string(json_data)
                                           
                            with dol2:  
                                if st.button("CLEAR DISPATCH QUEUE!"):
                                    dispatch={}
                                    json_data = json.dumps(dispatch)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket("olym_suzano")
                                    blob = bucket.blob(rf"dispatched.json")
                                    blob.upload_from_string(json_data)
                                    st.markdown(f"**CLEARED ALL DISPATCHES**")   
                            with dol3:
                                dispatch=gcp_download("olym_suzano",rf"dispatched.json")
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
                        pass
                        #sales_orders_completed=[k for k in targets if target[k]['remaining']<=0]
                        
                                
        
                        
        
        ##########  LOAD OUT  ##############
        
        
        
        if select=="LOADOUT" :
        
            
            bill_mapping=gcp_download("olym_suzano","bill_mapping.json")
            bill_mapping=json.loads(bill_mapping)
            mill_info_=gcp_download("olym_suzano",rf"mill_info.json")
            mill_info=json.loads(mill_info_)
            no_dispatch=0
            number=None
            if number not in st.session_state:
                st.session_state.number=number
            try:
                dispatched=gcp_download("olym_suzano","dispatched.json")
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
                    file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=7),key="file_dates",disabled=True)
                    if file_date not in st.session_state:
                        st.session_state.file_date=file_date
                    file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=7),step=60,disabled=False)
                    delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=7),key="delivery_date",disabled=True)
                    eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                    
                with col2:
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
                    file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=7),step=60,disabled=False,key="popop")
                    b=file_time.strftime("%H%M%S")
                    b_=file_time.strftime("%H:%M:%S")
                    c=datetime.datetime.strftime(eta_date,"%Y%m%d")
                else:     
                    
                    a=datetime.datetime.strftime(file_date,"%Y%m%d")
                    a_=datetime.datetime.strftime(file_date,"%Y-%m-%d")
                    b=file_time.strftime("%H%M%S")
                    b=(datetime.datetime.now()-datetime.timedelta(hours=7)).strftime("%H%M%S")
                    b_=(datetime.datetime.now()-datetime.timedelta(hours=7)).strftime("%H:%M:%S")
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
                                suzano_report_=gcp_download("olym_suzano",rf"suzano_report.json")
                                suzano_report=json.loads(suzano_report_)
                            except:
                                suzano_report={}
                            consignee=destination.split("-")[0]
                            consignee_city=mill_info[destination]["city"]
                            consignee_state=mill_info[destination]["state"]
                            vessel_suzano,voyage_suzano=vessel.split("-")
                            if manual_time:
                                eta=datetime.datetime.strftime(file_date+datetime.timedelta(hours=mill_info[destination]['hours']-7)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")
                            else:
                                eta=datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=mill_info[destination]['hours']-7)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")
    
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
                                                     "ETA":eta,"Ocean BOL#":ocean_bill_of_lading,"Warehouse":"OLYM","Vessel":vessel_suzano,"Voyage #":voyage_suzano,"Grade":wrap,"Quantity":quantity,
                                                     "Metric Ton": quantity*2, "ADMT":admt,"Mode of Transportation":transport_type}})
                                suzano_report=json.dumps(suzano_report)
                                storage_client = storage.Client()
                                bucket = storage_client.bucket("olym_suzano")
                                blob = bucket.blob(rf"suzano_report.json")
                                blob.upload_from_string(suzano_report)
    
                              
                                mill_progress=json.loads(gcp_download("olym_suzano",rf"mill_progress.json"))
                                map={8:"SEP 2023",9:"SEP 2023",10:"OCT 2023",11:"NOV 2023",12:"DEC 2023"}
                                mill_progress[destination][map[file_date.month]]["Shipped"]=mill_progress[destination][map[file_date.month]]["Shipped"]+len(textsplit)*2
                                json_data = json.dumps(mill_progress)
                                storage_client = storage.Client()
                                bucket = storage_client.bucket("olym_suzano")
                                blob = bucket.blob(rf"mill_progress.json")
                                blob.upload_from_string(json_data)       
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
                            bucket = storage_client.bucket("olym_suzano")
                            blob = bucket.blob(rf"release_orders/{vessel}/{current_release_order}.json")
                            blob.upload_from_string(json_data)
    
                            try:
                                release_order_database=gcp_download("olym_suzano",rf"release_orders/RELEASE_ORDERS.json")
                                release_order_database=json.loads(release_order_database)
                            except:
                                release_order_database={}
                           
                            release_order_database[current_release_order][current_sales_order]["remaining"]=release_order_database[current_release_order][current_sales_order]["remaining"]-quantity
                            release_order_database=json.dumps(release_order_database)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket("olym_suzano")
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
                            #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                            recipients = ["afsiny@portolympia.com"]
                            password = "xjvxkmzbpotzeuuv"
                    
                  
                    
                    
                            with open('temp_file.txt', 'w') as f:
                                f.write(file_content)
                    
                            file_path = 'temp_file.txt'  # Use the path of the temporary file
                    
                            send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                            upload_cs_file("olym_suzano", 'temp_file.txt',rf"EDIS/{vessel}/{file_name}") 
                            
                        else:   ###cancel bill of lading
                            pass
                
                            
        
            
                        
            else:
                st.subheader("**Nothing dispatched!**")
                    
            
                    
                        
        
        
                
                        
        if select=="INVENTORY" :
            Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
           
            mill_info=json.loads(gcp_download("olym_suzano",rf"mill_info.json"))
            inv1,inv2,inv3,inv4,inv5=st.tabs(["DAILY ACTION","SUZANO DAILY REPORTS","EDI BANK","MAIN INVENTORY","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS"])
            with inv1:
                data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
                bill_of_ladings=json.loads(data)
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
                                #st.write(f"Truck No : {truck} is Enroute to {destination} with ETA {estimated_arrival_string}")
                                enroute_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                                 "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}","ETA":estimated_arrival_string}
                            else:
                                with daily3:
                                    #st.write(f"Truck No : {truck} arrived at {destination} at {estimated_arrival_string}")
                                    arrived_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                                 "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}","ARRIVAL TIME":estimated_arrival_string}
                             
                    arrived_vehicles=pd.DataFrame(arrived_vehicles)
                    arrived_vehicles=arrived_vehicles.rename_axis('TRUCK NO')               
                    enroute_vehicles=pd.DataFrame(enroute_vehicles)
                    enroute_vehicles=enroute_vehicles.rename_axis('TRUCK NO')
                    st.dataframe(enroute_vehicles.T)                      
                    for i in enroute_vehicles:
                        st.write(f"Truck No : {i} is Enroute to {enroute_vehicles[i]['DESTINATION']} at {enroute_vehicles[i]['ETA']}")
                with daily3:
                    st.table(arrived_vehicles.T)
            
            with inv2:
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')
                try:
                    now=datetime.datetime.now()-datetime.timedelta(hours=7)
                    suzano_report_=gcp_download("olym_suzano",rf"suzano_report.json")
                    suzano_report=json.loads(suzano_report_)
                    suzano_report=pd.DataFrame(suzano_report).T
                    suzano_report=suzano_report[["Date Shipped","Vehicle", "Shipment ID #", "Consignee","Consignee City","Consignee State","Release #","Carrier","ETA","Ocean BOL#","Warehouse","Vessel","Voyage #","Grade","Quantity","Metric Ton", "ADMT","Mode of Transportation"]]
                    suzano_report["Shipment ID #"]=[str(i) for i in suzano_report["Shipment ID #"]]
                    daily_suzano=suzano_report.copy()
                    daily_suzano["Date"]=[datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S").date() for i in suzano_report["Date Shipped"]]
                    daily_suzano=daily_suzano[daily_suzano["Date"]==now.date()]
                    choose = st.radio(
                                    "Select Daily or Accumulative Report",
                                    ["DAILY", "ACCUMULATIVE"])
                    if choose=="DAILY":
                        st.dataframe(daily_suzano)
                        csv=convert_df(daily_suzano)
                    else:
                        st.dataframe(suzano_report)
                        csv=convert_df(suzano_report)
                    
                    
                    
                   
                    
                
                    st.download_button(
                        label="DOWNLOAD REPORT AS CSV",
                        data=csv,
                        file_name=f'OLYMPIA_DAILY_REPORT{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%Y_%m_%d")}.csv',
                        mime='text/csv')
                except:
                    st.write("NO REPORTS RECORDED")
                

            with inv3:
                edi_files=list_files_in_subfolder("olym_suzano", rf"EDIS/KIRKENES-2304/")
                requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
                try:
                    requested_edi=gcp_download("olym_suzano", rf"EDIS/KIRKENES-2304/{requested_edi_file}")
                    st.text_area("EDI",requested_edi,height=400)
                    st.download_button(
                        label="DOWNLOAD EDI",
                        data=requested_edi,
                        file_name=f'{requested_edi_file}',
                        mime='text/csv')

                except:
                    st.write("NO EDI FILES IN DIRECTORY")
                

                
            with inv4:
                     
                dab1,dab2=st.tabs(["IN WAREHOUSE","SHIPPED BY DATE"])
                df=Inventory[(Inventory["Location"]=="OLYM")|(Inventory["Location"]=="PARTIAL")][["Lot","Bales","Shipped","Remaining","Batch","Ocean B/L","Grade","DryWeight","ADMT","Location","Warehouse_In"]]
                zf=Inventory[(Inventory["Location"]=="ON TRUCK")|(Inventory["Location"]=="PARTIAL")][["Lot","Bales","Shipped","Remaining","Batch","Ocean B/L","Grade","DryWeight","ADMT","Release_Order_Number","Carrier_Code","Terminal B/L",
                                                              "Vehicle_Id","Warehouse_In","Warehouse_Out"]]
           
                items=df["Ocean B/L"].unique().tolist()
                
                with dab1:
                    
                    inv_col1,inv_col2,inv_col3=st.columns([2,6,2])
                    with inv_col1:
                        wrh=df["Remaining"].sum()*250/1000
                        shp=zf["Shipped"].sum()*250/1000
                        
                        st.markdown(f"**IN WAREHOUSE = {wrh} tons**")
                        st.markdown(f"**TOTAL SHIPPED = {shp} tons**")
                        st.markdown(f"**TOTAL OVERALL = {wrh+shp} tons**")
                    with inv_col2:
                        #st.write(items)
                        inhouse=[df[df["Ocean B/L"]==i]["Remaining"].sum()*250/1000 for i in items]
                        shipped=[df[df["Ocean B/L"]==i]["Shipped"].sum()*250/1000 for i in items]
                        
                        wrap_=[df[df["Ocean B/L"]==i]["Grade"].unique()[0] for i in items]
                       # st.write(wrap_)
                        tablo=pd.DataFrame({"Ocean B/L":items,"Grade":wrap_,"In Warehouse":inhouse,"Shipped":shipped},index=[i for i in range(1,len(items)+1)])
                        total_row={"Ocean B/L":"TOTAL","In Warehouse":sum(inhouse),"Shipped":sum(shipped)}
                        tablo = tablo.append(total_row, ignore_index=True)
                        tablo["TOTAL"] = tablo.loc[:, ["In Warehouse", "Shipped"]].sum(axis=1)
                        st.markdown(f"**IN METRIC TONS -- AS OF {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),'%b %d -  %H:%M')}**")
                        st.dataframe(tablo)
                    if st.checkbox("CLICK TO SEE INVENTORY LIST"):
                        st.dataframe(df)
                with dab2:
                    
                    
                    filter_date=st.date_input("Choose Warehouse OUT Date",datetime.datetime.today(),min_value=None, max_value=None,disabled=False,key="filter_date")
                    
                    
                   
                    zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]]=zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]].astype("str")
                    
                    zf["Warehouse_Out"]=[datetime.datetime.strptime(j,"%Y-%m-%d %H:%M:%S") for j in zf["Warehouse_Out"]]
                    filtered_zf=zf.copy()
                    
                    filtered_zf["Warehouse_Out"]=[i.date() for i in filtered_zf["Warehouse_Out"]]
                        
                    filtered_zf=filtered_zf[filtered_zf["Warehouse_Out"]==filter_date]
                        
                    
                    col1,col2=st.columns([2,8])
                    with col2:
                        st.dataframe(zf)
                        
                               
                        
                    with col1:
                        st.markdown(f"**SHIPPED ON THIS DAY = {zf['Shipped'].sum()*0.250} TONS**")
                        
                           
                        
                        
                   
            with inv5:
                mill_progress=json.loads(gcp_download("olym_suzano",rf"mill_progress.json"))
                reformed_dict = {}
                for outerKey, innerDict in mill_progress.items():
                    for innerKey, values in innerDict.items():
                        reformed_dict[(outerKey,innerKey)] = values
                mill_prog_col1,mill_prog_col2=st.columns([2,4])
                with mill_prog_col1:
                    st.dataframe(pd.DataFrame(reformed_dict).T)
                with mill_prog_col2:
                    chosen_month=st.selectbox("SELECT MONTH",["SEP 2023","OCT 2023","NOV 2023","DEC 2023"])
                    mills = mill_progress.keys()
                    targets = [mill_progress[i][chosen_month]["Planned"] for i in mills]
                    shipped = [mill_progress[i][chosen_month]["Shipped"] for i in mills]
                    
                    # Create a figure with a horizontal bar chart
                    fig = go.Figure()
                    
                    for mill, target, shipped_qty in zip(mills, targets, shipped):
                        fig.add_trace(
                            go.Bar(
                                y=[mill],
                                x=[shipped_qty],  # Darker shade indicating shipped
                                orientation="h",
                                name="Shipped",
                                marker=dict(color='rgba(0, 128, 0, 0.7)')
                            )
                        )
                        fig.add_trace(
                            go.Bar(
                                y=[mill],
                                x=[target],  # Lighter shade indicating target
                                orientation="h",
                                name="Target",
                                marker=dict(color='rgba(0, 128, 0, 0.3)')
                            )
                        )
                    
                    # Customize the layout
                    fig.update_layout(
                                barmode='stack',  # Stack the bars on top of each other
                                xaxis_title="Quantity",
                                yaxis_title="Mills",
                                title=f"Monthly Targets and Shipped Quantities - {chosen_month}",
                                legend=dict(
                                    x=1.02,  # Move the legend to the right
                                    y=1.0,
                                    xanchor="left",  # Adjust legend position
                                    yanchor="top",
                                    font=dict(size=12)  # Increase legend font size
                                ),
                                xaxis=dict(tickfont=dict(size=10)),  # Increase x-axis tick label font size
                                yaxis=dict(tickfont=dict(size=12)),  # Increase y-axis tick label font size
                                title_font=dict(size=16),  # Increase title font size and weight
                                 height=600,  # Adjust the height of the chart (in pixels)
                                width=800 
                            )
    
                    st.plotly_chart(fig)

                requested_mill=st.selectbox("**SELECT MILL TO SEE PROGRESS**",mill_progress.keys())
                def cust_business_days(start, end):
                    business_days = pd.date_range(start=start, end=end, freq='B')
                    return business_days
                target=mill_progress[requested_mill]["SEP 2023"]["Planned"]
                shipped=mill_progress[requested_mill]["SEP 2023"]["Shipped"]
                daily_needed_rate=int(target/len(cust_business_days(datetime.date(2023,9,1),datetime.date(2023,10,1))))
                days_passed=len(cust_business_days(datetime.date(2023,8,1),datetime.datetime.today()))
                days_left=len(cust_business_days(datetime.datetime.today(),datetime.date(2023,9,1)))
                #shipped=800
                reference=daily_needed_rate*days_passed
                
               
                fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = shipped,
                        mode = "gauge+number+delta",
                        title={'text': f"<span style='font-weight:bold; color:blue;'>TONS SHIPPED TO {requested_mill} - SEPT TARGET {target} MT</span>", 'font': {'size': 20}},
                        delta = {'reference': reference},
                        gauge = {'axis': {'range': [None, target]},
                                 'steps' : [
                                     {'range': [0, reference], 'color': "lightgray"},
                                  ],
                                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target}}))

                st.plotly_chart(fig)

                st.markdown(f"**SHOULD HAVE SHIPPED SO FAR : {reference} TONS (GRAY SHADE ON CHART)**")
                st.markdown(f"**SHIPPED SO FAR : {shipped} TONS (GREEN LINE ON CHART) - DAYS PASSED : {days_passed}**")
                st.markdown(f"**LEFT TO GO : {target-shipped} TONS (WHITE SHADE)- DAYS TO GO : {days_left}**")
                
                



    ########################                                WAREHOUSE                            ####################
    
    elif username == 'warehouse':
        bill_mapping=gcp_download("olym_suzano","bill_mapping.json")
        bill_mapping=json.loads(bill_mapping)
        mill_info_=gcp_download("olym_suzano",rf"mill_info.json")
        mill_info=json.loads(mill_info_)
        no_dispatch=0
        number=None
        if number not in st.session_state:
            st.session_state.number=number
        try:
            dispatched=gcp_download("olym_suzano","dispatched.json")
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
                file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=7),key="file_dates",disabled=True)
                if file_date not in st.session_state:
                    st.session_state.file_date=file_date
                file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=7),step=60,disabled=False)
                delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=7),key="delivery_date",disabled=True)
                eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                
            with col2:
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
                    
                
                    faults=[]
                    bale_faults=[]
                    fault_messaging={}
                    bale_fault_messaging={}
                    textsplit={}
                    bale_textsplit
                    yes=False
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
                                    
                                else:
                                    st.write(f"**Unit No : {i+1}-{x}**")
                                    faults.append(0)
                                    yes=True
                            else:
                                st.markdown(f"**:red[Unit No : {i+1}-{x}]**",unsafe_allow_html=True)
                                st.write(f"**:red[WRONG B/L, DO NOT LOAD UNIT {x}]**")
                                faults.append(1)
                       
                                
                            seen.add(x)
                    if bale_load_input is not None:
                        yes=False
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
                                    yes=True
                            else:
                                st.markdown(f"**:red[Bale No : {i+1}-{x}]**",unsafe_allow_html=True)
                                st.write(f"**:red[WRONG B/L, DO NOT LOAD UNIT {x}]**")
                                bale_faults.append(1)
                            seen.add(x)
                   
                       
                    loads={}
                    pure_loads={}
                    
                    if 1 in faults or 1 in bale_faults:
                        yes=False
                    else:
                        yes=True
                    
                    if yes:
                        pure_loads={**{k:0 for k in textsplit},**{k:0 for k in bale_textsplit}}
                        loads={**{k[:-2]:0 for k in textsplit},**{k[:-2]:0 for k in bale_textsplit}}
                        for k in textsplit:
                            loads[k[:-2]]+=1
                            pure_loads[k]+=1
                        for k in bale_textsplit:
                            loads[k[:-2]]+=0.125
                            pure_loads[k]+=0.125
            
                
            a=datetime.datetime.strftime(file_date,"%Y%m%d")
            a_=datetime.datetime.strftime(file_date,"%Y-%m-%d")
            b=file_time.strftime("%H%M%S")
            b=(datetime.datetime.now()-datetime.timedelta(hours=7)).strftime("%H%M%S")
            b_=(datetime.datetime.now()-datetime.timedelta(hours=7)).strftime("%H:%M:%S")
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
                            suzano_report_=gcp_download("olym_suzano",rf"suzano_report.json")
                            suzano_report=json.loads(suzano_report_)
                        except:
                            suzano_report={}
                        consignee=destination.split("-")[0]
                        consignee_city=mill_info[destination]["city"]
                        consignee_state=mill_info[destination]["state"]
                        vessel_suzano,voyage_suzano=vessel.split("-")
                        if manual_time:
                            eta=datetime.datetime.strftime(file_date+datetime.timedelta(hours=mill_info[destination]['hours']-7)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")
                        else:
                            eta=datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=mill_info[destination]['hours']-7)+datetime.timedelta(minutes=mill_info[destination]['minutes']+30),"%Y-%m-%d  %H:%M:%S")

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
                                                 "ETA":eta,"Ocean BOL#":ocean_bill_of_lading,"Warehouse":"OLYM","Vessel":vessel_suzano,"Voyage #":voyage_suzano,"Grade":wrap,"Quantity":quantity,
                                                 "Metric Ton": quantity*2, "ADMT":admt,"Mode of Transportation":transport_type}})
                            suzano_report=json.dumps(suzano_report)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket("olym_suzano")
                            blob = bucket.blob(rf"suzano_report.json")
                            blob.upload_from_string(suzano_report)

                          
                            mill_progress=json.loads(gcp_download("olym_suzano",rf"mill_progress.json"))
                            map={8:"SEP 2023",9:"SEP 2023",10:"OCT 2023",11:"NOV 2023",12:"DEC 2023"}
                            mill_progress[destination][map[file_date.month]]["Shipped"]=mill_progress[destination][map[file_date.month]]["Shipped"]+len(textsplit)*2
                            json_data = json.dumps(mill_progress)
                            storage_client = storage.Client()
                            bucket = storage_client.bucket("olym_suzano")
                            blob = bucket.blob(rf"mill_progress.json")
                            blob.upload_from_string(json_data)       
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
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"release_orders/{vessel}/{current_release_order}.json")
                        blob.upload_from_string(json_data)

                        try:
                            release_order_database=gcp_download("olym_suzano",rf"release_orders/RELEASE_ORDERS.json")
                            release_order_database=json.loads(release_order_database)
                        except:
                            release_order_database={}
                       
                        release_order_database[current_release_order][current_sales_order]["remaining"]=release_order_database[current_release_order][current_sales_order]["remaining"]-quantity
                        release_order_database=json.dumps(release_order_database)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
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
                        #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                        recipients = ["afsiny@portolympia.com"]
                        password = "xjvxkmzbpotzeuuv"
                
              
                
                
                        with open('temp_file.txt', 'w') as f:
                            f.write(file_content)
                
                        file_path = 'temp_file.txt'  # Use the path of the temporary file
                
                        send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                        upload_cs_file("olym_suzano", 'temp_file.txt',rf"EDIS/{vessel}/{file_name}") 
                        
                    else:   ###cancel bill of lading
                        pass
            
                        
    
        
                    
        else:
            st.subheader("**Nothing dispatched!**")
                        
    
        
                    ###########################       SUZANO INVENTORY BOARD    ###########################
 
    elif username == 'olysuzanodash':
        
        Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
           
        mill_info=json.loads(gcp_download("olym_suzano",rf"mill_info.json"))
        inv1,inv2,inv3,inv4,inv5=st.tabs(["DAILY ACTION","REPORTS","EDI BANK","MAIN INVENTORY","SUZANO MILL SHIPMENT SCHEDULE/PROGRESS"])
        with inv1:
            data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
            bill_of_ladings=json.loads(data)
            daily1,daily2,daily3=st.tabs(["TODAY'SHIPMENTS","TRUCKS ENROUTE","TRUCKS AT DESTINATION"])
            with daily1:
                now=datetime.datetime.now()-datetime.timedelta(hours=7)
                st.markdown(f"**SHIPPED TODAY ON {datetime.datetime.strftime(now.date(),'%b %d, %Y')}**")     
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
                            #st.write(f"Truck No : {truck} is Enroute to {destination} with ETA {estimated_arrival_string}")
                            enroute_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                             "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}","ETA":estimated_arrival_string}
                        else:
                            with daily3:
                                #st.write(f"Truck No : {truck} arrived at {destination} at {estimated_arrival_string}")
                                arrived_vehicles[truck]={"DESTINATION":destination,"CARGO":bill_of_ladings[i]["ocean_bill_of_lading"],
                                             "QUANTITY":f'{2*bill_of_ladings[i]["quantity"]} TONS',"LOADED TIME":f"{ship_date.date()}---{ship_time}","ARRIVAL TIME":estimated_arrival_string}
                         
                arrived_vehicles=pd.DataFrame(arrived_vehicles)
                arrived_vehicles=arrived_vehicles.rename_axis('TRUCK NO')               
                enroute_vehicles=pd.DataFrame(enroute_vehicles)
                enroute_vehicles=enroute_vehicles.rename_axis('TRUCK NO')
                st.dataframe(enroute_vehicles.T)                      
                for i in enroute_vehicles:
                    st.write(f"Truck No : {i} is Enroute to {enroute_vehicles[i]['DESTINATION']} at {enroute_vehicles[i]['ETA']}")
            with daily3:
                st.table(arrived_vehicles.T)
        
        with inv2:
            try:
                suzano_report_=gcp_download("olym_suzano",rf"suzano_report.json")
                suzano_report=json.loads(suzano_report_)
                suzano_report=pd.DataFrame(suzano_report).T
                suzano_report=suzano_report[["Date Shipped","Vehicle", "Shipment ID #", "Consignee","Consignee City","Consignee State","Release #","Carrier","ETA","Ocean BOL#","Warehouse","Vessel","Voyage #","Grade","Quantity","Metric Ton", "ADMT","Mode of Transportation"]]
                suzano_report["Shipment ID #"]=[str(i) for i in suzano_report["Shipment ID #"]]
                st.dataframe(suzano_report)
                
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')
                
                csv = convert_df(suzano_report)
                
            
                st.download_button(
                    label="DOWNLOAD REPORT AS CSV",
                    data=csv,
                    file_name=f'OLYMPIA_DAILY_REPORT{datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%Y_%m_%d")}.csv',
                    mime='text/csv')
            except:
                st.write("NO REPORTS RECORDED")
           

        with inv3:
            edi_files=list_files_in_subfolder("olym_suzano", rf"EDIS/KIRKENES-2304/")
            requested_edi_file=st.selectbox("SELECT EDI",edi_files[1:])
            try:
                requested_edi=gcp_download("olym_suzano", rf"EDIS/KIRKENES-2304/{requested_edi_file}")
                st.text_area("EDI",requested_edi,height=400)
            except:
                st.write("NO EDI FILES IN DIRECTORY")
            


            
        with inv4:
                 
            dab1,dab2=st.tabs(["IN WAREHOUSE","SHIPPED"])
            df=Inventory[(Inventory["Location"]=="OLYM")|(Inventory["Location"]=="PARTIAL")][["Lot","Bales","Batch","Ocean B/L","Grade","DryWeight","ADMT","Location","Warehouse_In"]]
            zf=Inventory[(Inventory["Location"]=="ON TRUCK")|(Inventory["Location"]=="PARTIAL")][["Lot","Bales","Batch","Ocean B/L","Grade","DryWeight","ADMT","Release_Order_Number","Carrier_Code","Terminal B/L",
                                                          "Vehicle_Id","Warehouse_In","Warehouse_Out"]]
            zf["Bales"]=[8-i if i<8 else i for i in zf["Bales"] ]
            items=df["Ocean B/L"].unique().tolist()
            
            with dab1:
                
                inv_col1,inv_col2,inv_col3=st.columns([2,6,2])
                with inv_col1:
                    wrh=df["Bales"].sum()*250/1000
                    shp=zf["Bales"].sum()*250/1000
                    
                    st.markdown(f"**IN WAREHOUSE = {wrh} tons**")
                    st.markdown(f"**TOTAL SHIPPED = {shp} tons**")
                    st.markdown(f"**TOTAL OVERALL = {wrh+shp} tons**")
                with inv_col2:
                    #st.write(items)
                    inhouse=[df[df["Ocean B/L"]==i]["Bales"].sum()*250/1000 for i in items]
                    shipped=[zf[zf["Ocean B/L"]==i]["Bales"].sum()*250/1000 for i in items]
                    
                    wrap_=[df[df["Ocean B/L"]==i]["Grade"].unique()[0] for i in items]
                   # st.write(wrap_)
                    tablo=pd.DataFrame({"Ocean B/L":items,"Grade":wrap_,"In Warehouse":inhouse,"Shipped":shipped},index=[i for i in range(1,len(items)+1)])
                    total_row={"Ocean B/L":"TOTAL","In Warehouse":sum(inhouse),"Shipped":sum(shipped)}
                    tablo = tablo.append(total_row, ignore_index=True)
                    tablo["TOTAL"] = tablo.loc[:, ["In Warehouse", "Shipped"]].sum(axis=1)
                    st.markdown(f"**IN METRIC TONS -- AS OF {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),'%b %d -  %H:%M')}**")
                    st.dataframe(tablo)
                if st.checkbox("CLICK TO SEE INVENTORY LIST"):
                    st.dataframe(df)
            with dab2:
                
                date_filter=st.checkbox("CLICK FOR DATE FILTER")
                if "disabled" not in st.session_state:
                    st.session_state.visibility = "visible"
                    st.session_state.disabled = True
                if date_filter:
                    st.session_state.disabled=False
                    
                else:
                    st.session_state.disabled=True
                    #min_value=min([i.date() for i in zf["Warehouse_Out"]])
                filter_date=st.date_input("Choose Warehouse OUT Date",datetime.datetime.today(),min_value=None, max_value=None,disabled=st.session_state.disabled,key="filter_date")
                
                
               
                zf[["Release_Order_Number","Carrier_Code","Terminal Bill Of Lading","Vehicle_Id"]]=zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]].astype("str")
                
                zf["Warehouse_Out"]=[datetime.datetime.strptime(j,"%Y-%m-%d %H:%M:%S") for j in zf["Warehouse_Out"]]
                filtered_zf=zf.copy()
                if date_filter:
                    filtered_zf["Warehouse_Out"]=[i.date() for i in filtered_zf["Warehouse_Out"]]
                    
                    filtered_zf=filtered_zf[filtered_zf["Warehouse_Out"]==filter_date]
                    
                filter_by=st.selectbox("SELECT FILTER",["Grade","Ocean B/L","Release_Order_Number","Terminal B/L","Carrier_Code","Vehicle_Id"])
                #st.write(filter_by)
                choice=st.selectbox(f"Filter By {filter_by}",[f"ALL {filter_by.upper()}"]+[str(j) for j in [str(i) for i in filtered_zf[filter_by].unique().tolist()]])
                
                
                col1,col2=st.columns([2,8])
                with col1:
                    st.markdown(f"**TOTAL SHIPPED = {len(zf)}**")
                    st.markdown(f"**IN WAREHOUSE = {len(df)}**")
                    st.markdown(f"**TOTAL OVERALL = {len(zf)+len(df)}**")
                try:
                    filtered_zf=filtered_zf[filtered_zf[filter_by]==choice]
                    filtered_df=filtered_zf[filtered_zf[filter_by]==choice]
                    
                except:
                    filtered_zf=filtered_zf
                    filtered_df=df.copy()
                    
                    pass
                with col2:
                    if date_filter:
                        st.markdown(f"**SHIPPED ON THIS DAY = {len(filtered_zf)}**")
                    else:
                        st.markdown(f"**TOTAL SHIPPED = {len(filtered_zf)}**")
                        st.markdown(f"**IN WAREHOUSE = {len(filtered_df)}**")
                        st.markdown(f"**TOTAL OVERALL = {len(filtered_zf)+len(filtered_df)}**")
                    
                    
                st.table(filtered_zf)
        with inv5:
            mill_progress=json.loads(gcp_download("olym_suzano",rf"mill_progress.json"))
            reformed_dict = {}
            for outerKey, innerDict in mill_progress.items():
                for innerKey, values in innerDict.items():
                    reformed_dict[(outerKey,innerKey)] = values
            mill_prog_col1,mill_prog_col2=st.columns([2,4])
            with mill_prog_col1:
                st.dataframe(pd.DataFrame(reformed_dict).T)
            with mill_prog_col2:
                chosen_month=st.selectbox("SELECT MONTH",["SEP 2023","OCT 2023","NOV 2023","DEC 2023"])
                mills = mill_progress.keys()
                targets = [mill_progress[i][chosen_month]["Planned"] for i in mills]
                shipped = [mill_progress[i][chosen_month]["Shipped"] for i in mills]
                
                # Create a figure with a horizontal bar chart
                fig = go.Figure()
                
                for mill, target, shipped_qty in zip(mills, targets, shipped):
                    fig.add_trace(
                        go.Bar(
                            y=[mill],
                            x=[shipped_qty],  # Darker shade indicating shipped
                            orientation="h",
                            name="Shipped",
                            marker=dict(color='rgba(0, 128, 0, 0.7)')
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            y=[mill],
                            x=[target],  # Lighter shade indicating target
                            orientation="h",
                            name="Target",
                            marker=dict(color='rgba(0, 128, 0, 0.3)')
                        )
                    )
                
                # Customize the layout
                fig.update_layout(
                            barmode='stack',  # Stack the bars on top of each other
                            xaxis_title="Quantity",
                            yaxis_title="Mills",
                            title=f"Monthly Targets and Shipped Quantities - {chosen_month}",
                            legend=dict(
                                x=1.02,  # Move the legend to the right
                                y=1.0,
                                xanchor="left",  # Adjust legend position
                                yanchor="top",
                                font=dict(size=12)  # Increase legend font size
                            ),
                            xaxis=dict(tickfont=dict(size=10)),  # Increase x-axis tick label font size
                            yaxis=dict(tickfont=dict(size=12)),  # Increase y-axis tick label font size
                            title_font=dict(size=16),  # Increase title font size and weight
                             height=600,  # Adjust the height of the chart (in pixels)
                            width=800 
                        )

                st.plotly_chart(fig)

            requested_mill=st.selectbox("**SELECT MILL TO SEE PROGRESS**",mill_progress.keys())
            def cust_business_days(start, end):
                business_days = pd.date_range(start=start, end=end, freq='B')
                return business_days
            target=mill_progress[requested_mill]["SEP 2023"]["Planned"]
            shipped=mill_progress[requested_mill]["SEP 2023"]["Shipped"]
            daily_needed_rate=int(target/len(cust_business_days(datetime.date(2023,9,1),datetime.date(2023,10,1))))
            days_passed=len(cust_business_days(datetime.date(2023,8,1),datetime.datetime.today()))
            days_left=len(cust_business_days(datetime.datetime.today(),datetime.date(2023,9,1)))
            #shipped=800
            reference=daily_needed_rate*days_passed
            
           
            fig = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = shipped,
                    mode = "gauge+number+delta",
                    title={'text': f"<span style='font-weight:bold; color:blue;'>TONS SHIPPED TO {requested_mill} - SEPT TARGET {target} MT</span>", 'font': {'size': 20}},
                    delta = {'reference': reference},
                    gauge = {'axis': {'range': [None, target]},
                             'steps' : [
                                 {'range': [0, reference], 'color': "lightgray"},
                              ],
                             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target}}))

            st.plotly_chart(fig)

            st.markdown(f"**SHOULD HAVE SHIPPED SO FAR : {reference} TONS (GRAY SHADE ON CHART)**")
            st.markdown(f"**SHIPPED SO FAR : {shipped} TONS (GREEN LINE ON CHART) - DAYS PASSED : {days_passed}**")
            st.markdown(f"**LEFT TO GO : {target-shipped} TONS (WHITE SHADE)- DAYS TO GO : {days_left}**")
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')




    
 
