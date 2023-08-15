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



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'

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
    blob = bucket.blob("Inventory.csv")
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
def store_release_order_data(vessel,release_order_number,sales_order_item,batch,ocean_bill_of_lading,dryness,quantity,tonnage,transport_type,carrier_code):
       
    # Create a dictionary to store the release order data
    release_order_data = { vessel: {
       
        release_order_number:{
        sales_order_item: {
        "batch": batch,
        "ocean_bill_of_lading": ocean_bill_of_lading,
        "dryness":dryness,
        "transport_type": transport_type,
        "carrier_code": carrier_code,
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

def edit_release_order_data(file,vessel,release_order_number,sales_order_item,batch,ocean_bill_of_lading,dryness,quantity,tonnage,transport_type,carrier_code):
       
    # Edit the loaded current dictionary.
    if sales_order_item not in file[vessel][release_order_number]:
        file[vessel][release_order_number][sales_order_item]={}
    file[vessel][release_order_number][sales_order_item]["batch"]= batch
    file[vessel][release_order_number][sales_order_item]["ocean_bill_of_lading"]= ocean_bill_of_lading
    file[vessel][release_order_number][sales_order_item]["dryness"]= dryness
    file[vessel][release_order_number][sales_order_item]["transport_type"]= transport_type
    file[vessel][release_order_number][sales_order_item]["carrier_code"]= carrier_code
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
    line2="2DTD:"+release_order_number+" "*(10-len(release_order_number))+"000"+sales_order_item+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(quantity*2000)+" "*(16-len(str(quantity*2000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+terminal_bill_of_lading+" "*(50-len(terminal_bill_of_lading))+c
               
    loadls=[]
    if double_load:
        for i in first_textsplit:
            loadls.append("2DEV:"+current_release_order+" "*(10-len(current_release_order))+"000"+current_sales_order+a+tsn+i[:-3]+" "*(10-len(i[:-3]))+"0"*16+str(2000))
        for k in second_textsplit:
            loadls.append("2DEV:"+next_release_order+" "*(10-len(next_release_order))+"000"+next_sales_order+a+tsn+k[:-3]+" "*(10-len(k[:-3]))+"0"*16+str(2000))
    else:
        for k in loads:
            loadls.append("2DEV:"+release_order_number+" "*(10-len(release_order_number))+"000"+sales_order_item+a+tsn+k[:-3]+" "*(10-len(k[:-3]))+"0"*16+str(2000))
        
    if double_load:
        number_of_lines=len(first_textsplit)+len(second_textsplit)+4
    else:
        number_of_lines=len(loads)+3
    end_initial="0"*(4-len(str(number_of_lines)))
    end=f"9TRL:{end_initial}{number_of_lines}"
    Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
    for i in loads:
        #st.write(i)
        try:
              
            Inventory.loc[Inventory["Lot"]==i,"Location"]="ON TRUCK"
            Inventory.loc[Inventory["Lot"]==i,"Warehouse_Out"]=datetime.datetime.combine(file_date,file_time)
            Inventory.loc[Inventory["Lot"]==i,"Vehicle_Id"]=str(vehicle_id)
            Inventory.loc[Inventory["Lot"]==i,"Release_Order_Number"]=str(release_order_number)
            Inventory.loc[Inventory["Lot"]==i,"Carrier_Code"]=str(carrier_code)
            Inventory.loc[Inventory["Lot"]==i,"Terminal Bill Of Lading"]=str(terminal_bill_of_lading)
        except:
            st.write("Check Unit Number,Unit Not In Inventory")
        #st.write(vehicle_id)

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
def generate_bill_of_lading(vessel,release_order,sales_order,carrier_id,vehicle,quantity):
    data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
    bill_of_ladings=json.loads(data)
    list_of_ladings=[]
    try:
        for key in bill_of_ladings:
            list_of_ladings.append(int(key))
        bill_of_lading_number=max(list_of_ladings)+1
    except:
        bill_of_lading_number=115240
    return bill_of_lading_number,bill_of_ladings

gty=1
authenticated_username='ayilmaz'
if gty==1:
    
    if authenticated_username == 'ayilmaz':
        #print(f'Welcome {name}')



    
    
        select=st.sidebar.radio("SELECT FUNCTION",
            ('ADMIN', 'LOADOUT', 'INVENTORY'))
        
            #tab1,tab2,tab3,tab4= st.tabs(["UPLOAD SHIPMENT FILE","ENTER LOADOUT DATA","INVENTORY","CAPTURE"])
            
        
            
        if select=="ADMIN" :
            admin_tab1,admin_tab2,admin_tab3=st.tabs(["RELEASE ORDERS","BILL OF LADINGS","SHIPMENT FILES"])
            with admin_tab2:
                bill_data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
                admin_bill_of_ladings=json.loads(bill_data)
                st.dataframe(pd.DataFrame.from_dict(admin_bill_of_ladings).T)
            
            with admin_tab3:
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
                                  [str(i)[20:28] for i in temp[7]])),columns=["Lot","Lot Qty","B/L"])
                        df["Lot Qty"]=[int(int(i)/2) for i in df["Lot Qty"]]
                        df["Wrap"]=[i[:3] for i in temp[1]]
                        df["Vessel"]=[i[-12:] for i in temp[7]]
                        df["DryWeight"]=[int(i) for i in temp[8]]
                        df["ADMT"]=[int(i)/0.9/100000 for i in temp[8]]
                        new_list=[]
                        lotq=[]
                        bl=[]
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
                                bl.append(str(df.loc[i,"B/L"]))
                                wrap.append(df.loc[i,"Wrap"])
                                vessel.append(df.loc[i,"Vessel"])
                                DryWeight.append(df.loc[i,"DryWeight"])
                                ADMT.append(df.loc[i,"ADMT"])
                        new_df=pd.DataFrame(list(zip(new_list,lotq,bl,wrap,vessel,DryWeight,ADMT)),columns=df.columns.to_list())
                        new_df["Location"]="OLYM"
                        new_df["Warehouse_In"]="8/24/2023"
                        new_df["Warehouse_Out"]=""
                        new_df["Vehicle_Id"]=""
                        new_df["Release_Order_Number"]=""
                        new_df["Carrier_Code"]=""
                        new_df["BL"]=""
                        bls=new_df["B/L"].value_counts()
                        wraps=[new_df[new_df["B/L"]==k]["Wrap"].unique()[0] for k in bls.keys()]
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
                            st.write(requested_shipping_file[["Lot","Lot Qty","B/L","Wrap","Ocean B/L","DryWeight","ADMT","Location","Warehouse_In","Warehouse_Out","Vehicle_Id","Release_Order_Number","Carrier_Code","BL"]])
            with admin_tab1:
                
                #st.markdown("RELEASE ORDERS") 
                
                #st.write(f'CURRENT RELEASE ORDERS : {list_files_in_folder("olym_suzano", "release_orders")[1:]}')
                release_order_tab1,release_order_tab2=st.tabs(["CREATE RELEASE ORDER","RELEASE ORDER DATABASE"])
                with release_order_tab1:
                    vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304"])
                    edit=st.checkbox("CHECK TO ADD TO EXISTING RELEASE ORDER")
                    batch_mapping=gcp_download("olym_suzano",rf"batch_mapping.json")
                    batch_mapping=json.loads(batch_mapping)
                    if edit:
                        #release_order_number=st.selectbox("SELECT RELEASE ORDER",(list_files_in_folder("olym_suzano", "release_orders/{vessel}")))
                        release_order_number=st.selectbox("SELECT RELEASE ORDER",([i.replace(".json","") for i in list_files_in_subfolder("olym_suzano", rf"release_orders/KIRKENES-2304/")]))
                    else:
                        
                        release_order_number=st.text_input("Release Order Number")
                    sales_order_item=st.text_input("Sales Order Item")
                    ocean_bill_of_lading=st.selectbox("Ocean Bill Of Lading",batch_mapping.keys())
                    batch=st.text_input("Batch No",batch_mapping[ocean_bill_of_lading]["batch"],disabled=True)
                    dryness=st.text_input("Dryness",batch_mapping[ocean_bill_of_lading]["dryness"],disabled=True)
                    quantity=st.number_input("Quantity of Units", min_value=1, max_value=800, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                    tonnage=2*quantity
                    #queue=st.number_input("Place in Queue", min_value=1, max_value=20, value=1, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
                    transport_type=st.radio("Select Transport Type",("TRUCK","RAIL"))
                    carrier_code=st.text_input("Carrier Code")            
                    
        
                    create_release_order=st.button("SUBMIT")
                    if create_release_order:
                        
                        if edit: 
                            data=gcp_download("olym_suzano",rf"release_orders/{vessel}/{release_order_number}.json")
                            to_edit=json.loads(data)
                            temp=edit_release_order_data(to_edit,vessel,release_order_number,sales_order_item,batch,ocean_bill_of_lading,dryness,quantity,tonnage,transport_type,carrier_code)
                            st.write(f"ADDED sales order item {sales_order_item} to release order {release_order_number}!")
                        else:
                            
                            temp=store_release_order_data(vessel,release_order_number,sales_order_item,batch,ocean_bill_of_lading,dryness,quantity,tonnage,transport_type,carrier_code)
                            #st.write(temp)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"release_orders/{vessel}/{release_order_number}.json")
                        blob.upload_from_string(temp)
                        st.write(f"Recorded Release Order - {release_order_number} for Item No: {sales_order_item}")
                        
                with release_order_tab2:
                    
                    vessel=st.selectbox("SELECT VESSEL",["KIRKENES-2304"],key="other")
                    rls_tab1,rls_tab2=st.tabs(["ACTIVE RELEASE ORDERS","COMPLETED RELEASE ORDERS"])
        
                    with rls_tab1:
                        
                                    
                        files_in_folder = [i.replace(".json","") for i in list_files_in_subfolder("olym_suzano", rf"release_orders/KIRKENES-2304/")]
                        requested_file=st.selectbox("ACTIVE RELEASE ORDERS",files_in_folder)
                        
                        nofile=0
                        try:
                            data=gcp_download("olym_suzano",rf"release_orders/{vessel}/{requested_file}.json")
                            release_order_json = json.loads(data)
                            
                            target=release_order_json[vessel][requested_file]
                            if len(target.keys())==0:
                                nofile=1
                           
                            number_of_sales_orders=len(target)
                            rel_col1,rel_col2,rel_col3=st.columns([2,2,2])
                        except:
                            nofile=1
                     
                        #### DISPATCHED CLEANUP  #######
                        
                        try:
                            dispatched=gcp_download("olym_suzano",rf"dispatched.json")
                            dispatched=json.loads(dispatched)
                            #st.write(dispatched)
                        except:
                            pass
                        to_delete=[]            
                        for i in dispatched.keys():
                            sales=dispatched[i]["sales_order"]
                            if target[sales]["remaining"]==0:
                                to_delete.append(i)
                        for k in to_delete:
                            dispatched.pop(k)
                            #st.write("deleted k")
                        if list(dispatched.keys())==["2","3"]:
                            dispatched["1"]=dispatched["2"]
                            dispatched["2"]=dispatched["3"]
                            del dispatched["3"]
                        if list(dispatched.keys())==["2"]:
                            dispatched["1"]=dispatched["2"]
                            del dispatched["2"]
                        
                        
                            
                        json_data = json.dumps(dispatched)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"dispatched.json")
                        blob.upload_from_string(json_data)
                        
                        
                        ###CLEAN DISPATCH
        
                        
                                              
                        if nofile!=1 :         
                                        
                            targets=[i for i in target] ####doing this cause we set jason path {downloadedfile[vessel][releaseorder] as target. i have to use one of the keys (release order number) that is in target list
                            sales_orders_completed=[k for k in targets if target[k]['remaining']<=0]
                            
                            with rel_col1:
                                
                                st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                if targets[0] in sales_orders_completed:
                                    st.markdown(f"**:orange[Sales Order Item : {targets[0]} - COMPLETED]**")
                                else:
                                    st.markdown(f"**:blue[Sales Order Item] : {targets[0]}**")
                                st.write(f"        Total Quantity-Tonnage : {target[targets[0]]['quantity']} Units - {target[targets[0]]['tonnage']} Metric Tons")
                                st.write(f"        Ocean Bill Of Lading : {target[targets[0]]['ocean_bill_of_lading']}")
                                st.write(f"        Batch : {target[targets[0]]['batch']}")
                                st.write(f"        Units Shipped : {target[targets[0]]['shipped']} Units - {2*target[targets[0]]['shipped']} Metric Tons")
                                if target[targets[0]]['remaining']<=10:
                                    st.markdown(f"**:red[Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Metric Tons]**")
                                else:
                                    st.write(f"       Units Remaining : {target[targets[0]]['remaining']} Units - {2*target[targets[0]]['remaining']} Metric Tons")
                            with rel_col2:
                                try:
                                
                                    st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                    st.markdown(f"**:blue[Sales Order Item] : {targets[1]}**")
                                    st.write(f"        Total Quantity-Tonnage : {target[targets[1]]['quantity']} Units - {target[targets[1]]['tonnage']} Metric Tons")                        
                                    st.write(f"        Ocean Bill Of Lading : {target[targets[1]]['ocean_bill_of_lading']}")
                                    st.write(f"        Batch : {target[targets[1]]['batch']}")
                                    st.write(f"        Units Shipped : {target[targets[1]]['shipped']} Units - {2*target[targets[1]]['shipped']} Metric Tons")
                                    st.write(f"        Units Remaining : {target[targets[1]]['remaining']} Units - {2*target[targets[1]]['remaining']} Metric Tons")
                                    
                                        
                                except:
                                    pass
                
                            with rel_col3:
                                try:
                                
                                    st.markdown(f"**:blue[Release Order Number] : {requested_file}**")
                                    st.markdown(f"**:blue[Sales Order Item] : {targets[2]}**")
                                    st.write(f"        Total Quantity-Tonnage : {target[targets[2]]['quantity']} Units - {target[targets[2]]['tonnage']} Metric Tons")
                                    st.write(f"        Ocean Bill Of Lading : {target[targets[1]]['ocean_bill_of_lading']}")
                                    st.write(f"        Batch : {target[targets[2]]['batch']}")
                                    st.write(f"        Units Shipped : {target[targets[2]]['shipped']} Units - {2*target[targets[2]]['shipped']} Metric Tons")
                                    st.write(f"        Units Remaining : {target[targets[2]]['remaining']} Units - {2*target[targets[2]]['remaining']} Metric Tons")
                                    
                                    
                                except:
                                    pass
            
                                   # dispatched={"vessel":vessel,"date":datetime.datetime.strftime(datetime.datetime.today()-datetime.timedelta(hours=7),"%b-%d-%Y"),
                                     #               "time":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%H:%M:%S"),
                                       #                 "release_order":requested_file,"sales_order":hangisi,"ocean_bill_of_lading":ocean_bill_of_lading,"batch":batch}
                            
                            hangisi=st.selectbox("SELECT SALES ORDER ITEM TO DISPATCH",([i for i in target if i not in sales_orders_completed]))
                            dol1,dol2,dol3,dol4=st.columns([2,2,2,2])
                            with dol1:
                               
                                       
                                if st.button("DISPATCH TO WAREHOUSE",key="lala"):
                                   
                                    
                                    
                                    dispatch=dispatched.copy()
                                    try:
                                        last=list(dispatch.keys())[-1]
                                        dispatch[str(int(last)+1)]={"vessel":vessel,"date":datetime.datetime.strftime(datetime.datetime.today()-datetime.timedelta(hours=7),"%b-%d-%Y"),
                                                    "time":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%H:%M:%S"),
                                                     "release_order":requested_file,"sales_order":hangisi,"ocean_bill_of_lading":target[hangisi]["ocean_bill_of_lading"],"batch":target[hangisi]["batch"]}
                                    except:
                                        dispatch["1"]={"vessel":vessel,"date":datetime.datetime.strftime(datetime.datetime.today()-datetime.timedelta(hours=7),"%b-%d-%Y"),
                                                    "time":datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7),"%H:%M:%S"),
                                                     "release_order":requested_file,"sales_order":hangisi,"ocean_bill_of_lading":target[hangisi]["ocean_bill_of_lading"],"batch":target[hangisi]["batch"]}
            
                                    
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
                                item=st.selectbox("CHOOSE ITEM",dispatch.keys())
                                if st.button("CLEAR DISPATCH ITEM"):                                       
                                    del dispatch[item]
                                    json_data = json.dumps(dispatch)
                                    storage_client = storage.Client()
                                    bucket = storage_client.bucket("olym_suzano")
                                    blob = bucket.blob(rf"dispatched.json")
                                    blob.upload_from_string(json_data)
                                    st.markdown(f"**CLEARED DISPATCH ITEM {item}**")   
                            st.markdown("**CURRENT DISPATCH QUEUE**")
                            try:
                                dispatch=gcp_download("olym_suzano",rf"dispatched.json")
                                dispatch=json.loads(dispatch)
                                try:
                                    for i in dispatch.keys():
                                        st.write(f'**ORDER:{i}**___Release Order = {dispatch[i]["release_order"]}, Item No: {dispatch[i]["sales_order"]}')
                                except:
                                    pass
                            except:
                                st.write("NO DISPATCH ITEMS")
                        
                        else:
                            st.write("NO RELEASE ORDERS IN DATABASE")
                    with rls_tab2:
                            pass
        
                        
        
        ##########  LOAD OUT  ##############
        
        
        
        if select=="LOADOUT" :
        
            
            bill_mapping=gcp_download("olym_suzano","bill_mapping.json")
            bill_mapping=json.loads(bill_mapping)
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
            #st.write(dispatched)
            
            double_load=False
            
            if len(dispatched.keys())>0 and not no_dispatch:
                vessel=dispatched["1"]["vessel"]
                current_release_order=dispatched['1']['release_order']
                current_sales_order=dispatched['1']['sales_order']
                try:
                    next_release_order=dispatched['2']['release_order']
                    next_sales_order=dispatched['2']['sales_order']
                    
                except:
                    
                    pass
                info=gcp_download("olym_suzano",rf"release_orders/{dispatched['1']['vessel']}/{dispatched['1']['release_order']}.json")
                info=json.loads(info)
                
                
                if st.checkbox("CLICK TO LOAD MIXED SKU"):
                    try:
                        next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                        double_load=True
                    except:
                        st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                    
            
                load_col1,load_col2,load_col3=st.columns([4,4,2])
                with load_col1:
                    st.markdown(rf'**:blue[CURRENTLY WORKING] : Release Order-{current_release_order}**')
                    st.markdown(rf'**Sales Order Item-{current_sales_order}**')
                    st.markdown(f'**Ocean Bill Of Lading : {info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"]}**')
                    st.markdown(rf'**Total Quantity : {info[vessel][current_release_order][current_sales_order]["quantity"]}**')
                    st.markdown(rf'**Shipped : {info[vessel][current_release_order][current_sales_order]["shipped"]}**')
                    remaining=info[vessel][current_release_order][current_sales_order]["remaining"]                #######      DEFINED "REMAINING" HERE FOR CHECKS
                    if remaining<10:
                        st.markdown(rf'**:red[CAUTION : Remaining : {info[vessel][current_release_order][current_sales_order]["remaining"]}]**')
                    st.markdown(rf'**Remaining : {info[vessel][current_release_order][current_sales_order]["remaining"]}**')
                    
                with load_col2:
                    if double_load:
                        
                        try:
                            st.markdown(rf'**NEXT ITEM : Release Order-{next_release_order}**')
                            st.markdown(rf'**Sales Order Item-{next_sales_order}**')
                            st.markdown(f'**Ocean Bill Of Lading : {info[vessel][next_release_order][next_sales_order]["ocean_bill_of_lading"]}**')
                            st.markdown(rf'**Total Quantity : {info[vessel][next_release_order][next_sales_order]["quantity"]}**')
                        except:
                            pass
                      
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
                    file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=7),disabled=True)
                    delivery_date=st.date_input("Delivery Date",datetime.datetime.today()-datetime.timedelta(hours=7),key="delivery_date",disabled=True)
                    eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date",disabled=True)
                    
                with col2:
                    if double_load:
                        release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                        sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                        ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                        batch=st.text_input("Batch",info[vessel][current_release_order][current_sales_order]["batch"],disabled=True)
                        #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                        pass
                    else:
                        release_order_number=st.text_input("Release Order Number",current_release_order,disabled=True,help="Release Order Number without the Item no")
                        sales_order_item=st.text_input("Sales Order Item (Material Code)",current_sales_order,disabled=True)
                        ocean_bill_of_lading=st.text_input("Ocean Bill Of Lading",info[vessel][current_release_order][current_sales_order]["ocean_bill_of_lading"],disabled=True)
                        batch=st.text_input("Batch",info[vessel][current_release_order][current_sales_order]["batch"],disabled=True)
                        #terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",disabled=False)
                   
                        
                    
                with col3: 
                    carrier_code=st.text_input("Carrier Code",info[vessel][current_release_order][current_sales_order]["carrier_code"],disabled=True)
                    transport_sequential_number=st.selectbox("Transport Sequential",["TRUCK","RAIL"],disabled=True)
                    transport_type=st.selectbox("Transport Type",["TRUCK","RAIL"],disabled=True)
                    vehicle_id=st.text_input("**:blue[Vehicle ID]**")
                
                    
               
                with col4:
                    updated_quantity=0
                    live_quantity=0
                    if updated_quantity not in st.session_state:
                        st.session_state.updated_quantity=updated_quantity
                    def audit_unit(x):
                            if len(x)==11:
                                #st.write(bill_mapping[x[:-3]]["Batch"])
                                st.write(Inventory_Audit[Inventory_Audit["Lot"]==x]["Location"])
                                if bill_mapping[x[:-3]]["Ocean_bl"]!=ocean_bill_of_lading and bill_mapping[x[:-3]]["Batch"]!=batch:
                                    st.write("WRONG B/L, DO NOT LOAD!")
                                
                                #if Inventory_Audit[Inventory_Audit["Lot"]==x]["Location"]!="OLYM":
                                   # st.write("THIS UNIT HAS BEEN SHIPPED")
                                
                                else:
                                    return True
                    def audit_split(release,sales):
                            if len(x)==11:
                                #st.write(bill_mapping[x[:-3]]["Batch"])
                                
                                if bill_mapping[x[:-3]]["Ocean_bl"]!=info[vessel][release][sales]["ocean_bill_of_lading"] and bill_mapping[x[:-3]]["Batch"]!=info[vessel][release][sales]["batch"]:
                                    st.write("WRONG B/L, DO NOT LOAD!")
                                if Inventory_Audit[Inventory_Audit["Lot"]==x]["Location"]!="OLYM":
                                    st.write("THIS UNIT HAS BEEN SHIPPED")
                                else:
                                    return True
                    
                    flip=False 
                    first_load_input=None
                    second_load_input=None
                    if double_load:
                        
                        try:
                            next_item=gcp_download("olym_suzano",rf"release_orders/{dispatched['2']['vessel']}/{dispatched['2']['release_order']}.json")
                            
                            first_load_input=st.text_area("**FIRST SKU LOADS**",height=300)
                            first_quantity=0
                            second_quantity=0
                            if first_load_input is not None:
                                first_textsplit = first_load_input.splitlines()
                                first_quantity=len(first_textsplit)
                            second_load_input=st.text_area("**SECOND SKU LOADS**",height=300)
                            if second_load_input is not None:
                                second_textsplit = second_load_input.splitlines()
                                second_quantity=len(second_textsplit)
                            updated_quantity=first_quantity+second_quantity
                            st.session_state.updated_quantity=updated_quantity
                        except Exception as e: 
                            st.write(e)
                            #st.markdown("**:red[ONLY ONE ITEM IN QUEUE ! ASK NEXT ITEM TO BE DISPATCHED!]**")
                            pass
                        
                    
                    else:
                        
        
        
                    
                        load_input=st.text_area("**LOADS**",height=300)#[:-3]
                        if load_input is not None:
                            textsplit = load_input.splitlines()
                            updated_quantity=len(textsplit)
                            st.session_state.updated_quantity=updated_quantity
                            #st.write(textsplit)
                            
                    quantity=st.number_input("**:blue[Quantity of Units]**",st.session_state.updated_quantity, key=None, help=None, on_change=None, disabled=True, label_visibility="visible")
                    st.markdown(f"{quantity*2} TONS - {round(quantity*2*2204.62,1)} Pounds")
                        
                        
                            
                    
                    
                    
                    
                 
                   
                with col5:
                    Inventory_Audit=gcp_csv_to_df("olym_suzano", "Inventory.csv")
                    st.write(Inventory_Audit)
                    if double_load:
                        first_faults=[]
                        if first_load_input is not None:
                            first_textsplit = first_load_input.splitlines()
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
                            #st.write(textsplit)
                            for i,x in enumerate(second_textsplit):
                                if audit_split(next_release_order,next_sales_order):
                                    st.text_input(f"Unit No : {j+1+i+1}",x)
                                    second_faults.append(0)
                                else:
                                    st.text_input(f"Unit No : {j+1+i+1}",x)
                                    second_faults.append(1)
        
                        loads=[]
                        for k in first_textsplit:
                            loads.append(k)
                        for l in second_textsplit:
                            loads.append(l)
                    
                    else:
                        
        
                    
                        faults=[]
                        if load_input is not None:
                            textsplit = load_input.splitlines()
                            #st.write(textsplit)
                            
                            for i,x in enumerate(textsplit):
                                st.write(Inventory_Audit[Inventory_Audit["Lot"]==x]["Location"])
                                if audit_unit(x):
                                    st.text_input(f"Unit No : {i+1}",x)
                                    faults.append(0)
                                else:
                                    st.text_input(f"Unit No : {i+1}",x)
                                    faults.append(1)
                        loads=[]
                        for k in textsplit:
                            loads.append(k)
                   
                                  
                a=datetime.datetime.strftime(file_date,"%Y%m%d")
                
                b=file_time.strftime("%H%M%S")
                c=datetime.datetime.strftime(eta_date,"%Y%m%d")
                
                
                    
                
                if st.button('SUBMIT EDI'):
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
                            bill_of_lading_number=115240
                        return bill_of_lading_number,bill_of_ladings
                    #st.write(bill_of_lading_number)
                    
                    
                    if double_load:
                        bill_of_lading_number,bill_of_ladings=gen_bill_of_lading()
                        bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"sales_order":current_sales_order,"carrier_id":carrier_code,"vehicle":vehicle_id,"quantity":len(first_textsplit)} 
                        bill_of_ladings[str(bill_of_lading_number+1)]={"vessel":vessel,"release_order":release_order_number,"sales_order":next_sales_order,"carrier_id":carrier_code,"vehicle":vehicle_id,"quantity":len(second_textsplit)} 
                    else:
                        bill_of_lading_number,bill_of_ladings=gen_bill_of_lading()
                        bill_of_ladings[str(bill_of_lading_number)]={"vessel":vessel,"release_order":release_order_number,"sales_order":sales_order_item,"carrier_id":carrier_code,"vehicle":vehicle_id,"quantity":st.session_state.updated_quantity}            
                        
                 
                    bill_of_ladings=json.dumps(bill_of_ladings)
                    storage_client = storage.Client()
                    bucket = storage_client.bucket("olym_suzano")
                    blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                    blob.upload_from_string(bill_of_ladings)
                    
                    
                    terminal_bill_of_lading=st.text_input("Terminal Bill of Lading",bill_of_lading_number,disabled=True)
                    
                    proceed=False
                    if double_load:
                        if 1 in first_faults or 1 in second_faults:
                            st.markdown(f"**:red[CAN NOT SUBMIT EDI!!] CHECK BELOW UNTIS**")
                            for i in first_faults:
                                if i==1:
                                    st.markdown(f"**Unit{first_faults.index(i)+1}**")
                            for i in second_faults:
                                if i==1:
                                    st.markdown(f"**Unit{second_faults.index(i)+1}**")
                        else:
                            proceed=True
                    else:
                        if 1 in faults:
                            proceed=False
                            for i in faults:
                                if i==1:
                                    st.markdown(f"**Unit{faults.index(i)+1}**")
                        else:
                            proceed=True
                    if remaining<0:
                        proceed=False
                        error="No more Items to ship on this Sales Order"
                        st.write(error)
                    if not vehicle_id: 
                        proceed=False
                        error="Please check Vehicle ID "
                        st.write(error)
                    if len(terminal_bill_of_lading)<6:
                        proceed=False
                        error="Please check Terminal Bill Of Lading. It should have 6 digits."
                        st.write(error)
                    if quantity<10:
                        proceed=False
                        error=f"{quantity} loads on this truck. Please check "
                        st.write(error)
                    if proceed:
                        
                        process()
                        if double_load:
                            info[vessel][current_release_order][current_sales_order]["shipped"]=info[vessel][current_release_order][current_sales_order]["shipped"]+len(first_textsplit)
                            info[vessel][current_release_order][current_sales_order]["remaining"]=info[vessel][current_release_order][current_sales_order]["remaining"]-len(first_textsplit)
                            info[vessel][next_release_order][next_sales_order]["shipped"]=info[vessel][next_release_order][next_sales_order]["shipped"]+len(second_textsplit)
                            info[vessel][next_release_order][next_sales_order]["remaining"]=info[vessel][next_release_order][next_sales_order]["remaining"]-len(second_textsplit)
                        else:
                            info[vessel][current_release_order][current_sales_order]["shipped"]=info[vessel][current_release_order][current_sales_order]["shipped"]+len(loads)
                            info[vessel][current_release_order][current_sales_order]["remaining"]=info[vessel][current_release_order][current_sales_order]["remaining"]-len(loads)
                        if info[vessel][current_release_order][current_sales_order]["remaining"]==0:
                            to_delete=[]
                            for i in dispatched.keys():
                                if dispatched[i]["release_order"]==current_release_order and dispatched[i]["sales_order"]==current_sales_order:
                                    to_delete.append(i)
                            for k in to_delete:
                                del dispatched[k]
                            if list(dispatched.keys())==["2","3"]:
                                dispatched["1"]=dispatched["2"]
                                dispatched["2"]=dispatched["3"]
                                del dispatched["3"]
                            if list(dispatched.keys())==["2"]:
                                dispatched["1"]=dispatched["2"]
                                del dispatched["2"]
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
                        with open('placeholder.txt', 'r') as f:
                            output_text = f.read()
                        st.markdown("**SUCCESS! EDI FOR THIS LOAD HAS BEEN SUBMITTED,THANK YOU**")
                        st.markdown("**EDI TEXT**")
                        st.text_area('', value=output_text, height=600)
                        with open('placeholder.txt', 'r') as f:
                            file_content = f.read()
                        newline="\n"
                        filename = f'{a}{b}OLYM'
                        file_name= f'{a}{b}OLYM.txt'
                        st.write(filename)
                        subject = f'Suzano_EDI_{a}_{release_order_number}'
                        body = f"EDI for Release Order Number {current_release_order} is attached.{newline}For Carrier Code:{carrier_code} and Bill of Lading: {terminal_bill_of_lading}, {len(loads)} loads were loaded to vehicle {vehicle_id}."
                        sender = "warehouseoly@gmail.com"
                        #recipients = ["alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
                        recipients = ["afsiny@portolympia.com"]
                        password = "xjvxkmzbpotzeuuv"
                
                          # Replace with the actual file path
                
                
                        with open('temp_file.txt', 'w') as f:
                            f.write(file_content)
                
                        file_path = 'temp_file.txt'  # Use the path of the temporary file
                
                        send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
                        upload_cs_file("olym_suzano", 'temp_file.txt',file_name) 
                    else:   ###cancel bill of lading
                        data=gcp_download("olym_suzano",rf"terminal_bill_of_ladings.json")
                        bill_of_ladings=json.loads(data)
                        del bill_of_ladings[str(bill_of_lading_number)]
                        bill_of_ladings=json.dumps(bill_of_ladings)
                        storage_client = storage.Client()
                        bucket = storage_client.bucket("olym_suzano")
                        blob = bucket.blob(rf"terminal_bill_of_ladings.json")
                        blob.upload_from_string(bill_of_ladings)
        
            
                        
            else:
                st.subheader("**Nothing dispatched!**")
                    
            
                    
                        
        
        
                
                        
        if select=="INVENTORY" :
            Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
            
            
            dab1,dab2=st.tabs(["IN WAREHOUSE","SHIPPED"])
            df=Inventory[Inventory["Location"]=="OLYM"][["Lot","Batch","Ocean B/L","Wrap","DryWeight","ADMT","Location","Warehouse_In"]]
            zf=Inventory[Inventory["Location"]=="ON TRUCK"][["Lot","Batch","Ocean B/L","Wrap","DryWeight","ADMT","Release_Order_Number","Carrier_Code","Terminal B/L",
                                                             "Vehicle_Id","Warehouse_In","Warehouse_Out"]]
            items=df["Ocean B/L"].unique().tolist()
            
            with dab1:
                
                inv_col1,inv_col2,inv_col3=st.columns([2,6,2])
                with inv_col1:
                    st.markdown(f"**IN WAREHOUSE = {len(df)}**")
                    st.markdown(f"**TOTAL SHIPPED = {len(zf)}**")
                    st.markdown(f"**TOTAL OVERALL = {len(zf)+len(df)}**")
                with inv_col2:
                    #st.write(items)
                    inhouse=[df[df["Ocean B/L"]==i].shape[0] for i in items]
                    shipped=[zf[zf["Ocean B/L"]==i].shape[0] for i in items]
                    tablo=pd.DataFrame({"Ocean B/L":items,"In Warehouse":inhouse,"Shipped":shipped},index=[i for i in range(1,len(items)+1)])
                    total_row={"Ocean B/L":"TOTAL","In Warehouse":sum(inhouse),"Shipped":sum(shipped)}
                    tablo = tablo.append(total_row, ignore_index=True)
                    tablo["TOTAL"] = tablo.loc[:, ["In Warehouse", "Shipped"]].sum(axis=1)
         
                    st.dataframe(tablo)
                if st.checkbox("CLICK TO SEE INVENTORY LIST"):
                    st.table(df)
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
                
                
                #st.write(zf)
                #zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]]=zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]].astype("int")
                zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]]=zf[["Release_Order_Number","Carrier_Code","Terminal B/L","Vehicle_Id"]].astype("str")
                
                zf["Warehouse_Out"]=[datetime.datetime.strptime(j,"%Y-%m-%d %H:%M:%S") for j in zf["Warehouse_Out"]]
                filtered_zf=zf.copy()
                if date_filter:
                    filtered_zf["Warehouse_Out"]=[i.date() for i in filtered_zf["Warehouse_Out"]]
                    
                    filtered_zf=filtered_zf[filtered_zf["Warehouse_Out"]==filter_date]
                    
                filter_by=st.selectbox("SELECT FILTER",["Wrap","Ocean B/L","Release_Order_Number","Terminal B/L","Carrier_Code","Vehicle_Id"])
                #st.write(filter_by)
                choice=st.selectbox(f"Filter By {filter_by}",[f"ALL {filter_by.upper()}"]+[str(i) for i in filtered_zf[filter_by].unique().tolist()])
                
                
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
        #with tab4:
        #     df=gcp_csv_to_df("olym_suzano", "Inventory.csv")
        #    st.write(df)
    elif username == 'rbriggs':
        st.write(f'Welcome *{name}*')
        st.title('Application 2')
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


