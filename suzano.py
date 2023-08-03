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

from google.cloud import storage
import os
import io



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



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'

st. set_page_config(layout="wide")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"



user="a"
    
if user :
    
    st.write(user.upper())
    tab1,tab2,tab3= st.tabs(["ENTER DATA","INVENTORY","CAPTURE"])
    
    
    if 'captured_units' not in st.session_state:
        st.session_state.captured_units =[]
        
        
        
    with tab1:
        col1, col2,col3,col4,col5= st.columns([2,2,2,2,2])
        with col1:
        
    
            file_date=st.date_input("File Date",datetime.datetime.today()-datetime.timedelta(hours=7),key="file_dates")
            if file_date not in st.session_state:
                st.session_state.file_date=file_date
            file_time = st.time_input('FileTime', datetime.datetime.now()-datetime.timedelta(hours=7))
            terminal_code=st.text_input("Terminal Code","OLYM")
            release_order_number=st.text_input("Release Order Number (FROM SUZANO)")
            
            if release_order_number not in st.session_state:
                st.session_state.release_order_number=release_order_number
            delivery_date=st.date_input("Delivery Date",datetime.datetime.today(),key="delivery_date")
        with col2:
            transport_sequential_number=st.selectbox("Transport Sequential",["TRUCK","RAIL"])
            transport_type=st.selectbox("Transport Type",["TRUCK","RAIL"])
            vehicle_id=st.text_input("Vehicle ID")
            quantity=st.number_input("Quantity In Tons", min_value=1, max_value=24, value=20, step=1,  key=None, help=None, on_change=None, disabled=False, label_visibility="visible")
            frame_placeholder = st.empty()
        with col3: 
            carrier_code=st.text_input("Carrier Code")
            bill_of_lading=st.text_input("Bill of Lading")
            eta_date=st.date_input("ETA Date (For Trucks same as delivery date)",delivery_date,key="eta_date")
            sales_order_item=st.text_input("Sales Order Item (Material Code)")
        with col4:
            
                
            
            try:
                load1=st.text_input("Unit No : 01",value=st.session_state.captured_units[0])
            except:
                load1=st.text_input("Unit No : 01")
            try:
                load2=st.text_input("Unit No : 02",value=st.session_state.captured_units[1])
            except:
                load2=st.text_input("Unit No : 02")
            try:
                load3=st.text_input("Unit No : 03",value=st.session_state.captured_units[2])
            except:
                load3=st.text_input("Unit No : 03")
            try:
                load4=st.text_input("Unit No : 04",value=st.session_state.captured_units[3])
            except:
                load4=st.text_input("Unit No : 04")
            try:
                load5=st.text_input("Unit No : 05",value=st.session_state.captured_units[4])
            except:
                load5=st.text_input("Unit No : 05")
            
            
         
           
        with col5:
            
            load6=st.text_input("Unit No : 06")
            load7=st.text_input("Unit No : 07")
            load8=st.text_input("Unit No : 08")
            load9=st.text_input("Unit No : 09")
            load10=st.text_input("Unit No : 10")
            
        gloads=[load1,load2,load3,load4,load5,load6,load7,load8,load9,load10]
        loads=[]
        for i in gloads:
            if i:
                loads.append(i)
                          
        a=datetime.datetime.strftime(file_date,"%Y%m%d")
        
        b=file_time.strftime("%H%M%S")
        c=datetime.datetime.strftime(eta_date,"%Y%m%d")
        
        #st.write(f'1HDR:{datetime.datetime.strptime(file_date,"%y%m%d")}')
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
            
        def process():
            line1="1HDR:"+a+b+terminal_code
            tsn="01" if transport_sequential_number=="TRUCK" else "02"
            tt="0001" if transport_type=="TRUCK" else "0002"
            line2="2DTD:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+tt+vehicle_id+" "*(20-len(vehicle_id))+str(quantity*1000)+" "*(16-len(str(quantity*1000)))+"USD"+" "*36+carrier_code+" "*(10-len(carrier_code))+bill_of_lading+" "*(50-len(bill_of_lading))+c
            loadls=[]
            if load1:
                loadl1="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load1+" "*(10-len(load1))+"0"*16+str(quantity*100)
                loadls.append(loadl1)
            if load2:
                loadl2="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load2+" "*(10-len(load2))+"0"*16+str(quantity*100)
                loadls.append(loadl2)
            if load3:
                loadl3="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load3+" "*(10-len(load3))+"0"*16+str(quantity*100)
                loadls.append(loadl3)
            if load4:
                loadl4="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load4+" "*(10-len(load4))+"0"*16+str(quantity*100)
                loadls.append(loadl4)
            if load5:
                loadl5="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load5+" "*(10-len(load5))+"0"*16+str(quantity*100)
                loadls.append(loadl5)
            if load6:
                loadl6="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load6+" "*(10-len(load6))+"0"*16+str(quantity*100)
                loadls.append(loadl6)
            if load7:
                loadl7="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load7+" "*(10-len(load7))+"0"*16+str(quantity*100)
                loadls.append(loadl7)
            if load8:
               loadl8="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load8+" "*(10-len(load8))+"0"*16+str(quantity*100)
               loadls.append(loadl8)
            if load9:
                loadl9="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load9+" "*(10-len(load9))+"0"*16+str(quantity*100)
                loadls.append(loadl9)
            if load10:
                loadl10="2DEV:"+release_order_number+" "*(10-len(release_order_number))+sales_order_item+a+tsn+load10+" "*(10-len(load10))+"0"*16+str(quantity*100)
                loadls.append(loadl10)
            end="9TRL:0013"
            Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
            for i in loads:
                #st.write(i)
                try:
                      
                    Inventory.loc[Inventory["Lot"]==i,"Location"]="ON TRUCK"
                    Inventory.loc[Inventory["Lot"]==i,"Warehouse_Out"]=datetime.datetime.combine(file_date,file_time)
                    Inventory.loc[Inventory["Lot"]==i,"Vehicle_Id"]=str(vehicle_id)
                    Inventory.loc[Inventory["Lot"]==i,"Release_Order_Number"]=str(release_order_number)
                    Inventory.loc[Inventory["Lot"]==i,"Carrier_Code"]=str(carrier_code)
                    Inventory.loc[Inventory["Lot"]==i,"BL"]=str(bill_of_lading)
                except:
                    st.write("Check Unit Number,Unit Not In Inventory")
                #st.write(vehicle_id)
    
                temp=Inventory.to_csv("temp.csv")
                upload_cs_file("olym_suzano", 'temp.csv',"Inventory.csv") 
            with open(f'placeholder.txt', 'w') as f:
                f.write(line1)
                f.write('\n')
                f.write(line2)
                f.write('\n')
                
                for i in loadls:
                    
                    f.write(i)
                    f.write('\n')
    
                f.write(end)
            
                
        try:
            down_button=st.download_button(label="Download EDI as TXT",on_click=process,data=output(),file_name=f'Suzano_EDI_{a}_{release_order_number}.txt')
        except:
            pass        
        if st.button('SAVE/DISPLAY EDI'):
            process()
            with open('placeholder.txt', 'r') as f:
                output_text = f.read()
            st.markdown("**EDI TEXT**")
            st.text_area('', value=output_text, height=600)
            with open('placeholder.txt', 'r') as f:
                file_content = f.read()
            newline="\n"
            filename = f'Suzano_EDI_{a}_{release_order_number}'
            file_name= f'Suzano_EDI_{a}_{release_order_number}.txt'
            st.write(filename)
            subject = f'Suzano_EDI_{a}_{release_order_number}'
            body = f"EDI for Release Order Number {release_order_number} is attached.{newline}For Carrier Code:{carrier_code} and Bill of Lading: {bill_of_lading}, {len(loads)} loads were loaded to vehicle {vehicle_id}."
            sender = "warehouseoly@gmail.com"
            #recipients = ["afsin1977@gmail.com","alexandras@portolympia.com","conleyb@portolympia.com", "afsiny@portolympia.com"]
            recipients = ["afsiny@portolympia.com"]
            password = "xjvxkmzbpotzeuuv"
    
              # Replace with the actual file path
    
    
            with open('temp_file.txt', 'w') as f:
                f.write(file_content)
    
            file_path = 'temp_file.txt'  # Use the path of the temporary file
    
            send_email_with_attachment(subject, body, sender, recipients, password, file_path,file_name)
            upload_cs_file("olym_suzano", 'temp_file.txt',file_name) 
    
            
    
            
                
    
    
            
                    
    with tab2:
        Inventory=gcp_csv_to_df("olym_suzano", "Inventory.csv")
        
        
        dab1,dab2=st.tabs(["IN WAREHOUSE","SHIPPED"])
        df=Inventory[Inventory["Location"]=="OLYM"][["Lot","Location","Warehouse_In"]]
        zf=Inventory[Inventory["Location"]=="ON TRUCK"][["Lot","Release_Order_Number","Carrier_Code","BL",
                                                         "Vehicle_Id","Warehouse_In","Warehouse_Out"]]
        with dab1:
            
            st.markdown(f"**IN WAREHOUSE = {len(df)}**")
            st.markdown(f"**TOTAL SHIPPED = {len(zf)}**")
            st.markdown(f"**TOTAL OVERALL = {len(zf)+len(df)}**")
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
            #zf[["Release_Order_Number","Carrier_Code","BL","Vehicle_Id"]]=zf[["Release_Order_Number","Carrier_Code","BL","Vehicle_Id"]].astype("int")
            zf[["Release_Order_Number","Carrier_Code","BL","Vehicle_Id"]]=zf[["Release_Order_Number","Carrier_Code","BL","Vehicle_Id"]].astype("str")
            zf["Warehouse_Out"]=[datetime.datetime.strptime(j,"%Y-%m-%d %H:%M:%S") for j in zf["Warehouse_Out"]]
            filtered_zf=zf.copy()
            if date_filter:
                filtered_zf["Warehouse_Out"]=[i.date() for i in filtered_zf["Warehouse_Out"]]
                filtered_zf=filtered_zf[filtered_zf["Warehouse_Out"]==filter_date]
            BL_filter=st.selectbox("Filter By Bill Of Lading",["ALL BILL OF LADINGS"]+[str(i) for i in filtered_zf["BL"].unique().tolist()])
            vehicle_filter=st.selectbox("Filter By Vehicle_Id",["ALL VEHICLES"]+[str(i) for i in filtered_zf["Vehicle_Id"].unique().tolist()])
            carrier_filter=st.selectbox("Filter By Carrier_Id",["ALL CARRIERS"]+[str(i) for i in filtered_zf["Carrier_Code"].unique().tolist()])
            
            col1,col2=st.columns([2,8])
            with col1:
                st.markdown(f"**TOTAL SHIPPED = {len(zf)}**")
                st.markdown(f"**IN WAREHOUSE = {len(df)}**")
                st.markdown(f"**TOTAL OVERALL = {len(zf)+len(df)}**")
            
            
                    
            if BL_filter!="ALL BILL OF LADINGS":
                filtered_zf=filtered_zf[filtered_zf["BL"]==BL_filter]
            if carrier_filter!="ALL CARRIERS":
                filtered_zf=filtered_zf[filtered_zf["Carrier_Code"]==carrier_filter]
            if vehicle_filter!="ALL VEHICLES":
                filtered_zf=filtered_zf[filtered_zf["Vehicle_Id"]==vehicle_filter]
            with col2:
                if date_filter:
                    st.markdown(f"**SHIPPED ON THIS DAY = {len(filtered_zf)}**")
            st.table(filtered_zf)
    with tab3:
        df=gcp_csv_to_df("olym_suzano", "Inventory.csv")
        st.write(df)
        
    
