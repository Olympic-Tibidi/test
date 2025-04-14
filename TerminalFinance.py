import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import sys
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Side, Alignment, Protection, Font
from collections import defaultdict
import csv
import datetime
from datetime import date
import calendar
import pickle
import json
import os
import copy
import plotly.graph_objs as go
#from more_itertools import one
pd.set_option("display.max_rows",3000)
pd.set_option("expand_frame_repr", True)
import plotly.express as px               #to create interactive charts
import plotly.graph_objects as go         #to create interactive charts
from ipywidgets import interact
from plotly.subplots import make_subplots
#from jupyter_dash import JupyterDash      #to build Dash apps from Jupyter environments
#from dash import dcc        #to get components for interactive user interfaces
#from dash import html
from plotly.offline import iplot, init_notebook_mode
import chart_studio.plotly as py
from ipywidgets import interactive, HBox, VBox, widgets, interact
#from d3blocks import D3Blocks
import seaborn as sns
import io

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_rows', None)

ship_accounts=[6311000,6312000,6313000,6314000,6313950,6315000,6315100,6315200,6316000,6317030,6318540,6318600,
               6318900,6329000,6373000,6381000,6389000,6888888,6999999,6999998,
                   7313015,7311015,7314300,7313949,7315000,7338700]

def prep_ledger(dosya,dosya1,yr1,month1,yr2,month2):
    
    tf=pd.DataFrame()
    for dos in [dosya,dosya1]:
        
        df=pd.read_csv(dos,header=None)

        checkdate=datetime.datetime.strptime(df.loc[1,14].split(" ")[-1],"%m/%d/%Y")

        a=df.iloc[:,41:45]
        b=df.iloc[:,49:61]

        df=pd.concat([a,b],axis=1)
        df.drop(columns=[43,54,57,59],inplace=True)

        columns=["Account","Name","Sub_Cat","Bat_No","Per_Entry","Ref_No","Date","Description","Debit","Credit","Job_No","Period"]
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
        df["Acc"]=[f"{i}-{j}" for i,j in zip(df["Account"],df["Sub_Cat"])]
        df["Month"]=df["Date"].dt.month
        tf=pd.concat([tf,df])
    return tf


def apply_general_corrections(df):
    ship_accounts=[6311000,6312000,6313000,6314000,6313950,6315000,6315100,6315200,6316000,6317030,6318540,6318600,
               6318900,6329000,6373000,6381000,6389000,6888888,6999999,6999998,
                   7313015,7311015,7314300,7313949,7315000,7338700]
    df["Acc"]=[f"{i}-{j}" for i,j in zip(df["Account"],df["Sub_Cat"])]
    
    #### WEYCO CREDIT
#     df.loc[df["Acc"]=="6313002-32","Account"]=6341000
#     df.loc[df["Acc"]=="6313002-32","Name"]="Real Prop Rent - Land"
#     df.loc[df["Acc"]=="6313002-32","Acc"]="6341000-32"
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
    ###df.loc[(df["Description"]=="Rev Accr. SSA Pacific 12.23")&(df["Name"]=="Equipment Rentals"),"Debit"]=0

    #### LOADOUTS TO HANDLING

    df.loc[df['Ref_No'].isin(["058498","058283","058710","058923","058924","059168","059386","059624",
                                      "059625","059881","060097","060298","060096"
                       ]),"Name"]="Handling"
    df.loc[df['Ref_No'].isin(["058498","058283","058710","058923","058924","059168","059386","059624",
                                      "059625","059881","060097","060298","060096"
                       ]),"Account"]=6316000
    df.loc[df['Ref_No'].isin(["058498","058283","058710","058923","058924","059168","059386","059624",
                                      "059625","059881","060097","060298","060096"
                       ]),"Acc"]="6316000-32"

    #### UNITED FROM 03 to 05

    df.loc[df['Ref_No'].isin(["059426"
                       ]),"Job_No"]="24UNITED / 05"

    #### SAGA ENVIRONMENTAL TO SF

    df.loc[(df['Ref_No']=="060106")&(df['Description']=="Environmental Fee"),"Name"]="SWTF Facility Charge"
    df.loc[(df['Ref_No']=="060106")&(df['Description']=="Environmental Fee"),"Account"]=6318540
    return df


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
def get_key(d,item):
    
    for k, v in d.items():
        if isinstance(v, dict):
            for a, b in v.items():
                if isinstance(b, dict):
                    for c, d in b.items():
                        if c==item:
                            return k,a
                else:
                    if a==item:
                        return k,b
        else:
            if v==item:
                return k,v
            
def get_value(source,item):
    
    for k, v in source.items():
            if isinstance(v, dict):
                for a, b in v.items():
                    if isinstance(b, dict):
                        for c, d in b.items():    
                            if c==item:
                                return d
                    else:
                        if a==item:
                            return a
            else:
                if v==item:
                    return v
                
def populate_main(main,df,key):
    
    for i in df["Acc"].unique():
        #cost_center=df[df["Account"]==i]["Sub_Cat"].values[0]
        #print(df[df["Account"]==i]["Name"].unique()[0])
        #acc=f"{i}-{cost_center}"
        main[i]={"Name":None,"Cat":None,"Sub_Cat":None,"Net":0}
        main[i]["Name"]=df[df["Acc"]==i]["Name"].unique()[0]
        try:
            main[i]["Sub_Cat"]=find_account_path(key,i)[1]
        except:
            main[i]["Sub_Cat"]=None
        try:
            main[i]["Cat"]=find_account_path(key,i)[0]
        except:
            main[i]["Cat"]=None
        main[i]["Net"]=round(df[df["Acc"]==i]["Net"].sum(),2)
    return main

def define_groups(ledger,key_dictionary):
    
    with open(fr'c:\Users\Afsiny\Desktop\BUDGET\TERMINAL\{key_dictionary}.json', 'r') as file:
        key = json.load(file)
    ledger=ledger.copy()
    ledger.reset_index(drop=True,inplace=True)
    for i in ledger.index:
        try:
            ledger.loc[i,"Group"]=get_key(key,ledger.loc[i,"Acc"])[0]
        except:
            pass
        try:
            ledger.loc[i,"Sub_Group"]=get_key(key,ledger.loc[i,"Acc"])[1]
        except:
            pass
    ignore=set(ledger[ledger["Group"]==0].Account.to_list())
    ledger=ledger[~ledger["Account"].isin(ignore)]
    return ledger

def weyco_changes(ledger):
    
    ledger=ledger.copy()
    ledger.loc[ledger["Acc"]=="6315100-32","Account"]=6315000
    ledger.loc[ledger["Acc"]=="6315100-32","Name"]="Loading & Unloading"
    ledger.loc[ledger["Acc"]=="6315100-32","Acc"]="6315000-32"
    ledger.loc[ledger["Acc"]=="6315200-32","Account"]=6315000
    ledger.loc[ledger["Acc"]=="6315200-32","Name"]="Loading & Unloading"
    ledger.loc[ledger["Acc"]=="6315200-32","Acc"]="6315000-32"

#     ledger.loc[ledger['Account'].isin([6315000,6317030,7313015,7311015]),"Name"]="Loading & Unloading"
#     ledger.loc[ledger['Account'].isin([6315000,6317030,7313015,7311015]),"Account"]=6315000

    ledger.loc[(ledger["Account"].isin(ship_accounts))&(ledger["Job_No"].str.contains("WEYCO")),"Account"]=6999999
    ledger.loc[(ledger["Account"].isin(ship_accounts))&(ledger["Job_No"].str.contains("WEYCO")),"Name"]="Tenant Ship Income"
    ledger.loc[(ledger["Account"].isin(ship_accounts))&(ledger["Job_No"].str.contains("WEYCO")),"Acc"]="6999999-32"

    ledger.loc[(ledger["Account"].isin(ship_accounts))&(ledger["Job_No"].str.contains("SUZANO")),"Account"]=6888888
    ledger.loc[(ledger["Account"].isin(ship_accounts))&(ledger["Job_No"].str.contains("SUZANO")),"Acc"]="6888888-32"
    ledger.loc[(ledger["Name"]=="Handling")&(ledger["Job_No"].str.contains("SUZANO")),"Account"]=6888889
    ledger.loc[(ledger["Name"]=="Handling")&(ledger["Job_No"].str.contains("SUZANO")),"Acc"]="6888889-32"

    ledger.loc[(ledger["Account"].isin(ship_accounts))&(ledger["Job_No"].str.contains("SUZANO")),"Name"]="Suzano Vessels"
    ledger.loc[(ledger["Name"]=="Handling")&(ledger["Job_No"].str.contains("SUZANO")),"Name"]="Suzano Warehouse"
    
    return ledger

def style_ledger(style,ledger,dep="key"):
    lookup={
        "normal":{"key":"terminal_budget","dep_key":"terminal_budget_dep","key_budget":"terminal_budget_2024",
                 "dep_key_budget":"terminal_budget_dep_2024"},
        "afsin":{"key":"better_budget","dep_key":"better_budget_dep",
                 "key_budget":"better_budget_2024","dep_key_budget":"budget_dep_detail_2024"},
        "weyco":{"key":"weyco_suzano_budget","dep_key":"weyco_suzano_budget_dep","key_budget":"weyco_suzano_budget_2024",
                       "dep_key_budget":"weyco_suzano_budget_2024",}
       }
    
    if dep=="key":
        ledger=ledger[ledger["Account"]>6000000]
        
        if style=="weyco":
            
            ledger=weyco_changes(ledger)
        ledger=define_groups(ledger,lookup[style]["key"])
            
#         if style=="normal":
#             ledger=define_groups(ledger,lookup[style]["dep_key"])
        return ledger
    else:
        ledger=ledger[~ledger["Acc"].isin(["7370000-32","7370010-32","7470000-40"])]
        ledger=ledger[(ledger["Account"]>1690000)&(~ledger["Account"].isin([2131001,2391030,2391040]))]
        ledger.loc[ledger["Name"].str.contains("A/D"),"Net"]=-ledger.loc[ledger["Name"].str.contains("A/D"),"Net"]
        ledger=define_groups(ledger,lookup[style]["dep_key"])
        return ledger


    
    ####  LABOR NORMALIZING
#     df.loc[df['Account'].isin([7311015,6315000,6316000,6317030,7313015,7311015]),"Name"]="Loading & Unloading"
#     df.loc[df['Account'].isin([7311015,6315000,6316000,6317030,7313015,7311015]),"Account"]=6315000
#     df["Acc"]=[f"{i}-{j}" for i,j in zip(df["Account"],df["Sub_Cat"])]
    

temp=pd.read_excel(fr"C:\Users\AfsinY\DESKTOP\BUDGET\budget2025.xlsx",header=4)
temp=temp.iloc[:,[0,4]]
temp.columns=["Account","Budget"]

temp["Account"]=temp["Account"].astype("str")
temp.replace(["nan", "Nan"], np.nan, inplace=True)
temp.dropna(subset=["Account"], inplace=True)

temp=pd.read_excel(fr"C:\Users\AfsinY\Desktop\BUDGET\budget2025.xlsx",header=4)
temp=temp.iloc[:,[0,4]]
temp.columns=["Account","Budget"]

temp["Account"]=temp["Account"].astype("str")
temp.replace(["nan", "Nan"], np.nan, inplace=True)
temp.dropna(subset=["Account"], inplace=True)

terminal_budget={"Revenues":{},"Operating Expenses":{},"Maintenance Expenses":{},
           "G&A Overhead":{},"Depreciation":{}}
terminal_budget_2024={"Revenues":{},"Operating Expenses":{},"Maintenance Expenses":{},
               "G&A Overhead":{},"Depreciation":{}}
group=None
subgroup=None
name=None
for i in temp["Account"]:
    if i[0] in ["6","7"]:
        first=str(i).split(" ")[0]
        second=str(i).split(" ")[1:]
        label=""
        for word in second:
            label+=word+ " "
        account=first.split("-")[0]
        acc=f'{account}-{first.split("-")[1][1:]}'
        if group in ["General & Administrative Overhead","G&A Overhead"]:
            subgroup="G&A Overhead"
        if acc== '7370000-32':
            subgroup="Depreciation Terminal"
        if acc== '7370010-32':
            subgroup="Depreciation Grants"
        if acc== '7470000-40':
            subgroup="Depreciation Stormwater"
        
        if subgroup not in terminal_budget[group]:
            terminal_budget[group][subgroup]={}
            terminal_budget_2024[group][subgroup]={}
        if name not in terminal_budget[group][subgroup]:
            terminal_budget[group][subgroup][acc]=label
            terminal_budget_2024[group][subgroup][acc]=temp.loc[temp["Account"]==i,"Budget"].values[0]


        #print(account,acc,name)
    elif any(keyword in i for keyword in ["Revenues", "Operating Expenses", "Maintenance Expenses",
                                          "General & Administrative Overhead",
                                          "Depreciation / Amortization"]) and "Total" not in i:

        group=i.split(":")[0]
        if group=="General & Administrative Overhead":
            group="G&A Overhead"
        elif group=="Depreciation / Amortization":
            group="Depreciation"
        #print(group)
    else:
        if i.endswith(":"):
            #print(i)
            subgroup=i.split(":")[0]

#     harita={"010-A":("airport_budget10","airport_budget10_2024"),
#             "012-A":("airport_budget12","airport_budget12_2024"),
#             "014-A":("airport_budget14","airport_budget14_2024")}
    with open(fr'c:\Users\Afsiny\Desktop\BUDGET\TERMINAL\terminal_budget.json', 'w') as json_file:
        json.dump(terminal_budget, json_file)    
    with open(fr'c:\Users\Afsiny\Desktop\BUDGET\TERMINAL\terminal_budget_2025.json', 'w') as json_file:
        json.dump(terminal_budget_2024, json_file)                 



with open(fr"C:\Users\Afsiny\Desktop\BUDGET\TERMINAL\weyco_suzano_budget.json", 'r') as file:
    weyco_suzano_budget = json.load(file)





Budget_Dict={"Revenues":
                 {"Vessel Operations":{"6311000-32":"Dockage - Vessel",
                                       "6311015-32":"Dockage - Layberth",
                                              "6311020-32":"Dockage LT - 30+ days",
                                              "6312000-32":"Wharfage",
                                              "6313000-32":"Service & Facilities",
                                              "6313950-32":"Security - Vessel",
                                              "6314000-32":"Storage",
                                              "6318540-32":"SWTF Facility Charge",
                                              "6318600-32":"Garbage Collection",
                                              "6373000-32":"Log Handler Rental with Operat",
                                              "6381000-32":"Crane Rental without Operator",
                                              "6389000-32":"Other Pers Prop Rental w/o Opr",
                                              "6318900-32":"Ship Services",
                                              "6319020-32":"Staging"                            },
                         
                         "Tenants":          {"6313001-32":"Service Fee - WEYCO",
                                              "6313003-32":"Flex Area Service Fee - WEYCO",
                                              "6313002-32":"WeyCo Service Fee Credit",
                                              "6313955-32":"Security - Log Tenant",
                                              "6318101-32":"Tenant Water",
                                              "6318301-32":"Sewer - Pass Thru Costs",
                                              "6318501-32":"WeyCo Stormwater",
                                              "6341000-32":"Real Prop Rent - Land",
                                              "6341010-32":"Temp Land Rental",
                                              "6351000-32":"Space Rental - General"},
                         
                         "Labor":            {"6315000-32":"Loading & Unloading",
                                              "6315100-32":"Dock Operators - Vessel",
                                              "6315200-32":"Dock Foreman - Vessel",
                                              "6316000-32":"Handling",
                                              "6317030-32":"Line Service"},
                         
                         "Other Revenue":    {"6319000-32":"CAM Charges",
                                              "6319040-32":"Scale Charges",
                                              "6319041-32":"Washrack Use Charges",
                                              "6319060-32":"Misc. Billable Costs",
                                              "6329000-32":"Other Misc. User Charges",
                                              "6399000-32":"Other Misc. Income",
                                              "6399010-32":"PMA Assess/Excess Tax Refunds-Terminal",
                                              "6318301-32":"Sewer Pass Through Costs",
                                              "6311070-32": "Interest Income",
                                              "6399002-32": "Scrap Metal Recycled"},                                       
                         
                         "Stormwater Revenue":{"6418500-40":"Stormwater Treatment Fee-Stormwater",
                                               "6418502-40":"Stormwater Treatment Fee-Port",
                                               "6411070-40": "Interest Income"}                             
                                              
                                              
                             
                         },##close revenue
            
             "Operating Expenses":
                     {"Operating Overhead":           {
                                                        "7311100-30":"Salaries: Regular",
                                                        "7311200-30":"Salaries: Overtime",
                                                        "7311300-30":"Salaries: Holiday",
                                                        "7311700-30":"Salaries: Vacation",
                                                        "7311800-30":"Salaries: Sick Leave",
                                                        "7312100-30":"Social Security (FICA)",
                                                        "7312200-30":"Ind. Ins (L&I)",
                                                        "7312300-30":"Misc. Benefits",
                                                        "7312303-30":"Misc. Benefits - PFML",
                                                        "7312400-30":"Pension (PERS)",
                                                        "7312500-30":"Med/Dent/Life Insurance",
                                                        "7313014-30":'Training & Development',},
                                                        
                                 
                        "Labor":                      {"7313015-32":"Longshore Services",
                                                       "7311015-32":"Longshore Fringe Benefits"},
                      
                        "Utilities":                  { '7318100-32':"Water",
                                                        '7318102-32':"Water-Purple Pipe",
                                                        '7318200-32':"Electricity",
                                                        '7318500-32':"Stormwater Permits & Utilities",
                                                        '7318540-32':"SWTF Fee Expense",
                                                        '7318580-32':"Environmental Maintenance Supp",
                                                        '7318581-32':"Environmental OS Maintenance",
                                                        '7318600-32':"Garbage Collection",
                                                        '7318700-32':"Natural Gas/Propane/Oil"},
                         
                         "Terminal Operating Expense": {'7314100-30':"Office Supplies & Printing-Admin",
                                                        '7314100-32':"Office Supplies & Printing-Cargo Ops",
                                                        '7313013-32':"Janitorial Services",
                                                        '7315000-30':"Equipment Rentals-Admin",
                                                        
                                                        '7313950-32':"Terminal Security",
                                                        '7313951-32':"Security Supplies",
                                                        '7316010-32':"Porta Potty Rental",
                                                        '7317000-30':"Purchase Card Clearing Account",
                                                        '7317100-30':'Promotional Hosting-Terminal - Admin.',
                                                        '7318800-30':"Telecommunications",
                                                        '7319000-32':"Other Operating Expense",
                                                        '7319005-32':"Keys and Locks",
                                                        '7319010-32':"Rodent & Pest Control",
                                                        '7319012-32':"Welding Expense",
                                                        '7319013-32':"Hardware Expense",
                                                        '7319014-32':"Safety Supplies",
                                                        '7319015-32':"Non-Stormwater Enviro Costs",
                                                        '7319060-32':"Misc. Billable Costs",
                                                        
                                                        "7313104-30":'Legal Svc',
                                                        "7313110-30":"Injuries and Damages-Admin",
                                                        "7313190-30":"Bad Debt Expense",
                                                        "7317000-30":"Purchase Card Clearing Account",
                                                        "7317200-30":"Travel",
                                                        "7317300-30":"Insurance",
                                                        "7317400-30":"Advertising",
                                                        "7317500-30":"Memberships",
                                                        "7317900-30":"Other G & A Direct Costs-Admin",
                                                        "7317910-30":"Meetings Expense",
                                                        "7317910-32":"Meetings Expense",
                                                        "7317911-30":"Publications/Subscriptions",
                                                        "7317912-30":"Customer Appreciation",
                         
                                                        "7317913-30":"Employee Development-Admin",
                                                        
                                                        "7317913-36":"Employee Development-Maint",
                                                        
                                                        "7317915-32":"B & O Excise Taxes"},
                          
                        "Outside Professional Services":{"7313010-30":"Outside Professional Services-Admin",
                                                         "7313010-32":"Outside Professional Services-Terminal"},
                        
                        "Vessel Operational Expenses":{'7314300-32':"Fuel & Lubricants",
                                                       "7338700-36":"Natural Gas/Propane/Oil",
                                                       '7313110-32':"Injuries and Damages-Cargo Ops",
                                                       
                                                       '7315000-32':"Equipment Rentals-Cargo Ops",
                                                    '7313949-32':"Vessel Ops Security"},
                      
                      "Stormwater Operating Expenses": {"7411100-40":"Salaries: Regular",
                                                        "7411300-40":"Salaries: Holiday",
                                                        "7411700-40":"Salaries: Vacation",
                                                        "7411800-40":"Salaries: Sick Leave",
                                                        "7412100-40":"Social Security (FICA)",
                                                        "7412200-40":"Ind. Ins (L&I)",
                                                        
                                                        "7412303-40":"Misc. Benefits - PFML",
                                                        "7412400-40":"Pension (PERS)",
                                                        "7412500-40":"Med/Dent/Life Insurance",
                                                        
                                                        "7413010-40":"Outside Professional Services-Stormwater",
                                                        "7413104-40":"Legal Svc-Spec Projects-Stormwater",
                                                        
                                                        
                                                        
                                                        "7414000-40":"Supplies",
                                                        "7414001-40":"Small Tools",
                                                        "7414010-40":"Chemical Expense",
                                                        "7414100-40":"Office Supplies",
                                                        
                                                        "7417300-40":"Insurance",
                                                        "7417900-40":"Other G&A Direct Costs",
                                                        "7417913-40":"Employee Development-Stormwater",
                                                        "7417915-40":"B&O Excise Taxes-Stormwater",
                                                        
                                                        "7418100-40":"Water-Stormwater",
                                                        "7418200-40":"Electricity-Stormwater",
                                                        "7418500-40":"Permits & Utilities-Stormwater",
                                                        "7418582-40":"Environmental Professional Services",
                                                        
                                                        "7418600-40":"Garbage Collection-Stormwater",
                                                        "7418601-40":"Sludge Collection",
                                                        "7418800-40":"Telecommunications-Stormwater",
                                                       
                                                       
                    
                                                        
                                                        "7450080-40":"Executive G&A Overhead",
                                                        
                                                        "7450082-40":"Marketing G&A Overhead",
                                                        "7450083-40":"Finance G&A Overhead",
                                                        "7450085-40":"Engineering G&A Overhead",
                                                        "7450087-40":"I/S G&A Overhead",
                                                        "7450088-40":"Administrative G&A Overhead"}
                                                        
                                                        
                                                       
                            },##close Operating expenses
             "Maintenance Expenses": {
                 
                 "Maintenance Overhead":  {"7331100-36":"Salaries",
                                           "7331200-36":"Overtime",
                                           "7331300-36":"Holiday",
                                           "7331700-36":"Vacation",
                                           "7331800-36":"Sick Leave",
                                           
                                           "7332100-36":"Social Security (FICA)",
                                           "7332200-36":"Ind. Ins (L&I)",
                                           "7332300-36":"Misc. Benefits",
                                           "7332303-36":"Misc. Benefits - PFML",
                                           "7332400-36":"Pension (PERS)",
                                           "7332500-36":"Med/Dent/Life Insurance"},
                                           
                 
                "Property Maintenance":  { "7339003-32":"Maint & Repair to Tenant Bldgs",
                                           "7339004-32":"Equipment Wash Facility-Terminal Cargo Op's",
                                           "7339006-32":"Maint & Repair Warehouse A",
                                           "7339007-32":"Maint & Repair Security Building",
                                           "7339008-32":"Maint & Repair Gatehouse",
                                           "7339009-36":"Maint & Repair Maint Shop",
                                           "7339020-32":"Maint & Repair Conveyor",
                                           "7339025-32":"Maint & Repair RR Tracks",
                                           "7339030-32":"Maint & Repair Roads",
                                           "7339040-32":"Maint & Repair Property",
                                           "7339070-32":"Maint & Repair Wharves/Docks",},
                 
                 "Equipment Maintenance": {"7339020-32":"Maint & Repair Conveyor",
                                           "7339050-36":"Vehicle Maintenance",
                                           "7339060-32":"Maint & Repair Misc Equipment-Terminal",
                                          
                                           "7339061-32":"Maint & Repair Lift Trucks",
                                           "7339062-32":"Maint & Repair Log Handlers",
                                           "7339063-32":"Maint & Repair Cranes",
                                           "7339064-32":"Maint & Repair Yard Equipment-Terminal",
                                           
                                           "7339065-36":"Maint & Repair Radios"},
                 
                 "Other Maintenance Expenses": {"7334001-36":"Small Tools-Maintenance",
                                                "7334260-36":"Laundry Expense",
                                                "7334261-36":"Safety Supplies",
                                                "7334300-36":"Fuel & Lubricants",
                                                "7337000-36":"Purchase Card Clearing Account",
                                                
                                                
                                                
                                                "7339000-36":"Other Maintenance Expense-Maintenance"},
                
                 "Stormwater Maintenance Expenses":    {"7431100-40":"Salaries: Regular",
                                                        "7431200-40":"Salaries: Overtime",
                                                        "7431300-40":"Salaries: Holiday",
                                                        "7431700-40":"Salaries: Vacation",
                                                        "7431800-40":"Salaries: Sick Leave",
                                                        "7432100-40":"Social Security (FICA)",
                                                        "7432200-40":"Ind. Ins (L&I)",
                                                        
                                                        "7432303-40":"Misc. Benefits - PFML",
                                                        "7432400-40":"Pension (PERS)",
                                                        "7432500-40":"Med/Dent/Life Insurance",
                                                       
                                                        
                                                        "7439000-40":"Other Maintenance Expense - Stormwater",
                                                        "7439010-40":"Maint & Repair Pump-Stormwater",
                                                        "7439031-40":"Maint & Repair Catch Basins-Stormwater",
                                                        "7439060-40":"Maint & Repair Facility Equipment"}    
                                             },##close Maintenance expenses}
             "G & A Overhead":    {"7350080-30":"Executive G&A Overhead",
                                   "7350082-30":"Marketing G&A Overhead",
                                   "7350083-30":"Finance G&A Overhead",
                                   "7350085-30":"Engineering G&A Overhead",
                                   
                                   "7350087-30":"I/S G&A Overhead",
                                   "7350088-30":"Administrative G&A Overhead"},
             "Depreciation":      
                    {"Depreciation Terminal":{"1712000-30":"A/D: Improvements to Land",
                                              "1721000-30":"A/D: Wharves, Docks, Piers &",
                                              "1721100-30":"A/D: Dredging",
                                              "1722000-30":"A/D: Utility Systems",
                                              "1731000-30":"A/D: Gen & Admin Buildings",
                                              "1758000-30":"A/D: Shop Equipment & Tools",
                                              "1765000-30":"A/D: Rail Mover",
                                              "1777000-30":"A/D: Trucks, General Purpose",
                                              "1778000-30":"A/D: Cranes",
                                              "1782000-30":"A/D: Load'g Equip & Lift Truk",
                                              "1782001-30":"A/D: Heavy Lift Trucks",
                                              "1785000-30":"A/D: Sweepers/Grounds Mainten",
                                              "1786001-30":"A/D: Container Handling Equip",
                                              "1787000-30":"A/D: Trailers & Freight Cars",
                                              "1796000-30":"A/D: Computer Hdware/Software",
                                              "1861000-30":"A/D: Roads & Grounds",
                                              "1865000-30":"A/D: Trackage"},
                     
                     "Depreciation Stormwater":{"1738000-40":"A/D: Other Buildings & Struct",
                                               "1757000-40":"A/D: Stormwater System Equipmt",
                                               "1761000-40":"A/D: Boats",
                                               "1796000-40":"A/D: Computer Hdware/Software",
                                               "1861000-40":"A/D: Roads & Grounds",
                                               "7439010-40":"Maint & Repair Pump"
                                               }
                                              
                 
                 
                 
                 
             }}
                                   
                                
                                   
with open(fr'c:\Users\Afsiny\Desktop\Budget\Terminal\better_budget.json', 'w') as json_file:
    json.dump(Budget_Dict, json_file)                 
                                                
                                                
                                                
                                                
ships_jobs=['Dockage - Vessel','Wharfage','Service & Facilities','SWTF Facility Charge','Line Service',
            'Loading & Unloading',"Handling",'Dock Foreman - Vessel','Dock Operators - Vessel','Crane Rental without Operator',
            'Log Handler Rental with Operat','Security - Vessel', 'Garbage Collection','Ship Services',
             'Storage','Other Misc. Income','Longshore Services','Equipment Rentals','Other Misc. User Charges',
            'Vessel Ops Security']
# ledger.loc[(ledger["Name"].isin(ships_jobs))&(ledger["Job_No"].str.contains("WEYCO")),"Account"]=6999999
# ledger.loc[(ledger["Name"].isin(ships_jobs))&(ledger["Job_No"].str.contains("WEYCO")),"Name"]="Tenant Ship Income"                     
                     
                        
       
                    
             

Budget_Dict1={"Revenues":
                 {"Vessel Operations":{"6311000-32":"Dockage - Vessel",
                                       "6311015-32":"Dockage - Layberth",
                                              "6311020-32":"Dockage LT - 30+ days",
                                              "6312000-32":"Wharfage",
                                              "6313000-32":"Service & Facilities",
                                              "6313950-32":"Security - Vessel",
                                              "6314000-32":"Storage",
                                              "6318540-32":"SWTF Facility Charge",
                                              "6318600-32":"Garbage Collection",
                                              "6373000-32":"Log Handler Rental with Operat",
                                              "6381000-32":"Crane Rental without Operator",
                                              "6389000-32":"Other Pers Prop Rental w/o Opr",
                                              "6318900-32":"Ship Services",
                                              "6319020-32":"Staging"                            },
                         
                         "Tenants":          {"6313001-32":"Service Fee - WEYCO",
                                              "6313003-32":"Flex Area Service Fee - WEYCO",
                                              "6313002-32":"WeyCo Service Fee Credit",
                                              "6313955-32":"Security - Log Tenant",
                                              "6318101-32":"Tenant Water",
                                              "6318301-32":"Sewer - Pass Thru Costs",
                                              "6318501-32":"WeyCo Stormwater",
                                              "6341000-32":"Real Prop Rent - Land",
                                              "6341010-32":"Temp Land Rental",
                                              "6351000-32":"Space Rental - General"},
                         
                         "Labor":            {"6315000-32":"Loading & Unloading",
                                              "6315100-32":"Dock Operators - Vessel",
                                              "6315200-32":"Dock Foreman - Vessel",
                                              "6316000-32":"Handling",
                                              "6317030-32":"Line Service"},
                         
                         "Other Revenue":    {"6319000-32":"CAM Charges",
                                              "6319040-32":"Scale Charges",
                                              "6319041-32":"Washrack Use Charges",
                                              "6319060-32":"Misc. Billable Costs",
                                              "6329000-32":"Other Misc. User Charges",
                                              "6399000-32":"Other Misc. Income",
                                              "6399010-32":"PMA Assess/Excess Tax Refunds-Terminal",
                                              "6318301-32":"Sewer Pass Through Costs",
                                              "6311070-32": "Interest Income",
                                              "6399002-32": "Scrap Metal Recycled"},                                       
                         
                         "Stormwater Revenue":{"6418500-40":"Stormwater Treatment Fee-Stormwater",
                                               "6418502-40":"Stormwater Treatment Fee-Port",
                                               "6411070-40": "Interest Income"}                             
                                              
                                              
                             
                         },##close revenue
            
             "Operating Expenses":
                     {"Operating Overhead":           {
                                                        "7311100-30":"Salaries: Regular",
                                                        "7311200-30":"Salaries: Overtime",
                                                        "7311300-30":"Salaries: Holiday",
                                                        "7311700-30":"Salaries: Vacation",
                                                        "7311800-30":"Salaries: Sick Leave",
                                                        "7312100-30":"Social Security (FICA)",
                                                        "7312200-30":"Ind. Ins (L&I)",
                                                        "7312300-30":"Misc. Benefits",
                                                        "7312303-30":"Misc. Benefits - PFML",
                                                        "7312400-30":"Pension (PERS)",
                                                        "7312500-30":"Med/Dent/Life Insurance",
                                                        "7313014-30":'Training & Development',},
                                                        
                                 
                        "Labor":                      {"7313015-32":"Longshore Services",
                                                       "7311015-32":"Longshore Fringe Benefits"},
                      
                        "Utilities":                  { '7318100-32':"Water",
                                                        '7318102-32':"Water-Purple Pipe",
                                                        '7318200-32':"Electricity",
                                                        '7318500-32':"Stormwater Permits & Utilities",
                                                        '7318540-32':"SWTF Fee Expense",
                                                        '7318580-32':"Environmental Maintenance Supp",
                                                        '7318581-32':"Environmental OS Maintenance",
                                                        '7318600-32':"Garbage Collection",
                                                        '7318700-32':"Natural Gas/Propane/Oil"},
                         
                         "Terminal Operating Expense": {'7314100-30':"Office Supplies & Printing-Admin",
                                                        '7314100-32':"Office Supplies & Printing-Cargo Ops",
                                                        '7313013-32':"Janitorial Services",
                                                        '7315000-30':"Equipment Rentals-Admin",
                                                        
                                                        '7313950-32':"Terminal Security",
                                                        '7313951-32':"Security Supplies",
                                                        '7316010-32':"Porta Potty Rental",
                                                        '7317000-30':"Purchase Card Clearing Account",
                                                        '7317100-30':'Promotional Hosting-Terminal - Admin.',
                                                        '7318800-30':"Telecommunications",
                                                        '7319000-32':"Other Operating Expense",
                                                        '7319005-32':"Keys and Locks",
                                                        '7319010-32':"Rodent & Pest Control",
                                                        '7319012-32':"Welding Expense",
                                                        '7319013-32':"Hardware Expense",
                                                        '7319014-32':"Safety Supplies",
                                                        '7319015-32':"Non-Stormwater Enviro Costs",
                                                        '7319060-32':"Misc. Billable Costs",
                                                        
                                                        "7313104-30":'Legal Svc',
                                                        "7313110-30":"Injuries and Damages-Admin",
                                                        "7313190-30":"Bad Debt Expense",
                                                        "7317000-30":"Purchase Card Clearing Account",
                                                        "7317200-30":"Travel",
                                                        "7317300-30":"Insurance",
                                                        "7317400-30":"Advertising",
                                                        "7317500-30":"Memberships",
                                                        "7317900-30":"Other G & A Direct Costs-Admin",
                                                        "7317910-30":"Meetings Expense",
                                                        "7317910-32":"Meetings Expense",
                                                        "7317911-30":"Publications/Subscriptions",
                                                        "7317912-30":"Customer Appreciation",
                         
                                                        "7317913-30":"Employee Development-Admin",
                                                      
                                                        "7317913-36":"Employee Development-Maint",
                                                        
                                                        "7317915-32":"B & O Excise Taxes"},
                          
                        "Outside Professional Services":{"7313010-30":"Outside Professional Services-Admin",
                                                         "7313010-32":"Outside Professional Services-Terminal"},
                        
                        "Vessel Operational Expenses":{'7314300-32':"Fuel & Lubricants",
                                                       "7338700-36":"Natural Gas/Propane/Oil",
                                                       '7313110-32':"Injuries and Damages-Cargo Ops",
                                                       
                                                       '7315000-32':"Equipment Rentals-Cargo Ops",
                                                    '7313949-32':"Vessel Ops Security"},
                      
                      "Stormwater Operating Expenses": {"7411100-40":"Salaries: Regular",
                                                        "7411300-40":"Salaries: Holiday",
                                                        "7411700-40":"Salaries: Vacation",
                                                        "7411800-40":"Salaries: Sick Leave",
                                                        "7412100-40":"Social Security (FICA)",
                                                        "7412200-40":"Ind. Ins (L&I)",
                                                        
                                                        
                                                        
                                                        "7412303-40":"Misc. Benefits - PFML",
                                                        "7412400-40":"Pension (PERS)",
                                                        "7412500-40":"Med/Dent/Life Insurance",
                                                        
                                                        "7413010-40":"Outside Professional Services-Stormwater",
                                                        "7413104-40":"Legal Svc-Spec Projects-Stormwater",
                                                        
                                                        
                                                        
                                                        "7414000-40":"Supplies",
                                                        "7414001-40":"Small Tools",
                                                        "7414010-40":"Chemical Expense",
                                                        "7414100-40":"Office Supplies",
                                                        
                                                        "7417300-40":"Insurance",
                                                        "7417900-40":"Other G&A Direct Costs",
                                                        "7417913-40":"Employee Development-Stormwater",
                                                        "7417915-40":"B&O Excise Taxes-Stormwater",
                                                        
                                                        "7418100-40":"Water-Stormwater",
                                                        "7418200-40":"Electricity-Stormwater",
                                                        "7418500-40":"Permits & Utilities-Stormwater",
                                                        "7418582-40":"Environmental Professional Services",
                                                        
                                                        "7418600-40":"Garbage Collection-Stormwater",
                                                        "7418601-40":"Sludge Collection",
                                                        "7418800-40":"Telecommunications-Stormwater",
                                                       
                                                       
                                           
                                                        
                                                        "7450080-40":"Executive G&A Overhead",
                                                        
                                                        "7450082-40":"Marketing G&A Overhead",
                                                        "7450083-40":"Finance G&A Overhead",
                                                        "7450085-40":"Engineering G&A Overhead",
                                                        "7450087-40":"I/S G&A Overhead",
                                                        "7450088-40":"Administrative G&A Overhead"}
                                                        
                                                        
                                                       
                            },##close Operating expenses
             "Maintenance Expenses": {
                 
                 "Maintenance Overhead":  {"7331100-36":"Salaries",
                                           "7331200-36":"Overtime",
                                           "7331300-36":"Holiday",
                                           "7331700-36":"Vacation",
                                           "7331800-36":"Sick Leave",
                                           
                                           "7332100-36":"Social Security (FICA)",
                                           "7332200-36":"Ind. Ins (L&I)",
                                           "7332300-36":"Misc. Benefits",
                                           "7332303-36":"Misc. Benefits - PFML",
                                           "7332400-36":"Pension (PERS)",
                                           "7332500-36":"Med/Dent/Life Insurance"},
                                           
                 
                "Property Maintenance":  { "7339003-32":"Maint & Repair to Tenant Bldgs",
                                           "7339004-32":"Equipment Wash Facility-Terminal Cargo Op's",
                                           "7339006-32":"Maint & Repair Warehouse A",
                                           "7339007-32":"Maint & Repair Security Building",
                                           "7339008-32":"Maint & Repair Gatehouse",
                                           "7339009-36":"Maint & Repair Maint Shop",
                                           "7339020-32":"Maint & Repair Conveyor",
                                           "7339025-32":"Maint & Repair RR Tracks",
                                           "7339030-32":"Maint & Repair Roads",
                                           "7339040-32":"Maint & Repair Property",
                                           "7339070-32":"Maint & Repair Wharves/Docks",},
                 
                 "Equipment Maintenance": {"7339020-32":"Maint & Repair Conveyor",
                                           "7339050-36":"Vehicle Maintenance",
                                           "7339060-32":"Maint & Repair Misc Equipment-Terminal",
                                           
                                           "7339061-32":"Maint & Repair Lift Trucks",
                                           "7339062-32":"Maint & Repair Log Handlers",
                                           "7339063-32":"Maint & Repair Cranes",
                                           "7339064-32":"Maint & Repair Yard Equipment-Terminal",
                                           
                                           "7339065-36":"Maint & Repair Radios"},
                 
                 "Other Maintenance Expenses": {"7334001-36":"Small Tools-Maintenance",
                                                "7334260-36":"Laundry Expense",
                                                "7334261-36":"Safety Supplies",
                                                "7334300-36":"Fuel & Lubricants",
                                                "7337000-36":"Purchase Card Clearing Account",
                                                
                                                
                                                "7339000-36":"Other Maintenance Expense-Maintenance"},
                
                 "Stormwater Maintenance Expenses":    {"7431100-40":"Salaries: Regular",
                                                        "7431200-40":"Salaries: Overtime",
                                                        "7431300-40":"Salaries: Holiday",
                                                        "7431700-40":"Salaries: Vacation",
                                                        "7431800-40":"Salaries: Sick Leave",
                                                        "7432100-40":"Social Security (FICA)",
                                                        "7432200-40":"Ind. Ins (L&I)",
                                                        
                                                        "7432303-40":"Misc. Benefits - PFML",
                                                        "7432400-40":"Pension (PERS)",
                                                        "7432500-40":"Med/Dent/Life Insurance",
                                                       
                                                        
                                                        "7439000-40":"Other Maintenance Expense - Stormwater",
                                                        "7439010-40":"Maint & Repair Pump-Stormwater",
                                                        "7439031-40":"Maint & Repair Catch Basins-Stormwater",
                                                        "7439060-40":"Maint & Repair Facility Equipment"}    
                                             },##close Maintenance expenses}
             "G & A Overhead":    {"7350080-30":"Executive G&A Overhead",
                                   "7350082-30":"Marketing G&A Overhead",
                                   "7350083-30":"Finance G&A Overhead",
                                   "7350085-30":"Engineering G&A Overhead",
                                   
                                   "7350087-30":"I/S G&A Overhead",
                                   "7350088-30":"Administrative G&A Overhead"},
             "Depreciation":      
                    {"Depreciation Terminal":{"7370000-32":"Depreciation Terminal"},
                     "Depreciation Grants":{"7370010-32":"Depreciation Grants"},
                     "Depreciation Stormwater":{"7470000-40":"Depreciation Stormwater"}
                                              
                 
                 
                 
                 
             }}
                                   
                                
                                   
                 
with open(fr'c:\Users\Afsiny\Desktop\Budget\Terminal\better_budget_dep.json', 'w') as json_file:
    json.dump(Budget_Dict1, json_file)                 
                                                
                                                
                                                
                                                
ships_jobs=['Dockage - Vessel','Wharfage','Service & Facilities','SWTF Facility Charge','Line Service',
            'Loading & Unloading',"Handling",'Dock Foreman - Vessel','Dock Operators - Vessel','Crane Rental without Operator',
            'Log Handler Rental with Operat','Security - Vessel', 'Garbage Collection','Ship Services',
             'Storage','Other Misc. Income','Longshore Services','Equipment Rentals','Other Misc. User Charges',
            'Vessel Ops Security']
# ledger.loc[(ledger["Name"].isin(ships_jobs))&(ledger["Job_No"].str.contains("WEYCO")),"Account"]=6999999
# ledger.loc[(ledger["Name"].isin(ships_jobs))&(ledger["Job_No"].str.contains("WEYCO")),"Name"]="Tenant Ship Income"                     
                     
                        
       
                    
             

Budget_Dict={"Revenues":
                 {"Vessel Operations":{"6311000-32":"Dockage - Vessel",
                                       "6311015-32":"Dockage - Layberth",
                                              "6311020-32":"Dockage LT - 30+ days",
                                              "6312000-32":"Wharfage",
                                              "6313000-32":"Service & Facilities",
                                              "6313950-32":"Security - Vessel",
                                              "6314000-32":"Storage",
                                              "6318540-32":"SWTF Facility Charge",
                                              "6318600-32":"Garbage Collection",
                                              "6373000-32":"Log Handler Rental with Operat",
                                              "6381000-32":"Crane Rental without Operator",                                              
                                              "6319020-32":"Staging"                            },
                         
                         "Weyco":          {"6313001-32":"Service Fee - WEYCO",
                                              "6313003-32":"Flex Area Service Fee - WEYCO",
                                              "6389000-32":"Other Pers Prop Rental w/o Opr",
                                              "6318900-32":"Ship Services",
                                              "6313002-32":"WeyCo Service Fee Credit",
                                              "6313955-32":"Security - Log Tenant",
                                              "6318101-32":"Tenant Water",
                                              "6318301-32":"Sewer - Pass Thru Costs",
                                              "6318501-32":"WeyCo Stormwater",
                                              "6341000-32":"Real Prop Rent - Land",
                                              "6341010-32":"Temp Land Rental",
                                              "6351000-32":"Space Rental - General",
                                              "6999999-32":"Tenant Ship Income"}, ### MADE UP FOR SHIPS
                         
                         "All Other Vessels":{"6315000-32":"Loading & Unloading",
                                              "6315100-32":"Dock Operators - Vessel",
                                              "6315200-32":"Dock Foreman - Vessel",
                                              "6316000-32":"Handling",
                                              "6317030-32":"Line Service"},
                        
                      "Suzano Operations":  {"6888888-32":"Suzano Vessels",### MADE UP FOR SHIPS
                                             "6888889-32":"Suzano Warehouse"},### MADE UP FOR SHIPS
                         
                         "Other Revenue":    {"6319000-32":"CAM Charges",
                                              "6319040-32":"Scale Charges",
                                              "6319041-32":"Washrack Use Charges",
                                              "6319060-32":"Misc. Billable Costs",
                                              "6329000-32":"Other Misc. User Charges",
                                              "6399000-32":"Other Misc. Income",
                                              "6399010-32":"PMA Assess/Excess Tax Refunds-Terminal",
                                              "6318301-32":"Sewer Pass Through Costs",
                                              "6311070-32": "Interest Income",
                                              "6411070-40": "Interest Income",
                                              "6399002-32": "Scrap Metal Recycled"},                                       
                         
                         "Stormwater Revenue":{"6418500-40":"Stormwater Treatment Fee-Stormwater",
                                               "6418502-40":"Stormwater Treatment Fee-Port"}                             
                                              
                                              
                             
                         },##close revenue
            
             "Operating Expenses":
                     {"Operating Overhead":           {
                                                        "7311100-30":"Salaries: Regular",
                                                        "7311200-30":"Salaries: Overtime",
                                                        "7311300-30":"Salaries: Holiday",
                                                        "7311700-30":"Salaries: Vacation",
                                                        "7311800-30":"Salaries: Sick Leave",
                                                        "7312100-30":"Social Security (FICA)",
                                                        "7312200-30":"Ind. Ins (L&I)",
                                                        "7312300-30":"Misc. Benefits",
                                                        "7312303-30":"Misc. Benefits - PFML",
                                                        "7312400-30":"Pension (PERS)",
                                                        "7312500-30":"Med/Dent/Life Insurance",
                                                        "7313014-30":'Training & Development'},
                                                        
                                 
                        "Labor":                      {"7313015-32":"Longshore Services",
                                                       "7311015-32":"Longshore Fringe Benefits"},
                      
                        "Utilities":                  { '7318100-32':"Water",
                                                        '7318102-32':"Water-Purple Pipe",
                                                        '7318200-32':"Electricity",
                                                        '7318500-32':"Stormwater Permits & Utilities",
                                                        '7318540-32':"SWTF Fee Expense",
                                                        '7318580-32':"Environmental Maintenance Supp",
                                                        '7318581-32':"Environmental OS Maintenance",
                                                        '7318600-32':"Garbage Collection",
                                                        '7318700-32':"Natural Gas/Propane/Oil"},
                         
                         "Terminal Operating Expense": {'7314100-30':"Office Supplies & Printing-Admin",
                                                        '7314100-32':"Office Supplies & Printing-Cargo Ops",
                                                        '7313013-32':"Janitorial Services",
                                                        
                                                        '7313950-32':"Terminal Security",
                                                        '7313951-32':"Security Supplies",
                                                        '7316010-32':"Porta Potty Rental",
                                                        '7317000-30':"Purchase Card Clearing Account",
                                                        '7317100-30':'Promotional Hosting-Terminal - Admin.',
                                                        '7318800-30':"Telecommunications",
                                                        '7319000-32':"Other Operating Expense",
                                                        '7319005-32':"Keys and Locks",
                                                        '7319010-32':"Rodent & Pest Control",
                                                        '7319012-32':"Welding Expense",
                                                        '7319013-32':"Hardware Expense",
                                                        '7319014-32':"Safety Supplies",
                                                        '7319015-32':"Non-Stormwater Enviro Costs",
                                                        '7319060-32':"Misc. Billable Costs",
                                                        
                                                        "7313104-30":'Legal Svc',
                                                        "7313110-30":"Injuries and Damages-Admin",
                                                        "7313190-30":"Bad Debt Expense",
                                                        "7317000-30":"Purchase Card Clearing Account",
                                                        "7317200-30":"Travel",
                                                        "7317300-30":"Insurance",
                                                        "7317400-30":"Advertising",
                                                        "7317500-30":"Memberships",
                                                        "7317900-30":"Other G & A Direct Costs-Admin",
                                                        "7317910-30":"Meetings Expense",
                                                        "7317910-32":"Meetings Expense",
                                                        "7317911-30":"Publications/Subscriptions",
                                                        "7317912-30":"Customer Appreciation",
                         
                                                        "7317913-30":"Employee Development-Admin",
                                                        "7317913-32":"Employee Development-Ops",
                                                        "7317913-36":"Employee Development-Maint",
                                                        
                                                        "7317915-32":"B & O Excise Taxes"},
                          
                        "Outside Professional Services":{"7313010-30":"Outside Professional Services-Admin",
                                                         "7313010-32":"Outside Professional Services-Terminal"},
                        
                        "Vessel Operational Expenses":{'7314300-32':"Fuel & Lubricants",
                                                       '7313110-32':"Injuries and Damages-Cargo Ops",
                                                       '7315000-30':"Equipment Rentals-Admin",
                                                       '7315000-32':"Equipment Rentals-Cargo Ops",
                                                    '7313949-32':"Vessel Ops Security"},
                      
                      "Stormwater Operating Expenses": {"7411100-40":"Salaries: Regular",
                                                        "7411300-40":"Salaries: Holiday",
                                                        "7411700-40":"Salaries: Vacation",
                                                        "7411800-40":"Salaries: Sick Leave",
                                                        "7412100-40":"Social Security (FICA)",
                                                        "7412200-40":"Ind. Ins (L&I)",
                                                        "7412300-40":"Misc. Benefits",
                                                        "7412303-40":"Misc. Benefits - PFML",
                                                        "7412400-40":"Pension (PERS)",
                                                        "7412500-40":"Med/Dent/Life Insurance",
                                                        
                                                        "7413010-40":"Outside Professional Services-Stormwater",
                                                        "7413104-40":"Legal Svc-Spec Projects-Stormwater",
                                                        "7413190-40":"Bad Debt Expense-Stormwater",
                                                        
                                                        
                                                        "7414000-40":"Supplies",
                                                        "7414001-40":"Small Tools",
                                                        "7414010-40":"Chemical Expense",
                                                        "7414100-40":"Office Supplies",
                                                        
                                                        "7417300-40":"Insurance",
                                                        "7417900-40":"Other G&A Direct Costs",
                                                        "7417913-40":"Employee Development-Stormwater",
                                                        "7417915-40":"B&O Excise Taxes-Stormwater",
                                                        
                                                        "7418100-40":"Water-Stormwater",
                                                        "7418200-40":"Electricity-Stormwater",
                                                        "7418500-40":"Permits & Utilities-Stormwater",
                                                        "7418582-40":"Environmental Professional Services",
                                                        
                                                        "7418600-40":"Garbage Collection-Stormwater",
                                                        "7418601-40":"Sludge Collection",
                                                        "7418800-40":"Telecommunications-Stormwater",
                                                       
                                                      
                                          
                                                        
                                                        "7450080-40":"Executive G&A Overhead",
                                                        
                                                        "7450082-40":"Marketing G&A Overhead",
                                                        "7450083-40":"Finance G&A Overhead",
                                                        "7450085-40":"Engineering G&A Overhead",
                                                        "7450087-40":"I/S G&A Overhead",
                                                        "7450088-40":"Administrative G&A Overhead",}
                                                        
                                                        
                                                       
                            },##close Operating expenses
             "Maintenance Expenses": {
                 
                 "Maintenance Overhead":  {"7331100-36":"Salaries",
                                           "7331200-36":"Overtime",
                                           "7331300-36":"Holiday",
                                           "7331700-36":"Vacation",
                                           "7331800-36":"Sick Leave",
                                           
                                           "7332100-36":"Social Security (FICA)",
                                           "7332200-36":"Ind. Ins (L&I)",
                                           "7332300-36":"Misc. Benefits",
                                           "7332303-36":"Misc. Benefits - PFML",
                                           "7332400-36":"Pension (PERS)",
                                           "7332500-36":"Med/Dent/Life Insurance"},
                                           
                 
                "Property Maintenance":  { "7339003-32":"Maint & Repair to Tenant Bldgs",
                                           "7339004-32":"Equipment Wash Facility-Terminal Cargo Op's",
                                           "7339006-32":"Maint & Repair Warehouse A",
                                           "7339007-32":"Maint & Repair Security Building",
                                           "7339008-32":"Maint & Repair Gatehouse",
                                           "7339009-36":"Maint & Repair Maint Shop",
                                           "7339020-32":"Maint & Repair Conveyor",
                                           "7339025-32":"Maint & Repair RR Tracks",
                                           "7339030-32":"Maint & Repair Roads",
                                           "7339040-32":"Maint & Repair Property",
                                           "7339070-32":"Maint & Repair Wharves/Docks"},
                 
                 "Equipment Maintenance": {"7339020-32":"Maint & Repair Conveyor",
                                           "7339050-36":"Vehicle Maintenance",
                                           "7339060-32":"Maint & Repair Misc Equipment-Terminal",
                                           
                                           "7339061-32":"Maint & Repair Lift Trucks",
                                           "7339062-32":"Maint & Repair Log Handlers",
                                           "7339063-32":"Maint & Repair Cranes",
                                           "7339064-32":"Maint & Repair Yard Equipment-Terminal",
                                           
                                           
                                           "7339065-36":"Maint & Repair Radios"},
                 
                 "Other Maintenance Expenses": {"7334001-36":"Small Tools-Maintenance",
                                                "7334260-36":"Laundry Expense",
                                                "7334261-36":"Safety Supplies",
                                                "7334300-36":"Fuel & Lubricants",
                                                "7337000-36":"Purchase Card Clearing Account",
                                                "7338700-36":"Natural Gas/Propane/Oil",
                                                "7339000-32":"Other Maintenance Expense-Terminal",
                                                "7339000-36":"Other Maintenance Expense-Maintenance"},
                 "Stormwater Maintenance Expenses":    {"7431100-40":"Salaries: Regular",
                                                        "7431200-40":"Salaries: Overtime",
                                                        "7431300-40":"Salaries: Holiday",
                                                        "7431700-40":"Salaries: Vacation",
                                                        "7431800-40":"Salaries: Sick Leave",
                                                        "7432100-40":"Social Security (FICA)",
                                                        "7432200-40":"Ind. Ins (L&I)",
                                                       
                                                        "7432303-40":"Misc. Benefits - PFML",
                                                        "7432400-40":"Pension (PERS)",
                                                        "7432500-40":"Med/Dent/Life Insurance",
                                                       
                                                        
                                                        "7439000-40":"Other Maintenance Expense - Stormwater",
                                                        "7439010-40":"Maint & Repair Pump-Stormwater",
                                                        "7439031-40":"Maint & Repair Catch Basins-Stormwater",
                                                        "7439060-40":"Maint & Repair Facility Equipment"}    
    
                                             },##close Maintenance expenses}
             "G & A Overhead":    {"7350080-30":"Executive G&A Overhead",
                                   "7350082-30":"Marketing G&A Overhead",
                                   "7350083-30":"Finance G&A Overhead",
                                   "7350085-30":"Engineering G&A Overhead",
                                   
                                   "7350087-30":"I/S G&A Overhead",
                                   "7350088-30":"Administrative G&A Overhead"},
             "Depreciation":      
                    {"Depreciation Terminal":{"1712000-30":"A/D: Improvements to Land",
                                              "1721000-30":"A/D: Wharves, Docks, Piers &",
                                              "1721100-30":"A/D: Dredging",
                                              "1722000-30":"A/D: Utility Systems",
                                              "1731000-30":"A/D: Gen & Admin Buildings",
                                              "1758000-30":"A/D: Shop Equipment & Tools",
                                              "1765000-30":"A/D: Rail Mover",
                                              "1777000-30":"A/D: Trucks, General Purpose",
                                              "1778000-30":"A/D: Cranes",
                                              "1782000-30":"A/D: Load'g Equip & Lift Truk",
                                              "1782001-30":"A/D: Heavy Lift Trucks",
                                              "1785000-30":"A/D: Sweepers/Grounds Mainten",
                                              "1786001-30":"A/D: Container Handling Equip",
                                              "1787000-30":"A/D: Trailers & Freight Cars",
                                              "1796000-30":"A/D: Computer Hdware/Software",
                                              "1861000-30":"A/D: Roads & Grounds",
                                              "1865000-30":"A/D: Trackage"},
                     "Depreciation Stormwater":{"1738000-40":"A/D: Other Buildings & Struct",
                                               "1757000-40":"A/D: Stormwater System Equipmt",
                                               "1761000-40":"A/D: Boats",
                                               "1796000-40":"A/D: Computer Hdware/Software",
                                               "1861000-40":"A/D: Roads & Grounds",
                                               }}
                                              
                 
                 
                 
                 
             }
                                   
                 
with open(fr'c:\Users\Afsiny\Desktop\Budget\Terminal\weyco_suzano_budget.json', 'w') as json_file:
    json.dump(Budget_Dict, json_file)                                                                                                     
                                                        
                 
                                                
                                                
                                                
                                                
              
                     
                     
