Dit notebook wordt gebruikt om de data op te halen vanaf de OSN (Open Sky Network). Als er data voor een langere tijd moet worden gedownload dan is het aan te raden om dit in kleinere stukjes op te delen. Dit vergt dus wel wat handmatig werk. Als er meerdere losse datasets worden gedownload, dan kan vervolgens het script: "Data samenvoegen" worden gebruikt.

import pandas as pd
import os
from shapely.geometry import Point, Polygon
from pyopensky import OpenskyImpalaWrapper, EHSHelper

CWD = os.getcwd()
opensky = OpenskyImpalaWrapper()
ehs = EHSHelper()

# OSN data ophalen

Eerst moeten de juiste waardes worden voor de input variabelen hieronder worden ingevuld. Dit kan worden verkregen via de onderstaande 2 linkjes.

creeer een bounding box online, https://boundingbox.klokantech.com/
UNIX timestamp converteren, https://www.unixtimestamp.com/index.php

# de latitude en longtitude worden gebruikt om het gebied te selecteren waarbinnen de data moet worden opgehaald.
lamin = 52.299498 #linker kant, minimum latitude
lomin = 4.7130477 #onderkant, minimum lontitude
lamax = 52.3400011 #rechter kant, maximum latitude
lomax = 4.7499549 #bovenkant, maximum lontitude

voor veel data is het makkelijker om de data in kleinere hoeveelheden op te halen. Pas de filename aan naar een gewenste naam. Deze naam wordt gebruikt voor het opslaan van de csv. De Situatie_...._unix wordt gebruikt om te defineren binnen welke tijd de data opgehaald worden.

filename = '2021-08-31'
Situatie_begin_unix = 1630360800 #start tijd voor data ophalen
Situatie_eind_unix = 1630447200 #eind tijd voor data ophalen

#selecteer alleen de bruikbare kolommen uit OSN
columns_to_select = 'time, icao24, lat, lon, velocity, heading, callsign, onground, squawk, hour'

request_str = 'SELECT ' + columns_to_select + ' FROM state_vectors_data4 WHERE lat<=' + str(lamax) + ' AND lat>=' + str(lamin) + ' AND lon>=' + str(lomin) + ' AND lon <=' + str(lomax) +  ' AND hour >=' + str(Situatie_begin_unix) + ' AND hour <=' + str(Situatie_eind_unix) + ' AND onground = True'

OSN_data = opensky.rawquery(request_str)

df_OSN_data = pd.DataFrame(OSN_data)

df_OSN_data.to_csv(path_or_buf=CWD + '/Data_Raw/' + filename + '.csv' , index=False)

import os
import pandas as pd
import numpy as np

CWD = os.getcwd()

# de filename wordt de naam van het bestand waarin alle datasets zijn samengevoegd
filename = 'adsb_data'

CWD

#Het samenvoegen van de data
path_to_data = CWD + '/Data_Raw/'
data = pd.DataFrame()

for file_name in [file for file in os.listdir(path_to_data) if file.endswith('.csv')]:
        try:
            with open(path_to_data + file_name) as csv_file:
                data = data.append(pd.read_csv(csv_file))
        except:
            print("ERROR loading file: " + path_to_data + file_name)

data.head()

data.shape

#Data omzetten naar CSV
data.to_csv(path_or_buf=CWD + '/Data_Clean/' + filename + '.csv' , index=False)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime
import calendar
import math
import statistics
import time
from datetime import date, datetime

import warnings
warnings.filterwarnings("ignore") 

CWD = os.getcwd()

pd.set_option('display.max_rows',None)

dynniq = pd.read_csv(filepath_or_buffer = CWD + '/Data_Dynniq/export_db.csv')

dynniq.info()

dynniq.SubSystem.unique()

dynniq.Terminal.unique()

dynniq.Origin.unique()

dynniq.Origin = dynniq.Origin.astype(str).str[14:16]

df = dynniq.drop(['Login','Reason', 'SubSystem'], axis = 1)

df.Time = pd.to_datetime(df['Time'], infer_datetime_format = True)

df.head()

#split de kolom "message" in 2 op de plek van de komma
df[['Message','Overig2']] = df['Message'].str.split(',', expand=True,n=1)

df.drop('Overig2',axis=1,inplace=True)

df.head()

# selecteer alleen de data van 2021, omdat we hier in gaan onderzoeken
df21 = df[df['Time'].dt.year == 2021]

# selecteer alleen maand 8 van 2021
df21 = df21[df21['Time'].dt.month == 8]

# terminal B is een back-up van A, dus deze zijn dubbel
df21.drop(df21.loc[df21['Terminal']=='NCU_T_B_ASPG'].index, inplace=True)

df21.drop(df21[df21['Origin']=='Z1'].index, inplace=True)

df21.reset_index(inplace=True)

df21.drop('index', axis=1, inplace=True)

df21.shape

df21.Origin.unique()

df21.Message.unique()

# maak een lijst met een 0 als de verlichting uit staat, een 1 als de verlichting aanstaat en een 9 als er iets anders mee is
onoff = []

for x in range(0, (len(df21))):
    if 'off' in df21['Message'].iloc[x][-3:]:
        onoff.append(0)
    elif 'on' in df21['Message'].iloc[x][-3:]:
        onoff.append(1)
    else:
        onoff.append(9)


        
df21['On_Off'] = onoff


df21.On_Off.unique()

df21.drop(df21.loc[df21['On_Off'] == 9].index, inplace=True)

df21.On_Off.unique()

df21.reset_index(drop = True , inplace = True)

df21.head()

df21.shape

# selecteer in welke richting de baan wordt gebruikt op basis van de approach verlichting (APH)
richting = []

for x in range(0, (len(df21))):
    if df21['Origin'].iloc[x] == 'Ccr APH0601' :
        richting.append('18C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0603':
        richting.append('18C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0605':
        richting.append('18C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0602':
        richting.append('36C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0604':
        richting.append('36C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0606':
        richting.append('36C')
    else:
        richting.append('Onbekend')
      
df21['richting'] = richting

df21 = df21.sort_values(by=['Time','Origin'])

df21.reset_index(inplace=True)

df21.drop(['index'], axis=1, inplace=True)

df21.head()

# maak een apparte lijsten per richting
testw = df21[df21['Origin']=='W5']
testy = df21[df21['Origin']=='Y']
testz = df21[df21['Origin']=='Z2']

# bereken per richting het tijdsverschil tussen events
testw.sort_values(by=['Origin','Time'])
testw['time_diff'] = testw['Time'].diff()
testw['time_last_action'] = testw['Time'].shift(1)

testy.sort_values(by=['Origin','Time'])
testy['time_diff'] = testy['Time'].diff()
testy['time_last_action'] = testy['Time'].shift(1)

testz.sort_values(by=['Origin','Time'])
testz['time_diff'] = testz['Time'].diff()
testz['time_last_action'] = testz['Time'].shift(1)

print(testw.shape)
print(testy.shape)
print(testz.shape)

test_filtered_1 = testw.append(testy)

test_filtered_2 = test_filtered_1.append(testz)

test_filtered_2.sort_values(by=['Time','Origin'], inplace=True)

test_filtered_2.reset_index(inplace=True)
test_filtered_2.drop(['index'], axis=1, inplace=True)

df21 = test_filtered_2

df21.head(100)

df21.drop(['TimeMilliSec','Terminal'], axis=1, inplace=True)

epoch = []
epoch2 = [0,0,0] 

p='%Y-%m-%d %H:%M:%S'

for x in range(len(df21)):
    epoch.append(int(calendar.timegm(time.strptime(str(df21.Time[x]),p))))


for x in range(3,len(df21)):
        epoch2.append(int(calendar.timegm(time.strptime(str(df21.time_last_action[x]),p))))

df21['time'] = epoch
df21['time_last_action'] = epoch2

df21.head()

df21.to_csv(path_or_buf='/Data_Clean/dynniq_systeem.csv' , index=False)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime
import calendar
import math
import statistics
import time
from datetime import date, datetime

import warnings
warnings.filterwarnings("ignore") 

CWD = os.getcwd()

dynniq = pd.read_csv(filepath_or_buffer = CWD + '/Data_Dynniq/Baangebruik_dynniq.csv')

dynniq.head()

dynniq.info()

dynniq.SubSystem.unique()

dynniq.Terminal.unique()

dynniq.Origin.unique()

dynniq.Origin = dynniq.Origin.astype(str).str[14:16]

df = dynniq.drop(['Login','Reason', 'SubSystem'], axis = 1)

df.Time = pd.to_datetime(df['Time'], infer_datetime_format = True)

# bepaal de lengte van elke "message"
y = []
for x in df['Message']:
    y.append(len(str(x)))
df['length'] = y

df.length.unique()

# filter op messages kleiner dan of gelijk aan 25, want hierin staat of een lamp aan of uit gaat
df = df[df['length']<=25]

# selecteer alleen de data van 2021, omdat we hier in gaan onderzoeken
df21 = df[df['Time'].dt.year == 2021]

# selecteer alleen maand 8 van 2021
df21 = df21[df21['Time'].dt.month == 8]

# terminal B is een back-up van A, dus deze zijn dubbel
df21.drop(df21.loc[df21['Terminal']=='NCU_F2_B_BLOI'].index, inplace=True)
df21.drop(df21.loc[df21['Terminal']=='NCU_P_B_BLOI'].index, inplace=True)

df21.reset_index(inplace=True)

df21.drop('index', axis=1, inplace=True)

df21.shape

pd.set_option('display.max_rows',None)

df21.Origin.unique()

df21.Message.unique()

# elke richting waarin de baan wordt gebruikt heeft een unieke combinatie van verlichting.
# APH0601 + REH0601 geeft aan dat de baan wordt gebruikt voor landen in 18C
# APH0602 + REH0601 geeft aan dat de baan wordt gebruikt voor landen in 36C
# REH0601 geeft aan dat de baan wordt gebruikt voor opstijgen
df21 = df21[(df21['Origin']=='Ccr APH0601')|(df21['Origin']=='Ccr APH0602')|(df21['Origin']=='Ccr REH0601')]

df21.Message.unique()

# maak een lijst met een 0 als de verlichting uit staat, een 1 als de verlichting aanstaat en een 9 als er iets anders mee is
onoff = []

for x in range(0, (len(df21))):
    if 'off' in df21['Message'].iloc[x][-3:]:
        onoff.append(0)
    elif 'on' in df21['Message'].iloc[x][-3:]:
        onoff.append(1)
    else:
        onoff.append(9)


        
df21['On_Off'] = onoff


df21.On_Off.unique()

df21.drop(df21.loc[df21['On_Off'] == 9].index, inplace=True)

df21.On_Off.unique()

df21.reset_index(drop = True , inplace = True)

df21.head()

# selecteer in welke richting de baan wordt gebruikt op basis van de approach verlichting (APH)
richting = []

for x in range(0, (len(df21))):
    if df21['Origin'].iloc[x] == 'Ccr APH0601' :
        richting.append('18C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0603':
        richting.append('18C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0605':
        richting.append('18C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0602':
        richting.append('36C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0604':
        richting.append('36C')
    elif df21['Origin'].iloc[x] == 'Ccr APH0606':
        richting.append('36C')
    else:
        richting.append('Onbekend')
      
df21['richting'] = richting

df21 = df21.sort_values(by=['Time','Origin'])

df21.reset_index(inplace=True)

df21.drop(['index'], axis=1, inplace=True)

df21.head()

# maak een apparte lijsten per richting
testaph01 = df21[df21['Origin']=='Ccr APH0601']
testaph02 = df21[df21['Origin']=='Ccr APH0602']
testreh = df21[df21['Origin']=='Ccr REH0601']

# bereken per richting het tijdsverschil tussen events
testaph01.sort_values(by=['Origin','Time'])
testaph01['time_diff'] = testaph01['Time'].diff()
testaph01['time_last_action'] = testaph01['Time'].diff(-1)
testaph01['time_last_action_shift'] = testaph01['Time'].shift(-1,fill_value=testaph01['Time'].iloc[0])

testaph02.sort_values(by=['Origin','Time'])
testaph02['time_diff'] = testaph02['Time'].diff()
testaph02['time_last_action'] = testaph02['Time'].diff(-1)
testaph02['time_last_action_shift'] = testaph02['Time'].shift(-1,fill_value=testaph02['Time'].iloc[0])

testreh.sort_values(by=['Origin','Time'])
testreh['time_diff'] = testreh['Time'].diff()
testreh['time_last_action'] = testreh['Time'].diff(-1)
testreh['time_last_action_shift'] = testreh['Time'].shift(-1,fill_value=testreh['Time'].iloc[0])

testaph01.info()

print(testaph01.shape)
print(testaph02.shape)
print(testreh.shape)

# verwijder messages als er 1 minuut of minder tussen 2 messages zit
testaph01.drop(testaph01[(testaph01['time_last_action']>='-1 days +23:59:00') & (testaph01['time_last_action']<='0 days 00:00:00')].index, inplace=True)
testaph01.drop(testaph01[(testaph01['time_diff']<='0 days 00:01:00') & (testaph01['time_diff']>='0 days 00:00:00')].index,inplace=True)

testaph02.drop(testaph02[(testaph02['time_last_action']>='-1 days +23:59:00') & (testaph02['time_last_action']<='0 days 00:00:00')].index, inplace=True)
testaph02.drop(testaph02[(testaph02['time_diff']<='0 days 00:01:00') & (testaph02['time_diff']>='0 days 00:00:00')].index,inplace=True)

print(testaph01.shape)
print(testaph02.shape)
print(testreh.shape)

test_filtered_1 = testaph01.append(testaph02)

test_filtered_2 = test_filtered_1.append(testreh)

test_filtered_2.sort_values(by=['Time','Origin'], inplace=True)

test_filtered_2.reset_index(inplace=True)
test_filtered_2.drop(['index'], axis=1, inplace=True)

df21 = test_filtered_2

landen = []

for x in range(0, (len(df21))):
    #Kijken of de regel approach lighting uit staat en of er op de volgende regel de runway edge lighting is en als de tijd niet hetzelfde is. Zoja, opstijgen invullen.
    if (df21['Origin'].iloc[x] == 'Ccr APH0601') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr REH0601') & (df21['Time'].iloc[x] != df21['Time'].shift(-1).iloc[x]) & (df21['On_Off'].iloc[x] == 0):
        landen.append('Opstijgen')
    #Kijken of de regel approach lighting uit staat en of er op de volgende regel de runway edge lighting is en als de tijd niet hetzelfde is. Zoja, opstijgen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr APH0602') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr REH0601') & (df21['Time'].iloc[x] != df21['Time'].shift(-1).iloc[x]) & (df21['On_Off'].iloc[x] == 0):
        landen.append('Opstijgen')
    #Er wordt eerst gekeken of de baan uit staat, als dit zo is wordt dit ingevuld.
    elif (df21['On_Off'].iloc[x] == 0): 
        landen.append('Niet in gebruik')
    #Als de baan ingebruik is, dan eerst kijken of er geland wordt vanuit het noorden. Zoja, dan landen invullen.
    elif df21['Origin'].iloc[x] == 'Ccr APH0601':
        landen.append('Landen')
    #Als de baan ingebruik is, dan eerst kijken of er geland wordt vanuit het zuiden. Zoja, dan landen invullen.
    elif df21['Origin'].iloc[x] == 'Ccr APH0602':
        landen.append('Landen')
    #Kijken of de regel runway edge lighting aanstaat en of dit op de volgende regel ook aanstaat. Zoja, dan opstijgen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr REH0601'):
        landen.append('Opstijgen')
    #Kijken of de regel runway edge lighting aanstaat en of er op de volgende regel de approach lightning aanstaat en als de tijd hetzelfde is en of de regel ervoor aanstaat. Zoja, landen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(1).iloc[x] == 'Ccr APH0601') & (df21['Time'].iloc[x] == df21['Time'].shift(1).iloc[x]) & (df21['On_Off'].shift(1).iloc[x] == 1):
        landen.append('Landen')
    #Kijken of de regel runway edge lighting aanstaat en of er op de volgende regel de approach lightning aanstaat en als de tijd hetzelfde is en of de regel ervoor aanstaat. Zoja, landen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(1).iloc[x] == 'Ccr APH0602') & (df21['Time'].iloc[x] == df21['Time'].shift(1).iloc[x]) & (df21['On_Off'].shift(1).iloc[x] == 1):
        landen.append('Landen')
    #Kijken of de regel runway edge lighting aanstaat en of er op de volgende regel de approach lightning aanstaat en als de tijd hetzelfde is en of de regel ernaar aanstaat. Zoja, opstijgen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr APH0601') & (df21['Time'].iloc[x] != df21['Time'].shift(-1).iloc[x]) & (df21['On_Off'].shift(-1).iloc[x] == 1):
        landen.append('Opstijgen')
    #Kijken of de regel runway edge lighting aanstaat en of er op de volgende regel de approach lightning aanstaat en als de tijd hetzelfde is en of de regel ernaar aanstaat. Zoja, opstijgen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr APH0602') & (df21['Time'].iloc[x] != df21['Time'].shift(-1).iloc[x]) & (df21['On_Off'].shift(-1).iloc[x] == 1):
        landen.append('Opstijgen')
    #Kijken of de regel runway edge lighting aanstaat en of er op de volgende regel de approach lightning uitstaat en als de tijd hetzelfde is en of de regel ernaar aanstaat. Zoja, landen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr APH0601') & (df21['Time'].iloc[x] != df21['Time'].shift(-1).iloc[x]) & (df21['On_Off'].shift(-1).iloc[x] == 0):
        landen.append('Landen')
    #Kijken of de regel runway edge lighting aanstaat en of er op de volgende regel de approach lightning uitstaat en als de tijd hetzelfde is en of de regel ernaar aanstaat. Zoja, landen invullen.
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(-1).iloc[x] == 'Ccr APH0602') & (df21['Time'].iloc[x] != df21['Time'].shift(-1).iloc[x]) & (df21['On_Off'].shift(-1).iloc[x] == 0):
        landen.append('Landen')
    
        
    #Als iets niet aan de voorwaarde voldoet, dan onbekend invullen.
    else:
        landen.append('onbekend')
    

df21['Beweging'] = landen

df21.head(300)

df21['Beweging'].value_counts()

df21.shape

# vul de richting van het baangebruik in op basis van welke verlichting aan staat
for x in range(0, (len(df21))):
    if (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(1).iloc[x] == 'Ccr APH0601') & (df21['Beweging'].iloc[x] == df21['Beweging'].shift(1).iloc[x]):
        df21['richting'].iloc[x] = df21['richting'].shift(1).iloc[x]
    elif (df21['Origin'].iloc[x] == 'Ccr REH0601') & (df21['Origin'].shift(1).iloc[x] == 'Ccr APH0602') & (df21['Beweging'].iloc[x] == df21['Beweging'].shift(1).iloc[x]):
        df21['richting'].iloc[x] = df21['richting'].shift(1).iloc[x]

df21.reset_index(inplace=True)

df21.drop(['index','TimeMilliSec','Terminal','Origin','length','time_last_action'], axis=1, inplace=True)

df21.head()

epoch = []
epoch2 = []

p='%Y-%m-%d %H:%M:%S'


for x in range(len(df21)):
    epoch.append(int(calendar.timegm(time.strptime(str(df21.Time[x]),p))))

for x in range(0,len(df21)):
    epoch2.append(int(calendar.timegm(time.strptime(str(df21.time_last_action_shift[x]),p))))

df21['time'] = epoch
df21['time_last_action_shift'] = epoch2

df21.head()

df21.to_csv(path_or_buf='/Data_Clean/dynniq_systeem_baangebruik.csv' , index=False)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import calendar
import math
import statistics
from datetime import date, datetime
from os import path
from fastkml import  kml
from shapely.geometry import Point, Polygon
import os

CWD = os.getcwd()

# unix tijd om te defineren welke data in welk jaar hoort
Situatie_2019_begin_unix = 1564610400
Situatie_2019_eind_unix = 1567288800
Situatie_2021_begin_unix = 1627768800
Situatie_2021_eind_unix = 1630447200

data = pd.read_csv(CWD + '/Data_Clean/adsb_data.csv')

data['situatie'] = np.where(((data['time'] >= Situatie_2019_begin_unix) & (data['time'] <= Situatie_2019_eind_unix)), '2019', '2021')

y = []
for x in data['callsign']:
    y.append(len(str(x)))

data['call_len'] = y #Kolom aanmaken van de lengte van de callsigns zodat daarop gefilterd kan worden

airplane_data = data.loc[(data['onground'] == True) & 
                         (data['call_len'] >= 5) & 
                         (data['lat'] >= 52.31887) & (data['lat'] <= 52.32386) & 
                         (data['lon'] >= 4.73072) & (data['lon'] <= 4.73552)]

airplane_data = airplane_data[airplane_data['squawk'].notnull()]
#Ground vehicles hebben over het algemeen een callsign gelijk of kleiner dan 4 en geen squawk code
#Overgebleven data moet van vliegtuigen zijn binnen de orgineel aangegeven bounding box en tijdspanne

airplane_data.reset_index(inplace = True)

# zet de tijd om in een datetime format
time = []

for z in airplane_data['time']:
    utc_time = datetime.utcfromtimestamp(z)
    real_time = utc_time.strftime("%Y-%m-%d %H:%M:%S")
    time.append(real_time)

airplane_data['datetime'] = pd.to_datetime(time, format="%Y-%m-%d %H:%M:%S")

df = airplane_data.drop(['onground', 'call_len'], axis= 1)

#Onnodige kolommen worden gedropt

dfy = df.copy()
dfy = df.set_index(['callsign', 'situatie'])

#importeren van een kml bestand. Dit wordt gebruikt om te filteren of de vliegtuigen op de taxibaan zijn.
kml_file = path.join('Knooppunt.kml')

with open(kml_file, 'rt', encoding="utf-8") as myfile:
    doc=myfile.read()


# vul de eerste en derde coördinaten in
mijn_1e_xcoord = str(4.731085113064375)
mijn_1e_zcoord = str(-3.266113346082403)
coord_1e_idx = doc.index(mijn_1e_xcoord)

coord_2e_idx = doc.index(mijn_1e_zcoord,coord_1e_idx+19*3)+19
kml_coords = doc[coord_1e_idx:coord_2e_idx]

coord_values = []
num = ""
for i in range(len(kml_coords)):
    i_char = kml_coords[i]
    if i_char.isnumeric() or i_char == '-' or i_char == '.':
        num = num+i_char
        if i == len(kml_coords)-1:
            coord_values.append(float(num))
    else:
        coord_values.append(float(num))
        num = ""
    #isolate longitude and lattitude
lon= []
lat= []

for i in range(0,len(coord_values),3):
    lon.append(coord_values[i])
for i in range(1,len(coord_values),3):
    lat.append(coord_values[i])

    #Zip longitude and longitude together
kml_xy =  Polygon(list(zip(lon,lat)))

print(coord_values,'\n')
print(lat,'\n')
print(lon,'\n')

# vul alle latitude en longtitude coördinaten in
lat = [52.32235929838046, 52.32126928282872, 52.32103881082556, 52.32070100339872, 52.32043500807058, 52.32030248497378, 52.32014451702346, 52.31995552472713, 52.31974078795366, 52.31947048021787, 52.31908734884821, 52.3174475756656, 52.31742844268029, 52.31851475150383, 52.31927784336468, 52.31953359105014, 52.31972359827604, 52.31989501742536, 52.32008062491258, 52.32041743996272, 52.32033866612916, 52.32069241769287, 52.32070593912082, 52.32072211987975, 52.32076261251542, 52.32081377132051, 52.32088968596717, 52.32097655749163, 52.3211520614492, 52.3213305878777, 52.32138515124168, 52.32143146028785, 52.32154002960341, 52.32169203944285, 52.32183314280488, 52.32195005273363, 52.32495080149144, 52.32496953830486, 52.32272550142944, 52.32191839242426, 52.32184532244251, 52.32180722469496, 52.32178022506725, 52.32175330251666, 52.32175353169699, 52.32168610570145, 52.32169968203196, 52.32184834929689, 52.32201397983169, 52.32254052535707, 52.32430124801767, 52.32407841307511, 52.32235929838046]
lon = [4.731085113064375, 4.732604636009365, 4.733035234546512, 4.733504979339676, 4.733751803674906, 4.733917533992975, 4.73407239629114, 4.734213486971739, 4.734309390119802, 4.73435402250453, 4.734341696994358, 4.734214935836167, 4.734785831969834, 4.734889012069738, 4.73495177648948, 4.734961000992302, 4.734930261606785, 4.73487230653924, 4.734778850973274, 4.734517208541433, 4.736750852254003, 4.736781873125455, 4.736371935063099, 4.735739621915698, 4.735107783359962, 4.73452588605614, 4.734161867581577, 4.733926494293637, 4.733824381109788, 4.734348629894831, 4.734601923511872, 4.734712954892002, 4.734912983577697, 4.735086456987414, 4.735162399094537, 4.735193399143364, 4.735459213569275, 4.734892718265089, 4.734695167901542, 4.734448994324199, 4.734403413326762, 4.734332577927316, 4.734247589604377, 4.734126883138194, 4.733890803152314, 4.733578913242695, 4.733086180969273, 4.73280314975147, 4.73239003083987, 4.731582080124818, 4.729112458336829, 4.728667778823836, 4.731085113064375]

kml_xy =  Polygon(list(zip(lon,lat)))

poly = Polygon(kml_xy)

y = airplane_data['lat']
x = airplane_data['lon']
assert len(y) == len(x) #dubbelcheck dat ze allebei dezelfde lengte hebben

opdeweg = [] 
lis = []

for point in range(len(y)):
    p = Point(x[point],y[point])
    lis.append(p)
    
    if p.within(poly) == False:
        opdeweg.append(0)
    else:
        opdeweg.append(1)

#Lijst aangemaakt waarin staat of een punt binnen de verwachtte kaders valt of er buiten zodat er op gefilterd kan worden

airplane_data['nuttig'] = opdeweg

airplane_data['nuttig'].unique()

df = airplane_data.loc[airplane_data['nuttig'] == 1]

#Alle vliegtuigbeweging punten vallen nu binnen de polygon 

df.reset_index(inplace = True)
df.drop(['level_0', 'index', 'nuttig'], inplace = True, axis=1) #nutteloze kollomen verwijderen

df.shape

df

df.to_csv(path_or_buf='/Data_Clean/df_knooppunt.csv')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import calendar
import math
import statistics
from datetime import date, datetime
from os import path
from fastkml import  kml
from shapely.geometry import Point, Polygon
import os

CWD = os.getcwd()

# unix tijd om te defineren welke data in welk jaar hoort
Situatie_2019_begin_unix = 1564610400
Situatie_2019_eind_unix = 1567288800
Situatie_2021_begin_unix = 1627768800
Situatie_2021_eind_unix = 1630447200

data = pd.read_csv(CWD + '/Data_Clean/adsb_data.csv')

data['situatie'] = np.where(((data['time'] >= Situatie_2019_begin_unix) & (data['time'] <= Situatie_2019_eind_unix)), '2019', '2021')

y = []
for x in data['callsign']:
    y.append(len(str(x)))

data['call_len'] = y #Kollom aanmaken van de lengte van de callsigns zodat daarop gefilterd kan worden

airplane_data = data.loc[(data['onground'] == True) & 
                         (data['call_len'] >= 5) & 
                         (data['lat'] >= 52.29) & (data['lat'] <= 52.34) & 
                         (data['lon'] >= 4.73072) & (data['lon'] <= 4.75)]

airplane_data = airplane_data[airplane_data['squawk'].notnull()]
#Ground vehicles hebben over het algemeen een callsign gelijk of kleiner dan 4 en geen squawk code
#Overgebleven data moet van vliegtuigen zijn binnen de orgineel aangegeven bounding box en tijdspanne

airplane_data.reset_index(inplace = True)

time = []

for z in airplane_data['time']:
    utc_time = datetime.utcfromtimestamp(z)
    real_time = utc_time.strftime("%Y-%m-%d %H:%M:%S")
    time.append(real_time)

airplane_data['datetime'] = pd.to_datetime(time, format="%Y-%m-%d %H:%M:%S")

df = airplane_data.drop(['onground', 'call_len'], axis= 1)

#Onnodige kolommen worden gedropt

dfy = df.copy()
dfy = df.set_index(['callsign', 'situatie'])

#importeren van een kml bestand. Dit wordt gebruikt om te filteren of de vliegtuigen op de taxibaan zijn.
kml_file = path.join('Zwanenburgbaan.kml')

with open(kml_file, 'rt', encoding="utf-8") as myfile:
    doc=myfile.read()

# vul de eerste en derde coördinaten in
mijn_1e_xcoord = str(4.736721035932952)
mijn_1e_zcoord = str(-4.002312753543082)
coord_1e_idx = doc.index(mijn_1e_xcoord)

coord_2e_idx = doc.index(mijn_1e_zcoord,coord_1e_idx+19*3)+19
kml_coords = doc[coord_1e_idx:coord_2e_idx]

coord_values = []
num = ""
for i in range(len(kml_coords)):
    i_char = kml_coords[i]
    if i_char.isnumeric() or i_char == '-' or i_char == '.':
        num = num+i_char
        if i == len(kml_coords)-1:
            coord_values.append(float(num))
    else:
        coord_values.append(float(num))
        num = ""
    #isolate longitude and lattitude
lon= []
lat= []

for i in range(0,len(coord_values),3):
    lon.append(coord_values[i])
for i in range(1,len(coord_values),3):
    lat.append(coord_values[i])

    #Zip longitude and longitude together
kml_xy =  Polygon(list(zip(lon,lat)))

print(coord_values,'\n')
print(lat,'\n')
print(lon,'\n')

# vul alle latitude en longtitude coördinaten in
lat = [52.30145709423931, 52.30144220852637, 52.33138642645944, 52.33143038217568, 52.30145709423931]
lon = [4.736721035932952, 4.73784725923357, 4.740583138104921, 4.73949071313122, 4.736721035932952]

kml_xy =  Polygon(list(zip(lon,lat)))

poly = Polygon(kml_xy)

y = airplane_data['lat']
x = airplane_data['lon']
assert len(y) == len(x) #dubbelcheck dat ze allebei dezelfde lengte hebben

opdeweg = [] 
lis = []

for point in range(len(y)):
    p = Point(x[point],y[point])
    lis.append(p)
    
    if p.within(poly) == False:
        opdeweg.append(0)
    else:
        opdeweg.append(1)

#Lijst aangemaakt waarin staat of een punt binnen de verwachtte kaders valt of er buiten zodat er op gefilterd kan worden

airplane_data['nuttig'] = opdeweg

airplane_data['nuttig'].unique()

df = airplane_data.loc[airplane_data['nuttig'] == 1]

#Alle vliegtuigbeweging punten vallen nu binnen de polygon 

df.reset_index(inplace = True)
df.drop(['level_0', 'index', 'nuttig'], inplace = True, axis=1) #nutteloze kollomen verwijderen

df.shape

df

df.to_csv(path_or_buf='/Data_Clean/df_zwanenburg.csv')

### Packages inladen

import pandas as pd
import numpy as np
import seaborn as sns
import folium
import datetime as dt
import math 
import pyproj
pd.options.display.max_rows = None
import warnings
import statistics
warnings.filterwarnings("ignore") 

### Dynniq data inladen

df = pd.read_csv('dynniq_systeem_baangebruik.csv')

df.head()

### ADSB zwanenburgbaan data

zwanen = pd.read_csv('df_zwanenburg.csv')

zwanen = zwanen.drop(columns='Unnamed: 0')
zwanen['datetime'] = pd.to_datetime(zwanen['datetime'])

zwanen['day'] = zwanen['datetime'].dt.day
zwanen['month'] = zwanen['datetime'].dt.month
zwanen['year'] = zwanen['datetime'].dt.year

zwanen = zwanen[zwanen['year']== 2021]

zwanen = zwanen[zwanen['month']== 8]

zwanen = zwanen.drop(columns = ['velocity', 'heading', 'hour', 'icao24', 'onground'])

zwanen = zwanen.sort_values(by=['callsign', 'year', 'month', 'day'], ignore_index=True)

# df_zwan is een dictionary met daarin unieke combinaties van callsign, year, month en day. 
df_zwan = {f'{i}': d for i, (g, d) in enumerate(zwanen.groupby(['callsign', 'year', 'month', 'day']))}

# define een functie "distance" die gebruikt kan worden om de afstand tussen 2 coordinaten te bepalen
def distance(origin, destination):
    """
    Martin Thoma https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude/43211266#43211266
    
    Calculate the Haversine distance.
    
    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# define een functie "calculate_initial_compass_bearing" zodat de bewegings richting "bearing" kan worden bepaald voor elk meetpunt
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#De code hieronder maakt een list met de huidige lon & lat (a0 & b0) en een list met de volgende lon & lat (a1 & b1). 
#Op basis van deze lists kan de bearing worden bepaald voor elk punt. Deze bearing wordt vervolgens opgeslagen in de list "bearing" 

bearing= []

geodesic = pyproj.Geod(ellps='WGS84')

for x in range(0, len(df_zwan)):
    a0 = df_zwan[str(x)].lat
    a1 = df_zwan[str(x)].lat.shift(-1, fill_value = np.nan)
    
    b0 = df_zwan[str(x)].lon
    b1 = df_zwan[str(x)].lon.shift(-1, fill_value = np.nan)
    
    le = range(0, (len(a0)))
    
    for z in le:
               
        initial_compass_bearing = calculate_initial_compass_bearing((a0.iloc[z], b0.iloc[z]), (a1.iloc[z], b1.iloc[z]))
        bearing.append(initial_compass_bearing)

zwanen['bearing'] = bearing

zwanen['bearing'] = zwanen['bearing'].replace(0, np.nan)

zwanen['bearing_gem'] = zwanen.groupby(['callsign', 'year', 'month', 'day']).bearing.transform('mean')

# df_zwanen is een dictionary met daarin unieke combinaties van callsign, year, month en day. 
df_zwanen = {f'{i}': d for i, (g, d) in enumerate(zwanen.groupby(['callsign', 'year', 'month','day']))}

# df_zwanen is momenteel een dictionary met voor elke unieke combinatie van callsign, year, month en day een list met alle meetpunten.
# Er wordt door dit dictionary heen gegaan met een for loop om de callsign (a), year (year), day(day), month(month), datetime(datetime), time(unixtime), bearing_gem(bearing_gem)
# te extraheren vanuit df_zwanen. Hierdoor is er 1 regel per unieke combinatie.
a = []
year = []
day = []
month = []
datetime = []
unixtime = []
bearing_gem = []

for x in range(0, (len(df_zwanen))):
    
    a.append(df_zwanen[str(x)].callsign.iloc[0])
    year.append(df_zwanen[str(x)].year.iloc[0])
    day.append(df_zwanen[str(x)].day.iloc[0])
    month.append(df_zwanen[str(x)].month.iloc[0])
    datetime.append(df_zwanen[str(x)].datetime.iloc[0])
    unixtime.append(df_zwanen[str(x)].time.iloc[0])
    bearing_gem.append(df_zwanen[str(x)].bearing_gem.iloc[0])

#Er wordt hieronder een dataframe aangemaakt op basis van de geextraheerde data. 

dicto = {'callsign' : a,
         'year' : year,
         'month' : month,
         'day' : day,
         'datetime' : datetime,
         'unixtime' : unixtime,
         'bearing_gem' : bearing_gem}
flights = pd.DataFrame(dicto)

flights.head()

flights.shape

df.head()

df.rename(columns={'time':'action_time'},inplace=True)

df.head()

import sqlite3

#gebruik sqlite3 lokaal
conn = sqlite3.connect(':memory:')
#gebruik "df" & "flights" voor de sqlite database
df.to_sql('df', conn, index=False)
flights.to_sql('flights', conn, index=False)

# voeg alle rijen waarbij de unixtime van df tussen de action_time en time_last_action_shift zitten van fligths
qry = '''
    select *
    from
        df join flights on
        unixtime between action_time and time_last_action_shift
    '''
df = pd.read_sql_query(qry, conn)

df.head()

df['Beweging'].unique()

df_merged = df[df['Message'].str.contains('REH')]

df_merged.shape

df_merged.reset_index(inplace=True)

df_merged.drop('index', axis=1, inplace=True)

df_merged.head()

# creeer een list met de richting waarin de zwanenburgbaan wordt gebruikt
richting = []
for x in range(0, (len(df_merged))):
    # als de bearing tussen 0 en 70 is dan is de richting "36C"
    if (df_merged.richting.iloc[x] == 'Onbekend') & (df_merged.bearing_gem.iloc[x] >= 0) & (df_merged.bearing_gem.iloc[x] <= 70):
        richting.append('36C')
    # als de bearing tussen 290 en 360 is dan is de richting "36C"
    elif (df_merged.richting.iloc[x] == 'Onbekend') & (df_merged.bearing_gem.iloc[x] >= 290) & (df_merged.bearing_gem.iloc[x] <= 360):
        richting.append('36C')
    # als de bearing tussen 110 en 250 is dan is de richting "18C"
    elif (df_merged.richting.iloc[x] == 'Onbekend') & (df_merged.bearing_gem.iloc[x] >= 110) & (df_merged.bearing_gem.iloc[x] <= 250):
        richting.append('18C')
    else:
        richting.append(df_merged.richting.iloc[x])

df_merged['richting'] = richting

df_merged['richting'].value_counts()

df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])



df_merged.head()

df_baangebruik = df_merged[['EventID','Time','Message','On_Off','richting','Beweging','time_last_action_shift']]

df_baangebruik['richting'].unique()

df_baangebruik['richting'].value_counts()

# df_richting is een dictionary met daarin unieke combinaties van time, message.
df_richting = {f'{i}': d for i, (g, d) in enumerate(df_baangebruik.groupby(['Time', 'Message']))}

# tel per unieke combinatie voor df_richting het aantal "18C" & "36C" & "Beweging"
# check vervolgens welke richtin het meest voor komt en verander alle afwijkende richting in deze richting.
new_richting = []

for x in range(0, (len(df_richting))):
    richting = df_richting[str(x)]
    count_18C = richting[richting['richting']=='18C'].shape[0]
    count_36C = richting[richting['richting']=='36C'].shape[0]
    beweging = richting['Beweging'].iloc[0]
    value=0
    
    if (count_18C>count_36C):
        value='18C'
    elif (count_36C>count_18C):
        value='36C'
    elif (count_36C==count_18C) & (beweging == 'Niet in gebruik'):
        value='Oversteken'
    else:
        value='Onbekend'
    
    le = range(0, (len(richting)))
    
    for z in le:
        new_richting.append(value)
       



test = pd.DataFrame(new_richting)

test.head()

test.value_counts()

df_baangebruik['richting'] = new_richting

df_baangebruik.head(5000)

df_baangebruik = df_baangebruik.drop_duplicates()

df_baangebruik = df_baangebruik.reset_index()

df_baangebruik.drop('index', axis=1, inplace=True)

df_baangebruik.head()

df_baangebruik['richting'].unique()

df_baangebruik['richting'].value_counts()

df_baangebruik.to_csv(path_or_buf='baangebruik_fligths.csv' , index=False)

# Hoofdscript

pip install pyopensky

pip install fastkml

pip install shapely

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import calendar
import math
import statistics
from datetime import date, datetime
from os import path
from fastkml import  kml
from shapely.geometry import Point, Polygon
from itertools import tee, chain
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import pyproj
import folium
import sqlite3
import geopandas as gpd
from os import path
import webbrowser
import selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from pathlib import Path

import warnings
warnings.filterwarnings("ignore") 

CWD = os.getcwd()

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

airplane_data = pd.read_csv(CWD + '/Data_Clean/df_knooppunt.csv', index_col=0)

airplane_data['datetime'] = pd.to_datetime(airplane_data['datetime'])

airplane_data['day'] = airplane_data['datetime'].dt.day
airplane_data['month'] = airplane_data['datetime'].dt.month
airplane_data['year'] = airplane_data['datetime'].dt.year

airplane_data = airplane_data[airplane_data['datetime'].dt.month==8]

# Het vliegtuig van de overheid vertrok op één dag meerdere keren, dit kon niet goed weergegeven worden in de data. 
airplane_data = airplane_data[airplane_data['callsign'] != 'PHGOV']

airplane_data.head()

airplane_data.drop_duplicates(inplace=True)

airplane_data.reset_index(inplace=True)

airplane_data.drop(['index'], axis=1, inplace=True)

airplane_data.sort_values(by=['callsign', 'year', 'month','day'],inplace=True)

airplane_data = airplane_data.reset_index()

airplane_data.drop('index',axis=1,inplace=True)

airplane_data.head()

# Tijdverschil tussen vluchten weergeven in een kolom gegroupeerd op callsign, jaar, maand en dag
airplane_data['time_diff'] = airplane_data.groupby(['callsign', 'year', 'month','day'])['time'].diff()

# variable aanmaken welke een dictionary maakt van de gegroepeerde dataframe
df_squawk = {f'{i}': d for i, (g, d) in enumerate(airplane_data.groupby(['callsign', 'year', 'month','day']))}

#Ga door de lijst met unieke combinaties van callsign, year, month en day. Voor elke waarden in de lijst,
# check of de squawk code veranderd en het tijdverschil kleiner is dan 600. Hierdoor worden de vliegtuigen
# waarbij de squawk codes in korte tijd veranderen eruit gefilterd en veranderd in de squawk code die 
# de hele tijd voorkomt. Hierdoor worden de wisselingen in squawk codes opgevangen.

squawk_new = []

for x in range(0, (len(df_squawk))):
    squawk = df_squawk[str(x)].squawk
    time_diff = df_squawk[str(x)].time_diff
    unique_squawk = df_squawk[str(x)].squawk.iloc[0]
    
    le = range(0, (len(squawk)))
    
    for z in le:
        
        if ((squawk.iloc[z] != unique_squawk) & (time_diff.iloc[z] <=600)):
            squawk_new.append(unique_squawk)
        else:
            squawk_new.append(squawk.iloc[z])


airplane_data['squawk'] = squawk_new

airplane_data = airplane_data.sort_values(by = ['callsign', 'year', 'month','day','squawk'], ignore_index = True)

airplane_data.shape

airplane_data.head(5)

#unieke lijst voor data wat gegroupeerd is op callsign, year, month, day en squawk

df_dict = {f'{i}': d for i, (g, d) in enumerate(airplane_data.groupby(['callsign', 'year', 'month','day','squawk']))}

print(df_dict.keys())

df_dict['1']

#Een voorbeeld van een van de visualisaties die uit de code hierboven kan komen
plt.scatter(x = df_dict['2060'].lon, y = df_dict['2060'].lat)
plt.title(str(df_dict['2060'].callsign.unique()) + " " + str(df_dict['2060'].hour.unique()))
plt.yticks([52.318, 52.320, 52.322, 52.324, 52.326, 52.328, 52.330])
plt.xticks([4.722, 4.724, 4.726, 4.728, 4.730, 4.732, 4.734, 4.736])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.show()

# route KML

#Importeren van een kml bestand. Dit wordt gebruikt om te filteren of de vliegtuigen zich op de #taxibaan bevinden

kml_file = path.join('Taxi.kml')

with open(kml_file, 'rt', encoding="utf-8") as myfile:
    doc=myfile.read()

#Vul de eerste en derde coordinaten in

mijn_1e_xcoord = str(4.734561558021221)
mijn_1e_zcoord = str(-2.304046490721269)
coord_1e_idx = doc.index(mijn_1e_xcoord)

coord_2e_idx = doc.index(mijn_1e_zcoord,coord_1e_idx+18*3)+18
kml_coords = doc[coord_1e_idx:coord_2e_idx]

print(kml_coords)


# Deze code geeft een lijst met latitude en longitude waardes.

coord_values = []
num = ""
for i in range(len(kml_coords)):
    i_char = kml_coords[i]
    if i_char.isnumeric() or i_char == '-' or i_char == '.':
        num = num+i_char
        if i == len(kml_coords)-1:
            coord_values.append(float(num))
    else:
        coord_values.append(float(num))
        num = ""
    #isolate longitude and lattitude
lon= []
lat= []

for i in range(0,len(coord_values),3):
    lon.append(coord_values[i])
for i in range(1,len(coord_values),3):
    lat.append(coord_values[i])

    #Zip longitude and longitude together
kml_xy =  Polygon(list(zip(lon,lat)))

print(coord_values,'\n')
print(lat,'\n')
print(lon,'\n')



# Met de lat en lon wordt een polygon gemaakt

lat = [52.32081099823717, 52.32083122452499, 52.3208797221147, 52.32092891532012, 52.32097631430403, 52.32106546287495, 52.32115357658748, 52.32170781421692, 52.32183519094455, 52.32185758852329, 52.32189176434333, 52.32192677701127, 52.32198685441944, 52.32203346105308, 52.32211609995174, 52.32230968494117, 52.32234564409856, 52.32242387125484, 52.32254124804669, 52.32282920144415, 52.32300644251436, 52.32336310328748, 52.32353818061527, 52.32374115000253, 52.32390448839891, 52.32410471009445, 52.32430014298379, 52.32407921984141, 52.32365830963205, 52.32308472634, 52.32263882694155, 52.32219047998462, 52.32185267164736, 52.32154519150711, 52.32136041823966, 52.32117484518434, 52.32117084279839, 52.32104291781623, 52.32071354472734, 52.32045009675006, 52.32044050977068, 52.3203360958636, 52.32022738212471, 52.32041468944384, 52.32081099823717]
lon = [4.734561558021221, 4.734430165642365, 4.734217749292435, 4.734054229663817, 4.733930539714095, 4.733867758133322, 4.733830844094875, 4.733092603661251, 4.732810713015326, 4.732756325236604, 4.732668239352997, 4.732574236423317, 4.732453801139451, 4.732371476279623, 4.732238964787343, 4.731936392611889, 4.731881726360383, 4.731767081718942, 4.731587257357601, 4.731175037645237, 4.730923029008518, 4.730423331409959, 4.730180686593679, 4.729896590300109, 4.729664502436462, 4.729382620921072, 4.729109882595768, 4.72867254977595, 4.7292694362211, 4.730072975366282, 4.730701038478459, 4.731333278156511, 4.731804107010616, 4.732237405903921, 4.7324988871207, 4.732762083583464, 4.732781924059505, 4.733034992357343, 4.733501285767008, 4.733747204020369, 4.733744136516378, 4.733880419954353, 4.733999693348854, 4.734519064996643, 4.734561558021221]
kml_xy =  Polygon(list(zip(lon,lat)))

polytaxi = Polygon(kml_xy)
polytaxi

#Importeren van een kml bestand. Dit wordt gebruikt om te filteren of de vliegtuigen zich op Yankee #bevinden

kml_file = path.join('Yankee.kml')

with open(kml_file, 'rt', encoding="utf-8") as myfile:
    doc=myfile.read()
 
#Vul de eerste en derde coordinaten in

mijn_1e_xcoord = str(4.733828634207462)
mijn_1e_zcoord = str(-2.128166778527121)
coord_1e_idx = doc.index(mijn_1e_xcoord)

coord_2e_idx = doc.index(mijn_1e_zcoord,coord_1e_idx+18*3)+18
kml_coords = doc[coord_1e_idx:coord_2e_idx]

print(kml_coords)


# Deze code geeft een lijst met latitude en longitude waardes.

coord_values = []
num = ""
for i in range(len(kml_coords)):
    i_char = kml_coords[i]
    if i_char.isnumeric() or i_char == '-' or i_char == '.':
        num = num+i_char
        if i == len(kml_coords)-1:
            coord_values.append(float(num))
    else:
        coord_values.append(float(num))
        num = ""
    #isolate longitude and lattitude
lon= []
lat= []

for i in range(0,len(coord_values),3):
    lon.append(coord_values[i])
for i in range(1,len(coord_values),3):
    lat.append(coord_values[i])

    #Zip longitude and longitude together
kml_xy =  Polygon(list(zip(lon,lat)))

print(coord_values,'\n')
print(lat,'\n')
print(lon,'\n')

lat = [52.32115366424473, 52.32117997306595, 52.32121129542061, 52.32125456236628, 52.32133446924168, 52.32137333972412, 52.32140727391502, 52.32144845717308, 52.32150356010667, 52.32156840378664, 52.32161507943068, 52.32169495072942, 52.32177118553047, 52.32182407735866, 52.32186711732152, 52.32189788711532, 52.32218776566705, 52.32238656551178, 52.32247279333779, 52.32253405683565, 52.32275333491528, 52.32337200185824, 52.32386239087198, 52.32429753153576, 52.32462972579992, 52.32495217824942, 52.32496850947957, 52.32388855061926, 52.32328682148115, 52.32309682431131, 52.32289803153824, 52.32272636542835, 52.322442647571, 52.32216326755802, 52.32194441158454, 52.32188571987531, 52.3218445053902, 52.32178004033797, 52.32176097157308, 52.32175541333786, 52.32174984804502, 52.32174272413108, 52.32169116282326, 52.32170624044937, 52.32115366424473]
lon = [4.733828634207462, 4.733886951600857, 4.733978973862407, 4.734119154970449, 4.734356666559005, 4.734536472210873, 4.73463761377134, 4.734747269750297, 4.734851986607063, 4.734946847734996, 4.735005022230796, 4.735077176630911, 4.735129210773026, 4.735153275934469, 4.735167391366513, 4.735173342308093, 4.735204767908774, 4.735221421820963, 4.735230829788469, 4.735235294339297, 4.735254081338174, 4.735315545072556, 4.735363325360861, 4.735403110871907, 4.735432289424002, 4.735463640432071, 4.734900770446851, 4.734801144442164, 4.73474092727785, 4.73471866781126, 4.734710216838518, 4.734693038254642, 4.734603663206121, 4.73452232429911, 4.734462243096917, 4.73443724056851, 4.734397305606752, 4.734279635896397, 4.734194802728567, 4.734121942130511, 4.733985466775494, 4.73386319672545, 4.733573583594033, 4.733093152742232, 4.733828634207462]
kml_xy =  Polygon(list(zip(lon,lat)))


poly1 = Polygon(kml_xy)
poly1

# Met de lat en lon wordt een polygon gemaakt

lat = [52.32115366424473, 52.32117997306595, 52.32121129542061, 52.32125456236628, 52.32133446924168, 52.32137333972412, 52.32140727391502, 52.32144845717308, 52.32150356010667, 52.32156840378664, 52.32161507943068, 52.32169495072942, 52.32177118553047, 52.32182407735866, 52.32186711732152, 52.32189788711532, 52.32218776566705, 52.32238656551178, 52.32247279333779, 52.32253405683565, 52.32275333491528, 52.32337200185824, 52.32386239087198, 52.32429753153576, 52.32462972579992, 52.32495217824942, 52.32496850947957, 52.32388855061926, 52.32328682148115, 52.32309682431131, 52.32289803153824, 52.32272636542835, 52.322442647571, 52.32216326755802, 52.32194441158454, 52.32188571987531, 52.3218445053902, 52.32178004033797, 52.32176097157308, 52.32175541333786, 52.32174984804502, 52.32174272413108, 52.32169116282326, 52.32170624044937, 52.32115366424473]
lon = [4.733828634207462, 4.733886951600857, 4.733978973862407, 4.734119154970449, 4.734356666559005, 4.734536472210873, 4.73463761377134, 4.734747269750297, 4.734851986607063, 4.734946847734996, 4.735005022230796, 4.735077176630911, 4.735129210773026, 4.735153275934469, 4.735167391366513, 4.735173342308093, 4.735204767908774, 4.735221421820963, 4.735230829788469, 4.735235294339297, 4.735254081338174, 4.735315545072556, 4.735363325360861, 4.735403110871907, 4.735432289424002, 4.735463640432071, 4.734900770446851, 4.734801144442164, 4.73474092727785, 4.73471866781126, 4.734710216838518, 4.734693038254642, 4.734603663206121, 4.73452232429911, 4.734462243096917, 4.73443724056851, 4.734397305606752, 4.734279635896397, 4.734194802728567, 4.734121942130511, 4.733985466775494, 4.73386319672545, 4.733573583594033, 4.733093152742232, 4.733828634207462]
kml_xy =  Polygon(list(zip(lon,lat)))


polyY = Polygon(kml_xy)
polyY

#Importeren van een kml bestand. Dit wordt gebruikt om te filteren of de vliegtuigen zich op Zulu #bevinden

kml_file = path.join('Zulu.kml')

with open(kml_file, 'rt', encoding="utf-8") as myfile:
    doc=myfile.read()
 

#Vul de eerste en derde coordinaten in

mijn_1e_xcoord = str(4.733999715516262)
mijn_1e_zcoord = str(-2.152172742232368)
coord_1e_idx = doc.index(mijn_1e_xcoord)

coord_2e_idx = doc.index(mijn_1e_zcoord,coord_1e_idx+18*3)+18
kml_coords = doc[coord_1e_idx:coord_2e_idx]

print(kml_coords)


# Deze code geeft een lijst met latitude en longitude waardes.

coord_values = []
num = ""
for i in range(len(kml_coords)):
    i_char = kml_coords[i]
    if i_char.isnumeric() or i_char == '-' or i_char == '.':
        num = num+i_char
        if i == len(kml_coords)-1:
            coord_values.append(float(num))
    else:
        coord_values.append(float(num))
        num = ""
    #isolate longitude and lattitude
lon= []
lat= []

for i in range(0,len(coord_values),3):
    lon.append(coord_values[i])
for i in range(1,len(coord_values),3):
    lat.append(coord_values[i])

    #Zip longitude and longitude together
kml_xy =  Polygon(list(zip(lon,lat)))

print(coord_values,'\n')
print(lat,'\n')
print(lon,'\n')



# Met de lat en lon wordt een polygon gemaakt

lat = [52.32022766961732, 52.32017320048103, 52.32010671241238, 52.32002611423525, 52.31992835864764, 52.31981438594046, 52.31971846280538, 52.31963269267853, 52.31955962297683, 52.31941756498416, 52.31930714064136, 52.31913806605495, 52.31899263375055, 52.31894725886897, 52.31880787913053, 52.31867327948181, 52.31853694551505, 52.31841205868361, 52.31832145948944, 52.31813990744968, 52.31799704875704, 52.31784442642994, 52.31761306519998, 52.3174488404272, 52.31742856188598, 52.31749571520244, 52.31761858195501, 52.31774975198148, 52.31783961305745, 52.31795625809794, 52.31810780762734, 52.31828138212916, 52.31843755848265, 52.31853643305961, 52.31867501746078, 52.31887173339508, 52.3189559402497, 52.3190761628984, 52.31916035795946, 52.3192269247368, 52.3194964317595, 52.31954266325193, 52.31959215551012, 52.31965545359028, 52.31971828936786, 52.31980904630455, 52.31987216120807, 52.31992037301269, 52.31994867272742, 52.31996730596746, 52.32000967705855, 52.32006263119644, 52.32013941507857, 52.32019367489632, 52.32027495629438, 52.32035059355762, 52.32041554661159, 52.32022766961732]
lon = [4.733999715516262, 4.734052047938238, 4.734112090013829, 4.734172021422285, 4.734231990588238, 4.734285254597221, 4.734318725649751, 4.734339712599009, 4.734348869830203, 4.734351987804781, 4.734354403165493, 4.734352477047928, 4.734350071869331, 4.734345623042085, 4.734330785132004, 4.734318641206778, 4.734308069965496, 4.734300803172228, 4.734292937597018, 4.7342718312843, 4.73425843315979, 4.734243779783158, 4.734224064950252, 4.734205685659683, 4.734774455407022, 4.734781936477066, 4.734794193388153, 4.734799847870375, 4.734803947930766, 4.734817125230957, 4.734834433276673, 4.734850563094075, 4.73486771382535, 4.734877563766194, 4.734888026766959, 4.734906638874675, 4.7349151407474, 4.73493020305618, 4.734938576375878, 4.734942222865708, 4.734952044235625, 4.734951358723858, 4.734943790803968, 4.734932579953539, 4.734917822498462, 4.734894433142669, 4.734873562437545, 4.734855509529337, 4.734844767915298, 4.734835195362588, 4.73481407357399, 4.734784539142689, 4.734736036714116, 4.734699176428013, 4.734638074154631, 4.734576222070917, 4.734517177471207, 4.733999715516262]
kml_xy =  Polygon(list(zip(lon,lat)))


polyZ = Polygon(kml_xy)
polyZ

lat = [52.32022766961732, 52.32017320048103, 52.32010671241238, 52.32002611423525, 52.31992835864764, 52.31981438594046, 52.31971846280538, 52.31963269267853, 52.31955962297683, 52.31941756498416, 52.31930714064136, 52.31913806605495, 52.31899263375055, 52.31894725886897, 52.31880787913053, 52.31867327948181, 52.31853694551505, 52.31841205868361, 52.31832145948944, 52.31813990744968, 52.31799704875704, 52.31784442642994, 52.31761306519998, 52.3174488404272, 52.31742856188598, 52.31749571520244, 52.31761858195501, 52.31774975198148, 52.31783961305745, 52.31795625809794, 52.31810780762734, 52.31828138212916, 52.31843755848265, 52.31853643305961, 52.31867501746078, 52.31887173339508, 52.3189559402497, 52.3190761628984, 52.31916035795946, 52.3192269247368, 52.3194964317595, 52.31954266325193, 52.31959215551012, 52.31965545359028, 52.31971828936786, 52.31980904630455, 52.31987216120807, 52.31992037301269, 52.31994867272742, 52.31996730596746, 52.32000967705855, 52.32006263119644, 52.32013941507857, 52.32019367489632, 52.32027495629438, 52.32035059355762, 52.32041554661159, 52.32022766961732]
lon = [4.733999715516262, 4.734052047938238, 4.734112090013829, 4.734172021422285, 4.734231990588238, 4.734285254597221, 4.734318725649751, 4.734339712599009, 4.734348869830203, 4.734351987804781, 4.734354403165493, 4.734352477047928, 4.734350071869331, 4.734345623042085, 4.734330785132004, 4.734318641206778, 4.734308069965496, 4.734300803172228, 4.734292937597018, 4.7342718312843, 4.73425843315979, 4.734243779783158, 4.734224064950252, 4.734205685659683, 4.734774455407022, 4.734781936477066, 4.734794193388153, 4.734799847870375, 4.734803947930766, 4.734817125230957, 4.734834433276673, 4.734850563094075, 4.73486771382535, 4.734877563766194, 4.734888026766959, 4.734906638874675, 4.7349151407474, 4.73493020305618, 4.734938576375878, 4.734942222865708, 4.734952044235625, 4.734951358723858, 4.734943790803968, 4.734932579953539, 4.734917822498462, 4.734894433142669, 4.734873562437545, 4.734855509529337, 4.734844767915298, 4.734835195362588, 4.73481407357399, 4.734784539142689, 4.734736036714116, 4.734699176428013, 4.734638074154631, 4.734576222070917, 4.734517177471207, 4.733999715516262]
kml_xy =  Polygon(list(zip(lon,lat)))


polyW = Polygon(kml_xy)
polyW


#Importeren van een kml bestand. Dit wordt gebruikt om te filteren of de vliegtuigen zich op Zulu #bevinden

kml_file = path.join('Whiskey.kml')

with open(kml_file, 'rt', encoding="utf-8") as myfile:
    doc=myfile.read()
 

#Vul de eerste en derde coordinaten in

mijn_1e_xcoord = str(4.734521337112558)
mijn_1e_zcoord = str(-1.862379545009929)
coord_1e_idx = doc.index(mijn_1e_xcoord)

coord_2e_idx = doc.index(mijn_1e_zcoord,coord_1e_idx+18*3)+18
kml_coords = doc[coord_1e_idx:coord_2e_idx]

print(kml_coords)


# Deze code geeft een lijst met latitude en longitude waardes.

coord_values = []
num = ""
for i in range(len(kml_coords)):
    i_char = kml_coords[i]
    if i_char.isnumeric() or i_char == '-' or i_char == '.':
        num = num+i_char
        if i == len(kml_coords)-1:
            coord_values.append(float(num))
    else:
        coord_values.append(float(num))
        num = ""
    #isolate longitude and lattitude
lon= []
lat= []

for i in range(0,len(coord_values),3):
    lon.append(coord_values[i])
for i in range(1,len(coord_values),3):
    lat.append(coord_values[i])

    #Zip longitude and longitude together
kml_xy =  Polygon(list(zip(lon,lat)))

print(coord_values,'\n')
print(lat,'\n')
print(lon,'\n')



# Deze code geeft een lijst met latitude en longitude waardes.

lat = [52.3204171654266, 52.3204092634928, 52.32038774001034, 52.32036608006575, 52.3203405511961, 52.32068469869168, 52.32068610906026, 52.32069681192147, 52.32070407580463, 52.32070869620352, 52.32071530733362, 52.3207155105887, 52.3207262138244, 52.3207357961742, 52.32075313556085, 52.32076610374131, 52.3207815599192, 52.32079290319398, 52.32081144773836, 52.3204171654266]
lon = [4.734521337112558, 4.734753518894119, 4.735424175175728, 4.736026718682298, 4.736747426671986, 4.736780040999484, 4.736712170313114, 4.736470367048941, 4.736283664722563, 4.736130428817098, 4.735985020733384, 4.735885549121097, 4.735647206004414, 4.735512886701714, 4.735263828328669, 4.735085344221894, 4.73485870011671, 4.734718809504077, 4.734558983725446, 4.734521337112558]
kml_xy =  Polygon(list(zip(lon,lat)))


polyW = Polygon(kml_xy)
polyW

### Data Manipulatie

df_shape = {f'{i}': d for i, (g, d) in enumerate(airplane_data.groupby(['callsign', 'year', 'month','day','squawk']))}

Yankee = [[4.737473791698852,52.33246857141998],
[4.73302114387778,52.33260482310985],
[4.732536299503412,52.32301767102151],
[4.734538359546841,52.32121520406333],
[4.73671173101028,52.32115017490013],
[4.737473791698852,52.33246857141998]]

Whiskey = [[4.736781449544704,52.32091404406341],
[4.734294109463661,52.32102049018291],
[4.734659759938959,52.32035450001977],
[4.736727644536456,52.32021469139551],
[4.736781449544704,52.32091404406341]]

Zulu = [[4.73669759397084,52.31986178746558],
[4.73203751865203,52.32075511718472],
[4.731826470817222,52.31782283791117],
[4.736594010958592,52.31767137286197],
[4.736668943794948,52.3192189339204],
[4.73669759397084,52.31986178746558]]

Taxi = [[4.72239035435514,52.32926799322635],
[4.720631073255319,52.32794178491736],
[4.732025282105015,52.32082503745605],
[4.734628433111727,52.32030169699405],
[4.734225523935425,52.32101352748153],
[4.734450288559698,52.32123270942335],
[4.732495482820438,52.32298212672291],
[4.72239035435514,52.32926799322635]]

# De code hieronder creeërt een lijst met daarin aangegeven of een punt binnen een van de drie richtingen opgaat

traject = []

yankee = polyY
whiskey = polyW
zulu = polyZ
taxi = polytaxi

for x in range(0, len(df_shape)):
    le = range(0 , (len(df_shape[str(x)])))
    
    for z in le:
        
        lat = list(df_shape[str(x)].lat)
        lon = list(df_shape[str(x)].lon)
                
        p = Point(lon[z] , lat[z])
        
        if p.within(yankee) == True:
            traject.append('Y')
        elif p.within(whiskey) == True:
            traject.append('W')
        elif p.within(zulu) == True:
            traject.append('Z')
        elif p.within(taxi) == True:
            traject.append('V')
        else:
            traject.append('onbekend')






dft = airplane_data.sort_values(by = ['callsign', 'year', 'month','day','squawk'], ignore_index = True)

dft['traject'] = traject

df_traj = {f'{i}': d for i, (g, d) in enumerate(dft.groupby(['callsign', 'year', 'month','day','squawk']))}

# Middels onderstaande code wordt de afstand tussen 2 coordinaten berekend
def distance(origin, destination):
    """
    Martin Thoma https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude/43211266#43211266
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# define een functie "calculate_initial_compass_bearing" zodat de bewegings richting "bearing" kan worden bepaald voor elk meetpunt
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#De code hieronder maakt een list met de huidige lon & lat (a0 & b0) en een list met de volgende lon & lat (a1 & b1). 
#Op basis van deze lists kan de bearing worden bepaald voor elk punt. Deze bearing wordt vervolgens opgeslagen in de list "bearing"
c= []
DELTAt = []
bearing= []

geodesic = pyproj.Geod(ellps='WGS84')


for x in range(0, len(df_traj)):
    a0 = df_traj[str(x)].lat
    a1 = df_traj[str(x)].lat.shift(-1, fill_value = np.nan)
    
    b0 = df_traj[str(x)].lon
    b1 = df_traj[str(x)].lon.shift(-1, fill_value = np.nan)
    
    tijd = df_traj[str(x)].time.max() - df_traj[str(x)].time.min()
    DELTAt.append(tijd)
    
    

    
    le = range(0, (len(a1)))
    
    for z in le:
        
        r = (distance((a0.iloc[z], b0.iloc[z]), (a1.iloc[z], b1.iloc[z])))*1000        
        
        initial_compass_bearing = calculate_initial_compass_bearing((a0.iloc[z], b0.iloc[z]), (a1.iloc[z], b1.iloc[z]))
        bearing.append(initial_compass_bearing)
        c.append(r)

 

dft['afstand'] = c
dft['bearing'] = bearing

dft.head()

dft['bearing'] = dft['bearing'].replace(0, np.nan)

dft['bearing_gem'] = dft.groupby(['callsign', 'year', 'month','day', 'squawk']).bearing.transform('mean')

# In onderstaande code wordt aan de kolom 'Cardinal' de richting oost of west toegevoegd op basis van het aantal graden in de 'bearing-gem'
conditions = [
(dft['bearing_gem'] >= 0) & (dft['bearing_gem'] <= 180) ,
(dft['bearing_gem'] > 180) & (dft['bearing_gem'] < 360)]

values = ['O','W']


dft['Cardinal'] = np.select(conditions, values)


dft.head()

# Richting bepalen

aircraft = '1'
xsu, ysu = zip(*Yankee)

xss, yss = zip(*Whiskey)

xsd, ysd = zip(*Zulu)

xst, yst = zip(*Taxi)

plt.plot(*polytaxi.exterior.xy)
plt.plot(*polyY.exterior.xy)
plt.plot(*polyW.exterior.xy)
plt.plot(*polyZ.exterior.xy)
sns.scatterplot(x = df_traj[aircraft].lon, y = df_traj[aircraft].lat, hue = df_traj[aircraft].traject, palette = 'PuOr')
plt.title(str(df_traj[aircraft].callsign.unique()) + str(df_traj[aircraft].year.unique())+ str(df_traj[aircraft].month.unique())+ str(df_traj[aircraft].day.unique()))
plt.ylabel('Latitude')
plt.xlabel('Longitude')

plt.show()


dft.head()

df_analysis = {f'{i}': d for i, (g, d) in enumerate(dft.groupby(['callsign', 'year', 'month','day', 'squawk']))}

# df_analysis is momenteel een dictionary met voor elke unieke combinatie van callsign, year, month en day een list met alle meetpunten.
# Er wordt door dit dictionary heen gegaan met een for loop om de callsign (a), year (year), day(day), month(month), traject(traject), afstand(tot_distance), cardinal(Cardinal)
# te extraheren vanuit df_analysis. Hierdoor is er 1 regel per unieke combinatie.
a = []
b = []
year = []
day = []
month = []
traject = []
tot_distance = []
datetime = []
cardinal = []

for x in range(0, (len(df_analysis))):
    
    a.append(df_analysis[str(x)].callsign.iloc[0])
    b.append(df_analysis[str(x)].datetime.dt.time.iloc[0])
    year.append(df_analysis[str(x)].year.iloc[0])
    day.append(df_analysis[str(x)].day.iloc[0])
    month.append(df_analysis[str(x)].month.iloc[0])
    tot_distance.append(df_analysis[str(x)].afstand.cumsum().max())
    datetime.append(df_analysis[str(x)].datetime.iloc[0])
    cardinal.append(df_analysis[str(x)].Cardinal.iloc[0])
    
    if df_analysis[str(x)].traject.iloc[0] == 'V':
        traject.append(df_analysis[str(x)].traject.iloc[-1])
    elif df_analysis[str(x)].traject.iloc[0] != 'V':
        traject.append(df_analysis[str(x)].traject.iloc[0])
        
    

    


#Er wordt hieronder een dataframe aangemaakt op basis van de geextraheerde data.
dicto = {'callsign' : a,
         'year' : year,
         'month' : month,
         'day' : day,
         'time' : b,
         'totale_tijd (sec)' : DELTAt,
         'traject' : traject,
         'Tot Distance' : tot_distance,
         'datetime' : datetime,
         'cardinal' : cardinal}


flights = pd.DataFrame(dicto)

flights.reset_index(drop = True, inplace = True)
flights['avg_speed']= flights['Tot Distance']/flights['totale_tijd (sec)']
flights.head(10)

flights['time'] = pd.to_datetime(flights['time'], format='%H:%M:%S')

flights['hour'] = flights['time'].dt.hour
flights['minutes'] = flights['time'].dt.minute
flights['seconds'] = flights['time'].dt.second

flights.head(10)

flights = flights[flights['totale_tijd (sec)']>=25]

flights = flights[flights['totale_tijd (sec)']<100]

pd.crosstab(index = flights['traject'],
columns= flights['year'])

2019 = 9639 (zonder de onbekend)
2021 = 5798
2021 = -39.8% tov 2019

flights = flights[flights['traject']!='onbekend']

# Met onderstaande code worden de wave tijden toegevoegd in de kolom 'wave'
# De wave tijden zijn gebaseerd op de schiphol website inbound piektijden
# De wave tijden zijn in UTC vermeld, maar oorspronkelijk van lokale tijd gehaald
wave = []
for x in range(0, len(flights)):
    tijd_h=flights.iloc[x].hour
    tijd_m=flights.iloc[x].minutes
    tijd_s=flights.iloc[x].seconds
            
    if (tijd_h >= 0) & (tijd_h < 5):
        wave.append('02 - 07')
    elif (tijd_h >= 5) & (tijd_h < 7):
        wave.append('07 - 09')
    elif (tijd_h >= 7) & (tijd_h < 9):
        wave.append('09 - 11')
    elif (tijd_h >= 9) & (tijd_h < 11):
        wave.append('11 - 13')
    elif (tijd_h >= 11) & (tijd_h < 13):
        wave.append('13 - 15')
    elif (tijd_h >= 13) & (tijd_h < 15):
        wave.append('15 - 17')
    elif (tijd_h >= 15) & (tijd_h < 17):
        wave.append('17 - 19')
    elif (tijd_h >= 17) & (tijd_h < 19):
        wave.append('19 - 21')
    elif (tijd_h >= 19) & (tijd_h < 24):
        wave.append('21 - 02')
    else:
        wave.append('9')

flights['wave'] = wave

data_non = flights.loc[(flights['traject'] == 'Y') |
                    (flights['traject'] == 'W') |
                    (flights['traject'] == 'Z') ]

data_west = data_non[data_non['cardinal']=='W']

# data_non wordt de dataframe met alle oostwaardse vliegtuig bewegingen.
# Deze dataframe zal gebruikt worden voor verder onderzoek
data_non = data_non[data_non['cardinal']=='O']

pd.crosstab(index = data_west['traject'],
columns= data_west['year'])

pd.crosstab(index = data_non['traject'],
columns= data_non['year'])

fig = px.box(data_frame = data_non, x='year', y='avg_speed', color='year', template='simple_white',   title='Gemiddelde snelheid over knooppunt W5', category_orders={ 'year':['2019', '2021']})
fig.update_traces(width=0.1)
fig.update_xaxes(title_text='Year')
fig.update_yaxes(title_text='Average velocity')
fig.show()


# Afsplitsen van data_non naar 2 verschillende jaartallen
data_non19 = data_non[data_non["year"]== 2019]
data_non21 = data_non[data_non["year"]== 2021]


fig = go.Figure()

fig.add_trace(go.Histogram(
    x=data_non19['avg_speed'].values,
    name='2019',
    marker_color='blue',
    histnorm = 'percent'
    ))

fig.add_trace(go.Histogram(
    x=data_non21['avg_speed'].values,
    name='2021',
    marker_color='red',
    histnorm = 'percent',
    nbinsx=50

    ))

fig.show()

fig = go.Figure()

fig.add_trace(go.Box(
    y=data_non19['avg_speed'].values,
    x=data_non19['traject'].values,
    name='2019',
    marker_color='blue',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=data_non21['avg_speed'].values,
    x=data_non21['traject'].values,
    name='2021',
    marker_color='red',
    boxmean = True
    ))

fig.update_layout(boxmode='group')

fig.update_layout({
    'xaxis':{'title':{'text': 'Traject richting'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'yaxis':{'title':{'text':'Gemiddelde snelheid (m/s)'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'title':{'text':'Vergelijking trajectsnelheden tussen 2019 & 2021', 
                            'x':0.48,
                            'font':{'size':30}},
    'width':1920,
    'height':1080,
    'paper_bgcolor':'#fff', #de kleur om de grafiek
    'plot_bgcolor':'rgb(242,242,242)', #de kleur in de grafiek
    'legend':{'yanchor':'top', 'y':1., 'xanchor':'left', 'x':1.01, 'font':{'size':30}}
    
})


                 
fig.show()

fig.write_image('fig_snelheid_19_21.png', scale=3, width=1920,height=1080)

fig = go.Figure()

fig.add_trace(go.Box(
    y=data_non19['avg_speed'].values,
    x=data_non19['wave'].values,
    name='2019',
    marker_color='blue'
    ))

fig.add_trace(go.Box(
    y=data_non21['avg_speed'].values,
    x=data_non21['wave'].values,
    name='2021',
    marker_color='red'
    ))

fig.update_layout(boxmode='group')
fig.update_xaxes(categoryorder='category ascending')
fig.show()

data_non['traject'].unique()

data_2019_y = data_non19[data_non19['traject'] == 'Y']
data_2019_w = data_non19[data_non19['traject'] == 'W']
data_2019_z = data_non19[data_non19['traject'] == 'Z']
data_2021_y = data_non21[data_non21['traject'] == 'Y']
data_2021_w = data_non21[data_non21['traject'] == 'W']
data_2021_z = data_non21[data_non21['traject'] == 'Z']

print('2019 Yankee: ' + str(len(data_2019_y)))
print('2019 Wiskey: ' + str(len(data_2019_w)))
print('2019 Zulu: ' + str(len(data_2019_z)))
print(' ')
print('2021 Yankee: ' + str(len(data_2021_y)))
print('2021 Wiskey: ' + str(len(data_2021_w)))
print('2021 Zulu: ' + str(len(data_2021_z)))

perc_2019_y =round(len(data_2019_y)/len(data_non19),ndigits=3)
perc_2019_w = round(len(data_2019_w)/len(data_non19),ndigits=3)
perc_2019_z = round(len(data_2019_z)/len(data_non19),ndigits=3)
perc_2021_y = round(len(data_2021_y)/len(data_non21),ndigits=3)
perc_2021_w = round(len(data_2021_w)/len(data_non21),ndigits=3)
perc_2021_z = round(len(data_2021_z)/len(data_non21),ndigits=3)

print('2019 Yankee: ' + str(perc_2019_y))
print('2019 Wiskey: ' + str(perc_2019_w))
print('2019 Zulu: ' + str(perc_2019_z))
print(' ')
print('2021 Yankee: ' + str(perc_2021_y))
print('2021 Wiskey: ' + str(perc_2021_w))
print('2021 Zulu: ' + str(perc_2021_z))

data_non.head()

px.histogram(data_frame=data_non, x='totale_tijd (sec)', color='year',histnorm='percent', barmode='overlay' )

data_non.reset_index(inplace=True)
data_non.drop('index', axis=1, inplace=True)

# De epoch lijst wordt aangemaakt waarbij de unix tijd wordt toegevoeg daan de data_non dataframe
epoch = []

p='%Y-%m-%d %H:%M:%S'

for x in range(len(data_non)):
    epoch.append(int(calendar.timegm(time.strptime(str(data_non.datetime[x]),p))))
data_non['unix'] = epoch

data_non.head()

## combineren systeem data en ads-b data

pd.set_option('display.max_columns',None)

data_dynniq = pd.read_csv(CWD + '/Data_Clean/dynniq_systeem.csv')

data_dynniq.head()

test_data_dynniq = data_dynniq[['EventID','time','time_last_action','Message','Origin','On_Off']]

test_data_dynniq = test_data_dynniq.rename(columns={'time': 'unixtime'})

test_data_dynniq.head()

test_data_non = data_non[['unix','datetime','callsign','totale_tijd (sec)','traject','Tot Distance', 'avg_speed','wave']]

test_data_non.sort_values(by=['datetime'],inplace=True)

test_data_non.reset_index(inplace=True)

test_data_non.drop('index',axis=1,inplace=True)

test_data_non.head()

test_data_dynniq.head()



#Gebruik sqlite3 lokaal
conn = sqlite3.connect(':memory:')
#Door gebruik van test_data_dynniq en test_data_non voor sql database
test_data_dynniq.to_sql('test_data_dynniq', conn, index=False)
test_data_non.to_sql('test_data_non', conn, index=False)
#Selecteer callsign, datetime, traject, avg_speed en wave uit de test_data_non en maak 3 nieuwe kolommen genaamd 'Y, W5, Z2'
#Voeg een 1 toe in Y, W5 of Z2 wanneer de unix tijd van test_data_non tussen de unixtime en time_last_action van test_data_dynniq zit,
#En sorteer op datetime en callsign.
qry = '''
    select a.callsign, a.datetime, a.traject, a.avg_speed, a.wave,
    b.on_off as 'Y',  
    c.on_off as 'W5',
    d.on_off as 'Z2'
    from test_data_non a
    join test_data_dynniq b on a.unix <= b.unixtime and a.unix >= b.time_last_action and b.origin = 'Y'
    join test_data_dynniq c on a.unix <= c.unixtime and a.unix >= c.time_last_action and c.origin = 'W5'
    join test_data_dynniq d on a.unix <= d.unixtime and a.unix >= d.time_last_action and d.origin = 'Z2'
    order by a.datetime, a.callsign
    '''
taxiway_data = pd.read_sql_query(qry, conn)

taxiway_data.head()

# uitput geeft een 1 wanneer 'inactief' en 0 wanneer 'actief'
taxiway_data['Y'] = taxiway_data['Y'].replace(1, 'Inactive')
taxiway_data['Y'] = taxiway_data['Y'].replace(0, 'Active')
taxiway_data['W5'] = taxiway_data['W5'].replace(1, 'Inactive')
taxiway_data['W5'] = taxiway_data['W5'].replace(0, 'Active')
taxiway_data['Z2'] = taxiway_data['Z2'].replace(1, 'Inactive')
taxiway_data['Z2'] = taxiway_data['Z2'].replace(0, 'Active')

# vervang de inactief voor 0 en actief voor 1 om logica te creëren
taxiway_data['Y'] = taxiway_data['Y'].replace('Inactive', 0)
taxiway_data['Y'] = taxiway_data['Y'].replace('Active', 1)
taxiway_data['W5'] = taxiway_data['W5'].replace('Inactive', 0)
taxiway_data['W5'] = taxiway_data['W5'].replace('Active', 1)
taxiway_data['Z2'] = taxiway_data['Z2'].replace('Inactive', 0)
taxiway_data['Z2'] = taxiway_data['Z2'].replace('Active', 1)

taxiway_data.head()

taxiway_data = taxiway_data[taxiway_data['datetime']>='2021-01-01 00:00:00']

taxiway_data.reset_index(inplace=True)

taxiway_data.drop('index', axis=1, inplace=True)

# als Y, W5, Z2 gelijk zijn aan 0, dan is het alcms_gebruik 0, anders 1, en als Y, W5, Z2 allen gelijk zijn aan 1, dan is de output 2
alcms_gebruik = []
for x in range(len(taxiway_data)):
    if (taxiway_data.Y[x]==0) & (taxiway_data.W5[x]==0) & (taxiway_data.Z2[x]==0):
        alcms_gebruik.append(0)
    elif (taxiway_data.Y[x]==1) & (taxiway_data.W5[x]==1) & (taxiway_data.Z2[x]==1):
        alcms_gebruik.append(2)    
    else:
        alcms_gebruik.append(1)
        

taxiway_data['ALCMS'] = alcms_gebruik

# Aantallen voor W5 en Z bij alcms in gebruik en niet in gebruik
df_alcms_gebruik = taxiway_data[(taxiway_data['ALCMS']==1) & (taxiway_data['traject']!='Y')]
df_alcms_niet_gebruik = taxiway_data[(taxiway_data['ALCMS']==0) & (taxiway_data['traject']!='Y') | (taxiway_data['ALCMS']==2) & (taxiway_data['traject']!='Y')]


print(len(df_alcms_gebruik))
print(len(df_alcms_niet_gebruik))


df_alcms_gebruik['wave'].value_counts()

df_alcms_niet_gebruik['wave'].value_counts()

fig = go.Figure()

fig.add_trace(go.Box(
    y=df_alcms_gebruik['avg_speed'].values,
    x=df_alcms_gebruik['traject'].values,
    name='ALCMS in gebruik',
    marker_color='blue',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=df_alcms_niet_gebruik['avg_speed'].values,
    x=df_alcms_niet_gebruik['traject'].values,
    name='ALCMS niet in gebruik',
    marker_color='red',
    boxmean = True
    ))

fig.update_layout(boxmode='group')

fig.update_layout({
    'xaxis':{'title':{'text': 'Traject richting'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'yaxis':{'title':{'text':'Gemiddelde snelheid (m/s)'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'title':{'text':'Vergelijking trajectsnelheden tussen ALCMS gebruik', 
                            'x':0.48,
                            'font':{'size':30}},
    'width':1920,
    'height':1080,
    'paper_bgcolor':'#fff', #de kleur om de grafiek
    'plot_bgcolor':'rgb(242,242,242)', #de kleur in de grafiek
    'legend':{'yanchor':'top', 'y':1., 'xanchor':'left', 'x':1.01, 'font':{'size':30}}
    
})

fig.show()

fig.write_image('fig_snelheid_gebruik_nietgebruik.png', scale=3, width=1920,height=1080)

fig = go.Figure()

fig.add_trace(go.Box(
    y=df_alcms_gebruik['avg_speed'].values,
    name='ALCMS in gebruik',
    marker_color='blue',
    offsetgroup=1,
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=df_alcms_niet_gebruik['avg_speed'].values,
    name='ALCMS niet in gebruik',
    marker_color='red',
    offsetgroup=1,
    boxmean = True
    ))

fig.update_layout(boxmode='group')

fig.update_layout({
    'xaxis':{'title':{'text': ''},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'yaxis':{'title':{'text':'Gemiddelde snelheid (m/s)'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'title':{'text':'Vergelijking gemiddelde snelheid tussen ALCMS gebruik', 
                            'x':0.48,
                            'font':{'size':30}},
    'width':1920,
    'height':1080,
    'paper_bgcolor':'#fff', #de kleur om de grafiek
    'plot_bgcolor':'rgb(242,242,242)', #de kleur in de grafiek
    'legend':{'yanchor':'top', 'y':1., 'xanchor':'left', 'x':1.01, 'font':{'size':30}}
    
})

fig.show()

fig.write_image('fig_totaal_snelheid_gebruik_nietgebruik.png', scale=3, width=1920,height=1080)



fig.update_layout({
    'xaxis':{'title':{'text': 'Traject richting'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'yaxis':{'title':{'text':'Gemiddelde snelheid (m/s)'},
                            'tickfont':{'size':25},
                            'titlefont':{'size':25}},
    'title':{'text':'Vergelijking gemiddelde snelheid tussen ALCMS gebruik', 
                            'x':0.48,
                            'font':{'size':30}},
    'width':1920,
    'height':1080,
    'paper_bgcolor':'#fff', #de kleur om de grafiek
    'plot_bgcolor':'rgb(242,242,242)', #de kleur in de grafiek
    'legend':{'yanchor':'top', 'y':1., 'xanchor':'left', 'x':1.01, 'font':{'size':30}}
    
})

df_alcms_gebruik_w = df_alcms_gebruik[df_alcms_gebruik['traject']=='W']
df_alcms_gebruik_z = df_alcms_gebruik[df_alcms_gebruik['traject']=='Z']

df_alcms_niet_gebruik_w = df_alcms_niet_gebruik[df_alcms_niet_gebruik['traject']=='W']
df_alcms_niet_gebruik_z = df_alcms_niet_gebruik[df_alcms_niet_gebruik['traject']=='Z']

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=df_alcms_gebruik_w['avg_speed'].values,
    name='gebruik W',
    marker_color='blue',
    histnorm = 'percent'
    ))

fig.add_trace(go.Histogram(
    x=df_alcms_niet_gebruik_w['avg_speed'].values,
    name='niet gebruik W',
    marker_color='red',
    histnorm = 'percent',
    nbinsx=50

    ))

fig.show()

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=df_alcms_gebruik_z['avg_speed'].values,
    name='gebruik Z',
    marker_color='blue',
    histnorm = 'percent'
    ))

fig.add_trace(go.Histogram(
    x=df_alcms_niet_gebruik_z['avg_speed'].values,
    name='niet gebruik Z',
    marker_color='red',
    histnorm = 'percent',
    nbinsx=50

    ))

fig.show()





taxiway_data['traject'].value_counts()

taxiway_data.head()

# mergen baangebruik data

flights_baangebruik = pd.read_csv(CWD + '/Data_Clean/baangebruik_fligths.csv')

flights_baangebruik['Time'] = pd.to_datetime(flights_baangebruik['Time'])

# De epoch lijst wordt aangemaakt waarbij de unix tijd wordt toegevoeg daan de flights_baangebruik dataframe action_time kolom
epoch = []

p='%Y-%m-%d %H:%M:%S'

for x in range(len(flights_baangebruik)):
    epoch.append(int(calendar.timegm(time.strptime(str(flights_baangebruik.Time[x]),p))))
flights_baangebruik['action_time'] = epoch

flights_baangebruik.head()

flights_baangebruik['richting'].unique()

flights_baangebruik['richting'].value_counts()

# De epoch lijst wordt aangemaakt waarbij de unix tijd wordt toegevoeg daan de taxiway_data dataframe unix kolom

epoch = []

p='%Y-%m-%d %H:%M:%S'

for x in range(len(taxiway_data)):
    epoch.append(int(calendar.timegm(time.strptime(str(taxiway_data.datetime[x]),p))))
taxiway_data['unix'] = epoch

taxiway_data.head()

#Gebruik sqlite3 lokaal
conn = sqlite3.connect(':memory:')
#Door gebruik van test_data_dynniq en test_data_non voor sql database
taxiway_data.to_sql('taxiway_data', conn, index=False)
flights_baangebruik.to_sql('flights_baangebruik', conn, index=False)
#Selecteer callsign, datetime, traject, avg_speed, Y, W5, Z2, ALCMS, unix en wave uit de taxiway_data en maak 3 nieuwe kolommen genaamd 'richting, beweging, on_off'
#voeg kolom toe aan taxiway_data, wat de richting, beweging en het baangebruik aangeeft,
#En sorteer op datetime.
qry = '''
    select a.callsign, a.datetime, a.traject, a.avg_speed, a.Y, a.W5, a.Z2, a.ALCMS, a.unix, a.wave,
    b.richting as 'richting',
    c.beweging as 'beweging',
    d.on_off as 'on_off'
    from taxiway_data a
    join flights_baangebruik b on a.unix <= b.time_last_action_shift and a.unix >= b.action_time
    join flights_baangebruik c on a.unix <= c.time_last_action_shift and a.unix >= c.action_time
    join flights_baangebruik d on a.unix <= d.time_last_action_shift and a.unix >= d.action_time
    order by a.datetime
    '''
data_taxi_baan = pd.read_sql_query(qry, conn)

data_taxi_baan.head(300)

#Maak lijsten die aangeven per richting of de zwanenburgbaan in gebruik is, niet in gebruik, en of hierbij het ALCMS aan staat.
w_runway_in_use_w = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==1)]
w_runway_not_in_use_w = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==0)]
w_runway_in_use_alcms_off_w = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==1)]
w_runway_not_in_use_alcms_off_w = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==0)]

w_runway_in_use_y = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==1)]
w_runway_not_in_use_y = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==0)]
w_runway_in_use_alcms_off_y = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==1)]
w_runway_not_in_use_alcms_off_y = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==0)]


w_runway_in_use_z = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1) & (data_taxi_baan['on_off']==1)]
w_runway_not_in_use_z = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1) & (data_taxi_baan['on_off']==0)]
w_runway_in_use_alcms_off_z = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==1)]
w_runway_not_in_use_alcms_off_z = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0) & (data_taxi_baan['on_off']==0)]


fig = go.Figure()

fig.add_trace(go.Box(
    y=w_runway_in_use_w['avg_speed'].values,
    x=w_runway_in_use_w['wave'].values,
    name='runway in gebruik',
    marker_color='blue',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=w_runway_in_use_alcms_off_w['avg_speed'].values,
    x=w_runway_in_use_alcms_off_w['wave'].values,
    name='runway in gebruik, alcms uit',
    marker_color='green',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=w_runway_not_in_use_w['avg_speed'].values,
    x=w_runway_not_in_use_w['wave'].values,
    name='runway niet in gebruik',
    marker_color='red',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=w_runway_not_in_use_alcms_off_w['avg_speed'].values,
    x=w_runway_not_in_use_alcms_off_w['wave'].values,
    name='runway niet in gebruik, alcms uit',
    marker_color='orange',
    boxmean = True
    ))

fig.update_layout(boxmode='group',
                 title='Gemiddelde snelheid over W5')
fig.update_xaxes(categoryorder='category ascending')

fig.update_layout({
    'xaxis':{'title':{'text': 'Inbound piek'},
                            'tickfont':{'size':15},
                            'titlefont':{'size':15}},
    'yaxis':{'title':{'text':'Gemiddelde snelheid (m/s)'},
                            'tickfont':{'size':15},
                            'titlefont':{'size':15}},
    'title':{'text':'Gemiddelde snelheid over W5', 
                            'x':0.5,
                            'font':{'size':30}},
    'width':960,
    'height':540,
    'paper_bgcolor':'#fff', #de kleur om de grafiek
    'plot_bgcolor':'rgb(242,242,242)', #de kleur in de grafiek
    
})

fig.show()

#bepaal voor runway not in use of er een keuze gemaakt moet worden tussen W5 en een andere richting
ywz=[]
for x in range(len(w_runway_not_in_use_w)):
    y = w_runway_not_in_use_w.iloc[x].Y
    w = w_runway_not_in_use_w.iloc[x].W5
    z = w_runway_not_in_use_w.iloc[x].Z2
    ywz.append(str(y)+str(w)+str(z))

w_runway_not_in_use_w['YWZ'] = ywz

w_runway_not_in_use_w['YWZ'].value_counts()

len(w_runway_not_in_use_alcms_off_w)

w_runway_in_use_w['wave'].value_counts()

fig = go.Figure()

fig.add_trace(go.Box(
    y=w_runway_in_use_z['avg_speed'].values,
    x=w_runway_in_use_z['wave'].values,
    name='runway in gebruik',
    marker_color='blue',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=w_runway_in_use_alcms_off_z['avg_speed'].values,
    x=w_runway_in_use_alcms_off_z['wave'].values,
    name='runway in gebruik, alcms uit',
    marker_color='green',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=w_runway_not_in_use_z['avg_speed'].values,
    x=w_runway_not_in_use_z['wave'].values,
    name='runway niet in gebruik',
    marker_color='red',
    boxmean = True
    ))

fig.add_trace(go.Box(
    y=w_runway_not_in_use_alcms_off_z['avg_speed'].values,
    x=w_runway_not_in_use_alcms_off_z['wave'].values,
    name='runway niet in gebruik, alcms uit',
    marker_color='orange',
    boxmean = True
    ))

fig.update_layout(boxmode='group',
                 title='Gemiddelde snelheid over Z')
fig.update_xaxes(categoryorder='category ascending')

fig.update_layout({
    'xaxis':{'title':{'text': 'Inbound piek'},
                            'tickfont':{'size':15},
                            'titlefont':{'size':15}},
    'yaxis':{'title':{'text':'Gemiddelde snelheid (m/s)'},
                            'tickfont':{'size':15},
                            'titlefont':{'size':15}},
    'title':{'text':'Gemiddelde snelheid over Zulu', 
                            'x':0.5,
                            'font':{'size':30}},
    'width':960,
    'height':540,
    'paper_bgcolor':'#fff', #de kleur om de grafiek
    'plot_bgcolor':'rgb(242,242,242)', #de kleur in de grafiek
    
})

fig.show()

## Aantal vliegtuig bewegingen inzien per wave, status van ALCMS en baangebruik

w_runway_not_in_use_alcms_off_z['wave'].value_counts()

print('data_taxi_baan : \n' + str(data_taxi_baan['wave'].value_counts()))

print('w_runway_in_use : ' + str(w_runway_in_use_w['avg_speed'].count()))
print('w_runway_not_in_use : ' + str(w_runway_not_in_use_w['avg_speed'].count()))
print('w_runway_in_use_alcms_off : ' + str(w_runway_in_use_alcms_off_w['avg_speed'].count()))
print('w_runway_not_in_use_alcms_off : ' + str(w_runway_not_in_use_alcms_off_w['avg_speed'].count()))

print('w_runway_in_use : \n' + str(w_runway_in_use_w['wave'].value_counts()))
print('w_runway_in_use_alcms_off : \n' + str(w_runway_in_use_alcms_off_w['wave'].value_counts()))

print('w_runway_not_in_use : \n' + str(w_runway_not_in_use_w['wave'].value_counts()))
print('w_runway_not_in_use_alcms_off : \n' + str(w_runway_not_in_use_alcms_off_w['wave'].value_counts()))

print('w_runway_in_use : ' + str(w_runway_in_use_z['avg_speed'].count()))
print('w_runway_not_in_use : ' + str(w_runway_not_in_use_z['avg_speed'].count()))
print('w_runway_in_use_alcms_off : ' + str(w_runway_in_use_alcms_off_z['avg_speed'].count()))
print('w_runway_not_in_use_alcms_off : ' + str(w_runway_not_in_use_alcms_off_z['avg_speed'].count()))

print('w_runway_in_use : \n' + str(w_runway_in_use_z['wave'].value_counts()))
print('w_runway_in_use_alcms_off : \n' + str(w_runway_in_use_alcms_off_z['wave'].value_counts()))

print('w_runway_not_in_use : \n' + str(w_runway_not_in_use_z['wave'].value_counts()))
print('w_runway_not_in_use_alcms_off : \n' + str(w_runway_not_in_use_alcms_off_z['wave'].value_counts()))

data_taxi_baan.head()

data_taxi_baan[(data_taxi_baan['Y']==1) & (data_taxi_baan['Z2']==1) & (data_taxi_baan['W5']==0)]

# Lijsten aanmaken per traject W, Y, Z op basis van kruispunt ALCMS gebruik
z_z2 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['ALCMS']==1) & (data_taxi_baan['Z2']==1)].shape[0]
y_y = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['ALCMS']==1) & (data_taxi_baan['Y']==1)].shape[0]
w_w5 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['ALCMS']==1) & (data_taxi_baan['W5']==1)].shape[0]

yw_z2 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1)].shape[0] + data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1)].shape[0]
zw_y = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0)].shape[0] + data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0)].shape[0]
zy_w5 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0)].shape[0] + data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0)].shape[0]

yz_yz2 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1)].shape[0] + data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1)].shape[0]
w_yz2 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==1)].shape[0]

wz_w5z2 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==1)].shape[0] + data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==1)].shape[0]
y_w5z2 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==1)].shape[0]

wy_w5y = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0)].shape[0] + data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0)].shape[0]
z_w5y2 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==0)].shape[0]



#Percentages berekenen van vliegtuigen die de juiste aangegeven richtingen hebben opgevolgd van het ALCMS
perc_z = z_z2/(z_z2+yw_z2)*100
perc_y = y_y/(y_y+zw_y)*100
perc_w = w_w5/(w_w5+zy_w5)*100

perc_yz = yz_yz2/(yz_yz2+w_yz2)*100
#perc_wz = wz_w5z2/(wz_w5z2+y_w5z2)*100
perc_wy = wy_w5y/(wy_w5y+z_w5y2)*100

totaal = data_taxi_baan['traject'].shape[0]
alles_uit = data_taxi_baan[(data_taxi_baan['Y']==0) & (data_taxi_baan['W5']==0) & (data_taxi_baan['Z2']==0)].shape[0]
alles_aan = data_taxi_baan[(data_taxi_baan['Y']==1) & (data_taxi_baan['W5']==1) & (data_taxi_baan['Z2']==1)].shape[0]

print('Er zijn ' + str(totaal) + ' bewegingen.')
print('Daarvan zijn er ' + str(alles_uit) + ' bewegingen terwijl alle lichten uit staan')
print('en ' + str(alles_aan) + ' bewegingen terwijl alle lichten aan staan.')

print('Voor 1 richting per keer worden de volgende percentages opgevolgd: \n')

print('Z : ' + str(round(perc_z,1)) + '% van de ' + str(z_z2+yw_z2) + ' bewegingen')
print('Y : ' + str(round(perc_y,1)) + '% van de ' + str(y_y+zw_y) + ' bewegingen')
print('W : ' + str(round(perc_w,1)) + '% van de ' + str(w_w5+zy_w5) + ' bewegingen')

print('\nEr zijn ook scenarios waarbij er 2 richtingen aan staan : \n')

print('Y en Z : ' + str(round(perc_yz,1)) + '% van de ' + str(yz_yz2+w_yz2) + ' bewegingen')
#print('W of Z : ' + str(round(perc_wz,1)) + '% van de ' + str(wz_w5z2+y_w5z2) + ' bewegingen')
print('W en Y : ' + str(round(perc_wy,1)) + '% van de ' + str(wy_w5y+z_w5y2) + ' bewegingen')

data_taxi_baan['beweging'].unique()

data_taxi_baan['beweging'].value_counts()

#Lijsten aangemaakt van het taxiverkeer per mogelijkheid van het baangebruik (landen / stijgen) en het gevolgde traject
z_landen_18 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Landen')].shape[0]
y_landen_18 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Landen')].shape[0]
w_landen_18 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Landen')].shape[0]

z_landen_36 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Landen')].shape[0]
y_landen_36 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Landen')].shape[0]
w_landen_36 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Landen')].shape[0]

z_stijgen_18 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Opstijgen')].shape[0]
y_stijgen_18 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Opstijgen')].shape[0]
w_stijgen_18 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Opstijgen')].shape[0]

z_stijgen_36 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Opstijgen')].shape[0]
y_stijgen_36 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Opstijgen')].shape[0]
w_stijgen_36 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Opstijgen')].shape[0]

z_niet_gebruik_18 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Niet in gebruik')].shape[0]
y_niet_gebruik_18 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Niet in gebruik')].shape[0]
w_niet_gebruik_18 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['richting']=='18C') & (data_taxi_baan['beweging']=='Niet in gebruik')].shape[0]

z_niet_gebruik_36 = data_taxi_baan[(data_taxi_baan['traject']=='Z') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Niet in gebruik')].shape[0]
y_niet_gebruik_36 = data_taxi_baan[(data_taxi_baan['traject']=='Y') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Niet in gebruik')].shape[0]
w_niet_gebruik_36 = data_taxi_baan[(data_taxi_baan['traject']=='W') & (data_taxi_baan['richting']=='36C') & (data_taxi_baan['beweging']=='Niet in gebruik')].shape[0]



#Percentage trajecten berekenen van het taxiverkeer wat een bepaald traject volgt bij baangebruik 
print(str(round((z_landen_18/(z_landen_18+y_landen_18+w_landen_18))*100,1)) + '% van de ' + str(z_landen_18+y_landen_18+w_landen_18) + ' taxiende vliegtuigen taxien over Zulu terwijl 18C wordt gebruikt voor landen')
print(str(round((y_landen_18/(z_landen_18+y_landen_18+w_landen_18))*100,1)) + '% van de ' + str(z_landen_18+y_landen_18+w_landen_18) + ' taxiende vliegtuigen taxien over Yankee terwijl 18C wordt gebruikt voor landen')
print(str(round((w_landen_18/(z_landen_18+y_landen_18+w_landen_18))*100,1)) + '% van de ' + str(z_landen_18+y_landen_18+w_landen_18) + ' taxiende vliegtuigen taxien over Whiskey 5 terwijl 18C wordt gebruikt voor landen' + '\n')

#print('Z landen 36C: ' + str(round((z_landen_36/(z_landen_36+y_landen_36+w_landen_36))*100,1)) + '%')
#print('Y landen 36C: ' + str(round((y_landen_36/(z_landen_36+y_landen_36+w_landen_36))*100,1)) + '%')
#print('W landen 36C: ' + str(round((w_landen_36/(z_landen_36+y_landen_36+w_landen_36))*100,1)) + '%')

#print('Z opstijgen 18C: ' + str(round((z_stijgen_18/(z_stijgen_18+y_stijgen_18+w_stijgen_18))*100,1)) + '%')
print(str(round((y_stijgen_18/(z_stijgen_18+y_stijgen_18+w_stijgen_18))*100,1)) + '% van de ' +str(z_stijgen_18+y_stijgen_18+w_stijgen_18) + ' taxiende vliegtuigen taxien over Yankee terwijl 18C wordt gebruikt voor opstijgen')
print(str(round((w_stijgen_18/(z_stijgen_18+y_stijgen_18+w_stijgen_18))*100,1)) + '% van de ' +str(z_stijgen_18+y_stijgen_18+w_stijgen_18) + ' taxiende vliegtuigen taxien over Whiskey 5 terwijl 18C wordt gebruikt voor opstijgen' + '\n')

#print('Z opstijgen 36C: ' + str(round((z_stijgen_36/(z_stijgen_36+y_stijgen_36+w_stijgen_36))*100,1)) + '%')
#print('Y opstijgen 36C: ' + str(round((y_stijgen_36/(z_stijgen_36+y_stijgen_36+w_stijgen_36))*100,1)) + '%')
#print('W opstijgen 36C: ' + str(round((w_stijgen_36/(z_stijgen_36+y_stijgen_36+w_stijgen_36))*100,1)) + '%')

print(str(round(((z_niet_gebruik_18+z_niet_gebruik_36)/(z_niet_gebruik_18+y_niet_gebruik_18+w_niet_gebruik_18+z_niet_gebruik_36+y_niet_gebruik_36+w_niet_gebruik_36))*100,1)) + '% van de ' + str(z_niet_gebruik_18+y_niet_gebruik_18+w_niet_gebruik_18+z_niet_gebruik_36+y_niet_gebruik_36+w_niet_gebruik_36) + ' taxiende vliegtuigen taxien over Zulu terwijl 18C/36C niet wordt gebruikt')
print(str(round(((y_niet_gebruik_18+y_niet_gebruik_36)/(z_niet_gebruik_18+y_niet_gebruik_18+w_niet_gebruik_18+z_niet_gebruik_36+y_niet_gebruik_36+w_niet_gebruik_36))*100,1)) + '% van de ' + str(z_niet_gebruik_18+y_niet_gebruik_18+w_niet_gebruik_18+z_niet_gebruik_36+y_niet_gebruik_36+w_niet_gebruik_36) + ' taxiende vliegtuigen taxien over Yankee terwijl 18C/36C niet wordt gebruikt')
print(str(round(((w_niet_gebruik_18+w_niet_gebruik_36)/(z_niet_gebruik_18+y_niet_gebruik_18+w_niet_gebruik_18+z_niet_gebruik_36+y_niet_gebruik_36+w_niet_gebruik_36))*100,1)) + '% van de ' + str(z_niet_gebruik_18+y_niet_gebruik_18+w_niet_gebruik_18+z_niet_gebruik_36+y_niet_gebruik_36+w_niet_gebruik_36) + ' taxiende vliegtuigen taxien over Whiskey 5 terwijl 18C/36C niet wordt gebruikt')


#Aparte dataframes aanmaken van baangebruik 18C landen voor de richtingen Zulu en Whiskey 5
runway_in_use_z_landen_18C = w_runway_in_use_z[(w_runway_in_use_z['richting']=='18C')&(w_runway_in_use_z['beweging']=='Landen')]
runway_in_use_w_landen_18C = w_runway_in_use_w[(w_runway_in_use_w['richting']=='18C')&(w_runway_in_use_w['beweging']=='Landen')]

w_runway_in_use_z.head()

print('Gemiddelde snelheid van vliegtuigen over Zulu terwijl 18C voor landen wordt gebruikt is ' + str(round(runway_in_use_z_landen_18C['avg_speed'].mean(),2))+' m/s')
print('Gemiddelde snelheid van vliegtuigen over Whiskey 5 terwijl 18C voor landen wordt gebruikt is ' + str(round(runway_in_use_w_landen_18C['avg_speed'].mean(),2))+' m/s')

runway_in_use_landen_18C = runway_in_use_z_landen_18C.append(runway_in_use_w_landen_18C)

print('Gemiddelde snelheid van alle vliegtuigen terwijl 18C wordt gebruikt voor landen is ' + str(round(runway_in_use_landen_18C['avg_speed'].mean(),2)) +' m/s. Er is dus ' + str(round((runway_in_use_z_landen_18C['avg_speed'].mean()-runway_in_use_landen_18C['avg_speed'].mean()),2))+' m/s winst te behalen als alle vliegtuigen over Zulu gaan.')



runway_in_use_landen_36C = data_taxi_baan[data_taxi_baan['beweging']=='Niet in gebruik']

runway_in_use_z_landen_36C = runway_in_use_landen_36C[(runway_in_use_landen_36C['traject']=='Z')]
runway_in_use_w_landen_36C = runway_in_use_landen_36C[runway_in_use_landen_36C['traject']=='W']

print('Gemiddelde snelheid van vliegtuigen over Zulu terwijl 18C/36C niet gebruikt wordt ' + str(round(runway_in_use_z_landen_36C['avg_speed'].mean(),2))+' m/s')
print('Gemiddelde snelheid van vliegtuigen over Whiskey 5 terwijl 18C/36C niet gebruikt wordt ' + str(round(runway_in_use_w_landen_36C['avg_speed'].mean(),2))+' m/s')

print('Gemiddelde snelheid van alle vliegtuigen terwijl 18C/36C niet gebruikt wordt ' + str(round(runway_in_use_landen_36C['avg_speed'].mean(),2)) +' m/s. Er is dus ' + str(round((runway_in_use_z_landen_36C['avg_speed'].mean()-runway_in_use_w_landen_36C['avg_speed'].mean()),2))+' m/s winst te behalen als alle vliegtuigen over Zulu gaan.')



#Kaart maken met de polygons
m = folium.Map(location=[52.323 , 4.73], zoom_start= 15, tiles='CartoDB positron')
folium.GeoJson(polytaxi, style_function=lambda x: {'fillColor': 'orange', 'color':'orange'}).add_to(m)
folium.GeoJson(polyY, style_function=lambda x: {'fillColor': 'green', 'color':'green'}).add_to(m)
folium.GeoJson(polyW, style_function=lambda x: {'fillColor': 'red', 'color':'red'}).add_to(m)
folium.GeoJson(polyZ, style_function=lambda x: {'fillColor': 'blue', 'color':'blue'}).add_to(m)
m

m.save('kruispunt_w5.html')

path = os.getcwd()

# Om de kaart op te slaan
delay=4
os.chdir(path)

fn='kruispunt_w5.html'
tmpurl='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=fn)

s=Service(ChromeDriverManager().install())
browser = webdriver.Chrome(service=s)
browser.maximize_window()
browser.get(tmpurl)

time.sleep(delay)
browser.save_screenshot('kruispunt_w5.png')
browser.quit()

