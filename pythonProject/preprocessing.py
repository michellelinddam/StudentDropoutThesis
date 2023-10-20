import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_studerende = pd.read_excel('Frafald_grunddata_20230921.xlsx')
df_eksamen = pd.read_excel('Frafald_grunddata_karakter_20230921.xlsx')

#%% FJERNER KOLONNER
#%%
# dropout related columnes
remove_dropout_related_columns = ['frafald_beslutter', 'frafald_begrundelse', 'udmeldbegrundelse', 'studie_status', 'studieslut', 'studie_event']
df_studerende.drop(columns=remove_dropout_related_columns, inplace=True)

# andre irrelevante columns fra grunddata (fra feature selection):
remove_columns = ['loadtime', 'nationalitet', 'studieretning', 'institutionslandekode', 'institutionskode', 'dimissionsalder', 'institutionsland', 'uddannelse']
df_studerende.drop(columns=remove_columns, inplace=True)

#drop irrelevante columns fra karakter data (fra feature selection):
remove_columns_exam = ['eksamen_navn', 'eksamen_kode', 'bedoemmelsesdato', 'resultat_id', 'loadtime', 'bestaaet']
df_eksamen.drop(columns=remove_columns_exam, inplace=True)

#%% MISSING VALUES TJEK I HVERT DATASÆT (INDEN SLETNING)
nan_count_studerende = df_studerende.isnull().sum()
print("Antal NaN værdier i df_studerende: ")
print(nan_count_studerende)
print("\n")

nan_count_eksamen = df_eksamen.isnull().sum()
print("Antal NaN værdier i df_eksamen: ")
print(nan_count_eksamen)
print("\n")

#%% SLETTER MISSING VALUES
#sletter de resterende missing values (sletter 425 rækker)
df_studerende.dropna(inplace=True)

#%% BEGYNDENDE SAMMENLÆGNING AF DATASÆT
#%% STEP 1: KARAKTERER

#opretter en pivot tabel for at tælle antallet af hver karakter for hver studerende
karakter_pivot = pd.pivot_table(df_eksamen, values='eksamenstype', index='studerende_id', columns='karakter', aggfunc='count', fill_value=0).reset_index()

#merge df_studerende og karakter_pivot
df = pd.merge(df_studerende, karakter_pivot, on='studerende_id', how='left')

#lister alle mulige karakterer
karakterer = ['IB', 'U', '-3', '00', '02', '4', '7', '10', '12', 'B']

#erstatter manglende værdier med 0
df[karakterer] = df[karakterer].fillna(0).astype(int)

#%% STEP 2: ORDINÆR- OG RE-EKSAMEN

#opretter en pivot tabel for at tælle antallet af ordinære og re-eksamener for hver studerende
eksamenstype_pivot = pd.pivot_table(df_eksamen, values='karakter', index='studerende_id', columns='eksamenstype', aggfunc='count', fill_value=0).reset_index()

#merge df_studerende og eksamenstype_pivot
df = pd.merge(df, eksamenstype_pivot, on='studerende_id', how='left')

#lister de forskellige typer
eksamentyper = ['o', 'r']

#erstatter NaN værdier med 0
df[eksamentyper] = df[eksamentyper].fillna(0)

df['o'] = df['o'].astype(int) #omkoder til int i stedet for float (nemmere for modellerne at håndtere)
df['r'] = df['r'].astype(int) #omkoder til int i stedet for float (nemmere for modellerne at håndtere)

#%% gruppering af ects
def grupper_ects(x):
    if 0.5 <= x <= 10:
        return '0-10'
    elif 11 <= x <= 20:
        return '11-20'
    elif 21 <= x <= 30:
        return '21-30'
    else:
        return 'other'

df_eksamen['ects_grouped'] = df_eksamen['eksamen_ects'].apply(grupper_ects)

#%% STEP 3: MERGER SKALA OG ECTS KOLONNER

#nedenstående merger de manglende kolonner og sætter missing values til at være 0

for col in ['skala_kode', 'ects_grouped']:
    pivot = pd.pivot_table(df_eksamen, values= 'karakter', index = 'studerende_id', columns=col, aggfunc='count', fill_value=0).reset_index()
    pivot.columns = [f"{col}_{x}" if x != 'studerende_id' else x for x in pivot.columns]
    df = pd.merge(df, pivot, on='studerende_id', how='left')

df.fillna(0, inplace=True)

#%% STEP 4: KONVERTERE VÆRDIER FRA FLOAT TIL INT

for col in df.columns:
    if "skala_kode_" in col or "ects_grouped_" in col:
        df[col] = df[col].astype(int)

#%% FEATURE ENGINEERING AND SELECTION
#%% LABEL ENCODING

df['gender'] = df['koen'].replace({'Mand': 0, 'Kvinde': 1})

df['priority'] = df['prioritet'].replace({'Lavere prioritet': 0, '1. prioritet': 1})

df['round_of_enrollment'] = df['optagsrunde'].replace({'1. runde': 0, '2. runde': 1})

df['type_of_bachelor'] = df['uddannelse_type'].replace({'Professionsbachelor': 0, 'Bachelor': 1})

#%%
df.drop(columns='koen', inplace=True)
df.drop(columns='prioritet', inplace=True)
df.drop(columns='optagsrunde', inplace=True)
df.drop(columns='uddannelse_type', inplace=True)

#%% KONVERTER FLOAT TIL INT
def convert_float_to_int(df, columns_to_int):
    for col in columns_to_int:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

columns_to_int = ['alder_ved_studiestart', 'prio_nr', 'adgangsgivende_eksamen_aar', 'gnms_a_bonus']

df = convert_float_to_int(df, columns_to_int)

#%%

df = df.rename(columns={'alder_ved_studiestart': 'age_when_enrolled', 'adgangsgivende_eksamen_aar':'year_for_qualifying_examination', 'gnms_a_bonus':'avg_after_bonus'})

#%% AGGREGERING AF KARAKTERER - KARAKTER_MEDIAN
def beregn_karakter_median(row):
    karakterer = [-3, 00, 2, 4, 7, 10, 12]
    frekvenser = [row['-3'], row['00'], row['02'], row['4'], row['7'], row['10'], row['12']]

    #generer en liste af karakterer baseret på deres frekvens
    karakter_liste = []
    for k, f in zip(karakterer, frekvenser):
        karakter_liste.extend([k] * f)

    #beregn medianen, hvis listen ikke er tom
    if karakter_liste:
        return np.median(karakter_liste)
    else:
        return np.nan #indikerer, at medianen ikke kunne beregnes

df['grade_median'] = df.apply(beregn_karakter_median, axis=1)

#%% Trin 1: gennemsnitlig karakter for hver studerende i hver periode

#konverter til numerisk hvor det er muligt
df_eksamen['karakter_numerisk'] = pd.to_numeric(df_eksamen['karakter'], errors='coerce')

#beregn gennemsnitlig karakter for hver tstuderende i hver period e
grouped_by_student_period = df_eksamen.groupby(['studerende_id', 'periode'])['karakter_numerisk'].mean().reset_index()

#omdøb kolonne til "gennemsnitlig karakter"
grouped_by_student_period.rename(columns={'karakter_numerisk' : 'gennemsnitlig_karakter'}, inplace = True)


#%% Trin 2: måle stigning eller fald i gennemsnit mellem perioder

correct_order_periods = ['E18', 'F19', 'E19', 'F20', 'E20', 'F21', 'E21', 'F22']

wide_format = grouped_by_student_period.pivot(index='studerende_id', columns='periode', values='gennemsnitlig_karakter').reset_index()

wide_format.fillna(0, inplace=True)

wide_format = wide_format[['studerende_id'] + correct_order_periods]

def calculate_total_diff(row):
    non_zero_values = [val for val in row if val != 0]
    if len(non_zero_values) < 2:
        return 0
    return non_zero_values[-1] - non_zero_values[0]

wide_format['grade_difference'] = wide_format[correct_order_periods].apply(calculate_total_diff, axis=1)

df = pd.merge(df, wide_format[['studerende_id', 'grade_difference']], on='studerende_id', how='left')

#df['karakter_diff'].fillna(0, inplace=True)

#%% FEATURE: ÅR MELLEM AFSLUTTET GYM TIL START UNI

#konverter float to int
df_studerende['adgangsgivende_eksamen_aar'] = df_studerende['adgangsgivende_eksamen_aar'].astype(int)

#eksktraher årstallet fra studiestart
df_studerende['studiestart_år'] = pd.to_datetime(df_studerende['studiestart']).dt.year

#beregn antallet af år mellem
df_studerende['sabbatical_years'] = df_studerende['studiestart_år'] - df_studerende['adgangsgivende_eksamen_aar']

#%%
df = df.merge(df_studerende[['studerende_id', 'sabbatical_years']], on='studerende_id', how='left')
df.drop(columns='studiestart', inplace=True) #overflødig

# %% Mapping af gymnasier til regioner

gym_mapping_dict = {
    'Nyborg Gymnasium': 'Region Syddanmark (Fyn)',
    'Tønder Handelsskole': 'Region Syddanmark',
    'Syddansk Erhvervsskole Odense-Vejle': 'Region Syddanmark (Fyn)',
    'Tietgenskolen': 'Region Syddanmark (Fyn)',
    'Rødkilde Gymnasium': 'Region Syddanmark',
    'EUC Lillebælt': 'Region Syddanmark',
    'Rosborg Gymnasium & HF': 'Region Syddanmark',
    'Esbjerg Gymnasium': 'Region Syddanmark',
    'Kolding Gymnasium, HF-Kursus og IB School': 'Region Syddanmark',
    'Mulernes Legatskole': 'Region Syddanmark (Fyn)',
    'Svendborg Erhvervsskole & - Gymnasier': 'Region Syddanmark (Fyn)',
    'Svendborg Erhvervsskole &  - Gymnasier': 'Region Syddanmark (Fyn)',
    'Skolerne i Oure - Sport & Performance': 'Region Syddanmark (Fyn)',
    'VUC Vest': 'Region Hovedstaden',
    'Kolding HF og VUC': 'Region Syddanmark',
    'Svendborg Gymnasium': 'Region Syddanmark (Fyn)',
    'Herningsholm Erhvervsskole og Gymnasier': 'Region Midtjylland',
    'Roskilde Handelsskole': 'Region Sjælland',
    'Tønder Gymnasium': 'Region Syddanmark',
    'Midtfyns Gymnasium': 'Region Syddanmark (Fyn)',
    'Odense Katedralskole': 'Region Syddanmark (Fyn)',
    'HF & VUC FYN': 'Region Syddanmark (Fyn)',
    'Tradium, Erhvervsskole og -gymnasier, Randers': 'Region Midtjylland',
    'Favrskov Gymnasium': 'Region Midtjylland',
    'Aalborg Katedralskole': 'Region Nordjylland',
    'Aalborghus Gymnasium': 'Region Nordjylland',
    'Vestfyns Gymnasium': 'Region Syddanmark (Fyn)',
    'TietgenSkolen (ELM)': 'Region Syddanmark (Fyn)',
    'Nordfyns Gymnasium': 'Region Syddanmark (Fyn)',
    'Rysensteen Gymnasium': 'Region Hovedstaden',
    'Vordingborg Gymnasium & HF': 'Region Sjælland',
    'Munkensdam Gymnasium': 'Region Syddanmark',
    'Fjerritslev Gymnasium': 'Region Nordjylland',
    'Himmelev Gymnasium': 'Region Sjælland',
    'Københavns VUC': 'Region Hovedstaden',
    'Ribe Katedralskole, HF/HHX/HTX/STX': 'Region Syddanmark',
    'Hf og VUC Roskilde-Køge': 'Region Sjælland',
    'Nykøbing Katedralskole': 'Region Sjælland',
    'Alssundgymnasiet Sønderborg': 'Region Syddanmark',
    'Borupgaard Gymnasium': 'Region Hovedstaden',
    'IBC International Business College': 'Region Syddanmark',
    'Færøisk studenter og HF': 'Udlandet',
    'Rødovre Gymnasium': 'Region Hovedstaden',
    'Haderslev Katedralskole': 'Region Syddanmark',
    'Grindsted Gymnasie- & Erhvervsskole, STX/HF': 'Region Syddanmark',
    'Tornbjerg Gymnasium': 'Region Syddanmark (Fyn)',
    'HANSENBERG' : 'Region Syddanmark',
    'HF & VUC Klar': 'Region Syddanmark',
    'Stenhus Gymnasium': 'Region Sjælland',
    'CELF - Center for erhv.rettede udd. Lolland-Falster': 'Region Sjælland',
    'Vejen Gymnasium og HF': 'Region Syddanmark',
    'Zealand Business College': 'Region Sjælland',
    'MSG-Haslev': 'Region Sjælland',
    'Rybners': 'Region Sjælland',
    'Middelfart Gymnasium & HF': 'Region Syddanmark (Fyn)',
    'Gefion Gymnasium': 'Region Hovedstaden',
    'Ikast-Brande Gymnasium': 'Region Midtjylland',
    'Aarhus Katedralskole': 'Region Midtjylland',
    'U/NORD': 'Region Hovedstaden',
    'Frederiksberg VUC & STX': 'Region Hovedstaden',
    'Midtsjællands Gymnasium': 'Region Sjælland',
    'Faaborg Gymnasium': 'Region Syddanmark (Fyn)',
    'Erhvervsskolen Nordsjælland': 'Region Sjælland',
    'Grindsted Gymnasie- & Erhvervsskole, HHX/HTX': 'Region Syddanmark',
    'Fredericia Gymnasium': 'Region Syddanmark',
    'Udenlandsk Skole': 'Udlandet',
    'TEC, Technical Education Copenhagen': 'Region Hovedstaden',
    'VUC Storstrøm': 'Region Sjælland',
    'Sorø Akademis Skole': 'Region Sjælland',
    'Køge Gymnasium': 'Region Sjælland',
    'Sønderborg Statsskole': 'Region Syddanmark',
    'Ribe Katedralskole, EUD/EUX': 'Region Syddanmark',
    'Virum Gymnasium': 'Region Sjælland',
    'Vestjysk Gymnasium Tarm': 'Region Midtjylland',
    'Frederikssund Gymnasium': 'Region Hovedstaden',
    'EUC Syd': 'Region Syddanmark',
    'VUC Lyngby': 'Region Hovedstaden',
    'ZBC Slagelse (Selandia)': 'Region Sjælland',
    'Birkerød Gymnasium HF IB & Kostskole': 'Region Hovedstaden',
    'Frederiksborg Gymnasium Og HF': 'Region Sjælland',
    'Espergærde Gymnasium og HF': 'Region Hovedstaden',
    'Struer Statsgymnasium': 'Region Midtjylland',
    'Campus Bornholm': 'Region Hovedstaden',
    'TH. LANGS HF & VUC': 'Region Midtjylland',
    'Roskilde Katedralskole': 'Region Sjælland',
    'Kalundborg Gymnasium og HF': 'Region Sjælland',
    'Bjerringbro Gymnasium': 'Region Midtjylland',
    'Nørre Gymnasium': 'Region Hovedstaden',
    'Sct. Knuds Gymnasium': 'Region Syddanmark (Fyn)',
    'Roskilde Tekniske Skole': 'Region Sjælland',
    'Niels Brock (Copenhagen Business College)': 'Region Hovedstaden',
    'HF & VUC FYN Odense': 'Region Syddanmark (Fyn)',
    'Christianshavns Gymnasium': 'Region Hovedstaden',
    'Roskilde Gymnasium': 'Region Sjælland',
    'Slagelse Gymnasium': 'Region Sjælland',
    'Herlufsholm Skole og Gods': 'Region Sjælland',
    'Egedal Gymnasium & HF': 'Region Hovedstaden',
    'Ordrup Gymnasium': 'Region Hovedstaden',
    'Nordvestsjællands HF & VUC': 'Region Sjælland',
    'Uddannelsescenter Holstebro': 'Region Midtjylland',
    'Vejlefjordskolen (gymnasium)': 'Region Syddanmark',
    'Sønderjyllands Gymnasium, Grundskole og Kostskole': 'Region Syddanmark',
    'VUC Syd - Tønder afdeling': 'Region Syddanmark',
    'Helsingør Gymnasium': 'Region Hovedstaden',
    'Frederiksværk Gymnasium og HF': 'Region Hovedstaden',
    'Næstved Gymnasium og HF': 'Region Sjælland',
    'AARHUS TECH': 'Region Midtjylland',
    'Prins Henriks Skole, Lycee Francais De Copenhague': 'Region Hovedstaden',
    'VUC Syd': 'Region Syddanmark',
    'Risskov gymnasium': 'Region Midtjylland',
    'Handelsgymnasiet Vestfyn': 'Region Syddanmark (Fyn)',
    'Greve Gymnasium': 'Region Sjælland',
    'Thisted Gymnasium, STX og HF': 'Region Nordjylland',
    'Aabenraa Statsskole': 'Region Syddanmark',
    'Frederiksberg Gymnasium': 'Region Hovedstaden',
    'Køge Handelsskole': 'Region Sjælland',
    'Silkeborg Gymnasium': 'Region Midtjylland',
    'Business College Syd': 'Region Syddanmark',
    'AARHUS GYMNASIUM, Tilst': 'Region Midtjylland',
    'Vejen Business College': 'Region Syddanmark',
    'Campus Vejle': 'Region Syddanmark',
    'Århus Akademi': 'Region Midtjylland',
    'Nærum Gymnasium': 'Region Hovedstaden',
    'Nordsjællands Grundskole og Gymnasium samt HF': 'Region Hovedstaden',
    'Gladsaxe Gymnasium': 'Region Hovedstaden',
    'HF-Centret Efterslægten': 'Region Hovedstaden',
    'Bagsværd Kostskole Og Gymnasium': 'Region Hovedstaden',
    'Duborg-Skolen': 'Udlandet',
    'Rungsted Gymnasium': 'Region Hovedstaden',
    'Kold college': 'Region Syddanmark (Fyn)',
    'NEXT - Sydkysten Gymnasium': 'Region Hovedstaden',
    'VUC Holstebro-Lemvig-Struer': 'Region Midtjylland',
    'Toftegård Statskursus': 'Region Hovedstaden',
    'Skanderborg Gymnasium': 'Region Midtjylland',
    'NEXT UDDANNELSE KØBENHAVN': 'Region Hovedstaden',
    'Gentofte HF': 'Region Hovedstaden',
    'Skanderborg-Odder Center for uddannelse': 'Region Midtjylland',
    'Varde Gymnasium': 'Region Midtjylland',
    'Allerød Gymnasium': 'Region Hovedstaden',
    'Herning Gymnasium': 'Region Midtjylland',
    'Sankt Annæ Gymnasium': 'Region Hovedstaden',
    'Varde Handelsskole og Handelsgymnasium': 'Region Midtjylland',
    'Skive Gymnasium': 'Region Midtjylland',
    'Gribskov Gymnasium': 'Region Hovedstaden',
    'Odense Tekniske Gymnasium': 'Region Syddanmark (Fyn)',
    'Aarhus HF & VUC': 'Region Syddanmark',
    'Horsens Gymnasium & HF, Højen 1': 'Region Midtjylland',
    'Frederikshavn Gymnasium': 'Region Nordjylland',
    'Viborg Gymnasium': 'Region Midtjylland',
    'A. P. Møllerskolen': 'Region Syddanmark',
    'Nordvestsjællands Erhvervs- og Gymnasieuddannelser': 'Region Hovedstaden',
    'Høje-Taastrup Gymnasium': 'Region Hovedstaden',
    'Århus Statsgymnasium': 'Region Midtjylland',
    'Det frie Gymnasium': 'Region Hovedstaden',
    'Tårnby Gymnasium': 'Region Hovedstaden',
    'Aurehøj gymnasium': 'Region Hovedstaden',
    'Holstebro Gymnasium og HF': 'Region Midtjylland',
    'Århus Købmandsskole, Handelsgymnasiet': 'Region Midtjylland',
    'Brønderslev Gymnasium og HF': 'Region Nordjylland',
    'Ørestad Gymnasium': 'Region Hovedstaden',
    'Kujataani, Ilinniarnertuunngorniarfik, Grønland': 'Udlandet',
    'Marselisborg Gymnasium': 'Region Midtjylland',
    'Viborg Katedralskole': 'Region Midtjylland',
    'College360 - Bredhøjvej 8': 'Region Midtjylland',
    'Paderup gymnasium': 'Region Midtjylland',
    'NEXT Uddannelse København': 'Region Hovedstaden',
    'Horsens Gymnasium & HF, Studentervænget 2': 'Region Midtjylland',
    'N. Zahles Gymnasieskole': 'Region Hovedstaden',
    'Tørring Gymnasium': 'Region Midtjylland',
    'Frederiksberg HF-Kursus': 'Region Hovedstaden',
    'Sankt Petri skole - Gymnasium': 'Region Hovedstaden',
    'Øregård Gymnasium': 'Region Hovedstaden',
    'Maribo Gymnasium': 'Region Sjælland',
    'Mercantec': 'Region Midtjylland',
    'Niels Steensens Gymnasium': 'Region Hovedstaden',
    'Ribe Katedralskole': 'Region Syddanmark',
    'Haderslev Handelsskole': 'Region Syddanmark',
    'Syddjurs Gymnasium': 'Region Midtjylland',
    'College360': 'Region Midtjylland',
    'HF & VUC Nordsjælland': 'Region Hovedstaden',
    'EUC Sjælland': 'Region Sjælland',
    'Gammel Hellerup Gymnasium': 'Region Hovedstaden',
    'Høng Gymnasium og HF': 'Region Sjælland',
    'Gentofte Gymnasium': 'Region Hovedstaden',
    'Lemvig Gymnasium': 'Region Midtjylland',
    'NEXT Uddannelse København, Ishøj': 'Region Hovedstaden',
    'Herlev Gymnasium og HF': 'Region Hovedstaden',
    'Støvring Gymnasium': 'Region Nordjylland',
    'Københavns åbne Gymnasium': 'Region Hovedstaden',
    'Dronninglund Gymnasium': 'Region Nordjylland',
    'EUC Syd, Christen Kolds Vej': 'Region Syddanmark',
    'Grenaa Gymnasium': 'Region Midtjylland',
    'Erhvervsakademiet København Nord': 'Region Hovedstaden',
    'Learnmark Horsens': 'Region Midtjylland',
    'VUC Vejle': 'Region Syddanmark',
    'Solrød Gymnasium': 'Region Hovedstaden',
    'TECHCOLLEGE': 'Region Nordjylland',
    'Nørrebro Gymnasium': 'Region Hovedstaden',
    'Brøndby Gymnasium': 'Region Hovedstaden',
    'Marie Kruses Skole': 'Region Hovedstaden',
    'UCRS': 'Region Midtjylland',
    'Deutsches Gymnasium Für Nordschleswig': 'Region Syddanmark',
    'Aarhus Business College': 'Region Midtjylland',
    'Teknisk Skole i Torshavn': 'Udlandet',
    'Hillerødgades Skole': 'Region Hovedstaden',
    'Odder Gymnasium': 'Region Midtjylland',
    'Viby Gymnasium': 'Region Sjælland',
    'Hvidovre Gymnasium & HF': 'Region Hovedstaden',
    'Randers HF & VUC': 'Region Midtjylland',
    'Nakskov Gymnasium og HF': 'Region Sjælland',
    'Ringkjøbing Gymnasium': 'Region Midtjylland',
    'Hellerup Skole': 'Region Hovedstaden',
    'EUC Nordvest': 'Region Sjælland',
    'Det Kristne Gymnasium': 'Region Midtjylland',
    'Egå Gymnasium': 'Region Midtjylland',
    'Randers Statsskole': 'Region Midtjylland',
    'Herning HF og VUC': 'Region Midtjylland',
    'HF & VUC NORD': 'Region Nordjylland',
    'Odsherred Gymnasium': 'Region Sjælland',
    'Rybners- STX- Grådybet': 'Region Midtjylland',
    'Frederiksberg Studenterkursus': 'Region Hovedstaden',
    'HF & VUC København Syd': 'Region Hovedstaden',
    'Viden Djurs': 'Region Midtjylland',
    'Københavns Private Gymnasium': 'Region Hovedstaden',
    'Horsens HF & VUC': 'Region Midtjylland',
    'Falkonergårdens Gymnasium og HF-Kursus': 'Region Hovedstaden',
    'Thy-Mors HF & VUC': 'Region Nordjylland',
    'Ingrid Jespersens Gymnasieskole': 'Region Hovedstaden',
    'Himmerlands Erhvervs- og Gymnasieuddannelser' : 'Region Nordjylland',
    'Rybners - HHX/STX - Grådybet': 'Region Syddanmark',
    'Mariagerfjord Gymnasium': 'Region Nordjylland',
    'Vesthimmerlands Gymnasium og HF': 'Region Nordjylland',
    'HF & VUC Vest, Esbjerg & Omegn': 'Region Syddanmark',
    'Nørresundby Gymnasium og HF': 'Region Nordjylland',
    'Johannesskolen': 'Region Hovedstaden',
    'Odense Tekniske Skole': 'Region Syddanmark (Fyn)',
    'Veggerby Friskole': 'Region Nordjylland',
    'Midtgrøndlands Gymnasiale Skole': 'Udlandet',
    'Professionshøjskolen VIA University College': 'Region Midtjylland',
    'Glostrup Kommune': 'Region Hovedstaden',
    'HF & VUC FYN Svendborg': 'Region Syddanmark (Fyn)',
    'NEXT Uddannelse København erhvervsuddannelser (EUD/EUX), Ishøj': 'Region Hovedstaden',
    'Hjørring Gymnasium/STX og HF': 'Region Nordjylland',
    'Skive College, Arvikavej': 'Region Midtjylland',
    'Hasseris Gymnasium': 'Region Nordjylland',
    'Metropolitanskolen': 'Region Hovedstaden',
    'Taastrup City Gymnasium': 'Region Hovedstaden',
    'Midtgrønlands gymnasiale kursus': 'Udlandet',
    'Rybners - EUD - Spangsbjerg Møllevej': 'Region Syddanmark',
    'Vestegnen HF & VUC': 'Region Hovedstaden',
    'FYNs HF + STX': 'Region Syddanmark (Fyn)',
    'NEXT - Albertslund Gymnasium': 'Region Hovedstaden',
    'Syddansk Universitet, Sønderborg': 'Region Syddanmark',
    'EUC Nord': 'Region Nordjylland',
    'Vestre Borgerdyd Kursuscenter Og Gymnasium': 'Region Hovedstaden',
    'EUC Syd, Syd Plantagevej': 'Region Syddanmark',
    'Føroya Handelsskuli': 'Udlandet',
    'Aalborg City Gymnasium': 'Region Nordjylland',
    'Dalum Landbrugsskole': 'Region Syddanmark (Fyn)',
    'Hotel- Og Restaurantskolen': 'Region Hovedstaden',
    'Østre Borgerdyd Gymnasium': 'Region Hovedstaden',
    'CEUS': 'Region Sjælland',
    'Herning HF': 'Region Syddanmark',
    'Færøsk Tekniske Gymnasium': 'Udlandet',
    'Vejle Tekniske Skole': 'Region Syddanmark',
    'Skive College, Kongsvingervej': 'Region Midtjylland',
    'Morsø Gymnasium': 'Region Nordjylland',
    'Svendborg Erhvervsskole & -Gymnasier, Skovsbovej': 'Region Syddanmark (Fyn)',
    'Tietgenskolen Handelshøjsk.Afd. I Odense (Kbh)': 'Region Hovedstaden',
    'Skive College': 'Region Midtjylland',
    'Sydgrøndlands Gymnasiale Skole': 'Udlandet',
    'College360 - Bindslev Plads 1': 'Region Midtjylland',
    'Aalborg Handelsskole, Hovedafdeling': 'Region Nordjylland',
    'EUC MIDT': 'Region Midtjylland',
    'Rønde Gymnasium': 'Region Hovedstaden',
    'VUC Syd - Sønderborg afdeling': 'Region Syddanmark',
    'ZBC Vordingborg': 'Region Sjælland',
    'Syddansk Universitet': 'Region Syddanmark (Fyn)',
    'Midnamsskulin I Suderoy': 'Udlandet',
    'Kolding Købmandsskole': 'Region Syddanmark',
    'Syddansk Erhvervsskole Odense-Vejle, Munkebjergvej 130': 'Region Syddanmark (Fyn)',
    'SOSU H': 'Region Hovedstaden',
    'Horsens Gymnasium & HF': 'Region Midtjylland',
    'Lemvig Gymnasium, STX og HHX': 'Region Midtjylland',
    'Føroya Studentaskuli Og HF-Skeid': 'Udlandet',
    'Stenløse Gymnasium og HF': 'Region Sjælland',
    'Erhvervsskolerne, Østre Boulevard 10, 9600 Års': 'Region Nordjylland',
    'EUC Nordvestsjælland': 'Region Hovedstaden',
    'U/NORD Hillerød Handelsskole': 'Region Hovedstaden',
    'Frederikshavn Handelsskole': 'Region Nordjylland',
    'UU Ikast-Brande': 'Region Midtjylland',
    'Lemvig Gymnasium , EUX og EUD': 'Region Midtjylland',
    'Skive-Viborg HF & VUC': 'Region Midtjylland',
    'Frederiksberg VUC': 'Region Hovedstaden',
    'NEXT - Sukkertoppen Gymnasium': 'Region Hovedstaden',
    'Campus Bornholm HF, HHX, HTX, STX': 'Region Hovedstaden',
    'Aarhus Private Gymnasium': 'Region Midtjylland',
    'Stige Friskole': 'Region Syddanmark (Fyn)',
    'Copenhagen International School': 'Region Hovedstaden',
    'NEXT- Baltorp Business Gymnasium': 'Region Hovedstaden',
    'Vejle Tekniske Gymnasium': 'Region Syddanmark',
    'Den Jyske Håndværkerskole': 'Region Midtjylland',
    'VIA University College, Nørre Nissum': 'Region Midtjylland',
    'VUC Syd - Aabenraa afdeling': 'Region Syddanmark',
    'Anden institution': 'Region Syddanmark',
    'Niels Brock Copenhagen Business College': 'Region Hovedstaden',
    'EUC-Syd': 'Region Syddanmark',
    'Bornholms Erhvervsskole': 'Region Hovedstaden',
    'H.C. Ørsted Gymnasiet, Lyngby': 'Region Hovedstaden',
    'Albertslund Amtsgymnasium': 'Region Hovedstaden',
    'Herningsholm Erhvervsgymnasium, HTX Herning': 'Region Syddanmark',
    'Svendborg Erhvervsskole & - Gymnasier, Porthusvej': 'Region Syddanmark (Fyn)',
    'Syddansk Erhvervsskole Odense-Vejle, Risingsvej': 'Region Syddanmark (Fyn)',
    'Frederikssund Handelsgymnasium og Teknisk Gymnasium': 'Region Hovedstaden',
    'Slotshaven Gymnasium': 'Region Sjælland',}

## definerer mapping funktion
def map_gymnasium_to_region(df, mapping_dict):
    df['region'] = df['institutionsnavn'].map(mapping_dict)
    return df

# Anvender "def_gymnasium_to_region" på dataframe
df = map_gymnasium_to_region(df, gym_mapping_dict)

df_mapping_check = df[['institutionsnavn', 'region']]

#%%
df.drop(columns='institutionsnavn', inplace=True) #overflødig

#%%

region_mapping = {
    'Region Syddanmark (Fyn)': 0,
    'Region Syddanmark': 1,
    'Region Midtjylland': 2,
    'Region Sjælland': 3,
    'Region Hovedstaden': 4,
    'Region Nordjylland': 5,
    'Udlandet': 6
}

df['region_encoded'] = df['region'].map(region_mapping)
df.drop(columns='region', inplace=True) #overflødig

#%%
df = df.rename(columns={'optagsmetode': 'enrollment_method'})
df = df.rename(columns={'adgangsgivende_eksamen': 'qualifying_diploma'})
#%%

def one_hot_encode (df, columns_to_encode):
    df_encoded = pd.get_dummies(df, columns= columns_to_encode, drop_first=True)
    return df_encoded
columns_to_encode = ['nationalitet_grp', 'campus', 'fakultet', 'enrollment_method', 'optag_selektion', 'qualifying_diploma']

df = one_hot_encode(df, columns_to_encode)

#%%
bool_cols = [col for col in df.columns if df[col].dtype == 'bool']
df[bool_cols] = df[bool_cols].astype(int)
#%% konverterer floats to int

def convert_float_to_int(df, columns_to_int):
    for col in columns_to_int:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

columns_to_int = ['age_when_enrolled', 'prio_nr', 'year_for_qualifying_examination']

df = convert_float_to_int(df, columns_to_int)

#%% KNN til at finde median til de studerende der ikke har numerisk data

from sklearn.impute import KNNImputer

#opret en KNNimputer-instans
df_subset = df[['grade_median', 'gender', 'age_when_enrolled','type_of_bachelor','round_of_enrollment','priority', 'grade_difference', 'avg_after_bonus', 'frafald_event', 'prio_nr', 'year_for_qualifying_examination','region_encoded', '-3', '00', '02', '4',
                '7', '10', '12', 'B', 'IB', 'U', 'o', 'r', 'skala_kode_7-trinsskala', 'skala_kode_Bestået','sabbatical_years']]

#isoler kolonnen med manglende værdier samt nogle andre relevante kolonner (numeriske)
imputer = KNNImputer(n_neighbors=5)

#udfør KNN imputering
imputed_data = imputer.fit_transform(df_subset)

#erstat den gamle karakter_median kolonne med den nye
df['grade_median'] = imputed_data[:, 0]

#%% KNN til at finde imputer for karakter_diff

#opret en KNNimputer-instans
df_subset = df[['grade_difference', 'gender', 'age_when_enrolled','type_of_bachelor','round_of_enrollment','priority', 'grade_median', 'avg_after_bonus', 'frafald_event', 'prio_nr', 'year_for_qualifying_examination','region_encoded', '-3', '00', '02', '4',
                '7', '10', '12', 'B', 'IB', 'U', 'o', 'r', 'skala_kode_7-trinsskala', 'skala_kode_Bestået','sabbatical_years']]

#isoler kolonnen med manglende værdier samt nogle andre relevante kolonner (numeriske)
imputer = KNNImputer(n_neighbors=5)

#udfør KNN imputering
imputed_data = imputer.fit_transform(df_subset)

#erstat den gamle karakter_median kolonne med den nye
df['grade_difference'] = imputed_data[:, 0]

#%%

df = df.rename(columns={'frafald_event': 'dropout_event', '-3': 'grade_-3', '00' : 'grade_00', '02': 'grade_02', '4':'grade_4', '7': 'grade_7', '10': 'grade_10', '12': 'grade_12'})
df = df.rename(columns={'B': 'passed_exams_count', 'IB': 'failed_exams_count', 'U': 'absent_exams_count', 'o': 'ordinary_exams_count', 'r': 'reexams_count'})
df = df.rename(columns={'skala_kode_7-trinsskala': 'numeric_grade_scale', 'skala_kode_Bestået': 'nunnumeric_grade_scale', 'nationalitet_grp_Norden (EU/EØS)': 'nationality_nordic', 'nationalitet_grp_Øvrige Europa (EU/EØS)': 'nationality_europe', 'nationalitet_grp_Øvrige udland (ej EU/EØS)': 'nationality_not_eu_eøs'})
df = df.rename(columns={'fakultet_Samfundsvidenskab': 'faculty_social_sciences', 'fakultet_Sundhedsvidenskab': 'faculty_health_sciences', 'fakultet_Teknik': 'faculty_engineering'})
df = df.rename(columns={'enrollment_method_Kvote 1': 'enrollment_method_quota1', 'enrollment_method_Kvote 2': 'enrollment_method_quota2', 'optag_selektion_Kvote 1 - Afvist kvote 2': 'enrollment_selection_q1_rejected_q1', 'optag_selektion_Kvote 1 - Indst. kvote 2': 'enrollment_selection_q1q2', 'optag_selektion_Kvote 1 - Kun kvote 1': 'enrollment_selection_quota1', 'optag_selektion_Kvote 2 - Kun kvote 2': 'enrollment_selection_quota2'})
df = df.rename(columns={'fakultet_Naturvidenskab' : 'faculty_science'})

#%% Nu skal de studerende anonymiseres og bruges som indeks

import hashlib

def hash_id(id):
    return hashlib.sha256(str(id).encode()).hexdigest()

df['studerende_id'] = df['studerende_id'].apply(hash_id)

#%% indekserer dem

df.set_index('studerende_id', inplace=True)

#%% Test for multikollinaritet

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_no_target = df.drop('dropout_event', axis=1)

#tilføjer konstant
X = add_constant(X_no_target)

#beregn VIF for hver hver variabel
vif = pd.DataFrame()
vif["Variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

#%% korrelationsmatrix

correlation_matrix = df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix before VIF")
plt.show()

#%% sletter på baggrund af vif værdier og korrelationsplot

df.drop(columns='year_for_qualifying_examination', inplace=True)
df.drop(columns='enrollment_method_quota1', inplace=True)
df.drop(columns='enrollment_method_quota2', inplace=True)
#df.drop(columns='enrollment_selection_q1_rejected_q1', inplace=True)
#df.drop(columns='enrollment_selection_q1q2', inplace=True)

#%%

X_no_target = df.drop('dropout_event', axis=1)

#tilføjer konstant
X = add_constant(X_no_target)

#beregn VIF for hver hver variabel
vif = pd.DataFrame()
vif["Variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

#%%
correlation_matrix = df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix")
plt.show()

#%% saving the dataframe
df.to_csv("df_pre_model_final_sletningaftrevariable.csv")


#%% last check for missing values

nan_count_df = df.isnull().sum()
print("Antal NaN værdier i df: ")
print(nan_count_df)
print("\n")

#%% Outliers

plt.figure(figsize=(20, 40))
for i, column in enumerate(df.columns, 1):
    plt.subplot(15, 4, i)
    sns.boxplot(y=df[column])
    plt.ylabel(column)

plt.tight_layout()
plt.show()

#%% Feature selection før modelbygning
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

y = df['frafald_event'].reset_index(drop=True)
X = df.drop('frafald_event', axis=1).reset_index(drop=True)

# Initialiser Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Placeholder for feature importance
feature_importance_values = np.zeros(X.shape[1])

# Iterate over hver fold
for train_index, val_index in skf.split(X, y):

    # Split data
    X_train_fold, X_val_fold = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_train_fold, y_val_fold = y.iloc[train_index].copy(), y.iloc[val_index].copy()

    # sikrer at længder matcher
    assert len(X_train_fold) == len(y_train_fold), "Training data mismatch!"
    assert len(X_val_fold) == len(y_val_fold), "Validation data mismatch!"

    # anvender SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)

    # træner Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_smote, y_train_smote)

    # måler base accuracy
    base_accuracy = accuracy_score(y_val_fold, clf.predict(X_val_fold))

    # udregner Mean Decrease Accuracy for hver feature
    for i in range(X.shape[1]):
        X_val_fold_copy = X_val_fold.copy()
        np.random.shuffle(X_val_fold_copy.iloc[:, i].values)  # Shuffle individual feature
        shuff_acc = accuracy_score(y_val_fold, clf.predict(X_val_fold_copy))
        feature_importance_values[i] += base_accuracy - shuff_acc

# genmmensnitlig importances over alle folds
feature_importance_values /= 5

# laver DataFrame til at vise resultater
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance_values
}).sort_values(by='importance', ascending=False)

print(feature_importance_df)

#%%

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='green')
plt.xlabel('Variable Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature importance', fontsize=16)
plt.show()

