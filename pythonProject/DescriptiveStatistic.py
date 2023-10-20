import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns


df_studerende = pd.read_excel("Frafald_grunddata_20230921_engelsk.xlsx")
df_eksamen = pd.read_excel("Frafald_grunddata_karakter_20230921_engelsk.xlsx")

#%%
xx = (df_studerende['dropoutEvent'] == 1).sum()
print(xx)
# facultet, uddannelsestype, alder, gnms_a_bonus, køn
#%% FØR TRANSFORMATION
#%% Plot som viser fordelingen af frafald_event
import seaborn as sns
import matplotlib.pyplot as plt

# Beregner antal af hver kategori
counts = df_studerende['dropoutEvent'].value_counts()

# Laver barplot
sns.barplot(x=counts.index, y=counts.values, palette="viridis")

plt.ylabel('Count')
plt.xlabel('dropoutEvent')
plt.xticks([0, 1], [0, 1])  # Sørger for kun at vise 0 og 1 på x-aksen
plt.show()

#%% Plot som viser fordelingen af studerende på fakultet
# Beregner antal af hver kategori
counts = df_studerende['faculty'].value_counts()

# Laver barplot
plt.figure(figsize=(12, 6))
sns.barplot(x=counts.index, y=counts.values, palette="viridis")


plt.ylabel('Count')
plt.xlabel('Faculty')
plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
plt.subplots_adjust(bottom=0.2)

plt.show()

#%% krydstabel periode og frafald
def format_date_to_fyear(date):
    year = date.year
    return f"f{year % 100}"


import pandas as pd
import matplotlib.pyplot as plt

# Opretter en krydstabel
ct = pd.crosstab(df_studerende['startDate'], df_studerende['dropoutEvent'])

# Henter farverne fra viridis paletten
colors = sns.color_palette('viridis', n_colors=2)

# Konverterer x-akse værdier til det ønskede format
formatted_index = [format_date_to_fyear(date) for date in ct.index]

# Plotter stablede barplots med frafald studerende i bunden
ct[[1, 0]].plot(kind='bar', stacked=True, figsize=(10, 5), color=colors)
plt.title('Crosstable of enrollment period and dropout event')

# Ændrer x-ticks baseret på det nye format
plt.xticks(ticks=range(len(formatted_index)), labels=formatted_index, rotation=0)

# Tilføjer antallet af frafald pr. startDate
for idx, value in enumerate(ct[1]):
    plt.text(idx, value/2, str(value), ha='center', va='center', fontweight='bold', color='white')

plt.xlabel("Period")

plt.show()

#%% krydstabel fakultet og frafald

# Opretter en krydstabel
ct = pd.crosstab(df_studerende['faculty'], df_studerende['dropoutEvent'])

# Sorterer fakulteterne baseret på antallet af studerende, der er droppet ud
ct = ct.sort_values(by=1, ascending=False)

plt.figure(figsize=(12, 6))

# Plotter stablede barplots med frafald studerende i bunden
ct[[1, 0]].plot(kind='bar', stacked=True, figsize=(10, 5), color=colors)
plt.title('Crosstable of faculty and dropout event')

# Mapping af fakulteter til deres engelske navne
faculty_mapping = {
    "Humanoira": "Humanities",
    "Samfundsvidenskab": "Social Science",
    "Teknik": "Engineering",
    "Naturvidenskab": "Natural Sciences",
    "Sundhedsvidenskab": "Health Sciences"
}
formatted_ticks = [faculty_mapping.get(tick, tick) for tick in ct.index]

# Anvender de formatterede ticks til x-aksen
plt.xticks(ticks=range(len(formatted_ticks)), labels=formatted_ticks, rotation=0)


# Tilføjer antallet af frafald pr. startDate
for idx, value in enumerate(ct[1]):
    plt.text(idx, value/2, str(value), ha='center', va='center', fontweight='bold', color='white')

plt.xlabel("Faculty", labelpad=20)
plt.subplots_adjust(bottom=0.2)

plt.show()

#%% Enrollment age droput
# 1. Opdel dataen i intervaller/bins
bins = [18, 20, 25, 30, 35, 40, 50, 60, 100]
labels = ["18-19", "20-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"]
df_studerende['age_group'] = pd.cut(df_studerende['enrollmentAge'], bins=bins, labels=labels, right=False)

# 2. Lav en krydstabel
ct_age = pd.crosstab(df_studerende['age_group'], df_studerende['dropoutEvent'])

# 3. Plot dataen
plt.figure(figsize=(12, 6))
ct_age[[1, 0]].plot(kind='bar', stacked=True, figsize=(10, 5), color=colors)
plt.title('Crosstable of enrollment age and dropout event')

# Tilføjer antallet af frafald pr. aldersgruppe
for idx, value in enumerate(ct_age[1]):
    plt.text(idx, value/2, str(value), ha='center', va='center', fontweight='bold', color='white')

plt.xlabel("Enrollment Age", labelpad=20)
plt.subplots_adjust(bottom=0.2)

plt.show()

#%%
# Beregn procentdel af studerende, der er droppet ud i hver aldersgruppe
ct_age['dropout_pct'] = ct_age[1] / (ct_age[0] + ct_age[1]) * 100
ct_age['total_students'] = ct_age[0] + ct_age[1]

plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x=ct_age.index, y=ct_age['dropout_pct'], palette=colors)

# Tilføj detaljer til plottet
plt.title('Dropout Percentage by Enrollment Age')
plt.ylabel('Dropout Percentage (%)')
plt.xlabel('Enrollment Age', labelpad=20)
plt.subplots_adjust(bottom=0.2)
plt.grid(axis='y')

# Tilføj tekst til hver søjle med det samlede antal studerende
for idx, patch in enumerate(bar_plot.patches):
    height = patch.get_height()
    bar_plot.text(patch.get_x() + patch.get_width() / 2., height + 1,
                 f'n={ct_age["total_students"].iloc[idx]}',
                 ha="center")

plt.show()


#%% GEnnemsnit efter bonus
import numpy as np

# Opretter bins
bins = np.arange(1, 14, 1)  # Bins fra -3 til 14 med intervaller på 1

# Grupperer data baseret på 'avgAfterBonus' bins og 'dropoutEvent'
grouped = df_studerende.groupby([pd.cut(df_studerende['avgAfterBonus'], bins), 'dropoutEvent']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))

# Plotter stablede barplots
grouped[[1, 0]].plot(kind='bar', stacked=True, figsize=(10, 5), color=colors)

plt.title('Crosstable of average gymnasium grade and dropout event')
plt.ylabel('Number of Students')
plt.xlabel('Gymnasium Average (After Bonus)')
plt.xticks(rotation=0)
plt.show()

#%% Køn
# Beregner antal af hver kategori
counts = df_studerende['dropoutEvent'].value_counts()

# Laver barplot
sns.countplot(x='sex', hue='dropoutEvent', data=df_studerende, palette="viridis")

plt.ylabel('Count')
plt.xlabel('Sex')
#plt.xticks([0, 1], [0, 1])  # Sørger for kun at vise 0 og 1 på x-aksen
plt.legend(title='dropoutEvent')
plt.show()
#%%
# Forbered data
total_counts = df_studerende['faculty'].value_counts()
dropout_counts = df_studerende[df_studerende['dropoutEvent'] == 1]['faculty'].value_counts()

# Sørger for, at rækkefølgen af fakulteter er den samme for begge tællinger
dropout_counts = dropout_counts.reindex(total_counts.index)

# Lav plot
plt.figure(figsize=(12, 6))

# Plotter alle studerende
sns.barplot(x=total_counts.index, y=total_counts.values, palette="viridis", label="Total Students")

# Plotter studerende, der er droppet ud
sns.barplot(x=dropout_counts.index, y=dropout_counts.values, palette="rocket", label="Dropouts")

plt.ylabel('Count')
plt.xlabel('Faculty')
plt.setp(plt.gca().get_xticklabels(), rotation=0)
plt.subplots_adjust(bottom=0.2)
plt.legend()

plt.show()

#%% FJERNER KOLONNER
#dropout related columnes
remove_dropout_related_columns = ['frafald_beslutter', 'frafald_begrundelse', 'udmeldbegrundelse', 'studie_status', 'studieslut', 'studie_event']
df_studerende.drop(columns=remove_dropout_related_columns, inplace=True)

#andre irrelevante columns fra grunddata:
remove_columns = ['loadtime', 'nationalitet', 'studieretning', 'institutionslandekode', 'institutionskode', 'dimissionsalder', 'institutionsland', 'uddannelse']
df_studerende.drop(columns=remove_columns, inplace=True)

#drop irrelevante columns fra karakter data:
remove_columns_exam = ['eksamen_navn', 'eksamen_kode', 'bedoemmelsesdato', 'resultat_id', 'loadtime', 'bestaaet']
df_eksamen.drop(columns=remove_columns_exam, inplace=True)

df_studerende['enrollmentAge'] = df_studerende['enrollmentAge'].astype(int)
# %% Beskriver grundlæggende information

## function til at beskrive dataframes
def describe_column(df):
    # Laver dataframe til at holde databeskrivelsen
    description = pd.DataFrame(
        columns=['Column', 'Datatype', 'N unique values', 'N missing values', 'Average', 'Median'])

    for col in df.columns:
        ## Beregn statistikkerne
        col_name = col
        datatype = df[col].dtype
        unique_values = df[col].nunique()
        missing_values = df[col].isnull().sum()
        mean = df[col].mean() if datatype in ['int64', 'float64'] else None
        median = df[col].median() if datatype in ['int64', 'float64'] else None

        ## Tilføj resultater til den beskrivende dataframe
        description = description._append({
            'Column': col_name,
            'DataType': datatype,
            'N unique values': unique_values,
            'N missing values': missing_values,
            'Average': mean,
            'Median': median
        }, ignore_index=True)

    ## Retuner den beskrivende dataframe
    return description


df_studerende_beskrivelse = describe_column(df_studerende)
df_eksamen_beskrivelse = describe_column(df_eksamen)

print("studerende:", df_studerende_beskrivelse)
print("studerende:", df_eksamen_beskrivelse)

# %% Gem de nye beskrivende DataFrames
df_studerende_beskrivelse.to_csv('beskrivelse_studerende1.csv', index=False, encoding='utf-8')
df_eksamen_beskrivelse.to_csv('beskrivelse_eksamen1.csv', index=False, encoding='utf-8')

# %% Lav flotte tabeller med det nye data
print(tabulate(df_eksamen_beskrivelse, headers='key',tablefmt='latex'))
print(tabulate(df_studerende_beskrivelse, headers='key',tablefmt='latex'))

#%% Visualisering af rå data

#%%
# Definer en dictionary med de gamle navne som nøgler og de nye navne som værdier
name_mapping = {
    'enrollmentAge': 'age at enrollment',
    'sex': 'sex',
    'avgAfterBonus': 'average after bonus',
    'qualifyingExamYear': 'year of qualifying exam',
    'faculty': 'faculty',
    'startDate': 'startDate'
}

# Din liste med variable
variables = ['enrollmentAge', 'sex', 'avgAfterBonus', 'qualifyingExamYear', 'faculty', 'startDate']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, var in enumerate(variables):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    # sæt forskellige bin størrelser
    bin_size = 20
    if var == 'sex':
        bin_size = 3

    ax.hist(df_studerende[df_studerende['dropoutEvent'] == 0][var], bins=bin_size, color='blue', alpha=0.5,
            label='Dropped out')
    ax.hist(df_studerende[df_studerende['dropoutEvent'] == 1][var], bins=bin_size, color='red', alpha=0.5,
            label='Not dropped out')

    # Brug de nye navne
    new_var = name_mapping[var]
    ax.set_title(f'Distribution of {new_var} by dropout event')
    ax.set_xlabel(new_var)
    ax.set_ylabel('Number of students')
    ax.legend(loc='upper right')

    # Roter x-aksetiketterne for at undgå overlap
    if var in ['faculty','startDate']:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    # Hvis variablen er 'startDate', indstil x-aksen til specifikke datoer
    if var == 'startDate':
        dates = ['2018-09', '2019-09', '2020-09', '2021-09']
        ax.set_xticks(dates)
        ax.set_xticklabels(dates)

# Justerer pladsen mellem plots for bedre læsbarhed
plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# mapping af variabelnavne
name_mapping = {
    'enrollmentAge': 'age at enrollment',
    'sex': 'sex',
    'avgAfterBonus': 'average after bonus',
    'qualifyingExamYear': 'year of qualifying exam',
    'faculty': 'faculty',
    'startDate': 'startDate'
}

# liste med variable
variables = ['enrollmentAge', 'sex', 'avgAfterBonus', 'qualifyingExamYear', 'faculty', 'startDate']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, var in enumerate(variables):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    new_var = name_mapping[var]

    # Sæt en specifik antal af bins for 'startDate'
    bins_param = 30
    if var == 'startDate':
        bins_param = 4
    if var == 'avgAfterBonus':
        bins_param = 12

    # Plot histogrammer
    sns.histplot(data=df_studerende, x=var, hue='dropoutEvent', multiple="stack", bins=bins_param, ax=ax)

    # Sæt titler og labels
    ax.set_title(f'Distribution of {new_var} by dropout event')
    ax.set_xlabel(new_var)
    ax.set_ylabel('Number of students')

    # Roter x-aksetiketterne for at undgå overlap
    if var in ['faculty', 'startDate']:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    if var == 'startDate':
        dates = ['2018-09', '2019-09', '2020-09', '2021-09']
        ax.set_xticks(dates)
        ax.set_xticklabels(dates)

plt.tight_layout()
plt.show()

#%%
from scipy import stats

df_studerende = df_studerende.dropna(subset=['avgAfterBonus'])
df_studerende['avgAfterBonus'].fillna(0, inplace=True)

dropouts = df_studerende[df_studerende['dropoutEvent'] == 1]['avgAfterBonus']
non_dropouts = df_studerende[df_studerende['dropoutEvent'] == 0]['avgAfterBonus']

t_stat, p_val = stats.ttest_ind(dropouts, non_dropouts)

print(f"T-statistik: {t_stat:.2f}")
print(f"P-værdi: {p_val:.2f}")

correlation_coefficient, _ = stats.pearsonr(df_studerende['avgAfterBonus'], df_studerende['dropoutEvent'])
print(f"Korrelationskoefficient: {correlation_coefficient:.2f}")

#%% Efter transformation
df_transformed = pd.read_csv("df_pre_model.csv")

df_transformed_beskrivelse = describe_column(df_transformed)
print(df_transformed_beskrivelse)

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Grupper data baseret på 'dropoutEvent' og beregn gennemsnittet for hver eksamenstype
grouped = df_transformed.groupby('frafald_event')[['skala_kode_7-trinsskala', 'skala_kode_Bestået']].mean().reset_index()
# Lav en barplot
fig, ax = plt.subplots()
grouped.plot(kind='bar', x='frafald_event', y=['skala_kode_7-trinsskala', 'skala_kode_Bestået'], ax=ax)
ax.set_title('Average Number of Exams by Exam Type and Dropout Event')
ax.set_xlabel('Dropout Event')
ax.set_ylabel('Average Number of Exams')
plt.show()

#%%
print(grouped.info)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Opret et scatter plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_transformed, x='karakter_median', y='år_mellem', hue='frafald_event', palette='coolwarm', alpha=0.6)

# Sæt titlen og labels
plt.title('Scatter Plot of B vs. karakter_median Colored by frafald_event')
plt.xlabel('karakter_median')
plt.ylabel('år_mellem')

# Viser plottet
plt.show()
#%%
# Opret et scatter plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_transformed, x='karakter_median', y='B', hue='frafald_event', palette='coolwarm', alpha=0.6)

# Sæt titlen og labels
plt.title('Scatter Plot of n passed exams vs. median grade, colored by frafald_event')
plt.xlabel('Median grade')
plt.ylabel('Number of passes exams')

# Begrænser x-aksen og sætter xticks
plt.xlim(-3.2, 13)  # x starter ved 0
plt.xticks(list(range(-3, 13)))  # sætter ticks fra -3 til 12

# Viser plottet
plt.show()

#%%
print(grouped.info)

#%% korrelationsmatrix
corr_matrix = df_transformed.corr()
corr_matrix_str = corr_matrix.round(2).to_string()
print(corr_matrix_str)

with open('DescriptiveStatistic/korrelationsmatrix.txt', 'w') as file:
    file.write(corr_matrix_str)

#%%
f, ax = plt.subplots(figsize=(30, 30))
corr = df_transformed.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)

plt.show()

#%%
sorted_correlations = corr_matrix['frafald_event'].sort_values(ascending=False)

print("Most positive correlating variables")
print(sorted_correlations.head())

print("Most negative correlating variables")
print(sorted_correlations.tail())

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Vælg de variable, du er interesseret i
selected_columns = ['frafald_event', 'U', 'IB', 'skala_kode_7-trinsskala', 'karakter_median']

# Beregn korrelationsmatrix for de valgte kolonner
corr_selected = df_transformed[selected_columns].corr()

# Opret et heatmap
plt.figure(figsize=(15, 8))  # Sæt figurens størrelse
sns.heatmap(corr_selected, annot=True, cmap='coolwarm', fmt=".2f")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

# Tilføj titel og vis plottet
plt.title('Heatmap over Korrelation mellem Variable')
plt.show()

#%%
corr = df_transformed[['frafald_event', 'U', 'IB', '-3', 'skala_kode_Bestået', 'o', '10', '7', 'karakter_median', 'skala_kode_7-trinsskala']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#%%
print(df_transformed.columns)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Opretter en scatter plot
plot = sns.scatterplot(
    data=df_transformed,
    x='r',  # x-akse
    y='o',  # y-akse
    hue='frafald_event',
)

# Sætter titlen på plottet
plot.set_title('Scatter Plot of alder vs år_mellem')

# Viser plottet
plt.show()

sns.countplot(x=df_transformed['frafald_event'], data=df_transformed, palette='viridis')
plt.xlabel("Target class")
plt.ylabel("N Dropouts")
plt.show()

#%%
df_test = df_transformed[['frafald_event', 'U', 'IB', '-3', 'skala_kode_Bestået', 'o', '10', '7', 'karakter_median', 'skala_kode_7-trinsskala']]
df_test.hist(figsize=(15,12),bins=15)
plt.title("Features distribution")
plt.show()
