import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns


df_studerende = pd.read_excel("Frafald_grunddata_20230921_engelsk.xlsx")
df_eksamen = pd.read_excel("Frafald_grunddata_karakter_20230921_engelsk.xlsx")

#%% Plot forarbejde
colors = sns.color_palette('viridis', n_colors=2)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

#%%
print(colors[1])
#%% Plot 1: Køn

# Erstat "Mand" og "Kvinde" med "Male" og "Female" i 'sex' kolonnen
df_studerende['sex'].replace({"Mand": "Male", "Kvinde": "Female"}, inplace=True)

# Beregner antal af hver kategori
counts = df_studerende['dropoutEvent'].value_counts()

# Laver barplot
sns.countplot(x='sex', hue='dropoutEvent', data=df_studerende, palette="viridis", ax=ax1)

ax1.set_title('Distribution of dropout event by sex')
ax1.set_ylabel('Count')
ax1.set_xlabel('Gender')
#plt.xticks([0, 1], [0, 1])  # Sørger for kun at vise 0 og 1 på x-aksen
ax1.get_legend().set_visible(False)
#plt.close(fig1)

#%% Plot 2: Gennemsnit

# Opretter bins
bins = np.arange(1, 14, 1)  # Bins fra -3 til 14 med intervaller på 1

# Grupperer data baseret på 'avgAfterBonus' bins og 'dropoutEvent'
grouped = df_studerende.groupby([pd.cut(df_studerende['avgAfterBonus'], bins), 'dropoutEvent']).size().unstack(fill_value=0)

# Beregner procentdelen af studerende, der er droppet ud for hver bin
grouped['dropout_pct'] = (grouped[1] / (grouped[0] + grouped[1])) * 100

# Henter farverne fra viridis paletten
colors = sns.color_palette('viridis', n_colors=2)

# Sørger for at dropoutEvent=1 altid er grøn
color_order = [colors[1], colors[0]]

# Plotter procentdelen af dem, der dropper ud som søjler på ax2
bars = grouped['dropout_pct'].plot(kind='bar', color=colors[1], ax=ax2, width=0.8)

# Tilføj tekst over hver søjle
for idx, bar in enumerate(bars.patches):
    # Beregn det samlede antal studerende for denne bin
    total_students = grouped[0].iloc[idx] + grouped[1].iloc[idx]
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'n={total_students}',
             ha='center', va='bottom', fontsize=8)

ax2.set_title('Dropout Percentage by high-school grade (after bonus)')
ax2.set_ylabel('Dropout Percentage (%)')
ax2.set_xlabel('High-school grade average (after bonus)')
ax2.set_xticks(range(len(grouped.index)))
ax2.set_xticklabels([str(interval).replace(", ", ",") for interval in grouped.index], rotation=0)
#%% Plot 3: Age


# Opretter bins og labels
bins = [18, 20, 25, 30, 35, 40, 50, 60, 100]
labels = ["18-19", "20-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"]
df_studerende['age_group'] = pd.cut(df_studerende['enrollmentAge'], bins=bins, labels=labels, right=False)

# Beregn procentdel af studerende, der er droppet ud i hver aldersgruppe
ct_age = pd.crosstab(df_studerende['age_group'], df_studerende['dropoutEvent'])
ct_age['dropout_pct'] = ct_age[1] / (ct_age[0] + ct_age[1]) * 100
ct_age['total_students'] = ct_age[0] + ct_age[1]

# Plotter barplot på ax3
bar_plot = sns.barplot(x=ct_age.index, y=ct_age['dropout_pct'], color=colors[1], ax=ax3)

# Tilføj detaljer til plottet
ax3.set_title('Dropout Percentage by age when enrolled')
ax3.set_ylabel('Dropout Percentage (%)')
ax3.set_xlabel('Enrollment Age')
ax3.grid(axis='y')

# Tilføj tekst til hver søjle med det samlede antal studerende
for idx, patch in enumerate(bar_plot.patches):
    height = patch.get_height()
    ax3.text(patch.get_x() + patch.get_width() / 2., height + 1,
             f'n={ct_age["total_students"].iloc[idx]}',
             ha="center")


#%% Plot 4: krydstabel fakultet og frafald


# Opretter en krydstabel
ct = pd.crosstab(df_studerende['faculty'], df_studerende['dropoutEvent'])

# Sorterer fakulteterne baseret på antallet af studerende, der er droppet ud (dvs. kolonne 1 i crosstabellen)
ct = ct.sort_values(by=1, ascending=False)

# Beregner procentdelen af studerende, der er droppet ud for hvert fakultet
ct['dropout_pct'] = (ct[1] / (ct[0] + ct[1])) * 100

# Henter farverne fra viridis paletten
colors = sns.color_palette('viridis', n_colors=2)

# Sørger for at dropoutEvent=1 altid er grøn
color_order = [colors[1], colors[0]]

# Plotter stablede barplots med frafald studerende i bunden på ax4
ct[[1, 0]].plot(kind='bar', stacked=True, ax=ax4, color=color_order, legend=False)
ax4.set_title('Distribution of dropout event by faculty')

# Mapping af fakulteter til deres engelske navne
faculty_mapping = {
    "Humaniora": "Humanities",
    "Samfundsvidenskab": "Social Sciences",
    "Teknik": "Engineering",
    "Naturvidenskab": "Science",
    "Sundhedsvidenskab": "Health Sciences"
}
formatted_ticks = [faculty_mapping.get(tick, tick) for tick in ct.index]

# Anvender de formatterede ticks til x-aksen
ax4.set_xticks(range(len(formatted_ticks)))
ax4.set_xticklabels(formatted_ticks, rotation=0)

# Tilføjer procentdelen af frafald pr. fakultet til søjlerne
for idx, value in enumerate(ct['dropout_pct']):
    ax4.text(idx, ct[1].iloc[idx]/2, f'{value:.2f}%', ha='center', va='center', color='black')

ax4.set_xlabel("Faculty")
ax4.set_ylabel("Count")


#%% Plot 5: krydstabel periode og frafald
def format_date_to_fyear(date):
    year = date.year
    return f"Fall {year % 100}"


import pandas as pd
import matplotlib.pyplot as plt


# Opretter en krydstabel
ct = pd.crosstab(df_studerende['startDate'], df_studerende['dropoutEvent'])

# Beregner procentdelen af studerende, der er droppet ud for hver startdato
ct['dropout_pct'] = (ct[1] / (ct[0] + ct[1])) * 100

# Henter farverne fra viridis paletten
colors = sns.color_palette('viridis', n_colors=2)

# Sørger for at dropoutEvent=1 altid er grøn
color_order = [colors[1], colors[0]]

# Konverterer x-akse værdier
formatted_index = [format_date_to_fyear(date) for date in ct.index]

# Plotter stablede barplots med frafald studerende i bunden på ax5
ct[[1, 0]].plot(kind='bar', stacked=True, ax=ax5, color=color_order, legend=False)
ax5.set_title('Distribution of dropout event by enrollment period')

# Ændrer x-ticks baseret på det nye format
ax5.set_xticks(range(len(formatted_index)))
ax5.set_xticklabels(formatted_index, rotation=0)

# Tilføjer procentdelen af frafald pr. startDate til søjlerne
for idx, value in enumerate(ct['dropout_pct']):
    ax5.text(idx, ct[1].iloc[idx]/2, f'{value:.2f}%', ha='center', va='center', color='black')

ax5.set_xlabel("Period")


#%% Plot 6: Distribution a dropout Event
# Opretter en figur og akse
#fig6, ax6 = plt.subplots(figsize=(10, 5))

# Beregner antal af hver kategori
counts = df_studerende['dropoutEvent'].value_counts()

# Laver barplot på ax6
sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax6)

ax6.set_title('Distribution of dropout event')
ax6.set_ylabel('Count')
ax6.set_xlabel('Dropout Event')
ax6.set_xticks([0, 1])  # Sørger for kun at vise 0 og 1 på x-aksen
ax6.set_xticklabels([0, 1])

#%%
# Tilføj en legende uden for plots
handles, labels = ax1.get_legend_handles_labels()
# Fjern duplikater
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
unique_handles, unique_labels = zip(*unique)

# Placer legenden øverst uden for figuren
fig.legend(unique_handles, unique_labels, loc='upper center', title="Dropout Event", ncol=2)

plt.subplots_adjust(top=0.85)

plt.tight_layout()
plt.show()