import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

df = pd.read_csv("C:/Users/alan_/Documents/ML.DataAnalitics/DataAnalitics/Curso udemy/Dados/Desafio Prático IV - Página1.csv")

df.columns = [c.strip() for c in df.columns]

#TREATMENT OF DATA#

for col in ['IDH', 'Gini', 'TaxaDesemprego', 'TaxaFecundidade']:
    df[col] = (
        df[col]
        .astype(str)                 # make sure it's string
        .str.replace(',', '.', regex=False)  # replace comma with dot
        .astype(float)               # convert to float
    )

df['PIB_per_capita'] = (
    df['PIB_per_capita']
    .astype(str)
    .str.replace(r'[R$\s]', '', regex=True)   # remove R$ and spaces
    .str.replace('.', '', regex=False)        # remove thousands separator
    .str.replace(',', '.', regex=False)       # convert comma to decimal
    .astype(float)
)

df['Populacao'] = (
    df['Populacao']
    .astype(str)
    .str.replace('.', '', regex=False)  # remove thousands separator
    .astype(int)
)

#EXERCISE 1#
print('EXERCISE 1')
threshold = 0.75
low_group = df[df['IDH'] < threshold]
high_group  = df[df['IDH'] >= threshold]

mean_low = low_group['PIB_per_capita'].mean()
mean_high = high_group['PIB_per_capita'].mean()

print("Média do PIB per capita >= 0.75: ", round(mean_low, 2))
print("Média do PIB per capita < 0.75: ", round(mean_high, 2))
print("N(HDI<0.75): ",len(low_group), "N(HDI>=0.75: ", len(high_group))

if 'Populacao' in df.columns:
    corr_pop = df[['PIB_per_capita', 'Populacao']].dropna().corr().iloc[0,1]
    print("Correlation (GDP x Population): ", round(corr_pop, 2))
if 'TaxaFecundidade' in df.columns:
    corr_fec = df[['PIB_per_capita', 'TaxaFecundidade']].dropna().corr().iloc[0,1]
    print("Correlation (GDP X Fertility rate): ", round(corr_fec, 2))

#EXERCISE 2#
print('EXERCISE 2')
def try_stat_test(a, b):
    a = a.dropna()
    b = b.dropna()

    normal_a = normal_b = False
    if len(a) > 2 and len(a) <= 5000:
        normal_a = stats.shapiro(a).pvalue > 0.05
    if len(b) > 2 and len(b) <= 5000:
        normal_b = stats.shapiro(b).pvalue > 0.05

    if normal_a and normal_b:
        tstat, pval = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        test_name = 't-test (Welch)'
    else:
        tstat, pval = stats.mannwhitneyu(a, b, alternative='two-sided')
        test_name = 'Mann-Whitney U'
    return test_name, tstat, pval

test_name, stat_val, p_value = try_stat_test(low_group['TaxaDesemprego'], high_group['TaxaDesemprego'])
print(f"Test for the difference in Unnemployed Rate -> {test_name}")
print("Statistic:", round(stat_val, 2), "p-value:", round(p_value, 4))
if p_value < 0.05:
    print("Result: There is significant evidence (alfa=0.05).")
else:
    print("Result: No significant evidence (alfa=0.05).")
print()

#EXERCISE 3#
print('EXERCISE 3')
sub = df[['Municipio','Gini','PIB_per_capita']].dropna()
pearson_r, pearson_p = stats.pearsonr(sub['Gini'], sub['PIB_per_capita'])
spearman_r, spearman_p = stats.spearmanr(sub['Gini'], sub['PIB_per_capita'])

print("Correlation Pearson (Gini x GDP): r =", round(pearson_r, 2), ", p =", round(pearson_p, 2))
print("Correlation Spearman (Gini x GDP): rho =", round(spearman_r, 2), ", p =", round(spearman_p, 2))
print()

# Gráfic
plt.figure(figsize=(8,6))
plt.scatter(sub['Gini'], sub['PIB_per_capita'], alpha=0.8)
m, b = np.polyfit(sub['Gini'], sub['PIB_per_capita'], 1)
xfit = np.linspace(sub['Gini'].min(), sub['Gini'].max(), 100)
plt.plot(xfit, m*xfit + b, linewidth=2)
plt.xlabel('Gini Index')
plt.ylabel('GDP')
plt.title('Gini vs GDP (scatter + regressão)')
out_png = os.path.join(os.getcwd(), "gini_vs_pib.png")
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Gráfico salvo em: {out_png}")
print()

#EXERCISE 4#
print('EXERCISE 4')
top3_gini = df[['Municipio','Gini','PIB_per_capita','TaxaDesemprego']].dropna().sort_values('Gini', ascending=False).head(3)
print("Top 3 municípios por Gini (mais desiguais):")
print(top3_gini.to_string(index=False))
print()

#EXERCISE 5#
print('EXERCISE 5')
sub = df[['Municipio','IDH','Gini','TaxaDesemprego','PIB_per_capita']].dropna()

# Ranking indicators: lower IDH and PIB per capita -> worse, higher Gini and Unemployment -> worse
sub['rank_IDH'] = sub['IDH'].rank(ascending=True)  # lower IDH -> higher vulnerability
sub['rank_Gini'] = sub['Gini'].rank(ascending=False)  # higher Gini -> higher vulnerability
sub['rank_Unemployment'] = sub['TaxaDesemprego'].rank(ascending=False)  # higher unemployment -> higher vulnerability
sub['rank_PIB'] = sub['PIB_per_capita'].rank(ascending=True)  # lower PIB -> higher vulnerability

# Composite vulnerability score
sub['vulnerability_score'] = sub[['rank_IDH','rank_Gini','rank_Unemployment','rank_PIB']].sum(axis=1)

# Select top 3 most vulnerable cities
top3_vulnerable = sub.sort_values(by='vulnerability_score', ascending=False).head(3)
print("Top 3 most vulnerable municipalities:")
print(top3_vulnerable[['Municipio','IDH','Gini','TaxaDesemprego','PIB_per_capita','vulnerability_score']])
print()

#EXERCISE 6#
print('EXERCISE 6')
# Save table for report / further analysis
out_csv = os.path.join(os.getcwd(), "top3_vulnerable_municipalities.csv")
top3_vulnerable.to_csv(out_csv, index=False)
print(f"Top 3 municipalities saved as CSV: {out_csv}")

# Optional: quick visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.barh(top3_vulnerable['Municipio'], top3_vulnerable['vulnerability_score'], color='tomato')
plt.xlabel("Vulnerability Score")
plt.title("Top 3 Most Vulnerable Municipalities")
plt.gca().invert_yaxis()
plt.tight_layout()
out_png = os.path.join(os.getcwd(), "top3_vulnerable_municipalities.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Bar chart saved as PNG: {out_png}")