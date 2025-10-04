import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# 1️⃣ Ler Excel e tratar colunas
# ----------------------
df = pd.read_excel("C:/Users/alan_/Documents/python.ML.dados/Curso udemy/Dados/Desafio prático - aula 27.xlsx")

# Padronizar nomes das colunas (remove espaços)
df.rename(columns=lambda x: x.strip().replace(' ', '_'), inplace=True)

# Limpar coluna Preco_R$
df['Preco_R$'] = pd.to_numeric(
    df['Preco_R$'].astype(str)
      .str.replace(r'[^\d,.-]', '', regex=True)  # remove tudo que não é número, vírgula ou ponto
      .str.replace('.', '', regex=False)
      .str.replace(',', '.', regex=False),
    errors='coerce'
)

# Garantir coluna numérica
df['Quantidade_Vendida'] = pd.to_numeric(df['Quantidade_Vendida'], errors='coerce')

# Remover linhas inválidas
df = df.dropna(subset=['Preco_R$', 'Quantidade_Vendida', 'Categoria'])

# ----------------------
# 2️⃣ Estatísticas sobre preços
# ----------------------
precos = df['Preco_R$']

media = precos.mean()
mediana = precos.median()
desvio_padrao = precos.std(ddof=0)
cv = (desvio_padrao / media) * 100
moda = precos.mode()

if len(moda) == 0:
    tipo_moda = "Amodal"
elif len(moda) == 1:
    tipo_moda = f"Unimodal (valor mais frequente: {moda.iloc[0]:.2f})"
else:
    tipo_moda = f"Multimodal (valores: {[round(v,2) for v in moda]})"

print(f"Media: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desvio padrão (pop): {desvio_padrao:.2f}")
print(f"Coeficiente de variação: {cv:.2f}%")
print("Moda:", tipo_moda)

# ----------------------
# 3️⃣ Gráficos de preços
# ----------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=precos, color='lightgreen')
plt.title("Boxplot dos Preços")
plt.xlabel("Preço (R$)")
plt.tight_layout()
plt.savefig("boxplot_precos.png")
plt.savefig("boxplot_precos.jpeg")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(precos, bins=20, kde=True, color='skyblue')
plt.title("Distribuição dos Preços")
plt.xlabel("Preço (R$)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# ----------------------
# 4️⃣ Estatísticas por categoria
# ----------------------
estatisticas = df.groupby('Categoria')['Quantidade_Vendida'].agg(['count', 'mean', 'median', 'std']).reset_index()
print(estatisticas)

media_categoria = df.groupby('Categoria')['Quantidade_Vendida'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=media_categoria.index, y=media_categoria.values, palette='pastel')
plt.title("Média da Quantidade Vendida por Categoria")
plt.xlabel("Categoria")
plt.ylabel("Média da Quantidade Vendida")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("media_categoria_quantidade.png")
plt.show()

# ----------------------
# 5️⃣ Faturamento e filtros
# ----------------------
df['Faturamento'] = df['Preco_R$'] * df['Quantidade_Vendida']

mediana_qtde = df['Quantidade_Vendida'].median()
filtro_qtde = df[df['Quantidade_Vendida'] > mediana_qtde]
filtro_final = filtro_qtde[(filtro_qtde['Preco_R$'] > 200) & (filtro_qtde['Faturamento'] > 6000)]

num_produtos = filtro_final.shape[0]
total_produtos = df.shape[0]
percentual = (num_produtos / total_produtos) * 100

print(f"Produtos que atendem aos critérios: {num_produtos}")
print(f"Porcentagem em relação ao total: {percentual:.2f}%")

# ----------------------
# 6️⃣ Gráfico 1: barras horizontais por faturamento
# ----------------------
filtro_final = filtro_final.sort_values(by='Faturamento', ascending=True)

plt.figure(figsize=(10,6))
sns.barplot(x='Faturamento', y='ID_Produto', data=filtro_final, hue='Categoria', dodge=False, palette='pastel')
plt.title("Faturamento dos Produtos Filtrados")
plt.xlabel("Faturamento (R$)")
plt.ylabel("ID do Produto")
plt.legend(title='Categoria', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("produtos_filtrados_bar.png")
plt.show()

# ----------------------
# 7️⃣ Gráfico 2: scatter quantidade x faturamento
# ----------------------
plt.figure(figsize=(10,6))
sns.scatterplot(data=filtro_final, x='Quantidade_Vendida', y='Faturamento',
                size='Preco_R$', hue='Categoria', palette='tab10', sizes=(50, 300), alpha=0.7)
plt.title("Quantidade Vendida x Faturamento (Produtos Filtrados)")
plt.xlabel("Quantidade Vendida")
plt.ylabel("Faturamento (R$)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("produtos_filtrados_scatter.png")
plt.savefig("produtos_filtrados_scatter.jpeg")
plt.show()
