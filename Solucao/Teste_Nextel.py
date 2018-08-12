
# coding: utf-8

# <img src='nextel.jpg'></img>
# <center><h1>Teste Engenheiro de Dados - Nextel Telecomunicações</h1></center>

# <h3>Nome: Dennis Cardoso</h2>
# <h3>Data: 12 de Agosto de 2018</h2>

# <H2> Objetivos </h2>
# 
# A base de house_sales.csv fornece os dados de vendas de casas em uma determinadar região dos Estados Unidos. Abaixo segue o dicionário de dados da base de dados:
# 
# <ul>
#     <li>**price** - The last price the house was sold for</li>
#     <li>**num_bed** - The number of bedrooms</li>
#     <li>**num_bath** - The number of bathrooms (fractions mean the house has a toilet-only or shower/bathtub-only bathroom)</li>
#     <li>**size_house** (includes basement) - The size of the house</li>
#     <li>**size_lot** - The size of the lot</li>
#     <li>**num_floors** - The number of floors</li>
#     <li>**is_waterfront** - Whether or not the house is a waterfront house (0 means it is not a waterfront house whereas 1 means that it is a waterfront house)</li>
#     <li>**condition** - How worn out the house is. Ranges from 1 (needs repairs all over the place) to 5 (the house is very well maintained)</li>
#     <li>**size_basement** - The size of the basement</li>
#     <li>**year_built** - The year the house was built</li>
#     <li>**renovation_date** - The year the house was renovated for the last time. 0 means the house has never been renovated</li>
#     <li>**zip** - The zip code</li>
#     <li>**latitude** - Latitude</li>
#     <li>**longitude** - Longitude</li>
#     <li>**avg_size_neighbor_houses** - The average house size of the neighbors</li>
#     <li>**avg_size_neighbor_lot** - The average lot size of the neighbors</li>
# </ul>
# 
# Utilizando o arquivo 'houses_sales.csv', serão realizadas análises preliminares dos valores para identificar hipóteses, e então, construir um modelo preditivo para identificar os melhores preços das casas dessa região.

# <h2>Ferramentas e Linguagem</h2>
# 
# Neste teste, estão sendo utilizadas as seguintes ferramentas e linguagens:
# 
# <ul>
#   <li>Jupyter Notebook</li>
#   <li>Anaconda</li>
#   <li>Python 3.0</li>
#   <li>sublime</li>
# </ul>

# In[ ]:


#iniciando as bibliotecas a serem utilizadas

#Biblioteca panda pra análises de dados
import pandas as pd

#biblioteca fundamental para computação cientifica em python
import numpy as np

#Teste de biblioteca
from sklearn.preprocessing import LabelEncoder

#Função para construção de uma base de treino e de teste para nosso modelo
from sklearn.model_selection import train_test_split

#Função para execução de uma regração Linear
from sklearn.linear_model import LinearRegression


# <h3> Input de dados</h2>

# In[ ]:


#Entrada de arquivo para análise para um dataframe
housedata = pd.read_csv('house_sales.csv')

#Nome de todas as colunas do arquivos
housedata.columns.tolist()


# In[ ]:


#Exibir informações preliminares (10 linhas):
housedata.head(10)


# In[ ]:


#Resumo dos campos número, com analises preliminares
housedata.describe()


# In[ ]:


#Exibir os tipos dos das colunas analisadas
housedata.dtypes


# In[ ]:


#Total de linhas:
print('A estrutura de dados contém {} linhas de registro e {} colunas.'.format(housedata.shape[0], housedata.shape[1]))


# <h3>Contrução do Modelo Preditivo - Regressão Linear</h3>

# <h4>Treinando o modelo</h4>

# Geraremos uma base de teste e de treino para adicionar dados ao modelo:

# In[ ]:


#variável Y com valores apenas do preço das casas - Coluna 'price'
y = housedata['price']

#Variável x contém as demais colunas, com exceção da coluna 'price'
x = housedata.drop(['price'],axis=1)

#criando variáveis de teste e treino para o modelo de regressão linear
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1) 


# In[ ]:


#realizar o modelo de regressão nas bases de treino e de teste
housemodel = LinearRegression()
housemodel.fit(x_train, y_train)


# <h4>Verificando a qualidade do modelo - Score e coeficientes</h4>

# In[ ]:


#Teste de pontuação do modelo na base de treino
housemodel.score(x_train, y_train)


# In[ ]:


#Teste de pontuação do modelo na base de teste
housemodel.score(x_test, y_test)


# In[ ]:


#Desmontração dos coeficientes
for idx, column_name in enumerate(x_train.columns):
    print("O coeficiente para {} é {}".format(column_name, housemodel.coef_[idx]))


# In[ ]:


#Intercepto do modelo

intercepto = housemodel.intercept_
print("O intercepto do modelo é {}".format(intercepto))


# <h4>Realizar previsões de preço</h4>

# In[ ]:


#Precisão de preço utilizando a base de teste - Calculo aleatório
precocasa = housemodel.predict(x_test)

print("O preço de venda: U$ {:0,.0f}".format(int(precocasa[random.randint(0, 100)])))


# In[ ]:


#Precisão de preço utilizando valores de exemplo
precocasa = housemodel.predict([[3,1,1181,5650,1,0,3,0,1955,0,98178,47.51123398,-122.2567754,1340,5650]])

print("O preço de venda: U$ {:0,.0f}".format(int(precocasa)))

