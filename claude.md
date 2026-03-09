# statMancer

Projeto baseado em TDD Crie dados sintéticos para poder realizar os testes


A ideia desse projeto é ter ferramentas de modelagem de dados para execução de modelos.

1 - Ferramentas de amostragem com up & down Sample baseados num ID de identificação e uma variável target (podendo ou não usar amostra estratificada com outras variáveis)
2 - Ferramentas de separação de dados para Treino e Teste
3 - Ferramentas de buscas estatísticas para poder indentificar possíveis variáveis relacionadas com o Target (tanto target número como de classes)
4 - Ferramentas de modelagem com Xgboost (grande parte já construída)
5 - Construção de relatórios com os resultados dos modelos usando Quarto e output para html
7 - Deve Saber lidar com variáveis categoricas, como "região", sexo e tudo mais. utilise factors para isso.  Modelos de árvore consegue lidar bem usando essas colunas como factors.
8 - Deve ter uma procura de variáveis relacionadas com target quando é classificação e o a variaǘel tb é categorica

Primariamente deve ser escrito em R SEMPRE.

Os pacotes do R que devem ser utilizados primariamente são:

- data.table: para manipulação de dados
- xgboost: para modelagens
- mlrMBO: para busca de parametros
- recipies: para criar receitas
- quarto: para criar relatórios

