# statMancer

Ferramentas de modelagem de dados em R para execução de modelos preditivos.
Projeto baseado em TDD — todos os módulos possuem testes com dados sintéticos.

---

## Estrutura

```
R/
  sampling/
    sampling_balance.R     # upsample() / downsample() por ID + target
    sampling_split.R       # train_test_split() / kfold_split() por ID
  stats/
    stats_search.R         # stats_search() — busca estatística de variáveis
  modeling/
    modeling_xgb_train.R   # xgb_train()
    modeling_xgb_predict.R # xgb_predict() / xgb_importancia()
  reporting/
    reporting_metrics.R    # metricas_binario() / tabela_decis() / curva_roc()
    reporting_render.R     # preparar_dados_relatorio() / renderizar_relatorio()
  utils/
    utils_format_columns.R # utilitários de formatação de colunas
    utils_modelagem.R      # xgb_select_params() — busca bayesiana de hiperparâmetros

report/
  template_modelo.qmd      # template Quarto → HTML embed-resources

tests/
  helpers/
    synthetic_data.R       # geradores de dados sintéticos reutilizáveis
  test_sampling_balance.R
  test_sampling_split.R
  test_sampling_split_kfold.R
  test_stats_search.R
  test_modeling_xgb.R
  test_reporting_metrics.R

exemplo_completo.R         # pipeline de ponta a ponta (dados → relatório HTML)
```

---

## Módulos

### `R/sampling/` — Amostragem e Split

**Princípio:** a unidade de amostragem é sempre o `var_id`.
Todos os registros de um mesmo ID ficam obrigatoriamente no mesmo conjunto.

| Função | Descrição |
|---|---|
| `upsample(dt, var_id, var_target, n_por_classe, var_estratificacao, seed)` | Sobreamostra a classe minoritária com reposição |
| `downsample(dt, var_id, var_target, n_por_classe, var_estratificacao, seed)` | Subamostra a classe majoritária sem reposição |
| `train_test_split(dt, var_id, prop_treino, var_estratificacao, seed)` | Divide dados em treino/teste por ID, sem vazamento |
| `kfold_split(dt, var_id, k, var_estratificacao, seed)` | K-Fold cross-validation por ID; retorna lista de k folds |

Todos suportam **estratificação proporcional** via `var_estratificacao`.

---

### `R/stats/` — Busca Estatística

| Função | Descrição |
|---|---|
| `stats_search(dt, var_target, vars_excluir, tipo_target, alpha, max_categorias)` | Testa cada variável contra o target e retorna ranking por relevância |

**Testes aplicados automaticamente:**

| Tipo variável | Tipo target | Teste | Métrica de efeito |
|---|---|---|---|
| Numérica | Classificação | KS (binário) / KS par-a-par | D de KS |
| Categórica | Classificação | Chi-quadrado | V de Cramér |
| Numérica | Regressão | Pearson / Spearman | \|r\| |
| Categórica | Regressão | ANOVA | η² (Eta²) |

---

### `R/modeling/` — Modelagem XGBoost

| Função | Descrição |
|---|---|
| `xgb_train(dt_treino, var_target, vars_excluir, params, nrounds)` | Treina modelo XGBoost |
| `xgb_predict(modelo_obj, dt_novo, var_id)` | Gera predições |
| `xgb_importancia(modelo_obj, top_n)` | Importância das features (Gain) |
| `xgb_select_params(...)` | Busca bayesiana de hiperparâmetros via mlrMBO *(utils_modelagem.R)* |

Os parâmetros de `xgb_select_params()$parametros` podem ser passados diretamente ao `xgb_train()`.

---

### `R/reporting/` — Relatórios

| Função | Descrição |
|---|---|
| `metricas_binario(dt, var_pred, var_target)` | AUC, KS, Gini, Precision, Recall, F1, Accuracy |
| `tabela_decis(dt, var_pred, var_target, n_decis)` | Lift, taxa de evento e captura por decil |
| `curva_roc(dt, var_pred, var_target)` | Pontos (FPR, TPR) para plotagem da curva ROC |
| `preparar_dados_relatorio(...)` | Coleta todos os resultados e salva `.rds` para o template |
| `renderizar_relatorio(template_qmd, dados_rds, output_file, output_dir)` | Renderiza HTML via `quarto::quarto_render()` |

O relatório HTML inclui:
- Sumário executivo com avaliação das métricas
- Distribuição do target (treino vs teste)
- Tabela de busca estatística (se disponível)
- Curva ROC + distribuição dos scores
- Tabela de decis + gráfico de lift + captura acumulada
- Importância das features
- Configuração do modelo

---

## Executar exemplo completo

```r
source("exemplo_completo.R")
# Saída em: output/relatorio_modelo.html
```

## Executar testes

```r
source("tests/test_sampling_balance.R")
source("tests/test_sampling_split.R")
source("tests/test_sampling_split_kfold.R")
source("tests/test_stats_search.R")
source("tests/test_modeling_xgb.R")
source("tests/test_reporting_metrics.R")
```

---

## Pacotes utilizados

- `data.table` — manipulação de dados
- `xgboost` — modelagem
- `mlrMBO` — busca bayesiana de hiperparâmetros
- `recipes` — pré-processamento
- `quarto` — renderização de relatórios
- `ggplot2` — visualizações no relatório HTML


---

## Estrutura

```
R/
  sampling/
    sampling_balance.R     # upsample() / downsample() por ID + target
    sampling_split.R       # train_test_split() por ID
  stats/
    stats_search.R         # stats_search() — busca estatística de variáveis
  modeling/
    modeling_xgb_train.R   # xgb_train()
    modeling_xgb_predict.R # xgb_predict() / xgb_importancia()
  utils/
    utils_format_columns.R # utilitários de formatação de colunas
    utils_modelagem.R      # xgb_select_params() — busca bayesiana de hiperparâmetros

tests/
  helpers/
    synthetic_data.R       # geradores de dados sintéticos reutilizáveis
  test_sampling_balance.R
  test_sampling_split.R
  test_stats_search.R
  test_modeling_xgb.R
```

---

## Módulos

### `R/sampling/` — Amostragem

**Princípio:** a unidade de amostragem é sempre o `var_id`.
Todos os registros de um mesmo ID ficam obrigatoriamente no mesmo conjunto.

| Função | Descrição |
|---|---|
| `upsample(dt, var_id, var_target, n_por_classe, var_estratificacao, seed)` | Sobreamostra a classe minoritária com reposição |
| `downsample(dt, var_id, var_target, n_por_classe, var_estratificacao, seed)` | Subamostra a classe majoritária sem reposição |
| `train_test_split(dt, var_id, prop_treino, var_estratificacao, seed)` | Divide dados em treino/teste por ID, sem vazamento |

Todos suportam **estratificação proporcional** via `var_estratificacao`.

---

### `R/stats/` — Busca Estatística

| Função | Descrição |
|---|---|
| `stats_search(dt, var_target, vars_excluir, tipo_target, alpha, max_categorias)` | Testa cada variável contra o target e retorna ranking por relevância |

**Testes aplicados automaticamente:**

| Tipo variável | Tipo target | Teste | Métrica de efeito |
|---|---|---|---|
| Numérica | Classificação | KS (binário) / KS par-a-par | D de KS |
| Categórica | Classificação | Chi-quadrado | V de Cramér |
| Numérica | Regressão | Pearson / Spearman | \|r\| |
| Categórica | Regressão | ANOVA | η² (Eta²) |

---

### `R/modeling/` — Modelagem XGBoost

| Função | Descrição |
|---|---|
| `xgb_train(dt_treino, var_target, vars_excluir, params, nrounds)` | Treina modelo XGBoost |
| `xgb_predict(modelo_obj, dt_novo, var_id)` | Gera predições |
| `xgb_importancia(modelo_obj, top_n)` | Importância das features (Gain) |
| `xgb_select_params(...)` | Busca bayesiana de hiperparâmetros via mlrMBO *(utils_modelagem.R)* |

Os objetos de `xgb_train` são compatíveis diretamente com `xgb_predict` e `xgb_importancia`.
Os parâmetros de `xgb_select_params()$parametros` podem ser passados diretamente ao `xgb_train()`.

---

## Pacotes utilizados

- `data.table` — manipulação de dados
- `xgboost` — modelagem
- `mlrMBO` — busca bayesiana de hiperparâmetros
- `recipes` — pré-processamento
- `quarto` — relatórios

---

## Executar testes

```r
source("tests/test_sampling_balance.R")
source("tests/test_sampling_split.R")
source("tests/test_stats_search.R")
source("tests/test_modeling_xgb.R")
```

