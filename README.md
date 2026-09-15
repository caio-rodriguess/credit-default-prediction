# Predição de Inadimplência com Machine Learning

Este projeto foi desenvolvido com o objetivo de praticar e aprender técnicas de Machine Learning aplicadas à classificação de clientes inadimplentes.

O estudo explora o tratamento de **desbalanceamento de classes com SMOTENC**, a busca de hiperparâmetros com **GridSearchCV** e o uso de **Pipelines** para organizar o pré-processamento e evitar data leakage durante a validação cruzada.

## Objetivos

- Explorar e visualizar os dados de clientes bancários;
- Treinar um modelo inicial de classificação com Random Forest;
- Avaliar o modelo usando accuracy, precision, recall e F1-score;
- Investigar o efeito do desbalanceamento entre clientes inadimplentes e não inadimplentes;
- Aplicar SMOTENC para gerar amostras sintéticas da classe minoritária;
- Comparar o desempenho antes e depois do balanceamento;
- Utilizar GridSearchCV para explorar diferentes hiperparâmetros do Random Forest;
- Utilizar um Pipeline para aplicar o SMOTENC dentro de cada divisão da validação cruzada, reduzindo o risco de data leakage.

## Dados

Os dados utilizados são a amostra de dados bancários disponibilizada por Cibele Russo:

[amostra_banco.csv](https://github.com/cibelerusso/Datasets/blob/main/amostra_banco.csv)

O notebook acessa os dados diretamente pelo GitHub.

## Metodologia

O projeto segue, de forma geral, as seguintes etapas:

1. **Análise exploratória:** inspeção dos dados, estatísticas descritivas, valores ausentes e distribuição da variável `Inadimplente`.
2. **Visualização:** comparação das distribuições das variáveis quantitativas entre clientes inadimplentes e não inadimplentes.
3. **Pré-processamento:** transformação das variáveis categóricas para utilização no modelo.
4. **Modelo inicial:** treinamento de um `RandomForestClassifier` e avaliação por meio da matriz de confusão, precision, recall e F1-score.
5. **Balanceamento:** aplicação do `SMOTENC` para aumentar a representação da classe minoritária no conjunto de treino.
6. **Novo modelo:** treinamento do Random Forest com os dados balanceados e comparação com o modelo inicial.
7. **Otimização:** utilização de `GridSearchCV` para testar diferentes combinações de hiperparâmetros.
8. **Avaliação final:** análise do desempenho do modelo otimizado no conjunto de teste.

## Principais resultados

No modelo inicial, a acurácia ficou em aproximadamente **73,6%**, mas o recall da classe inadimplente foi de apenas **14,81%**. Isso mostra que uma acurácia relativamente alta não significa necessariamente que o modelo esteja identificando bem a classe de interesse.

Após a aplicação do SMOTENC, o recall aumentou para aproximadamente **29,63%**, enquanto o F1-score passou de cerca de **0,20 para 0,28**. Em contrapartida, houve uma redução da acurácia e da precisão.

Esse resultado mostra o principal efeito do balanceamento neste estudo: o modelo passou a identificar uma parcela maior dos inadimplentes, mas também passou a gerar mais falsos positivos.

Por isso, o **recall da classe inadimplente** é uma métrica especialmente relevante neste problema. A análise também mostra por que avaliar apenas a acurácia pode ser insuficiente em conjuntos de dados desbalanceados.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## Como executar

Clone o repositório e abra o notebook `credit_default_prediction.ipynb` em Jupyter Notebook, JupyterLab ou Google Colab.

As principais bibliotecas necessárias são:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Em seguida, execute as células na ordem apresentada no notebook.

## Conclusões e próximos passos

O projeto mostra que o Random Forest consegue identificar alguns casos de inadimplência, mas encontra dificuldades principalmente por causa do desbalanceamento da classe alvo. O SMOTENC melhora a identificação da classe minoritária, embora esse ganho venha acompanhado de uma redução em outras métricas.

Como próximos passos, seria interessante testar outros algoritmos de classificação, avaliar diferentes estratégias de balanceamento, explorar outros limiares de classificação e aprofundar a análise das variáveis utilizadas pelo modelo.

> **Observação:** este é um projeto de estudo desenvolvido para praticar conceitos de Machine Learning e não pretende representar um sistema de concessão de crédito pronto para uso em produção.
