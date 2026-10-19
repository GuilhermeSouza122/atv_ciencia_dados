# Spotify em Dados

**Do histórico de streaming à análise descritiva e à classificação preditiva.**

Projeto de ciência de dados com tratamento de dados do Spotify, modelagem relacional, visualização em Power BI e experimentos de machine learning em notebooks Python.

## Percurso da análise

1. **Preparação:** tratamento do histórico geral e dos dados SoundCapsule.
2. **Modelagem:** scripts SQL e modelo conceitual para organizar os dados.
3. **Exploração:** análise descritiva em Power BI e apresentação dos resultados.
4. **Predição:** classificação com regressão logística e exploração de explicações com SHAP.

## Explore as entregas

- [`Tratamento de dados`](entrega/Tratamento%20de%20dados) — notebooks de preparação.
- [`Banco`](entrega/Banco) — scripts SQL e modelo conceitual.
- [`Análise Descritiva`](entrega/An%C3%A1lise%20Descritiva) — arquivo Power BI.
- [`Análise Diagnóstica & Apresentação`](entrega/An%C3%A1lise%20Diagn%C3%B3stica%20%26%20Apresenta%C3%A7%C3%A3o) — apresentação em PDF.
- [`Modelo preditivo final`](entrega/An%C3%A1lise%20Preditiva/Modelo_preditivo_Entrega.ipynb) — notebook principal da etapa preditiva.

## Stack

Python · pandas · NumPy · SQL · SQLAlchemy · scikit-learn · SHAP · matplotlib · Jupyter · Power BI

## Como explorar

Os notebooks podem ser lidos diretamente no GitHub. Para executar as análises, abra-os em um ambiente Jupyter e prepare as bibliotecas, as fontes de dados e a conexão de banco utilizadas nas células. A etapa preditiva consulta dados relacionais; os arquivos publicados não constituem um ambiente completo de reprodução automática.

O arquivo `.pbix` exige Power BI Desktop para edição. O PDF permite acompanhar a apresentação sem abrir o relatório no Power BI.
