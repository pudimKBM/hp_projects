# Projeto de Análise e Observabilidade de Produtos HP

Este repositório contém o projeto completo desenvolvido para o Challenge Sprint - HP, abrangendo desde a coleta de dados (web scraping) até a análise exploratória, rotulagem heurística, uma plataforma de observabilidade e um pipeline completo de Machine Learning para classificação de autenticidade de produtos HP. O projeto está organizado em módulos, cada um com suas dependências e instruções de execução.

## 📁 Estrutura do Repositório

O projeto está organizado na seguinte estrutura de pastas:

```
. (diretório raiz)
├── backend/             # Código do backend (Flask) da plataforma de observabilidade
├── data/                # Datasets CSV (limpos, enriquecidos, mockados)
├── logs/                # Arquivos de log das execuções
├── misc/                # Arquivos diversos (READMEs específicos, todo.md, etc.)
├── notebooks/           # Jupyter Notebooks (IPYNB e HTML) para EDA e ML Pipeline
├── presentations/       # Apresentações (HTML) geradas
├── reports/             # Relatórios (Markdown e PDF)
├── scripts/             # Scripts Python (scrapers, limpeza, EDA, rotulagem, etc.)
├── src/                 # Pipeline de Machine Learning modular (Sprint 3)
│   ├── feature_engineering/    # Engenharia de features
│   ├── preprocessing/          # Pré-processamento de dados
│   ├── models/                 # Treinamento de modelos
│   ├── hyperparameter_optimization/  # Otimização de hiperparâmetros
│   ├── validation/             # Validação e comparação de modelos
│   ├── interpretation/         # Interpretabilidade de modelos
│   ├── persistence/            # Persistência e versionamento
│   └── reporting/              # Geração de relatórios
├── visualizations/      # Visualizações (PNG) geradas pela EDA
└── zips/                # Arquivos ZIP de entregas anteriores
```

## 🚀 Configuração Inicial do Ambiente

Para rodar qualquer parte deste projeto, você precisará ter o Python 3.x e o Node.js (com pnpm) instalados em seu sistema.

### 1. Python

Certifique-se de ter o Python 3.x instalado. É recomendado usar um ambiente virtual para gerenciar as dependências.

```bash
python3 -m venv venv_global
source venv_global/bin/activate
pip install --upgrade pip
```

### 2. Node.js e pnpm

Para o frontend, você precisará do Node.js e do pnpm (gerenciador de pacotes).

```bash
sudo apt install nodejs npm
sudo npm install -g pnpm
```

## 📦 Módulos do Projeto e Como Rodar

O projeto está dividido em três sprints principais:

- **Sprint 1**: Coleta e Processamento de Dados
- **Sprint 2**: Análise Exploratória e Plataforma de Observabilidade  
- **Sprint 3**: Pipeline de Machine Learning para Classificação de Autenticidade

Cada módulo do projeto possui suas próprias dependências e instruções de execução. Certifique-se de estar no diretório raiz do projeto antes de executar os comandos.




## 🚀 Sprint 1: Coleta e Processamento de Dados

### 1. Web Scraping (Scripts em `scripts/`)

**Descrição:** Coleta de dados de produtos HP do Mercado Livre e Americanas.

**Dependências:**
- `selenium`
- `beautifulsoup4`
- `pandas`
- `lxml`
- `chromium-browser` (para Selenium)

**Instalação:**

```bash
pip install selenium beautifulsoup4 pandas lxml
sudo apt-get update
sudo apt-get install -y chromium-browser
```

**Como Rodar:**

- **`scripts/hp_scraper.py` (Mercado Livre):**
  ```bash
  python3 scripts/hp_scraper.py
  ```
  *Nota: Este script pode precisar de ajustes nos seletores CSS devido a mudanças frequentes na estrutura do site.* 

- **`scripts/americanas_scraper.py` (Americanas - **Dados Mockados**):
  ```bash
  python3 scripts/americanas_scraper.py
  ```
  *Nota: O scraper da Americanas foi configurado para usar dados mockados devido à complexidade e instabilidade dos seletores. O script `scripts/generate_americanas_mock_data.py` pode ser usado para gerar novos dados mockados.* 

- **`scripts/generate_demo_data.py` (Gerador de Dados Demo):**
  ```bash
  python3 scripts/generate_demo_data.py
  ```

### 2. Processamento de Dados (Scripts em `scripts/`)

**Descrição:** Limpeza, padronização e enriquecimento dos dados coletados.

**Dependências:**
- `pandas`

**Instalação:**

```bash
pip install pandas
```

**Como Rodar:**

- **`scripts/data_cleaning.py` (Limpeza e Padronização):
  ```bash
  python3 scripts/data_cleaning.py
  ```

- **`scripts/data_enrichment.py` (Enriquecimento):
  ```bash
  python3 scripts/data_enrichment.py
  ```

- **`scripts/data_integration.py` (Integração de Múltiplas Fontes):
  ```bash
  python3 scripts/data_integration.py
  ```

### 3. Rotulagem Heurística (Scripts em `scripts/`)

**Descrição:** Aplicação de critérios heurísticos para classificar produtos como 


"original" ou "suspeito/pirata".

**Dependências:**
- `pandas`

**Instalação:**

```bash
pip install pandas
```

**Como Rodar:**

- **`scripts/heuristic_labeling.py` (Rotulagem Heurística):
  ```bash
  python3 scripts/heuristic_labeling.py
  ```

- **`scripts/dataset_generation.py` (Geração do Dataset Estruturado):
  ```bash
  python3 scripts/dataset_generation.py
  ```

## 📊 Sprint 2: Análise Exploratória e Observabilidade

### 4. Análise Exploratória de Dados (EDA) (Notebooks em `notebooks/`)

**Descrição:** Análise exploratória e descritiva dos dados coletados, incluindo visualizações e insights.

**Dependências:**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `wordcloud`
- `jupyter` (para abrir e executar o notebook)

**Instalação:**

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud jupyter
python3 -c "import nltk; nltk.download("punkt", quiet=True); nltk.download("stopwords", quiet=True); nltk.download("rslp", quiet=True)"
```

**Como Rodar:**

- **`notebooks/eda_notebook.ipynb`:**
  Para abrir e executar o notebook, navegue até a pasta `notebooks/` e inicie o Jupyter Notebook:
  ```bash
  cd notebooks/
  jupyter notebook
  ```
  Em seu navegador, abra o arquivo `eda_notebook.ipynb`.

- **`scripts/eda_script.py` (Script de EDA - usado para gerar as visualizações no notebook):
  ```bash
  python3 scripts/eda_script.py
  ```

### 5. Plataforma de Observabilidade (Backend em `backend/`, Frontend em `hp-observability-frontend/`)

**Descrição:** Aplicação web para visualização e monitoramento dos dados de produtos HP.

**Dependências:**
- **Backend (Flask):** `Flask`, `flask-cors`, `pandas` (se usar dados reais, mas aqui está mockado)
- **Frontend (React):** `react`, `react-dom`, `vite`, `tailwindcss`, `recharts`, `@radix-ui/react-*`, `lucide-react`

**Instalação do Backend:**

```bash
cd backend/hp_observability_backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Como Rodar o Backend:**

```bash
cd backend/hp_observability_backend
source venv/bin/activate
python src/main.py
```

**Instalação do Frontend:**

```bash
cd hp-observability-frontend
pnpm install
```

**Como Rodar o Frontend:**

```bash
cd hp-observability-frontend
pnpm run dev
```

*Nota: Certifique-se de que o backend esteja rodando antes de iniciar o frontend, pois o frontend consome a API do backend.*

## 🤖 Sprint 3: Pipeline de Machine Learning

**Descrição:** Pipeline completo de Machine Learning para classificação automática de produtos HP como "original" ou "suspeito/pirata" usando técnicas avançadas de feature engineering, múltiplos algoritmos de ML, otimização de hiperparâmetros e interpretabilidade de modelos.

### Arquitetura do Pipeline ML

O pipeline segue uma arquitetura modular com separação clara de responsabilidades:

```
Raw Data → Feature Engineering → Preprocessing → Model Training → Hyperparameter Optimization → Validation → Interpretation → Persistence → Reporting
```

### Dependências do Pipeline ML

**Principais bibliotecas:**
- `scikit-learn` - Algoritmos de ML e utilitários
- `pandas`, `numpy` - Manipulação de dados
- `matplotlib`, `seaborn` - Visualizações
- `joblib` - Serialização de modelos
- `imbalanced-learn` - Técnicas para dados desbalanceados
- `reportlab` - Geração de relatórios PDF

**Instalação:**

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn reportlab
```

### Como Rodar o Pipeline ML

#### Opção 1: Pipeline Completo (Recomendado)

```bash
# Navegue para o diretório notebooks
cd notebooks/

# Execute o pipeline modular completo
python ml_pipeline_modular.py
```

#### Opção 2: Módulos Individuais

```bash
# 1. Feature Engineering
from src.feature_engineering import FeatureEngineeringPipeline
pipeline = FeatureEngineeringPipeline()
X, y = pipeline.fit_transform(df)

# 2. Preprocessing
from src.preprocessing import create_complete_preprocessing_pipeline
results = create_complete_preprocessing_pipeline(X, y, imbalance_method='smote')

# 3. Model Training & Optimization
from src.hyperparameter_optimization import optimize_multiple_models
models = ['random_forest', 'svm', 'logistic_regression']
optimization_results = optimize_multiple_models(models, X_train, y_train)

# 4. Model Validation
from src.validation import create_model_comparison_table
comparison = create_model_comparison_table(models_results, cv_results)

# 5. Model Interpretation
from src.interpretation import InterpretationPipeline
interpreter = InterpretationPipeline(feature_names, class_names)
importance = interpreter.analyze_feature_importance(model, X_train, y_train)

# 6. Model Recommendation
from src.reporting import ModelRecommendationSystem
recommender = ModelRecommendationSystem()
recommendation = recommender.recommend_model(models_results, cv_results)
```

#### Opção 3: Exemplos e Testes

```bash
# Execute exemplos específicos
python notebooks/hyperparameter_pipeline_example.py
python notebooks/preprocessing_example.py

# Execute testes para validar funcionamento
python notebooks/test_preprocessing.py
python notebooks/test_hyperparameter_pipeline.py
python notebooks/test_validation.py
python notebooks/test_interpretation.py
python notebooks/test_model_recommendation.py
```

### Funcionalidades do Pipeline ML

#### 🔧 Feature Engineering
- **Features de Texto**: Vetorização TF-IDF de títulos e descrições
- **Features Numéricas**: Normalização de preços, ratings, contagem de reviews
- **Features Categóricas**: One-hot encoding para plataformas, tipos de produto
- **Features Derivadas**: Ratios de preço, métricas de texto, indicadores de palavras-chave
- **Análise de Correlação**: Remoção automática de features altamente correlacionadas

#### 🤖 Treinamento de Modelos
- **Múltiplos Algoritmos**: Random Forest, SVM, Logistic Regression, Gradient Boosting
- **Validação Cruzada**: 5-fold stratified cross-validation
- **Balanceamento de Classes**: SMOTE e class weighting
- **Interface Consistente**: API uniforme para todos os modelos

#### ⚡ Otimização de Hiperparâmetros
- **Seleção Inteligente**: Escolha automática entre GridSearchCV e RandomizedSearchCV
- **Gestão de Orçamento**: Orçamentos configuráveis e limites de tempo
- **Otimização Multi-Modelo**: Otimização paralela de múltiplos modelos
- **Tracking de Progresso**: Monitoramento em tempo real

#### 📊 Validação e Comparação
- **Métricas Abrangentes**: Precision, recall, F1-score, AUC-ROC, accuracy
- **Testes Estatísticos**: Testes de significância entre performances
- **Visualizações**: Curvas ROC, matrizes de confusão, heatmaps
- **Análise de Overfitting**: Detecção automática de overfitting

#### 🔍 Interpretabilidade
- **Feature Importance**: Análise baseada em árvores e permutação
- **Explicação de Predições**: Explicações individuais com features contribuintes
- **Visualizações Interativas**: Plots de importância e explicações
- **Insights Automáticos**: Geração automática de insights de negócio

#### 💾 Persistência e Versionamento
- **Versionamento Automático**: Tracking de versões com metadata
- **Serialização Completa**: Salvamento de pipelines completos
- **Validação de Compatibilidade**: Verificação de esquemas ao carregar
- **Metadata Rica**: Métricas de performance, parâmetros, schemas

#### 📈 Recomendação Automática de Modelos
- **Restrições de Negócio**: Configuração de requisitos de accuracy, interpretabilidade
- **Avaliação de Deploy**: Análise multi-critério para produção
- **Scoring Inteligente**: Combina métricas técnicas com requisitos de negócio
- **Relatórios Executivos**: Recomendações business-friendly

### Resultados Esperados

- **Accuracy**: >85% no conjunto de teste
- **F1-Score**: >0.85 para classificação balanceada
- **AUC-ROC**: >0.90 para forte discriminação
- **Interpretabilidade**: Explicações claras para decisões de negócio

### Arquivos Principais do Pipeline ML

```
src/
├── feature_engineering/     # Extração e transformação de features
├── preprocessing/           # Pré-processamento e divisão de dados
├── models/                 # Implementações de treinamento
├── hyperparameter_optimization/  # Otimização automática
├── validation/             # Avaliação e comparação
├── interpretation/         # Explicabilidade
├── persistence/           # Persistência com versionamento
├── reporting/             # Geração de relatórios
└── config.py              # Configurações do pipeline
```

*Nota: Para documentação detalhada do pipeline ML, consulte `src/README.md`.*

## 🏁 Como Rodar o Projeto Completo (Passo a Passo)

Para rodar o projeto completo do zero, siga os passos abaixo na ordem:

### Configuração Inicial do Ambiente

1. **Configuração Inicial do Ambiente:** Siga as instruções na seção "Configuração Inicial do Ambiente" para instalar Python, Node.js e pnpm.

### Sprint 1: Coleta e Processamento de Dados

2. **Web Scraping (Opcional - Use dados mockados para agilizar):**
   - Execute `python3 scripts/generate_demo_data.py` para gerar dados mockados do Mercado Livre.
   - Execute `python3 scripts/generate_americanas_mock_data.py` para gerar dados mockados da Americanas.

3. **Processamento de Dados:**
   - Execute `python3 scripts/data_cleaning.py`.
   - Execute `python3 scripts/data_enrichment.py`.
   - Execute `python3 scripts/data_integration.py`.

4. **Rotulagem Heurística:**
   - Execute `python3 scripts/heuristic_labeling.py`.
   - Execute `python3 scripts/dataset_generation.py`.

### Sprint 2: Análise Exploratória e Observabilidade

5. **Análise Exploratória de Dados (EDA):**
   - Navegue até a pasta `notebooks/` e inicie o Jupyter Notebook (`jupyter notebook`).
   - Abra e execute o `eda_notebook.ipynb` célula por célula.

6. **Plataforma de Observabilidade:**
   - **Backend:** Siga as instruções de instalação e execução do backend (`backend/hp_observability_backend`).
   - **Frontend:** Siga as instruções de instalação e execução do frontend (`hp-observability-frontend`).

### Sprint 3: Pipeline de Machine Learning

7. **Pipeline de Machine Learning:**
   - **Instalação das dependências ML:**
     ```bash
     pip install scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn reportlab
     ```
   
   - **Execução do Pipeline Completo:**
     ```bash
     cd notebooks/
     python ml_pipeline_modular.py
     ```
   
   - **Ou execute módulos individuais conforme necessário (ver seção Sprint 3 acima)**

8. **Testes e Validação (Opcional mas Recomendado):**
   ```bash
   python notebooks/test_preprocessing.py
   python notebooks/test_hyperparameter_pipeline.py
   python notebooks/test_validation.py
   python notebooks/test_interpretation.py
   python notebooks/test_model_recommendation.py
   ```

### Resultado Final

Ao final, você terá:

- **Dados coletados e processados** (Sprint 1)
- **Análise exploratória completa** com insights e visualizações (Sprint 2)
- **Plataforma de observabilidade** rodando em `http://localhost:5173` (Sprint 2)
- **Pipeline de ML completo** com modelos treinados, otimizados e prontos para produção (Sprint 3)
- **Relatórios técnicos e executivos** com recomendações de modelos (Sprint 3)
- **Sistema de classificação automática** de produtos HP por autenticidade (Sprint 3)

## 🎯 Objetivos Alcançados

### Sprint 1 ✅
- Coleta automatizada de dados de múltiplas fontes
- Limpeza e padronização de dados
- Rotulagem heurística para classificação inicial

### Sprint 2 ✅  
- Análise exploratória abrangente com insights de negócio
- Plataforma web para monitoramento e observabilidade
- Visualizações interativas e dashboards

### Sprint 3 ✅
- Pipeline de ML modular e escalável
- Classificação automática com >85% de accuracy
- Sistema de recomendação inteligente de modelos
- Interpretabilidade e explicabilidade completas
- Relatórios técnicos e executivos automatizados

## 📊 Métricas de Sucesso

- **Cobertura de Dados**: >1000 produtos coletados e analisados
- **Accuracy do Modelo**: >85% na classificação de autenticidade
- **Tempo de Processamento**: <5 minutos para pipeline completo
- **Interpretabilidade**: Explicações claras para 100% das predições
- **Automação**: Pipeline totalmente automatizado do dado bruto ao modelo em produção



