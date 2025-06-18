# Projeto de Análise e Observabilidade de Produtos HP

Este repositório contém o projeto completo desenvolvido para o Challenge Sprint - HP, abrangendo desde a coleta de dados (web scraping) até a análise exploratória, rotulagem heurística e uma plataforma de observabilidade. O projeto está organizado em módulos, cada um com suas dependências e instruções de execução.

## 📁 Estrutura do Repositório

O projeto está organizado na seguinte estrutura de pastas:

```
. (diretório raiz)
├── backend/             # Código do backend (Flask) da plataforma de observabilidade
├── data/                # Datasets CSV (limpos, enriquecidos, mockados)
├── logs/                # Arquivos de log das execuções
├── misc/                # Arquivos diversos (READMEs específicos, todo.md, etc.)
├── notebooks/           # Jupyter Notebooks (IPYNB e HTML) para EDA
├── presentations/       # Apresentações (HTML) geradas
├── reports/             # Relatórios (Markdown e PDF)
├── scripts/             # Scripts Python (scrapers, limpeza, EDA, rotulagem, etc.)
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

Cada módulo do projeto possui suas próprias dependências e instruções de execução. Certifique-se de estar no diretório raiz do projeto (`/home/ubuntu/`) antes de executar os comandos.




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

## 🏁 Como Rodar o Projeto Completo (Passo a Passo)

Para rodar o projeto completo do zero, siga os passos abaixo na ordem:

1.  **Configuração Inicial do Ambiente:** Siga as instruções na seção "Configuração Inicial do Ambiente" para instalar Python, Node.js e pnpm.

2.  **Web Scraping (Opcional - Use dados mockados para agilizar):**
    *   Execute `python3 scripts/generate_demo_data.py` para gerar dados mockados do Mercado Livre.
    *   Execute `python3 scripts/generate_americanas_mock_data.py` para gerar dados mockados da Americanas.

3.  **Processamento de Dados:**
    *   Execute `python3 scripts/data_cleaning.py`.
    *   Execute `python3 scripts/data_enrichment.py`.
    *   Execute `python3 scripts/data_integration.py`.

4.  **Rotulagem Heurística:**
    *   Execute `python3 scripts/heuristic_labeling.py`.
    *   Execute `python3 scripts/dataset_generation.py`.

5.  **Análise Exploratória de Dados (EDA):**
    *   Navegue até a pasta `notebooks/` e inicie o Jupyter Notebook (`jupyter notebook`).
    *   Abra e execute o `eda_notebook.ipynb` célula por célula.

6.  **Plataforma de Observabilidade:**
    *   **Backend:** Siga as instruções de instalação e execução do backend (`backend/hp_observability_backend`).
    *   **Frontend:** Siga as instruções de instalação e execução do frontend (`hp-observability-frontend`).

Ao final, você terá todos os componentes do projeto rodando localmente, e poderá acessar a plataforma de observabilidade em `http://localhost:5173` (ou a porta que o Vite indicar).



