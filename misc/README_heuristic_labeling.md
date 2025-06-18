# Projeto de Rotulagem Heurística de Produtos HP

## Descrição do Projeto

Este projeto implementa um sistema de rotulagem heurística para classificar produtos HP como "originais" ou "suspeitos/piratas" com base em critérios automatizados. O objetivo é criar um dataset estruturado para aplicações de Machine Learning e Processamento de Linguagem Natural.

## Estrutura do Projeto

### Scripts Principais

- `heuristic_labeling.py` - Script principal para aplicação dos critérios heurísticos
- `dataset_generation.py` - Geração do dataset estruturado com features selecionadas
- `hp_products_labeled_*.csv` - Dataset com rótulos aplicados
- `hp_products_structured_dataset_*.csv` - Dataset final estruturado para ML

### Apresentação

- Apresentação interativa documentando todo o processo, critérios e resultados
- 9 slides cobrindo desde objetivos até conclusões
- Inclui 5 exemplos práticos de dados rotulados

## Critérios Heurísticos Implementados

### Produtos Originais
- Vendedores oficiais (HP Oficial, Loja Oficial HP, etc.)
- Palavras-chave de autenticidade (original, genuíno, lacrado, nota fiscal)
- Faixa de preço premium (> R$ 200)
- Score de confiança: 0.3 a 1.0

### Produtos Suspeitos
- Palavras-chave suspeitas (compatível, genérico, remanufaturado)
- Preços muito baixos (< R$ 30)
- Avaliações inconsistentes (rating baixo com muitas avaliações)
- Vendedores genéricos
- Score de confiança: 0.1 a 0.4

## Features do Dataset

### Features Textuais
- `title` - Título do produto
- `description` - Descrição detalhada
- `seller_name` - Nome do vendedor
- `platform` - Plataforma de origem
- `product_type` - Tipo do produto

### Features Numéricas
- `price_numeric` - Preço normalizado
- `rating_numeric` - Avaliação (0-5)
- `reviews_count` - Número de avaliações
- `confidence` - Score de confiança do algoritmo

### Features Booleanas (Derivadas)
- `is_original` - Indica se é original
- `is_xl` - Versão XL do cartucho
- `has_sealed_info` - Menciona produto lacrado
- `has_invoice_info` - Menciona nota fiscal
- `has_warranty_info` - Menciona garantia

### Target Variable
- `target_is_original` - Variável target binária (1 = original, 0 = suspeito)

## Resultados

- **Total de produtos analisados:** 10
- **Produtos classificados como originais:** 5 (50%)
- **Produtos classificados como suspeitos:** 1 (10%)
- **Produtos com sinais mistos:** 2 (20%)
- **Produtos não classificados:** 2 (20%)

## Execução

```bash
# Aplicar rotulagem heurística
python3 heuristic_labeling.py

# Gerar dataset estruturado
python3 dataset_generation.py
```

## Próximos Passos

1. **Machine Learning:** Treinar modelos supervisionados com o dataset rotulado
2. **PLN Avançado:** Análise de sentimento e extração de entidades
3. **Expansão de Dados:** Coletar mais produtos e outras categorias
4. **Validação:** Teste com especialistas e refinamento dos critérios

## Arquivos Gerados

- `hp_products_labeled_20250615_233023.csv` - Dataset com rótulos aplicados
- `hp_products_structured_dataset_20250615_233410.csv` - Dataset estruturado final
- Apresentação interativa com 9 slides documentando o processo completo

