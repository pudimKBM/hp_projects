# Seletores CSS Identificados no Mercado Livre

## Página de Busca
- Container de resultados: elementos com links contendo "/p/MLB" ou "mercadolivre.com.br/"
- Links de produtos: elementos <a> com href contendo "/p/MLB" ou "/MLB-"

## Página de Produto
- Título: h1 ou elemento com texto principal do produto
- Preço: elementos com classes relacionadas a preço (R$ 66,90)
- Avaliação: elemento com texto "(3317)" ou similar
- Vendedor: elemento com texto "Vendido por" seguido do nome
- Descrição: conteúdo da página de produto
- Especificações: informações técnicas listadas

## Observações
- A estrutura do Mercado Livre usa muito JavaScript dinâmico
- Seletores podem variar entre produtos
- Necessário usar Selenium para aguardar carregamento completo
- Implementar fallbacks para diferentes estruturas de página

