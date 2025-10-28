# INTELIGÊNCIA ARTIFICIAL - 2025.1

## Table of Contents
- [Introdução](#introdução)
- [Como rodar o projeto](#como-rodar-o-projeto)

## Introdução

Este repositório contém códigos referentes à implementação de uma àrvore de decisões comum e de uma random forest juntamente com a comparação da acurácia de ambas as implementações, contando com testes em uma base de dados fictícia e uma base de dados real sobre câncer de mama.

## Como rodar o projeto

1. Crie um ambiente virtual Python:
     ```bash
     python3 -m venv venv
     ```
2. Ative o ambiente virtual:
     - Linux/macOS:
         ```bash
         source venv/bin/activate
         ```
     - Windows:
         ```bash
         venv\Scripts\activate
         ```
3. Instale as dependências:
     ```bash
     pip install -r requirements.txt
     ```
4. Navegue até a pasta do código:
     ```bash
     cd caminho/para/o/repositorio/
     ```
5. Execute o arquivo desejado:
     ```bash
     python3 nome-do-arquivo.py
     ```

Obs.: caso esteja rodando o código utilizando o wsl provavelmente não será plotado um gráfico, pelo fato do wsl não fornecer um suporte direto a ferramentas visuais, como o matplotlib, é recomendado rodar o código diretamente no seu sistema operacioal ou dentro de uma máquina virtual com suporte a ferramentas visuais.
