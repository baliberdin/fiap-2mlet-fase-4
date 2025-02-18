# Projeto para o módulo 4 da Pós Tech FIAP
### Machine Learning Engineering

## O projeto
O projeto consiste em um conjunto de aplicações para demonstrar a criação, teste e acompanhamento de um modelo LSTM. Dentre os componentes do projeto temos:
- API de predição e histórico de valores de ações da Petrobras
- Client APP - Uma aplicação client (apenas para consumo da API e exibição dos dados)
- Modelo LSTM
- Containers auxiliares - MlflowServer, MySQL, Prometheus e Grafana
- Scripts: Makefile para automação e scripts de treino do modelo.

## Como funciona?
O modelo LSTM é criado e treinado. Durante o processo de treino é utilizando GridSearch e cross validation para encontrar os melhores hiperparâmetros. Todos os resultados de treinos são enviados para o MlflowServer onde podemos comparar o resultados dos testes e encontrar qual a melhor versão do nosso modelo. O servidor do Mlflow também serve como repositório de artefatos onde tanto o modelo como o scaler são armazenados.

A API de predição, por sua vez, é configurada para utilizar uma versão específica do Modelo já armazenado no MlflowServer. Durante o startup da aplicação o modelo é serializado e fica disponível para as consultas.
O banco de dados MySQL é exclusivo da API servindo para armazenar os valores históricos de valores reais e valores preditos pelo Modelo.
A API também conta com um job interno que verifica automaticamente (em um intervalo configurado de tempo) dados novos sobre as ações utilizando o Yahoo Finance. A cada nova entrada de dados reais uma nova predição é armazenada no banco.

O endpoint principal da API (http://localhost:8000) é utilizado para retornar os dados históricos e a previsão dos valores futuros, para isso, o modelo faz um *roll forward prediction* onde a previsão de D+1 é utilizada para compor a previsão de D+2 e assim por diante. Apenas as previsões D+1 são armazenadas no banco de dados.

A aplicação client é apenas um front-end onde se pode observar os dados reais, históricos e as previsões futuras.

## Pré-requisitos
- Docker
- Python 3.10
- Makefile

## Containers
- stock-prediction-api - http://localhost:8000
- mysql - mysql://localhost:3306
- client-app - http://localhost:3001
- prometheus - http://localhost:9090
- grafana - http://localhost:3000
- mlflow-server - http://localhost:5000

## Automações

Este projeto utiliza tasks de Makefile para automatizar algumas operações, para a lista completa vide o arquivo Makefile
- **run-all**: Inicia todos os containers de uma só vez. (Depende das imagens docker)
- **build-api**: Cria a imagem docker da API
- **build-client-app**: Cria a imagem docker do Client App
- **rebuild-api**: Recria a imagem da API e restarta o container
- **run-cross-validation-training**: Executa uma bateria de treinos combinando os hiperparâmetros contidos no script e publicando todos os artefatos no Mlflow Server

Para executar qualquer uma das tasks utilize um terminal bash na raiz do projeto e execute, por exemplo:
```sh
make run-all
```

## Rodando localmente
1. Inicie o mlflow-server com a task do makefile:
```sh
make run-mlflow-server
```
Certifique-se de que o mlflow-server foi iniciado acessando http://localhost:5000

2. Crie e ative o seu virtualenv local utilizando um terminal na raiz do projeto
```sh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Inicie o treino do modelo:
```sh
make run-cross-validation-training
```
Essa task deve durar muito tempo, mas você não precisa esperar até a sua finalização, ao invéz disso, vá até o mlflow-server e verifique se a primeira versão do modelo foi publicada (http://localhost:5000/#/models), se sim, você pode interromper esse script.

4. Construa as imagens dos containers
```sh
make build-api
make build-client-app
```

5. Inicie o restante dos containers e aguarde até que o container da API (stock-predictions-api) esteja iniciado por completo. Durante a primeira inicialização o job interno da API vai importar os dados do Yahoo Finance e isso deve demorar alguns minutos. Você pode observar os logs com **docker logs -f stock-predictions-api** assim que os logs de import pararem e o endereço da porta do servidor for exibido você já poderá passar para o próximo passo.
```sh
make run-all
```

6. Abra o client App (http://localhost:3001)

7.  Para acompanhar as métricas no grafana basta acessar http://localhost:3000 e fazer o login (username: admin, password: admin) e então acessar a sessão de dashboards, no menu lateral, e então acessar o dashboard **Stock Preiction API**


