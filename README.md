# tech-challenge-4: Sistema de Detecção de Violência Doméstica com IA Multimodal (AWS Serverless)

## Visão Geral

Este projeto implementa um sistema de Inteligência Artificial (IA) multimodal para detecção de sinais de violência doméstica em contextos de saúde da mulher, utilizando uma arquitetura **serverless** na Amazon Web Services (AWS). A solução analisa vídeos e áudios para identificar padrões comportamentais e emocionais indicativos de abuso, gerando um relatório clínico detalhado.

## Características Principais

-   **Análise de Vídeo**: Utiliza **Amazon Rekognition** para detecção de emoções faciais e atividades.
-   **Análise de Áudio**: Converte áudio em texto com **Amazon Transcribe**.
-   **Análise de Texto**: Analisa sentimento e entidades na transcrição com **Amazon Comprehend**.
-   **Fusão Multimodal e Geração de Relatório**: Utiliza **Amazon Bedrock** (com Claude 3) para interpretar os resultados de vídeo, áudio e texto, gerando um score de risco e um relatório clínico.
-   **Arquitetura Serverless**: Implementado com **AWS Lambda**, **S3**, **API Gateway** e **Cognito Identity** para escalabilidade, baixo custo e fácil manutenção.
-   **Frontend Simples**: Uma página HTML/JS para upload de vídeos e visualização dos resultados.

## Arquitetura

```mermaid
graph TD
    A[Frontend Web (HTML/JS)] --> B(API Gateway: Upload Video);
    B --> C[AWS S3: Video Upload Bucket];
    C -- Evento: New Object Created --> D(AWS Lambda: Orchestrator Function);

    subgraph Análise de Vídeo
        D -- Chama --> E[Amazon Rekognition: StartFaceDetection];
        E -- Resultados Assíncronos --> F[AWS S3: Rekognition Output Bucket];
    end

    subgraph Análise de Áudio
        D -- Extrai Áudio (via FFmpeg na Lambda) --> G[AWS S3: Audio Extract Bucket];
        G -- Chama --> H[Amazon Transcribe: StartTranscriptionJob];
        H -- Resultados Assíncronos --> I[AWS S3: Transcribe Output Bucket];
    end

    subgraph Análise de Texto
        D -- Lê Transcrição do S3 --> J[Amazon Comprehend: DetectSentiment/Entities];
    end

    subgraph Fusão Multimodal e Geração de Relatório
        D -- Agrega Resultados --> K[Amazon Bedrock: InvokeModel (Claude 3)];
        K -- Gera Relatório Clínico --> L[AWS S3: Report Output Bucket];
    end

    L --> M[API Gateway: Get Report];
    M --> A;
```

## Tecnologias Utilizadas

-   **AWS S3**: Armazenamento de objetos.
-   **AWS Lambda**: Computação serverless para orquestração.
-   **Amazon API Gateway**: Exposição de endpoints HTTP.
-   **Amazon Rekognition**: Análise de vídeo (detecção de faces e emoções).
-   **Amazon Transcribe**: Transcrição de áudio para texto.
-   **Amazon Comprehend**: Análise de sentimento e entidades em texto.
-   **Amazon Bedrock**: Geração de texto (Claude 3) para fusão multimodal e relatórios.
-   **Amazon Cognito Identity**: Autenticação para acesso ao S3 via frontend.
-   **Python (Boto3)**: SDK para interação com serviços AWS.
-   **HTML/JavaScript**: Frontend para interação do usuário.