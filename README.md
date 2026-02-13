# 🛡️ Tech Challenge 4  
## Sistema de Detecção de Violência Doméstica com IA Multimodal (AWS Serverless)

---

## 📌 Visão Geral

Este projeto implementa um sistema de **Inteligência Artificial Multimodal** para detecção de possíveis sinais de violência doméstica em contextos relacionados à saúde da mulher, utilizando uma arquitetura totalmente **serverless na Amazon Web Services (AWS)**.

A solução realiza análise integrada de **vídeo, áudio e texto**, combinando reconhecimento de emoções faciais, análise de sentimento e interpretação contextual para gerar:

- 📊 **Score de risco (0–100)**
- 🧠 **Classificação do nível de risco**
- 📝 **Relatório analítico detalhado**
- 🎥 **Frames relevantes que fundamentam a análise**

O objetivo é demonstrar como a IA pode apoiar processos de triagem e avaliação de risco de forma escalável, automatizada e auditável.

---

## 🚀 Principais Funcionalidades

### 🎥 Análise de Vídeo
Utiliza **Amazon Rekognition** para:
- Detecção de faces
- Identificação de emoções faciais
- Extração de timestamps relevantes

### 🎙️ Análise de Áudio
Utiliza **Amazon Transcribe** para:
- Conversão de fala em texto (pt-BR)

### 📖 Análise de Texto
Utiliza **Amazon Comprehend** para:
- Análise de sentimento
- Identificação de polaridade emocional

### 🧠 Fusão Multimodal
Utiliza **Amazon Bedrock (Claude 3)** para:
- Interpretação integrada dos dados de vídeo, áudio e texto
- Geração de score de risco
- Produção de relatório explicativo
- Justificativa baseada em evidências visuais e textuais

### 🖼️ Evidências Visuais
- Extração automática de frames do vídeo utilizando FFmpeg
- Exibição de frames relevantes alinhados ao nível de risco identificado

---

## 🏗️ Arquitetura

A solução foi construída utilizando uma arquitetura **100% serverless**, garantindo escalabilidade, baixo custo operacional e simplicidade de manutenção.

### Serviços AWS Utilizados:

- **Amazon S3** – Armazenamento de vídeos, relatórios e frames  
- **AWS Lambda** – Orquestração do pipeline de processamento  
- **Amazon API Gateway** – Exposição de endpoints HTTP  
- **Amazon Rekognition** – Análise de vídeo  
- **Amazon Transcribe** – Transcrição de áudio  
- **Amazon Comprehend** – Análise de sentimento  
- **Amazon Bedrock (Claude 3)** – Geração de relatório e análise multimodal  
- **Amazon Cognito Identity** – Autenticação no frontend  
- **Python (Boto3)** – Integração com serviços AWS  
- **HTML / JavaScript** – Interface do usuário  

---

## 🧪 Como Utilizar

1. Acesse a URL disponibilizada no PDF do Tech Challenge.
2. Faça upload de um vídeo em formato `.mp4`.
3. Aguarde o processamento da análise multimodal.
4. Visualize:
   - Score de risco
   - Relatório detalhado
   - Frames relevantes que embasam a avaliação

---

## 📁 Vídeos de Exemplo

O repositório contém a pasta: /video_samples

Com vídeos utilizados nos testes do sistema:

- `relato_real_1.mp4` – Trecho de relato real extraído do YouTube  
- `relato_real_2.mp4` – Trecho de relato real extraído do YouTube  
- `relato_IA_risco_alto.mp4` – Vídeo gerado por IA (HeyGen) simulando alto risco  
- `relato_IA_risco_medio.mp4` – Vídeo gerado por IA (HeyGen) simulando risco médio  
- `relato_IA_risco_baixo.mp4` – Vídeo gerado por IA (HeyGen) simulando baixo risco  

Esses arquivos podem ser utilizados para validação e demonstração do funcionamento do sistema.

---

## ⚠️ Observações

- O sistema tem finalidade educacional e demonstrativa.
- Não substitui avaliação profissional especializada.
- Os vídeos gerados por IA foram utilizados para simulação controlada de cenários.