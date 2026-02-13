import json
import boto3
import time
import os
import logging

# Configuração de Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Inicialização dos Clientes AWS
s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')
transcribe_client = boto3.client('transcribe')
comprehend_client = boto3.client('comprehend')
bedrock_runtime = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    """
    Função Lambda orquestradora para análise multimodal de vídeo e áudio.
    Acionada por upload de vídeo no S3.
    """
    try:
        # 1. Extrair informações do evento S3
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Iniciando análise para o arquivo: {file_key} no bucket: {bucket_name}")

        # 2. Iniciar Amazon Rekognition (Análise de Vídeo - Emoções)
        rekognition_job_id = start_video_analysis(bucket_name, file_key)
        
        # 3. Iniciar Amazon Transcribe (Análise de Áudio - Transcrição)
        transcribe_job_name = start_transcription(bucket_name, file_key)
        
        # 4. Aguardar e Coletar Resultados do Rekognition
        video_analysis_results = get_video_analysis_results(rekognition_job_id)
        
        # 5. Aguardar e Coletar Resultados do Transcribe
        transcription_text = get_transcription_results(transcribe_job_name)
        
        # 6. Amazon Comprehend (Análise de Texto)
        text_analysis = analyze_text(transcription_text)
        
        # 7. Amazon Bedrock (Fusão Multimodal e Geração de Relatório)
        final_report = generate_multimodal_report(video_analysis_results, transcription_text, text_analysis)
        
        # 8. Salvar Relatório Final no S3
        report_key = f"reports/{file_key.split('/')[-1].split('.')[0]}_report.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=report_key,
            Body=json.dumps(final_report, indent=4, ensure_ascii=False),
            ContentType='application/json'
        )
        
        logger.info(f"Análise concluída com sucesso. Relatório salvo em: {report_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Análise concluída', 'report_url': report_key})
        }

    except Exception as e:
        logger.error(f"Erro na execução da Lambda: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def start_video_analysis(bucket, key):
    """Inicia o job de detecção de faces e emoções no Rekognition Video."""
    response = rekognition_client.start_face_detection(
        Video={'S3Object': {'Bucket': bucket, 'Name': key}},
        FaceAttributes='ALL'
    )
    return response['JobId']

def get_video_analysis_results(job_id):
    """Aguarda a conclusão do job do Rekognition e retorna as emoções detectadas."""
    while True:
        response = rekognition_client.get_face_detection(JobId=job_id)
        status = response['JobStatus']
        if status == 'SUCCEEDED':
            # Simplificar resultados para focar em emoções predominantes
            emotions_summary = []
            for face in response['Faces']:
                if 'Emotions' in face['Face']:
                    # Pega a emoção com maior confiança
                    top_emotion = max(face['Face']['Emotions'], key=lambda x: x['Confidence'])
                    emotions_summary.append({
                        'Timestamp': face['Timestamp'],
                        'Emotion': top_emotion['Type'],
                        'Confidence': top_emotion['Confidence']
                    })
            return emotions_summary
        elif status == 'FAILED':
            raise Exception("Rekognition Video analysis failed")
        time.sleep(5)

def start_transcription(bucket, key):
    """Inicia o job de transcrição no Amazon Transcribe."""
    job_name = f"transcription_{int(time.time())}"
    media_uri = f"s3://{bucket}/{key}"
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': media_uri},
        LanguageCode='pt-BR'
    )
    return job_name

def get_transcription_results(job_name):
    """Aguarda a conclusão do job do Transcribe e retorna o texto transcrito."""
    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
            # O Transcribe retorna uma URL assinada para o JSON do resultado
            import urllib.request
            import json
            with urllib.request.urlopen(transcript_uri) as response:
                transcript_json = json.load(response)
            return transcript_json['results']['transcripts'][0]['transcript']
        elif status == 'FAILED':
            raise Exception("Transcription job failed")
        time.sleep(5)

def analyze_text(text):
    """Realiza análise de sentimento usando Amazon Comprehend."""
    if not text: return {}
    sentiment_response = comprehend_client.detect_sentiment(Text=text, LanguageCode='pt')
    return {
        'Sentiment': sentiment_response['Sentiment'],
        'Scores': sentiment_response['SentimentScore']
    }

def generate_multimodal_report(video_data, transcript, text_analysis):
    """Utiliza Amazon Bedrock (Claude 3) para realizar a fusão multimodal e gerar o relatório."""
    
    # Preparar o prompt para o Bedrock
    prompt_text = f"""
    Você é um assistente especializado em saúde da mulher e segurança. 
    Analise os seguintes dados multimodais coletados de um atendimento e identifique sinais de risco de violência doméstica.

    DADOS DE VÍDEO (Emoções detectadas):
    {json.dumps(video_data[:10])} (amostra dos primeiros 10 frames)

    DADOS DE ÁUDIO (Transcrição da fala):
    "{transcript}"

    DADOS DE TEXTO (Análise de sentimento):
    {json.dumps(text_analysis)}

    Com base nesses dados, gere um relatório estruturado em Português contendo:
    1. Score de Risco (0 a 100)
    2. Nível de Risco (Baixo, Moderado, Alto, Crítico)
    3. Resumo das evidências encontradas (Vídeo, Áudio e Texto)
    4. Recomendações clínicas e de segurança.
    5. Disclaimer ético.
    """

    # Payload para o Claude 3 no Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
    })

    model_id = "anthropic.claude-3-haiku-20240307-v1:0" # Modelo econômico e rápido

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')
