import json
import boto3
import time
import logging
import urllib.request
from urllib.parse import unquote_plus

# Configuração de Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Inicialização dos Clientes AWS
s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')
transcribe_client = boto3.client('transcribe')
comprehend_client = boto3.client('comprehend')
bedrock_runtime = boto3.client('bedrock-runtime')

def update_status(bucket, file_id, step, message, details=None):
    """Salva um arquivo de status no S3 para o frontend monitorar."""
    status_data = {
        "step": step,
        "message": message,
        "timestamp": time.time(),
        "details": details,
        "status": "processing" if step != "COMPLETED" else "finished"
    }
    status_key = f"status/{file_id}.json"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=status_key,
            Body=json.dumps(status_data, indent=4, ensure_ascii=False),
            ContentType='application/json'
        )
    except Exception as e:
        logger.error(f"Erro ao atualizar status: {str(e)}")

def lambda_handler(event, context):
    try:
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        raw_file_key = event['Records'][0]['s3']['object']['key']
        file_key = unquote_plus(raw_file_key)
        
        logger.info(f"EVENTO RECEBIDO - Bucket: {bucket_name}, Key: {file_key}")
        
        # --- FILTRO DE SEGURANÇA REFORÇADO (EVITAR LOOP) ---
        is_video = any(file_key.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv'])
        is_in_upload = file_key.startswith('uploads/')
        
        if not (is_in_upload and is_video):
            logger.info(f"FILTRO ATIVADO: Ignorando arquivo '{file_key}' pois não é um vídeo na pasta uploads/.")
            return {'statusCode': 200, 'body': 'Ignored'}
        # --------------------------------------------------

        file_id = file_key.split('/')[-1].split('.')[0]
        update_status(bucket_name, file_id, "INIT", "Iniciando análise multimodal...")

        # 1. Iniciar Jobs
        update_status(bucket_name, file_id, "VIDEO_START", "Iniciando Amazon Rekognition (Vídeo)...")
        rek_job_id = start_video_analysis(bucket_name, file_key)
        
        update_status(bucket_name, file_id, "AUDIO_START", "Iniciando Amazon Transcribe (Áudio)...")
        trans_job_name = start_transcription(bucket_name, file_key)
        
        # 2. Aguardar Vídeo
        update_status(bucket_name, file_id, "VIDEO_WAIT", "Aguardando processamento de frames e emoções...")
        video_results = get_video_analysis_results(rek_job_id)
        update_status(bucket_name, file_id, "VIDEO_DONE", "Análise de vídeo concluída.", {"emotions_count": len(video_results)})
        
        # 3. Aguardar Áudio
        update_status(bucket_name, file_id, "AUDIO_WAIT", "Aguardando transcrição de áudio para texto...")
        transcript = get_transcription_results(trans_job_name)
        update_status(bucket_name, file_id, "AUDIO_DONE", "Transcrição concluída.", {"transcript_preview": transcript[:100] + "..."})
        
        # 4. Comprehend
        update_status(bucket_name, file_id, "TEXT_ANALYSIS", "Analisando sentimento e linguagem no texto...")
        text_analysis = analyze_text(transcript)
        
        # 5. Bedrock
        update_status(bucket_name, file_id, "FUSION", "Realizando fusão multimodal e identificando evidências...")
        final_report = generate_multimodal_report(video_results, transcript, text_analysis)
        
        # 6. Finalizar
        report_key = f"reports/{file_id}_report.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=report_key,
            Body=json.dumps({"report": final_report, "transcript": transcript, "video_data": video_results}, indent=4, ensure_ascii=False),
            ContentType='application/json'
        )
        
        update_status(bucket_name, file_id, "COMPLETED", "Análise completa! Relatório gerado.", {"report_key": report_key})
        
        return {'statusCode': 200, 'body': 'Success'}

    except Exception as e:
        error_msg = f"Erro: {str(e)}"
        logger.error(error_msg)
        try: 
            if 'file_id' in locals():
                update_status(bucket_name, file_id, "ERROR", error_msg)
        except: pass
        return {'statusCode': 500, 'body': error_msg}

def start_video_analysis(bucket, key):
    return rekognition_client.start_face_detection(Video={'S3Object': {'Bucket': bucket, 'Name': key}}, FaceAttributes='ALL')['JobId']

def get_video_analysis_results(job_id):
    while True:
        res = rekognition_client.get_face_detection(JobId=job_id)
        if res['JobStatus'] == 'SUCCEEDED':
            return [{"Timestamp": f['Timestamp'], "Emotion": max(f['Face']['Emotions'], key=lambda x: x['Confidence'])['Type']} for f in res['Faces'] if 'Emotions' in f['Face']]
        if res['JobStatus'] == 'FAILED': raise Exception(f"Rekognition Failed: {res.get('StatusMessage', 'Unknown')}")
        time.sleep(5)

def start_transcription(bucket, key):
    # CORREÇÃO DA SINTAXE AQUI: usando list comprehension de forma correta
    clean_name = "".join([c for c in key.split('/')[-1] if c.isalnum()])[:10]
    job_name = f"trans_{int(time.time())}_{clean_name}"
    transcribe_client.start_transcription_job(TranscriptionJobName=job_name, Media={'MediaFileUri': f"s3://{bucket}/{key}"}, LanguageCode='pt-BR')
    return job_name

def get_transcription_results(job_name):
    while True:
        res = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = res['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            transcript_uri = res['TranscriptionJob']['Transcript']['TranscriptFileUri']
            with urllib.request.urlopen(transcript_uri) as response:
                transcript_json = json.load(response)
            return transcript_json['results']['transcripts'][0]['transcript']    
        if status == 'FAILED': raise Exception(f"Transcribe Failed: {res['TranscriptionJob'].get('FailureReason', 'Unknown')}")
        time.sleep(5)

def analyze_text(text):
    if not text: return {}
    res = comprehend_client.detect_sentiment(Text=text, LanguageCode='pt')
    return {'Sentiment': res['Sentiment'], 'Scores': res['SentimentScore']}

def generate_multimodal_report(video_data, transcript, text_analysis):
    prompt_text = f"""
    VOCÊ É UM ASSISTENTE ESPECIALIZADO EM SAÚDE DA MULHER E SEGURANÇA.
    Sua tarefa é analisar dados multimodais e justificar o nível de risco encontrado.

    DADOS DE VÍDEO (Emoções predominantes):
    {json.dumps(video_data[:20])}

    DADOS DE ÁUDIO (Transcrição):
    "{transcript}"

    DADOS DE TEXTO (Sentimento):
    {json.dumps(text_analysis)}

    POR FAVOR, GERE UM RELATÓRIO COM O SEGUINTE FORMATO:

    1. SCORE DE RISCO: (0-100)
    2. NÍVEL DE RISCO: (Baixo, Moderado, Alto ou Crítico)
    
    3. EVIDÊNCIAS QUE EMBASAM O SCORE (LISTA DE TÓPICOS):
       - [Ex: Voz trêmula detectada]
       - [Ex: Expressão facial de medo constante]

    4. ANÁLISE DETALHADA.
    5. RECOMENDAÇÕES E DISCLAIMER ÉTICO.
    """
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 2000, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]})
    res = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-haiku-20240307-v1:0", accept="application/json", contentType="application/json")
    return json.loads(res.get('body').read()).get('content')[0].get('text')
