import json
import boto3
import time
import logging
import urllib.request
import subprocess
import os
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
        
        # 3. Aguardar Áudio
        update_status(bucket_name, file_id, "AUDIO_WAIT", "Aguardando transcrição de áudio para texto...")
        transcript = get_transcription_results(trans_job_name)
        update_status(bucket_name, file_id, "AUDIO_DONE", "Transcrição concluída.", {"transcript_preview": transcript[:100] + "..."})
        
        # 4. Comprehend
        update_status(bucket_name, file_id, "TEXT_ANALYSIS", "Analisando sentimento e linguagem no texto...")
        text_analysis = analyze_text(transcript)
        
        # 5. Bedrock
        update_status(bucket_name, file_id, "FUSION", "Realizando fusão multimodal e gerando justificativas...")
        final_report = generate_multimodal_report(video_results, transcript, text_analysis, [])
        
        import re

        match = re.search(r'Score de Risco:\s*(\d+)', final_report)
        risk_score = int(match.group(1)) if match else 50

        print("Risk Score Extraído:", risk_score)

        coherent_frames = select_frames_by_risk(video_results, risk_score)

        coherent_frames = extract_and_upload_frames(
            bucket_name,
            file_key,
            file_id,
            coherent_frames
        )

        # 6. Finalizar
        report_key = f"reports/{file_id}_report.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=report_key,
            Body=json.dumps({
                "report": final_report, 
                "transcript": transcript, 
                "video_data": video_results,
                "critical_frames": coherent_frames
            }, indent=4, ensure_ascii=False),
            ContentType='application/json'
        )
        
        update_status(bucket_name, file_id, "COMPLETED", "Análise completa! Relatório com frames críticos gerado.", {"report_key": report_key})
        
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
            results = []
            for f in res['Faces']:
                if 'Emotions' in f['Face']:
                    top_emotion = max(f['Face']['Emotions'], key=lambda x: x['Confidence'])
                    results.append({
                        "Timestamp": f['Timestamp'],
                        "Emotion": top_emotion['Type'],
                        "Confidence": top_emotion['Confidence']
                    })
            return results
        if res['JobStatus'] == 'FAILED': raise Exception(f"Rekognition Failed: {res.get('StatusMessage', 'Unknown')}")
        time.sleep(5)

def select_critical_frames(video_results, count=6):
    """Seleciona os frames mais relevantes baseados em emoções de risco."""
    critical_emotions = ['FEAR', 'SADNESS', 'ANGRY', 'CONFUSED']
    # Filtrar apenas emoções críticas
    filtered = [r for r in video_results if r['Emotion'] in critical_emotions]
    # Se não houver críticas, pega as de maior confiança geral
    if not filtered: filtered = video_results
    
    # Ordenar por confiança e pegar os top N únicos por segundo aproximado
    sorted_res = sorted(filtered, key=lambda x: x['Confidence'], reverse=True)
    seen_seconds = set()
    selected = []
    for r in sorted_res:
        sec = round(r['Timestamp'] / 1000)
        if sec not in seen_seconds:
            selected.append(r)
            seen_seconds.add(sec)
        if len(selected) >= count: break
    
    return selected

def extract_and_upload_frames(bucket, file_key, file_id, critical_frames):
    """
    Extrai frames usando FFmpeg e salva no S3.
    """
    local_video = "/tmp/video.mp4"

    # Baixar vídeo para /tmp
    s3_client.download_file(bucket, file_key, local_video)

    enriched_frames = []

    for frame in critical_frames:
        timestamp_sec = frame["Timestamp"] / 1000
        output_image = f"/tmp/frame_{int(frame['Timestamp'])}.jpg"

        command = [
            "/opt/bin/ffmpeg",  # já validamos que funciona
            "-ss", str(timestamp_sec),
            "-i", local_video,
            "-frames:v", "1",
            "-q:v", "2",
            output_image
        ]

        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Confirma se o frame foi criado
        if not os.path.exists(output_image):
            print("ERRO: Frame não gerado:", output_image)
            continue

        s3_frame_key = f"frames/{file_id}_{int(frame['Timestamp'])}.jpg"

        # Upload para S3
        s3_client.upload_file(
            output_image,
            bucket,
            s3_frame_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )

        # Gerar URL assinada (1 hora)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': s3_frame_key},
            ExpiresIn=3600
        )

        frame["FrameImage"] = presigned_url
        enriched_frames.append(frame)

    return enriched_frames

def select_frames_by_risk(video_results, risk_score, count=6):
    if not video_results:
        return []

    high_risk_emotions = ['FEAR', 'SADNESS', 'ANGRY', 'CONFUSED', 'SURPRISED']
    low_risk_emotions = ['CALM', 'HAPPY']

    if risk_score >= 70:
        candidates = [r for r in video_results if r['Emotion'] in high_risk_emotions]
        if not candidates:
            candidates = sorted(video_results, key=lambda x: x['Confidence'])
    elif risk_score <= 30:
        candidates = [r for r in video_results if r['Emotion'] in low_risk_emotions]
        candidates = sorted(candidates, key=lambda x: x['Confidence'], reverse=True)
    else:
        # risco médio → mistura equilibrada
        candidates = video_results

    seen_seconds = set()
    selected = []

    for r in candidates:
        sec = round(r['Timestamp'] / 1000)
        if sec not in seen_seconds:
            selected.append(r)
            seen_seconds.add(sec)
        if len(selected) >= count:
            break

    return selected


def start_transcription(bucket, key):
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

def generate_multimodal_report(video_data, transcript, text_analysis, critical_frames):
    prompt_text = f"""
    VOCÊ É UM ASSISTENTE ESPECIALIZADO EM SAÚDE DA MULHER E SEGURANÇA.
    Sua tarefa é analisar dados multimodais e justificar o nível de risco encontrado.

    DADOS DE VÍDEO (Resumo): {json.dumps(video_data[:10])}    
    TRANSCRIÇÃO: "{transcript}"
    SENTIMENTO: {json.dumps(text_analysis)}

    POR FAVOR, GERE UM RELATÓRIO COM:
    1. SCORE DE RISCO (0-100) E NÍVEL.
    2. EVIDÊNCIAS QUE EMBASAM O SCORE.
    3. EVIDÊNCIAS VISUAIS: Explique quais tipos de expressões faciais ou momentos do vídeo embasam o score. Caso não haja expressões de risco explícitas, explique como a neutralidade ou estabilidade emocional contribui para a avaliação.
    4. ANÁLISE DETALHADA E RECOMENDAÇÕES.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    })
    res = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-haiku-20240307-v1:0", accept="application/json", contentType="application/json")
    return json.loads(res.get('body').read()).get('content')[0].get('text')
