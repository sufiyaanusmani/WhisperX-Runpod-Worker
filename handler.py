import runpod
import whisperx
import base64
import json
import tempfile
from utils.gpu_helpers import deallocate_gpu_memory_if_low, check_gpu_availability
from utils.temp_envvar import temp_envvar
from utils.hf_helpers import get_huggingface_token
from utils.file_helpers import download_file

batch_size = 16 # reduce if low on GPU mem
language_code = "en"

def base64_to_tempfile(base64_data):
    """
    Decode base64 data and write it to a temporary file.
    Returns the path to the temporary file.
    """
    # Decode the base64 data to bytes
    audio_data = base64.b64decode(base64_data)

    # Create a temporary file and write the decoded data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    with open(temp_file.name, 'wb') as file:
        file.write(audio_data)

    return temp_file.name

def get_settings() -> tuple[str, str]:
    # device = "cuda"
    # batch_size = 16 # reduce if low on GPU mem
    # compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    if check_gpu_availability():
        return "cuda", "float16"

    return "cpu", "int8"

def handler():
    """
    Run inference on the model.

    Args:
        event (dict): The input event containing the audio data.
            The event should have the following structure:
            {
                "input": {
                    "audio_base_64": str,  # Base64-encoded audio data (optional)
                    "audio_url": str       # URL of the audio file (optional)
                }
            }
            Either "audio_base_64" or "audio_url" must be provided.
    """
    audio_file_path = "./yt-0pyalp198h8_SpPFdmXj.mp3"
    # convert it to base 64

    with open(audio_file_path, "rb") as file:
        audio_base64 = base64.b64encode(file.read()).decode()
    event = {
            "input": {
                "audio_base_64": audio_base64,  # Base64-encoded audio data (optional)
            }
    }
    job_input = event['input']
    job_input_audio_base_64 = job_input.get('audio_base_64')
    job_input_audio_url = job_input.get('audio_url')

    if job_input_audio_base_64:
        # If there is base64 data
        audio_input = base64_to_tempfile(job_input_audio_base_64)
    elif job_input_audio_url and job_input_audio_url.startswith('http'):
        # If there is an URL
        audio_input = download_file(job_input_audio_url)
    else:
        return json.dumps({"error": "Invalid audio"}, indent=4)
    
    model_files_root = "../models"

    try:
        deallocate_gpu_memory_if_low()
        device, compute_type = get_settings()

        audio = whisperx.load_audio(audio_input)

        model = whisperx.load_model(
              "large-v2",
              device,
              compute_type=compute_type,
              download_root=model_files_root,
              asr_options={
                  "word_timestamps": False,  # set to True if you want word timestamps
                  "max_new_tokens": None,
                  "clip_timestamps": None,
                  "hallucination_silence_threshold": None,
              },
          )
        
        result = model.transcribe(audio, batch_size=batch_size, language="en")
        
        deallocate_gpu_memory_if_low([model])

        model_a, metadata = whisperx.load_align_model(
              language_code=result["language"],
              device=device,
              model_dir=model_files_root,
          )
        
        resulta = whisperx.align(
              result["segments"],
              model_a,
              metadata,
              audio,
              device,
              return_char_alignments=False,
          )
        
        deallocate_gpu_memory_if_low(model_a)

        with temp_envvar("PYANNOTE_CACHE", model_files_root):
            # Can add min/max number of speakers if known
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            hf_token = get_huggingface_token()
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device,
            )

        diarize_segments = diarize_model(audio)
        result_diarized = whisperx.assign_word_speakers(diarize_segments, resulta)

        deallocate_gpu_memory_if_low(diarize_model)
        del result_diarized["word_segments"]
        return json.dumps(result_diarized, indent=4)
    except Exception as e:
        return json.dumps({"error": f"Error in diarizing: {str(e)}"}, indent=4)

runpod.serverless.start({
    "handler": handler
})