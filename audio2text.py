import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings


class AudioTranscriber:
    def __init__(self, chunk_length=30):
        """
        Initializes the AudioTranscriber with a pre-trained Whisper model.
        The model is loaded onto the GPU if available, otherwise it falls back to CPU.
        The model is set to use half-precision floating point format (float16) if a GPU is available.

        Arguments:
        chunk_length_s (int): Length of audio chunks to process in seconds. Default is 30 seconds.
        """
        # inpsired by https://huggingface.co/openai/whisper-large-v3-turbo

        warnings.filterwarnings("ignore")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        # model.generation_config.language = "<|en|>"
        # model.generation_config.task = "transcribe"
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=chunk_length,  # for overlap rather then sequence for audio>30s
        )

    def transcribe_audio(self, input_audio, return_timestamps=True, batch_size=1):
        """
        Retrieves the transcription of the input audio using the Whisper model.

        Example:
            result = transcriber.transcribe_audio("audio.mp3")
            result["text"]         # full transcription
            result["chunks"][0]    # first transcribed segment
        Example with 2 audios result = transcribe_audio(["audio_1.mp3", "audio_2.mp3"], batch_size=2)

        Arguments:
        ----------
        input_audio (str or list of str):
            Path to a single audio file (e.g., "audio.mp3") or a list of file paths.
        return_timestamps (bool or str):
            If True, returns sentence-level timestamps.
            If "word", returns word-level timestamps.
            If False, returns only raw text. Default is True.
        batch_size (int):
            Number of audio files to process in parallel. Default is 1.

        Returns:
        -------
        dict or list of dicts:
            If one input file:
                {
                    'text': 'Full transcription as a string',
                    'chunks': [
                        {
                            'text': 'Segment transcription',
                            'timestamp': (start_time_in_sec, end_time_in_sec)
                        },
                        ...
                    ]
                }

            If multiple files:
                List of such dictionaries, one per file.
        """
        return self.pipe(
            input_audio,
            return_timestamps=return_timestamps,
            batch_size=batch_size,
            # generate_kwargs={"language": "en"}, #if needed to explicitly set language
        )

    def print_segments_with_timestamps(self, result):
        """
        Prints each transcribed sentence along with its start and end timestamps.

        Arguments:
        result (dict): A single transcription result from the pipeline (dict with 'text' and 'chunks' keys).
        """
        # print(result["text"])
        # print("\nSegments with timestamps:")
        for chunk in result.get("chunks", []):
            sentence = chunk["text"]
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            print(f"sentence: {sentence}, Start: {start}s, End: {end}s")
