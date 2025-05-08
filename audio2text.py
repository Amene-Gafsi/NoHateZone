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

    def transcribe_audio(self, input_audio, return_timestamps="word", batch_size=1):
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

    def get_text_in_interval(self, result, start_time, end_time):
        """
        Filters and returns transcribed text within a specific time interval.

        Arguments:
        result (dict): A single transcription result with 'chunks'.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

        Returns:
        str: Combined transcription text from the specified interval.
        """
        selected_chunks = [
            chunk["text"]
            for chunk in result.get("chunks", [])
            if chunk["timestamp"][0] >= start_time and chunk["timestamp"][1] <= end_time
        ]
        return " ".join(selected_chunks)
    
    def extract_segments_with_timestamps(self, result):
        segments = []
        for chunk in result.get("chunks", []):
            segments.append({
                "text": chunk.get("text", ""),
                "start": chunk.get("timestamp", (None,))[0],
                "end": chunk.get("timestamp", (None, None))[1]
            })
        return segments
    
    def extract_sentences_with_timestamps(self, result: dict):
        """
        Build sentence‐level segments from word‐level chunks.
        Each word chunk is a dict with 'text' and 'timestamp': (start, end).
        """
        sentences = []
        current_words = []
        current_start = None
        current_end = None

        for wc in result.get("chunks", []):
            word = wc["text"].strip()
            start, end = wc["timestamp"]
            if current_start is None:
                current_start = start
            current_words.append(word)
            current_end = end

            # flush on terminal punctuation
            if word.endswith((".", "?", "!")):
                sentence = " ".join(current_words)
                sentences.append({
                    "text": sentence,
                    "start": current_start,
                    "end": current_end
                })
                current_words = []
                current_start = None
                current_end = None

        # flush any trailing words (no ending punctuation)
        if current_words:
            sentence = " ".join(current_words)
            sentences.append({
                "text": sentence,
                "start": current_start,
                "end": current_end
            })

        return sentences
    
    def transcribe_sentences(self, input_audio, batch_size=1):
        """
        Directly returns sentence-level chunks for input_audio.
        """
        raw = self.transcribe_audio(input_audio, return_timestamps="word", batch_size=batch_size)
        return self.extract_sentences_with_timestamps(raw)

