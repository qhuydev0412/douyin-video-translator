"""Translation service using OpenAI GPT-4o for high-quality contextual translation."""

import os
import re
import time

from openai import OpenAI

from app.core.config import settings
from app.models.pipeline import (
    TranscriptionResult,
    TranslatedSegment,
    TranslationResult,
)


class TranslationError(Exception):
    """Base exception for translation errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class EmptyTextError(TranslationError):
    """Raised when input text is empty or contains no Chinese content."""

    def __init__(self):
        super().__init__(
            message="Không có nội dung tiếng Trung để dịch",
            retryable=False,
        )


class Translator:
    """Translates Chinese text to Vietnamese using OpenAI GPT-4o.

    GPT-4o provides contextual, natural translation — much better than
    Google Translate for idioms, slang, and conversational Chinese.
    Uses 2-pass approach: first understand context, then translate.
    """

    def __init__(
        self,
        max_retries: int | None = None,
        backoff_base: int | None = None,
        model: str = "gpt-4o",
    ):
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRY_ATTEMPTS
        self.backoff_base = backoff_base if backoff_base is not None else settings.RETRY_BACKOFF_BASE
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def translate(self, transcription: TranscriptionResult) -> TranslationResult:
        """Translate Chinese text segments to Vietnamese using GPT-4o.

        Sends all segments at once for better context understanding.
        GPT also identifies speakers based on dialogue context.

        Args:
            transcription: TranscriptionResult with Chinese text segments.

        Returns:
            TranslationResult with translated Vietnamese segments.

        Raises:
            EmptyTextError: If text is empty or contains no Chinese characters.
            TranslationError: If the API fails after retries.
        """
        self._validate_chinese_content(transcription.full_text)
        self._last_speakers: list[str] = []

        # Batch translate all segments in one API call for better context
        translated_texts = self._batch_translate_with_retry(
            [seg.text for seg in transcription.segments]
        )

        translated_segments: list[TranslatedSegment] = []
        for i, segment in enumerate(transcription.segments):
            # Use GPT-detected speaker instead of Whisper's gap-based detection
            speaker = self._last_speakers[i] if i < len(self._last_speakers) else "speaker_1"
            translated_segments.append(
                TranslatedSegment(
                    start=segment.start,
                    end=segment.end,
                    original_text=segment.text,
                    translated_text=translated_texts[i] if i < len(translated_texts) else segment.text,
                    speaker=speaker,
                )
            )

        full_text_translated = " ".join(translated_texts)

        return TranslationResult(
            segments=translated_segments,
            full_text_original=transcription.full_text,
            full_text_translated=full_text_translated,
        )

    def _batch_translate_with_retry(self, texts: list[str]) -> list[str]:
        """Translate a batch of texts in one API call with retry."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return self._call_gpt(texts)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    backoff_time = self.backoff_base ** (attempt + 1)
                    time.sleep(backoff_time)

        raise TranslationError(
            message=f"Lỗi dịch thuật sau {self.max_retries} lần thử: {last_exception}",
            retryable=True,
        )

    def _call_gpt(self, texts: list[str]) -> list[str]:
        """Call GPT-4o to translate Chinese texts to Vietnamese.
        
        Uses a 2-pass approach in a single call:
        - First, GPT analyzes the full context (what's happening in the video)
        - Then translates each line with that context in mind
        """
        numbered_input = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia dịch thuật Trung-Việt, chuyên dịch lời thoại video Douyin/TikTok.\n\n"
                        "QUY TRÌNH:\n"
                        "1. Đọc toàn bộ transcript, hiểu NGỮ CẢNH: video nói về gì, ai đang nói với ai, tình huống gì\n"
                        "2. Sửa lỗi ASR: text gốc từ nhận dạng giọng nói nên có thể sai chính tả, thiếu từ, hoặc nghe nhầm. "
                        "Hãy đoán từ đúng dựa vào ngữ cảnh\n"
                        "3. Dịch sát nghĩa, tự nhiên như người Việt nói. KHÔNG dịch từng từ máy móc\n"
                        "4. Giữ giọng điệu: hài hước → dịch hài, nghiêm túc → dịch nghiêm túc\n"
                        "5. Slang/internet: 太卷了→quá cuốn, 666→hay quá, 牛→dữ, 老铁→bro/anh em, 安排→lo hết, 摸鱼→lướt mạng/chơi\n"
                        "6. Câu nào chỉ là tiếng ồn/nhạc/không rõ nghĩa → dịch thành '...'\n\n"
                        "FORMAT OUTPUT:\n"
                        "Mỗi dòng: [SPEAKER_ID] bản dịch tiếng Việt\n"
                        "- Video 1 người: tất cả [1]\n"
                        "- Video nhiều người đối thoại: [1], [2], [3]... phân biệt dựa vào nội dung\n"
                        "- ĐÚNG số dòng như input. KHÔNG giải thích."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Transcript video Douyin ({len(texts)} câu). "
                        f"Hãy hiểu ngữ cảnh rồi dịch:\n\n{numbered_input}"
                    ),
                },
            ],
            temperature=0.2,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse response — each line is "[SPEAKER_ID] translated text"
        lines = [line.strip() for line in result_text.split("\n") if line.strip()]

        # Extract speaker and text from each line
        cleaned = []
        speakers = []
        for line in lines:
            # Remove numbering if GPT added it (e.g., "1. [1] Xin chào")
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            # Extract speaker: [1], [2], etc.
            speaker_match = re.match(r'\[(\d+)\]\s*(.*)', cleaned_line)
            if speaker_match:
                speakers.append(f"speaker_{speaker_match.group(1)}")
                cleaned.append(speaker_match.group(2))
            else:
                speakers.append("speaker_1")
                cleaned.append(cleaned_line)

        # Pad or trim to match input count
        while len(cleaned) < len(texts):
            cleaned.append(texts[len(cleaned)])
            speakers.append("speaker_1")
        cleaned = cleaned[:len(texts)]
        speakers = speakers[:len(texts)]

        # Store speakers for later use
        self._last_speakers = speakers

        return cleaned

    def _validate_chinese_content(self, text: str) -> None:
        """Validate that text contains Chinese characters."""
        if not text or not text.strip():
            raise EmptyTextError()
        if not self._contains_chinese(text):
            raise EmptyTextError()

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Check if text contains at least one Chinese character."""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
