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
    """

    def __init__(
        self,
        max_retries: int | None = None,
        backoff_base: int | None = None,
        model: str = "gpt-4o-mini",
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
        """Call GPT-4o to translate Chinese texts to Vietnamese."""
        # Format texts as numbered list for structured output
        numbered_input = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia dịch thuật Trung-Việt chuyên dịch lời thoại video. "
                        "Nhiệm vụ:\n"
                        "1. Sửa lỗi nhận dạng giọng nói (text có thể bị sai do ASR) — đoán nghĩa đúng dựa vào ngữ cảnh\n"
                        "2. Dịch tự nhiên, sát nghĩa, giữ nguyên giọng điệu và cảm xúc\n"
                        "3. Thuật ngữ internet/slang Trung Quốc dịch sang tương đương tiếng Việt\n"
                        "4. Giữ ngắn gọn — câu dịch nên có độ dài tương đương câu gốc\n"
                        "5. Nếu câu gốc vô nghĩa hoặc chỉ là tiếng ồn, dịch thành '...'\n\n"
                        "FORMAT OUTPUT: Mỗi dòng gồm [SPEAKER_ID] bản dịch\n"
                        "- Nếu video chỉ có 1 người nói: tất cả dùng [1]\n"
                        "- Nếu có nhiều người đối thoại: đánh số [1], [2], [3]... dựa vào ngữ cảnh (ai hỏi, ai trả lời)\n"
                        "- Phân biệt speaker dựa vào nội dung: câu hỏi vs trả lời, giọng điệu, vai trò trong hội thoại\n\n"
                        "Ví dụ output:\n"
                        "[1] Nhanh lên đi\n"
                        "[2] Đợi tôi chút\n"
                        "[1] Không đợi được\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Dịch {len(texts)} câu thoại từ video Trung Quốc sang tiếng Việt:\n\n{numbered_input}",
                },
            ],
            temperature=0.3,
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
