import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_WHITESPACE_RE = re.compile(r"\s+")
# Emoji vb. geniş aralık (tam kusursuz değil ama pratikte işe yarar)
_NON_TEXT_RE = re.compile(r"[^\w\s\.\,\!\?\-\'\"]", flags=re.UNICODE)

def clean_text(text: str) -> str:
    """
    Basit ama anlamı bozmayan temizlik:
    - html tag ve url kaldır
    - çoklu boşlukları tekleştir
    - aşırı garip karakterleri ayıkla
    - lower()
    """
    if text is None:
        return ""
    text = str(text)

    text = re.sub(_HTML_RE, " ", text)
    text = re.sub(_URL_RE, " ", text)

    # emoji/özel sembolleri hafifçe buda (kelime karakterleri kalsın)
    text = re.sub(_NON_TEXT_RE, " ", text)

    text = text.lower()
    text = re.sub(_WHITESPACE_RE, " ", text).strip()
    return text


 