"""Versioned deterministic byte tokenizer, vocabulary256, no hidden training."""
import hashlib


class ByteTokenizer:
    version='morph-byte-utf8-v1'
    vocab_size=256
    fingerprint=hashlib.sha256(version.encode()).hexdigest()

    def encode(self,text):
        return list(text.encode('utf8'))

    def decode(self,ids):
        return bytes(ids).decode('utf8',errors='strict')
