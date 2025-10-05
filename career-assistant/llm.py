# llm.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

class KeyExhaustedError(RuntimeError):
    pass

class LLMProvider:
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"  # OpenAI-compatible

    def __init__(self, provider="openai", model=DEFAULT_OPENAI_MODEL, temperature=0.2, max_tokens=800):
        self.provider = (provider or "openai").lower().strip()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load keys
        self._openai_keys = self._parse_keys(os.getenv("OPENAI_API_KEYS", os.getenv("OPENAI_API_KEY", "")))
        self._groq_keys   = self._parse_keys(os.getenv("GROQ_API_KEYS", ""))

        # Keep legacy compatibility
        self.api_key = self._openai_keys[0] if self._openai_keys else os.getenv("OPENAI_API_KEY", "")

        self._set_base_url()

    # ---------------- Internals ----------------
    @staticmethod
    def _parse_keys(keys_raw: str):
        return [k.strip() for k in (keys_raw or "").split(",") if k.strip()]

    def _set_base_url(self):
        if self.provider == "openai":
            self.base_url = self.OPENAI_URL
        elif self.provider == "groq":
            self.base_url = self.GROQ_URL
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _provider_keys(self, who: str):
        return self._openai_keys if who == "openai" else self._groq_keys

    # --------------- Public API ---------------
    def chat(self, messages):
        """
        Try all keys for preferred provider; if exhausted, fall back to the other provider.
        """
        preferred = self.provider
        other = "groq" if preferred == "openai" else "openai"

        try:
            return self._try_provider(messages, preferred)
        except KeyExhaustedError:
            if self._provider_keys(other):
                self.provider = other
                self._set_base_url()
                return self._try_provider(messages, other)
            raise RuntimeError(f"All {preferred} API keys failed and no {other} keys available.")

    # --------------- Provider key loop ---------------
    def _try_provider(self, messages, who: str):
        keys = list(self._provider_keys(who))
        if not keys:
            raise KeyExhaustedError(f"No keys configured for {who}.")

        last_err = None
        for key in keys:
            try:
                return self._call_api(messages, key, who)
            except RuntimeError as e:
                msg = str(e)
                if (" HTTP 401" in msg) or (" HTTP 403" in msg) or (" HTTP 429" in msg):
                    last_err = e
                    continue
                raise
        raise KeyExhaustedError(last_err or f"All {who} keys exhausted.")

    # --------------- API call ---------------
    def _call_api(self, messages, api_key, who: str):
        url = self.OPENAI_URL if who == "openai" else self.GROQ_URL

        # --- NEW: ensure correct model for provider ---
        model = self.model
        ml = (model or "").lower()
        if who == "groq" and ("gpt" in ml or "o-" in ml):
            model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        elif who == "openai" and "llama" in ml:
            model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        # ----------------------------------------------

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error calling {who}: {e}")

        if resp.status_code != 200:
            try:
                err = resp.json().get("error", {})
                code = err.get("code")
                msg  = err.get("message") or resp.text[:400]
                raise RuntimeError(f"{who} HTTP {resp.status_code}: {code or ''} {msg}")
            except Exception:
                raise RuntimeError(f"{who} HTTP {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # -------- Legacy-compatibility method --------
    def _openai_chat(self, messages):
        return self.chat(messages)
