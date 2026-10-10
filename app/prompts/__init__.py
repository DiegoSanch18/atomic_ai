from functools import lru_cache
from pathlib import Path
from typing import Optional, TYPE_CHECKING

_PROMPTS_DIR = Path(__file__).parent

_PROMPT_VAR_MAP = {
    "CALLER_SYSTEM_PREAMBLE": "caller_system_preamble",
    "DECOMPOSITION_SYSTEM_PROMPT": "decomposition_system",
    "DECOMPOSITION_USER_PROMPT": "decomposition_user",
    "EXECUTE_ATOMIC_SYSTEM_PROMPT": "execute_atomic_system",
    "EXECUTE_ATOMIC_USER_PROMPT": "execute_atomic_user",
    "SYNTHESIS_SYSTEM_PROMPT": "synthesis_system",
    "SYNTHESIS_USER_PROMPT": "synthesis_user",
}


@lru_cache(maxsize=128)
def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, profile: Optional[str] = None) -> str:
    """Carga un prompt por nombre y perfil, con fallback transparente al directorio base (default)."""
    clean_name = name[:-3] if name.endswith(".md") else name

    if profile is None:
        try:
            from app.config import settings
            profile = getattr(settings, "prompt_profile", "default")
        except Exception:
            profile = "default"

    profile_clean = (profile or "default").strip().lower()

    # Si se especifica un perfil distinto a default, buscar en la subcarpeta correspondiente
    if profile_clean != "default":
        profile_file = _PROMPTS_DIR / profile_clean / f"{clean_name}.md"
        if profile_file.is_file():
            return _read_file(profile_file)

    # Fallback al directorio base de prompts (default)
    default_file = _PROMPTS_DIR / f"{clean_name}.md"
    if default_file.is_file():
        return _read_file(default_file)

    raise FileNotFoundError(
        f"Prompt template '{clean_name}.md' no encontrado ni en perfil '{profile_clean}' ni en default."
    )


def clear_prompt_cache() -> None:
    """Limpia la caché de lectura de plantillas de prompts."""
    _read_file.cache_clear()


def __getattr__(name: str) -> str:
    if name in _PROMPT_VAR_MAP:
        return load_prompt(_PROMPT_VAR_MAP[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return list(globals().keys()) + list(_PROMPT_VAR_MAP.keys())


__all__ = [
    "load_prompt",
    "clear_prompt_cache",
    *list(_PROMPT_VAR_MAP.keys()),
]


if TYPE_CHECKING:
    CALLER_SYSTEM_PREAMBLE: str
    DECOMPOSITION_SYSTEM_PROMPT: str
    DECOMPOSITION_USER_PROMPT: str
    EXECUTE_ATOMIC_SYSTEM_PROMPT: str
    EXECUTE_ATOMIC_USER_PROMPT: str
    SYNTHESIS_SYSTEM_PROMPT: str
    SYNTHESIS_USER_PROMPT: str
