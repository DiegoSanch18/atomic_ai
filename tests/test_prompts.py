import json
import re
from pathlib import Path
import pytest

from app.config import settings
from app import prompts
from app.prompts import load_prompt, clear_prompt_cache
from app.engine import _parse_decomposition


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback conservador si tiktoken no estuviera disponible
        return int(len(text.split()) * 1.4)


class TestPromptTokenLimits:
    @pytest.mark.parametrize(
        "template_name",
        ["decomposition_system", "execute_atomic_system", "synthesis_system"],
    )
    def test_slm_prompts_under_180_tokens(self, template_name: str):
        content = load_prompt(template_name, profile="slm")
        tokens = count_tokens(content)
        assert tokens < 180, (
            f"Template '{template_name}' excedió el límite estricto de 180 tokens: {tokens} tokens encontrados"
        )

    def test_slm_prompts_substantially_smaller_than_default(self):
        for name in ["decomposition_system", "execute_atomic_system", "synthesis_system"]:
            default_content = load_prompt(name, profile="default")
            slm_content = load_prompt(name, profile="slm")
            default_tokens = count_tokens(default_content)
            slm_tokens = count_tokens(slm_content)
            assert slm_tokens < default_tokens * 0.35, (
                f"Template {name} ({slm_tokens} tokens) debería ser sustancialmente menor al default ({default_tokens} tokens)"
            )


class TestPromptProfileLoading:
    def test_default_profile_loads_base_files(self):
        content = load_prompt("decomposition_system", profile="default")
        assert "<criterio_de_atomicidad>" in content
        assert "Eres un planificador de tareas" in content

    def test_slm_profile_loads_slm_files(self):
        content = load_prompt("decomposition_system", profile="slm")
        assert "Eres un planificador" in content
        assert "<criterio_de_atomicidad>" not in content

    def test_slm_fallback_for_non_overridden_templates(self):
        # caller_system_preamble no se redefine en slm/ y debe heredar de default
        preamble_slm = load_prompt("caller_system_preamble", profile="slm")
        preamble_default = load_prompt("caller_system_preamble", profile="default")
        assert preamble_slm == preamble_default

        # decomposition_user no se redefine en slm/ y debe heredar de default
        user_slm = load_prompt("decomposition_user", profile="slm")
        user_default = load_prompt("decomposition_user", profile="default")
        assert user_slm == user_default

    def test_dynamic_profile_switching_via_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "prompt_profile", "default")
        assert "<criterio_de_atomicidad>" in prompts.DECOMPOSITION_SYSTEM_PROMPT

        monkeypatch.setattr(settings, "prompt_profile", "slm")
        assert "<criterio_de_atomicidad>" not in prompts.DECOMPOSITION_SYSTEM_PROMPT
        assert "Eres un planificador" in prompts.DECOMPOSITION_SYSTEM_PROMPT

        monkeypatch.setattr(settings, "prompt_profile", "default")
        assert "<criterio_de_atomicidad>" in prompts.DECOMPOSITION_SYSTEM_PROMPT

    def test_missing_prompt_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("non_existent_template_xyz", profile="slm")

    def test_clear_prompt_cache(self):
        content_before = load_prompt("decomposition_system", profile="slm")
        clear_prompt_cache()
        content_after = load_prompt("decomposition_system", profile="slm")
        assert content_before == content_after


class TestPromptSchemaAndSemantics:
    def test_decomposition_slm_contains_valid_json_examples(self):
        content = load_prompt("decomposition_system", profile="slm")
        matches = re.findall(r"\{[^{}]*\}", content)
        assert len(matches) >= 2, f"Se esperaban al menos 2 bloques JSON en decomposition_system.md, encontrados: {len(matches)}"
        for match in matches:
            parsed = json.loads(match)
            assert "atomic" in parsed, f"Falta clave 'atomic' en bloque JSON: {match}"
            assert isinstance(parsed["atomic"], bool), f"'atomic' debe ser bool en: {match}"
            assert "subtasks" in parsed, f"Falta clave 'subtasks' en bloque JSON: {match}"
            assert isinstance(parsed["subtasks"], list), f"'subtasks' debe ser list en: {match}"

    def test_engine_parses_slm_decomposition_outputs(self):
        raw_atomic = '{"atomic": true, "subtasks": []}'
        res_atomic = _parse_decomposition(raw_atomic)
        assert res_atomic["atomic"] is True
        assert res_atomic["subtasks"] == []

        raw_non_atomic = '{"atomic": false, "subtasks": ["subtarea 1", "subtarea 2"]}'
        res_non_atomic = _parse_decomposition(raw_non_atomic)
        assert res_non_atomic["atomic"] is False
        assert res_non_atomic["subtasks"] == ["subtarea 1", "subtarea 2"]

    def test_execute_atomic_retains_marker(self):
        content = load_prompt("execute_atomic_system", profile="slm")
        assert "[[NECESITA_HERRAMIENTA:" in content
        assert "]]" in content

    def test_synthesis_retains_marker(self):
        content = load_prompt("synthesis_system", profile="slm")
        assert "[[NECESITA_HERRAMIENTA:" in content
        assert "]]" in content
