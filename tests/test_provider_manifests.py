"""Tests for provider manifest loading: the has_setup_flow flag, the tier and the features."""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
from typing import TYPE_CHECKING

import pytest
from music_assistant_models.enums import ProviderFeature, ProviderTier
from music_assistant_models.provider import ProviderManifest

from music_assistant import mass as mass_module
from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.plex import helpers as plex_helpers

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


async def _load_manifests(mass_minimal: MusicAssistant) -> None:
    """Run the (name-mangled, private) manifest loader against PROVIDERS_PATH."""
    mass_minimal._provider_manifests.clear()
    await mass_minimal._MusicAssistant__load_provider_manifests()  # type: ignore[attr-defined]


def _write_provider(
    providers_dir: pathlib.Path,
    domain: str,
    *,
    setup_flow_source: str | None,
    extra_manifest_keys: dict[str, object] | None = None,
) -> None:
    """Write a minimal provider package with an optional setup_flow.py module."""
    provider_dir = providers_dir / domain
    provider_dir.mkdir()
    (provider_dir / "__init__.py").write_text("")
    (provider_dir / "manifest.json").write_text(
        json.dumps(
            {
                "type": "music",
                "domain": domain,
                "name": domain,
                "description": "test provider",
                "codeowners": [],
                **(extra_manifest_keys or {}),
            }
        )
    )
    if setup_flow_source is not None:
        (provider_dir / "setup_flow.py").write_text(setup_flow_source)


@pytest.fixture
def fake_providers_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Redirect PROVIDERS_PATH to a throwaway directory for the duration of the test."""
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    monkeypatch.setattr(mass_module, "PROVIDERS_PATH", str(providers_dir))
    return providers_dir


async def test_has_setup_flow_true_when_module_present(
    mass_minimal: MusicAssistant, fake_providers_dir: pathlib.Path
) -> None:
    """A provider shipping a setup_flow.py module is flagged as having one, including over the API."""
    _write_provider(fake_providers_dir, "with_flow", setup_flow_source="")
    await _load_manifests(mass_minimal)
    manifest = mass_minimal.get_provider_manifest("with_flow")
    assert manifest.has_setup_flow is True
    assert manifest.to_dict()["has_setup_flow"] is True
    assert manifest in mass_minimal.get_provider_manifests()


async def test_has_setup_flow_false_when_module_absent(
    mass_minimal: MusicAssistant, fake_providers_dir: pathlib.Path
) -> None:
    """A provider without a setup_flow.py module is flagged as not having one, including over the API."""
    _write_provider(fake_providers_dir, "without_flow", setup_flow_source=None)
    await _load_manifests(mass_minimal)
    manifest = mass_minimal.get_provider_manifest("without_flow")
    assert manifest.has_setup_flow is False
    assert manifest.to_dict()["has_setup_flow"] is False


async def test_has_setup_flow_ignores_internal_import_errors(
    mass_minimal: MusicAssistant, fake_providers_dir: pathlib.Path
) -> None:
    """
    A setup_flow.py that itself fails to import still counts as present.

    The flag is derived from the file's mere existence, not from actually
    importing it, so a bug inside the module must not produce a false negative
    here; that failure only ever surfaces later, when the flow is started.
    """
    _write_provider(
        fake_providers_dir, "broken_flow", setup_flow_source="raise ImportError('boom')"
    )
    await _load_manifests(mass_minimal)
    assert mass_minimal.get_provider_manifest("broken_flow").has_setup_flow is True


async def test_shipped_provider_is_core_whatever_its_manifest_says(
    mass_minimal: MusicAssistant, fake_providers_dir: pathlib.Path
) -> None:
    """A provider loaded from the server's own providers folder is always marked core."""
    _write_provider(
        fake_providers_dir,
        "claims_community",
        setup_flow_source=None,
        extra_manifest_keys={"tier": "community"},
    )
    await _load_manifests(mass_minimal)
    manifest = mass_minimal.get_provider_manifest("claims_community")
    assert manifest.tier == ProviderTier.CORE
    assert manifest.to_dict()["tier"] == "core"


@pytest.mark.parametrize(
    "domain",
    [
        "airplay_receiver",
        "ariacast_receiver",
        "sendspin_source",
        "spotify_connect",
        "vban_receiver",
        "yandex_ynison",
    ],
)
async def test_audio_source_plugins_declare_the_feature(domain: str) -> None:
    """A plugin that always offers an audio source declares it in its manifest."""
    manifest_path = pathlib.Path(mass_module.PROVIDERS_PATH, domain, "manifest.json")
    manifest = await ProviderManifest.parse(str(manifest_path))
    assert ProviderFeature.AUDIO_SOURCE in manifest.features


def _music_provider_domains() -> list[str]:
    """Return the domains of the built-in music providers."""
    return sorted(
        path.parent.name
        for path in pathlib.Path(mass_module.PROVIDERS_PATH).glob("*/manifest.json")
        if json.loads(path.read_text())["type"] == "music"
    )


def _features_in_every_config(domain: str) -> set[ProviderFeature]:
    """
    Return the features the code of a built-in music provider offers in every configuration.

    :param domain: Domain of the music provider.
    """
    if domain == "plex":
        # the library type chosen at setup picks one of these sets as a whole
        return (
            plex_helpers.SUPPORTED_FEATURES
            & plex_helpers.AUDIOBOOK_FEATURES
            & plex_helpers.PODCAST_FEATURES
        )
    package = importlib.import_module(f"music_assistant.providers.{domain}")
    classes = {
        value
        for value in vars(package).values()
        if isinstance(value, type) and issubclass(value, MusicProvider)
    }
    # setup builds the most derived one, the package may also import the classes it builds on
    (provider_class,) = (
        cls
        for cls in classes
        if not any(other is not cls and issubclass(other, cls) for other in classes)
    )
    # the provider is created with SUPPORTED_FEATURES, passed by its setup or by its class (or a
    # class it builds on); its config or account can only add to these
    for module in (package, *(sys.modules[cls.__module__] for cls in provider_class.__mro__)):
        if (features := vars(module).get("SUPPORTED_FEATURES")) is not None:
            return set(features)
    pytest.fail(f"no SUPPORTED_FEATURES found for {domain}")


@pytest.mark.parametrize("domain", _music_provider_domains())
async def test_music_provider_manifest_lists_the_features_it_always_offers(domain: str) -> None:
    """A built-in music provider's manifest lists exactly the features its code always offers."""
    manifest_path = pathlib.Path(mass_module.PROVIDERS_PATH, domain, "manifest.json")
    manifest = await ProviderManifest.parse(str(manifest_path))
    # compared as the providers/manifests API hands them out
    assert set(manifest.to_dict()["features"]) == {
        feature.value for feature in _features_in_every_config(domain)
    }
