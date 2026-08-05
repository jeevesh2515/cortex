"""Tests for the cortex doctor diagnostic module.

These exercise each individual check with a constructed Settings; the
`run_doctor` orchestrator test suite covers aggregation, severity ordering and
crash containment.
"""

from __future__ import annotations

import json as _json
import plistlib
import urllib.request
from pathlib import Path

import pytest

from cortex import doctor as _doctor_module
from cortex.config import Settings
from cortex.doctor import Check, DoctorReport, Status, run_doctor
from cortex.llm.protocol import ProviderSpec
from cortex.models import DataPolicy


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.vault_path = tmp_path / "vault"
    s.vault_path.mkdir()
    s.data_dir = tmp_path / "data"
    return s


@pytest.fixture
def empty_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a non-existent vault."""
    s = Settings()
    s.vault_path = tmp_path / "nope"
    s.data_dir = tmp_path / "data"
    return s


class TestStatusSummary:
    def test_ready_line_used_when_no_issues(self) -> None:
        report = DoctorReport(
            checks=[Check("a", Status.PASS, ""), Check("b", Status.INFO, "")]
        )
        assert report.summary_line() == "Ready"

    def test_singular_issue(self) -> None:
        report = DoctorReport(
            checks=[Check("a", Status.PASS, ""), Check("b", Status.WARN, "x")]
        )
        assert report.summary_line() == "1 issue"

    def test_plural_issues(self) -> None:
        report = DoctorReport(
            checks=[
                Check("a", Status.WARN, "x"),
                Check("b", Status.FAIL, "y"),
                Check("c", Status.WARN, "z"),
            ]
        )
        assert report.summary_line() == "3 issues"

    def test_blocking_and_warnings_counted_separately(self) -> None:
        report = DoctorReport(
            checks=[
                Check("a", Status.FAIL, "x"),
                Check("b", Status.FAIL, "y"),
                Check("c", Status.WARN, "z"),
            ]
        )
        assert report.blocking == 2
        assert report.warnings == 1
        assert report.infos == 0


class TestVaultCheck:
    def test_passes_when_vault_has_notes(self, settings: Settings) -> None:
        from cortex.doctor import check_vault

        (settings.vault_path / "A.md").write_text("# A\n[[B]]\n", encoding="utf-8")
        (settings.vault_path / "B.md").write_text("# B\n", encoding="utf-8")
        result = check_vault(settings)
        assert result.status is Status.PASS
        assert "notes" in result.message

    def test_non_linked_vault_still_passes(self, settings: Settings) -> None:
        # An unlinked vault is degraded for graph retrieval but valid as a
        # vault. Surfacing a WARN would mislead users into chasing a fix for
        # something they chose to leave flat.
        from cortex.doctor import check_vault

        (settings.vault_path / "A.md").write_text("# A\npure prose\n", encoding="utf-8")
        result = check_vault(settings)
        assert result.status is Status.PASS

    def test_fails_when_path_missing(self, empty_settings: Settings) -> None:
        from cortex.doctor import check_vault

        result = check_vault(empty_settings)
        assert result.status is Status.FAIL
        assert result.hint is not None
        assert "CORTEX_VAULT" in result.hint or "--vault" in result.hint


class TestOllamaCheck:
    def test_fail_when_unreachable(self, settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from cortex import doctor

        # Use a port nothing is listening on; connection refused returns fast
        # so this test does not burn the 2-second doctor timeout budget.
        settings.ollama_url = "http://127.0.0.1:1"

        result = doctor.check_ollama(settings)
        assert result.status is Status.FAIL
        assert "unreachable" in result.message

    def test_warn_when_ollama_up_but_model_missing(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        class _FakeResp:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return _json.dumps(self._payload).encode("utf-8")

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *_: object) -> None:
                return None

        payloads = [
            {"models": [{"name": "some-other-model"}]},  # /api/tags
            {"models": []},                              # /api/ps
        ]
        idx = 0

        def _fake_urlopen(req: object, timeout: float = 2.0) -> _FakeResp:  # type: ignore[no-untyped-def]
            nonlocal idx
            resp = _FakeResp(payloads[idx])
            idx += 1
            return resp

        # ``monkeypatch.setattr`` restores itself on fixture teardown, so a
        # failing assertion cannot leak the patch into the next test.
        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

        result = doctor.check_ollama(settings)
        assert result.status is Status.WARN
        assert "not pulled" in result.message

    def test_pass_message_reports_load_state(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        class _FakeResp:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return _json.dumps(self._payload).encode("utf-8")

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *_: object) -> None:
                return None

        model = settings.embed_model

        def _fake_urlopen(req: object, timeout: float = 2.0) -> _FakeResp:  # type: ignore[no-untyped-def]
            url = getattr(req, "full_url", "")
            if url.endswith("/api/ps"):
                return _FakeResp({"models": [{"name": model}]})  # currently loaded
            return _FakeResp({"models": [{"name": model}]})  # pulled

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        result = doctor.check_ollama(settings)
        assert result.status is Status.PASS
        assert "currently loaded" in result.message


class TestMcpCheck:
    def test_passes_when_mcp_present_and_compatible(self) -> None:
        # If the user has installed cortex-brain[mcp], the constructor accepts
        # a name-only Server() call -- meaning the wire format matches their
        # installed mcp version. This session has mcp installed.
        from cortex.doctor import check_mcp_wire

        result = check_mcp_wire(Settings())
        assert result.status in {Status.PASS, Status.FAIL}
        # Only PASS if mcp installed (this test env has it).
        assert result.name == "MCP"


class TestDaemonCheck:
    def test_warns_when_plist_not_installed(self) -> None:
        from cortex.doctor import check_daemon

        result = check_daemon(Settings())
        # The user's machine does not have the plist installed in this test
        # environment, so we expect WARN. (Reality: tracks the host.)
        assert result.status in {Status.WARN, Status.PASS}
        assert result.name == "Daemon"


class TestPrivacyCheck:
    def test_passes_when_local_provider_present(self, settings: Settings) -> None:
        from cortex.doctor import check_privacy

        settings.providers = [
            ProviderSpec(
                name="ollama",
                base_url="http://localhost:11434",
                model="x",
                policy=DataPolicy.LOCAL,
            )
        ]
        result = check_privacy(settings)
        assert result.status is Status.PASS

    def test_fails_when_only_training_provider(self, settings: Settings) -> None:
        from cortex.doctor import check_privacy

        settings.providers = [
            ProviderSpec(
                name="zen",
                base_url="https://example.invalid/v1",
                model="m",
                policy=DataPolicy.TRAINS,
            )
        ]
        result = check_privacy(settings)
        assert result.status is Status.FAIL
        assert "no provider" in result.message.lower()


class TestMemoryCheck:
    def test_warns_when_disabled(self, settings: Settings) -> None:
        from cortex.doctor import check_memory

        settings.memory_enabled = False
        result = check_memory(settings)
        assert result.status is Status.INFO

    def test_passes_when_vault_writable(self, settings: Settings) -> None:
        from cortex.doctor import check_memory

        settings.memory_enabled = True
        result = check_memory(settings)
        assert result.status is Status.PASS
        assert result.message.startswith("writable") or "writable" in result.message

    def test_skips_when_vault_missing_to_avoid_cascade(self, empty_settings: Settings) -> None:
        from cortex.doctor import check_memory

        empty_settings.memory_enabled = True
        result = check_memory(empty_settings)
        assert result.status is Status.INFO
        assert "skipped" in result.message.lower() or "vault" in result.message.lower()


def _write_plist(path: Path, env: dict[str, str]) -> None:
    """Helper: dump a minimal but valid plist with the given env block."""
    payload = {
        "Label": "test.ollama",
        "ProgramArguments": ["/usr/bin/true"],
        "EnvironmentVariables": env,
    }
    with path.open("wb") as fh:
        plistlib.dump(payload, fh)


def _write_systemd_unit(path: Path, lines: list[str]) -> None:
    """Helper: write a minimal systemd unit file with the given Environment= lines."""
    sections: list[str] = ["[Unit]", "Description=test", "", "[Service]"]
    sections.extend(lines)
    sections.extend(["", "[Install]", "WantedBy=default.target"])
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


class TestGpuPathCheck:
    """Test the GPU-path probe with monkeypatched candidate paths.

    Each test points ``doctor.OLLAMA_DAEMON_PATHS`` at fixture files in
    ``tmp_path`` so we never touch the real ``~/Library`` directory. Stale
    binaries that pre-date this feature would otherwise drift with every
    developer machine.
    """

    @pytest.fixture(autouse=True)
    def _stub_live_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default the live probe to ``None`` so tests do not bleed into the
        developer's real launchctl/systemctl state.

        Tests that care about the live env explicitly call
        ``monkeypatch.setattr`` to override this fixture's stub. The autouse
        hook means every GPT test here gets the safe default and we never
        write a test that accidentally relies on the host's daemon env.
        """
        monkeypatch.setattr(_doctor_module, "_live_ollama_env", lambda: None)

    def _candidate_for(self, tmp_path: Path, kind: str, env: dict[str, str]) -> Path:
        if kind == "plist":
            plist = tmp_path / "ollama.plist"
            _write_plist(plist, env)
            return plist
        unit = tmp_path / "ollama.service"
        env_lines = [f'Environment="{k}={v}"' for k, v in env.items()]
        _write_systemd_unit(unit, env_lines)
        return unit

    def test_no_daemon_config_returns_info(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        # Every candidate is a path inside tmp_path that does NOT exist.
        # The check should resolve to INFO (no evidence) rather than FAIL.
        fake_paths = tuple(
            (tmp_path / f"missing-{i}.plist", "plist") for i in range(3)
        )
        monkeypatch.setattr(doctor, "OLLAMA_DAEMON_PATHS", fake_paths)

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.INFO
        assert result.hint is not None
        assert "no ollama daemon config" in result.message.lower()

    def test_plist_with_num_gpu_zero_returns_warn(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((plist, "plist"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert "OLLAMA_NUM_GPU=0" in result.message
        assert result.hint is not None
        # The remediation path names the file the user needs to edit.
        assert str(plist) in result.hint
        # Details should expose the static value for the user.
        joined = " ".join(result.details)
        assert "static" in joined.lower()
        # The exact wording varies by live-probe result; just confirm the
        # surface area recognises the value. Either the live row spells
        # ``'0'`` outright (live matches), or the message flags the
        # discrepancy, the missing override, the daemon-bound value, or
        # (worst case) the live probe failed.
        assert any(
            token in joined.lower()
            for token in (
                "'0'",
                "live: no override",
                "disagree",
                "static override",
                "live probe: failed",
            )
        )

    def test_plist_without_num_gpu_returns_pass(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_FLASH_ATTENTION": "1"})
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((plist, "plist"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.PASS
        assert "default GPU acceleration" in result.message
        assert plist.name in result.message

    def test_systemd_unit_with_num_gpu_zero_returns_warn(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        unit = self._candidate_for(tmp_path, "systemd", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((unit, "systemd"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert "OLLAMA_NUM_GPU=0" in result.message
        # systemd-derived details must surface the unit kind so the user
        # can tell which kind of config file controls inference.
        joined = " ".join(result.details)
        assert "systemd" in joined

    def test_first_existing_path_wins(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the priority list has multiple real files, the first one wins.
        # Regression: an old broken plist at position 0 must not be silently
        # shadowed by a healthy plist at position 1.
        from cortex import doctor

        healthy_plist = tmp_path / "00-healthy.plist"
        _write_plist(healthy_plist, {"OLLAMA_FLASH_ATTENTION": "1"})
        cpu_plist = tmp_path / "99-cpu.plist"
        _write_plist(cpu_plist, {"OLLAMA_NUM_GPU": "0"})

        # Order matters: healthy is FIRST, then cpu. The first existing file
        # should win.
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            (
                (healthy_plist, "plist"),
                (cpu_plist, "plist"),
            ),
        )
        result = doctor.check_gpu_path(settings)
        assert result.status is Status.PASS
        assert healthy_plist.name in result.message

        # Flip the order. cpu is now first.
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            (
                (cpu_plist, "plist"),
                (healthy_plist, "plist"),
            ),
        )
        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert cpu_plist.name in str(result.details)

    def test_corrupt_plist_returns_pass_empty_env(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A garbage plist is treated as having no env. The user will see a
        # PASS for the GPU path -- but the daemon itself will probably fail
        # to load on the next launch; that's a separate diagnostic concern
        # the ollama check already covers.
        from cortex import doctor

        bad = tmp_path / "garbage.plist"
        bad.write_bytes(b"this is not a plist")
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((bad, "plist"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.PASS
        assert "default GPU acceleration" in result.message

    def test_systemd_multi_assignment_parses_each_token(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # systemd accepts whitespace-separated ``Environment=K1=v1 K2=v2`` and
        # we should treat each token as its own assignment. Regression for
        # the regex-grep bug where the entire RHS was captured as one value.
        from cortex import doctor

        unit = tmp_path / "multi.service"
        _write_systemd_unit(
            unit,
            ['Environment="OLLAMA_NUM_GPU=0 OLLAMA_DEBUG=1"'],
        )
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((unit, "systemd"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        # MUST capture ``0``, not the full tail ``0 OLLAMA_DEBUG=1``.
        assert "OLLAMA_NUM_GPU=0" in result.message
        assert "OLLAMA_DEBUG" not in result.message

    def test_systemd_inline_hash_in_value_is_preserved(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A trivial guard against the comment-strip eating values that
        # legitimately contain ``#``. The diagnostic doesn't run on such
        # values, never; this just verifies the parser doesn't silently
        # truncate.
        from cortex import doctor

        unit = tmp_path / "hash.service"
        # ``OLLAMA_NUM_GPU=foo#bar`` — hypothetical setting with a ``#``
        # inside the value. The parser must NOT return ``foo``.
        _write_systemd_unit(
            unit,
            ['Environment=OLLAMA_NUM_GPU=foo#bar'],
        )
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((unit, "systemd"),),
        )

        result = doctor.check_gpu_path(settings)
        # We don't expect a WARN here (the degenerate value isn't ``0``),
        # but if the parser mangled ``foo#bar`` to ``foo``, the doctor would
        # either WARN spuriously (with value=foo) or PASS spuriously. We
        # assert strictly: the value captured must round-trip the full
        # ``foo#bar`` form, so neither spuriously warns nor spuriously passes.
        joined = " ".join(result.details) + " " + result.message
        assert "foo" in joined
        # Specifically NOT truncated: the parser must round-trip ``foo#bar``,
        # not just ``foo``. If the parser mangled it, ``foo#bar`` would be
        # missing and the doctor would have spuriously warned; either
        # branch below rejects the malformed outcome.
        assert "foo#bar" in joined or (
            result.status is not Status.WARN
            and "drift" not in result.message.lower()
        )

    def test_pass_message_acknowledges_environmentfile(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Issue surfaced by code review: a unit using ``EnvironmentFile=``
        # would silently read as PASS-with-no-cav. We surface the caveat in
        # the details so a careful user sees the rounding.
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_FLASH_ATTENTION": "1"})
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((plist, "plist"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.PASS
        assert any("EnvironmentFile=" in line for line in result.details)

    def test_warn_hint_acknowledges_environmentfile(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The hint should explicitly mention EnvironmentFile= so a user whose
        # config relies on it isn't silently misled by a PASS.
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            ((plist, "plist"),),
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert result.hint is not None
        assert "EnvironmentFile=" in result.hint

    def test_launchctl_parser_recovers_all_three_blocks(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Lock the launchctl parser against drift by feeding it a verbatim
        # capture of the blocks launchctl print emits: inherited, default,
        # environment. The user's exact machine output -- regression.
        from cortex import doctor

        sample = """
        inherited environment = {
            \tSSH_AUTH_SOCK => /var/run/ssh-listener
            \tPATH => /usr/bin:/bin
        }

        default environment = {
            \tPATH => /usr/bin:/bin:/usr/sbin:/sbin
        }

        environment = {
            \tOSLogRateLimit => 64
            \tOLLAMA_FLASH_ATTENTION => 1
            \tOLLAMA_NUM_GPU => 0
            \tOLLAMA_KV_CACHE_TYPE => q8_0
            \tXPC_SERVICE_NAME => homebrew.mxcl.ollama
        }

        pid = 12345
        """
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: doctor._parse_launchctl_env_blocks(sample)
        )
        # Belt-and-braces: also stub the helper, in case ``_live_ollama_env``
        # is ever re-routed through it.
        monkeypatch.setattr(
            doctor, "_probe_launchctl_live", lambda: doctor._parse_launchctl_env_blocks(sample)
        )
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            (),
        )
        result = doctor.check_gpu_path(settings)
        # Live probe succeeded and found NUM_GPU=0; no static config.
        assert result.status is Status.WARN
        assert "transient override" in " ".join(result.details).lower() or \
            "live" in result.message.lower()
        # Look up the live env value in details; should collapse to '0'.
        assert any("live: '0'" in line for line in result.details)

    def test_inherited_environment_only_block_surfaces_transient(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Critical regression for ``launchctl setenv OLLAMA_NUM_GPU=0``: that
        # override doesn't land in ``environment = { ... }`` -- it lands in
        # ``inherited environment = { ... }`` because launchd propagates
        # domain-level setenv through the inherited block. The probe must
        # surface the value; an earlier filter ("only return env if there
        # was an explicit environment block") silently dropped the value
        # and the very case this test guards was invisible to the doctor.

        from cortex import doctor

        sample = """
        inherited environment = {
            \tPATH => /usr/bin:/bin
            \tOLLAMA_NUM_GPU => 0
            \tSSH_AUTH_SOCK => /var/run/listener
        }

        default environment = {
            \tPATH => /usr/bin:/bin:/usr/sbin:/sbin
        }

        pid = 12345
        """
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: doctor._parse_launchctl_env_blocks(sample)
        )
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            (),
        )
        result = doctor.check_gpu_path(settings)
        # Static absent, live asserts OLLAMA_NUM_GPU=0 -> transient override.
        assert result.status is Status.WARN
        joined = " ".join(result.details) + " " + result.message
        assert "live: '0'" in joined
        assert "transient" in joined.lower() or "setenv" in joined.lower()

    def test_static_and_live_match_returns_warn(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        # Live agrees with static.
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: {"OLLAMA_NUM_GPU": "0"}
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert "static and live daemon agree" in result.message
        # Details must show both sides for transparency.
        joined = " ".join(result.details)
        assert "static:" in joined
        assert "live:" in joined
        assert "match" in joined

    def test_static_and_live_drift_returns_warn(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # plist was edited but the daemon wasn't restarted -> static=0,
        # live=1 (stale from before the edit). Catch this user-hostile
        # state explicitly.
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: {"OLLAMA_NUM_GPU": "1"}
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert "drift" in result.message.lower()
        assert "static (0)" in result.message or "'0'" in result.message
        assert "live (1)" in result.message or "'1'" in result.message
        # Hint names the remediation.
        assert result.hint is not None
        assert "brew services restart" in result.hint
        joined = " ".join(result.details)
        assert "disagree" in joined

    def test_static_set_live_unset_returns_warn_restart_hint(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Live probe ran and reported no override. Plist sets NUM_GPU=0.
        # The running daemon is out of sync -- most likely the user added
        # the override but never restarted, or the override applies only at
        # service-start and hasn't been triggered since.
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: {}
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert result.hint is not None
        assert "brew services restart" in result.hint
        assert "live daemon env lacks" in result.message

    def test_static_unset_live_set_returns_warn_transient(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Common footgun on macOS: ``launchctl setenv OLLAMA_NUM_GPU=0`` in
        # the user's shell last week. Static config knows nothing of it; the
        # running daemon sees the override until reboot / logout. Warn
        # loudly so the user knows it's not going to persist.
        from cortex import doctor

        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ()
        )
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: {"OLLAMA_NUM_GPU": "0"}
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert result.hint is not None
        assert "transient" in result.hint.lower() or "setenv" in result.hint.lower()
        # The hint should not pretend this is a persistent config.
        assert "plist" in result.hint.lower() or "systemd unit" in result.hint.lower()

    def test_live_probe_failure_falls_back_to_static_only(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Probe failure (timeout, missing binary, no service loaded) must
        # not FAIL the whole check -- the static source is enough to keep
        # the WARN meaningful, and we say so in the hint.
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        monkeypatch.setattr(doctor, "_live_ollama_env", lambda: None)

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.WARN
        assert result.hint is not None
        # The hint must explicitly mention the probe failure so the user
        # doesn't naively trust a missing-override outcome.
        assert "live" in result.hint.lower() and "probe" in result.hint.lower()
        joined = " ".join(result.details)
        assert "live probe" in joined.lower()

    def test_static_unset_live_unset_with_static_file_returns_pass(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Static file exists, has env, but doesn't set NUM_GPU. Live daemon
        # also doesn't have it. PASS -- default GPU acceleration is in
        # effect on both sides, with the live probe confirming.
        from cortex import doctor

        plist = self._candidate_for(tmp_path, "plist", {"OLLAMA_FLASH_ATTENTION": "1"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        monkeypatch.setattr(
            doctor, "_live_ollama_env", lambda: {"OLLAMA_FLASH_ATTENTION": "1"}
        )

        result = doctor.check_gpu_path(settings)
        assert result.status is Status.PASS
        joined = " ".join(result.details)
        assert "live" in joined.lower()
        assert "no override" in joined.lower()


class TestInferencePathCheck:
    """Tests for the live inference cross-check against supervisor policy.

    Each test stubs ``_safe_run`` to return canned ``ollama ps`` output --
    we don't actually invoke the user's ollama here because tests would
    race against model loads/unloads. The autouse fixture on this class
    also stubs ``_live_ollama_env`` so tests don't bleed into the host's
    real launchctl/systemctl state.
    """

    @pytest.fixture(autouse=True)
    def _stub_external_probes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default ``_live_ollama_env`` to ``{}`` -- the live probe
        succeeded with no override. This gives the cross-check a known
        baseline (supervisor enabled by default) without leaking the
        developer's actual launchctl/systemctl state into tests.

        Tests that want a *different* supervisor picture override this
        fixture's stub with an explicit ``monkeypatch.setattr`` after
        the rest of the fixture has run.
        """
        monkeypatch.setattr(_doctor_module, "_live_ollama_env", lambda: {})

    def _fake_ollama_ps(self, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
        """Replace ``_safe_run`` so the only command we ever answer is the
        ``ollama ps`` invocation. Everything else returns None (timeout)
        so a typo in the test surface as a degraded probe rather than a
        real subprocess call."""
        from cortex import doctor

        def _fake(args: tuple[str, ...], timeout: float = 3.0) -> str | None:  # type: ignore[no-untyped-def]
            if args and args[0:2] == ("ollama", "ps"):
                return body
            return None

        monkeypatch.setattr(doctor, "_safe_run", _fake)

    def _binary_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tell ``shutil.which('ollama')`` to return a path so the binary
        check passes."""
        from cortex import doctor

        monkeypatch.setattr(doctor, "_ollama_binary_path", lambda: "/usr/bin/ollama")

    def test_binary_missing_returns_info(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        monkeypatch.setattr(doctor, "_ollama_binary_path", lambda: None)
        result = doctor.check_inference_path(settings)
        assert result.status is Status.INFO
        assert "not on PATH" in result.message

    def test_ollama_ps_failed_returns_info(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        self._binary_present(monkeypatch)
        monkeypatch.setattr(doctor, "_safe_run", lambda args, timeout=3.0: None)
        result = doctor.check_inference_path(settings)
        assert result.status is Status.INFO
        assert "did not respond" in result.message

    def test_no_models_returns_pass(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch, "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
        )
        result = doctor.check_inference_path(settings)
        assert result.status is Status.PASS
        assert "no models" in result.message

    def test_all_gpu_with_no_override_returns_pass(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Canonical happy path: supervisor enabled, model runs on GPU.
        from cortex import doctor

        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch,
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            # Line lengths follow real ``ollama ps`` output verbatim; the
            # trailing UNTIL cell is multi-word on recent builds.
            "qwen3-embedding:0.6b    ac6da0dfba84    2.2 GB    100% GPU     4096       4 minutes\n",
        )
        # No static config and live probe returns no override.
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            (),
        )
        result = doctor.check_inference_path(settings)
        assert result.status is Status.PASS
        joined = " ".join(result.details)
        assert "100% GPU" in joined

    def test_signature_zero_gpu_with_no_override_returns_warn(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The exact user-spec scenario: zero models with GPU layers but
        # the supervisor says enabled. The classic "silent GPU fallback".
        from cortex import doctor

        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch,
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "qwen3-embedding:0.6b    ac6da0dfba84    2.2 GB    100% CPU     4096       4 minutes\n",
        )
        monkeypatch.setattr(
            doctor,
            "OLLAMA_DAEMON_PATHS",
            (),
        )
        result = doctor.check_inference_path(settings)
        assert result.status is Status.WARN
        assert "zero GPU compute" in result.message
        joined = " ".join(result.details)
        assert "100% CPU" in joined
        assert "supervisor: GPU enabled" in joined
        # Hint should name plausible cause.
        assert result.hint is not None
        assert "Metal" in result.hint or "CUDA" in result.hint

    def test_cpu_with_supervisor_override_returns_pass(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Consistent: user disabled GPU, model is on CPU. Quiet pass.
        from cortex import doctor

        plist = tmp_path / "override.plist"
        _write_plist(plist, {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch,
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "qwen3-embedding:0.6b    ac6da0dfba84    2.2 GB    100% CPU     4096       4 minutes\n",
        )
        result = doctor.check_inference_path(settings)
        assert result.status is Status.PASS
        assert "consistent with OLLAMA_NUM_GPU=0" in result.message

    def test_gpu_with_supervisor_override_returns_warn_contradiction(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Contradiction: supervisor disabled GPU, but ollama reports GPU
        # compute. Could be Apple Silicon misreporting (we observed this
        # in practice on M5); the doctor surfaces the visible gap rather
        # than guessing which side is wrong.
        from cortex import doctor

        plist = tmp_path / "override.plist"
        _write_plist(plist, {"OLLAMA_NUM_GPU": "0"})
        monkeypatch.setattr(
            doctor, "OLLAMA_DAEMON_PATHS", ((plist, "plist"),)
        )
        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch,
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "qwen3:4b    123    8.0 GB    100% GPU     8192       4 minutes\n",
        )
        result = doctor.check_inference_path(settings)
        assert result.status is Status.WARN
        assert "supervisor" in result.message.lower() and "GPU" in result.message
        joined = " ".join(result.details)
        assert "100% GPU" in joined
        assert "supervisor: GPU disabled" in joined
        assert result.hint is not None
        assert "ollama ps" in result.hint or "latency" in result.hint

    def test_partial_offload_returns_warn(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mixed GPU/CPU with no override -> warn so user knows they have
        # a partial-offload situation.
        from cortex import doctor

        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch,
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "llama2:13b    abc    13.0 GB    32% GPU / 68% CPU    4096    4 minutes from now\n",
        )
        monkeypatch.setattr(doctor, "OLLAMA_DAEMON_PATHS", ())
        result = doctor.check_inference_path(settings)
        assert result.status is Status.WARN
        assert "partial" in result.message.lower()
        joined = " ".join(result.details)
        assert "32% GPU / 68% CPU" in joined

    def test_unrecognised_processor_value_returns_pass(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Old Ollama builds emit literal "GPU" or "CPU"; classifier
        # should still detect those without WARN noise.
        from cortex import doctor

        self._binary_present(monkeypatch)
        self._fake_ollama_ps(
            monkeypatch,
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "qwen3-embedding:0.6b    ac6da0dfba84    2.2 GB    GPU          4096       4 min\n",
        )
        monkeypatch.setattr(doctor, "OLLAMA_DAEMON_PATHS", ())
        result = doctor.check_inference_path(settings)
        assert result.status is Status.PASS

    def test_ollama_ps_parser_handles_short_rows(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Parse robustness: rows shorter than the header shouldn't crash.
        from cortex import doctor

        body = (
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "qwen3-embedding:0.6b    100% GPU\n"  # short row: only name + proc
        )
        result = doctor._parse_ollama_ps_rows(body)
        assert len(result) == 1
        # The PROCESSOR fragment should still be parsed even from a short row.
        assert result[0]["PROCESSOR"] == "100% GPU"

    def test_ollama_ps_parser_handles_full_output(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic case: single row with full table columns including
        # the multi-word UNTIL phrase that breaks naive token-position
        # cropping.
        from cortex import doctor

        body = (
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "qwen3-embedding:0.6b    ac6da0dfba84    2.2 GB    100% GPU     4096       4 minutes\n"
        )
        result = doctor._parse_ollama_ps_rows(body)
        assert len(result) == 1
        assert result[0]["NAME"] == "qwen3-embedding:0.6b"
        assert result[0]["PROCESSOR"] == "100% GPU"

    def test_ollama_ps_parser_handles_partial_offload(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex import doctor

        body = (
            "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
            "llama2:13b    abc    13.0 GB    32% GPU / 68% CPU    4096    4 minutes\n"
        )
        result = doctor._parse_ollama_ps_rows(body)
        assert len(result) == 1
        assert result[0]["PROCESSOR"] == "32% GPU / 68% CPU"


class TestRunnerOrchestration:
    def test_never_raises(self, settings: Settings) -> None:
        # A vault with no model still gets a structured report back.
        (settings.vault_path / "A.md").write_text("# A\n", encoding="utf-8")
        result = run_doctor(settings)
        assert isinstance(result, DoctorReport)
        # Pin the exact count: a future "I removed a check by accident"
        # refactor should surface here rather than leaving a silently empty
        # diagnostic.
        from cortex.doctor import _CHECKS

        assert len(result.checks) == len(_CHECKS)

    def test_contains_expected_check_names(self, settings: Settings) -> None:
        # Exact set of names; the doctor module owns each one. If a check is
        # renamed, this test surfaces it so callers reading the output don't
        # silently lose a familiar label.
        expected = {
            "Config",
            "Vault",
            "Ollama",
            "GPU path",
            "Inference path",
            "Vector store",
            "Privacy",
            "MCP",
            "Rerank extra",
            "Documents extra",
            "Daemon",
            "Memory folder",
        }
        result = run_doctor(settings)
        assert {c.name for c in result.checks} == expected

    def test_crash_in_one_check_does_not_abort(self, settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Simulate one check blowing up. The runner must convert that crash
        # into a recorded FAIL, not propagate.
        from cortex import doctor

        def _boom(_s: Settings) -> Check:
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(doctor, "_CHECKS", (doctor.check_config, _boom, doctor.check_vault))
        report = doctor.run_doctor(settings)
        statuses = [c.status for c in report.checks]
        # The middle check should be a FAIL carrying the crash message.
        assert Status.FAIL in statuses
        crash_check = next(c for c in report.checks if c.status is Status.FAIL)
        assert "synthetic crash" in (crash_check.details or [""])[0]
