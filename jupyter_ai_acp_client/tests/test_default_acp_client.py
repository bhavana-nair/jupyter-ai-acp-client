"""Tests for content block building and session management in JaiAcpClient."""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CurrentModeUpdate,
    DeniedOutcome,
    FileEditToolCallContent,
    ResourceContentBlock,
    TextContentBlock,
    ToolCall,
    ToolCallLocation,
    Usage,
    UsageUpdate,
)

from jupyterlab_chat.models import FileAttachment, NotebookAttachment

from jupyter_ai_persona_manager.persona_events import PersonaSessionState

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_acp_client.default_acp_client import JaiAcpClient


SESSION_ID = "sess-1"


def _state() -> PersonaSessionState:
    """A real PersonaSessionState with no event logger, so its typed properties
    store values in memory without emitting. The report_*/get_* methods
    round-trip through the real state object."""
    return PersonaSessionState(
        event_logger=None,
        chat_id="test-room",
        persona_id="test-persona",
        log=logging.getLogger("test"),
    )


def _make_client_and_persona():
    """Create a minimal mock JaiAcpClient with a persona wired for testing."""
    client = object.__new__(JaiAcpClient)
    client._prompt_locks_by_session = {}
    client._cancel_requested = {}
    client._permission_manager = MagicMock()

    # Mock connection
    conn = AsyncMock()
    conn.prompt = AsyncMock(return_value=MagicMock())
    client.get_connection = AsyncMock(return_value=conn)

    # Mock persona
    persona = MagicMock()
    persona.log = MagicMock()
    persona.state = MagicMock()
    persona.chat = MagicMock()
    persona.chat.get_message.return_value = None

    # Mock tool call manager
    client._tool_call_manager = MagicMock()

    client._personas_by_session = {SESSION_ID: persona}

    return client, conn, persona


class TestPromptAndReplyContentBlocks:
    """Tests for how prompt_and_reply builds ACP content blocks."""

    async def test_text_only(self):
        """Without attachments, sends a single TextContentBlock."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContentBlock)
        assert blocks[0].text == "hello"

    async def test_file_attachment_produces_resource_block(self):
        """A file attachment produces a ResourceContentBlock with file:// URI."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check this",
            attachments=[FileAttachment(value="src/main.py", mimetype="text/x-python")],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 2
        assert isinstance(blocks[1], ResourceContentBlock)
        assert blocks[1].uri == Path("/home/user/notebooks/src/main.py").resolve().as_uri()
        assert blocks[1].name == "main.py"
        assert blocks[1].mime_type == "text/x-python"

    async def test_notebook_attachment_default_mime_type(self):
        """Notebook attachments get application/x-ipynb+json when mimetype is None."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[NotebookAttachment(value="analysis.ipynb")],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "application/x-ipynb+json"

    async def test_notebook_explicit_mimetype_preserved(self):
        """When notebook has explicit mimetype, it is preserved."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review",
            attachments=[NotebookAttachment(value="nb.ipynb", mimetype="custom/type")],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type == "custom/type"

    async def test_multiple_attachments_in_order(self):
        """Multiple attachments produce ResourceContentBlocks in order after text."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="review all",
            attachments=[
                FileAttachment(value="a.py"),
                NotebookAttachment(value="b.ipynb"),
            ],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 3
        assert blocks[0].text == "review all"
        assert blocks[1].name == "a.py"
        assert blocks[2].name == "b.ipynb"
        assert blocks[2].mime_type == "application/x-ipynb+json"

    async def test_none_attachments(self):
        """None attachments produces only the text block."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="hello",
            attachments=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1

    async def test_empty_list_attachments(self):
        """Empty attachment list produces only the text block."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="hello",
            attachments=[],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert len(blocks) == 1

    async def test_empty_value_fallback_name(self):
        """When attachment value is empty, name falls back to '<attachment>'."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="")],
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].name == "<attachment>"

    async def test_mimetype_none_for_file(self):
        """File attachment with no mimetype gets None mime_type."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="data.csv")],
            root_dir="/tmp",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].mime_type is None

    async def test_no_root_dir_uses_relative_path(self):
        """When root_dir is None, URI is the raw relative path."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="subdir/file.py")],
            root_dir=None,
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "subdir/file.py"

    async def test_file_uri_format(self):
        """file:// URI has correct RFC 8089 format with three slashes."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="test.py")],
            root_dir="/home/user",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri.startswith("file:///")

    async def test_path_traversal_blocked(self):
        """Attachment path escaping root_dir falls back to raw relative path."""
        client, conn, _ = _make_client_and_persona()

        await client.prompt_and_reply(
            session_id=SESSION_ID,
            prompt="check",
            attachments=[FileAttachment(value="../../../etc/passwd")],
            root_dir="/home/user/notebooks",
        )

        blocks = conn.prompt.call_args.kwargs["prompt"]
        assert blocks[1].uri == "../../../etc/passwd"


def _real_usage_persona():
    """
    A `BaseAcpPersona` created without `__init__` (no subprocess or session),
    carrying the real usage setters and properties so tests cover the actual
    store-then-read round trip. Collaborators the client touches are mocked.
    """

    class _ConcreteAcpPersona(BaseAcpPersona):
        @property
        def defaults(self):  # pragma: no cover - never called in these tests
            return None

    persona = _ConcreteAcpPersona.__new__(_ConcreteAcpPersona)
    persona._acp_context_usage = None
    persona._acp_session_usage = None
    persona.log = logging.getLogger("test")
    # A real state slot so `_sync_awareness_usage` -> `report_usage`
    # round-trips through the real typed properties.
    persona.state = _state()
    persona.chat = MagicMock()
    # `set_status()` (called incidentally by `prompt_and_reply`) builds
    # `as_user()`, which reads `self.defaults`; this persona has none, so mock it.
    persona.as_user = MagicMock()
    return persona


class TestUsageStorage:
    """A usage report received by the client ends up readable on the persona."""

    async def test_usage_update_is_stored_as_context_usage(self):
        client, _, _ = _make_client_and_persona()
        client._loading_sessions = {}
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        update = UsageUpdate(sessionUpdate="usage_update", used=41_000, size=200_000)

        await client.session_update(SESSION_ID, update)

        assert persona.acp_context_usage is update

    async def test_prompt_response_usage_is_stored_as_session_usage(self):
        client, conn, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        usage = Usage(inputTokens=900, outputTokens=340, totalTokens=1_240)
        conn.prompt = AsyncMock(return_value=MagicMock(usage=usage))

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        assert persona.acp_session_usage is usage

    async def test_prompt_response_without_usage_stores_nothing(self):
        client, conn, _ = _make_client_and_persona()
        persona = _real_usage_persona()
        client._personas_by_session[SESSION_ID] = persona
        conn.prompt = AsyncMock(return_value=MagicMock(usage=None))

        await client.prompt_and_reply(session_id=SESSION_ID, prompt="hello")

        assert persona.acp_session_usage is None


class TestExtNotification:
    """The generic client is agent-agnostic: every ext notification (including
    vendor `kiro.dev/*` methods, now handled only by KiroAcpClient) is unknown
    to it and rejected as JSON-RPC method-not-found."""

    async def test_ext_notification_raises_method_not_found(self):
        client, _, _ = _make_client_and_persona()

        for method in ("kiro.dev/metadata", "kiro.dev/commands/available", "other.vendor/thing"):
            with pytest.raises(RequestError) as exc_info:
                await client.ext_notification(method, {"sessionId": SESSION_ID})
            assert exc_info.value.code == -32601, method

    async def test_ext_method_raises_method_not_found(self):
        client, _, _ = _make_client_and_persona()

        with pytest.raises(RequestError) as exc_info:
            await client.ext_method("kiro.dev/metadata", {"sessionId": SESSION_ID})
        assert exc_info.value.code == -32601


class TestAwarenessPush:
    """The client pushes ACP updates onto the persona's awareness API too."""

    async def test_available_commands_update_advertises_over_awareness(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona.report_slash_commands = MagicMock()
        update = AvailableCommandsUpdate(
            sessionUpdate="available_commands_update",
            availableCommands=[
                AvailableCommand(name="compact", description="Compact context"),
                AvailableCommand(name="/clear", description="Clear"),
            ],
        )

        await client.session_update(SESSION_ID, update)

        commands = persona.report_slash_commands.call_args[0][0]
        # Names are leading-slash normalized.
        assert [(c.name, c.description) for c in commands] == [
            ("/compact", "Compact context"),
            ("/clear", "Clear"),
        ]

    async def test_current_mode_update_rebuilds_awareness_config(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_config = MagicMock()
        update = CurrentModeUpdate(sessionUpdate="current_mode_update", currentModeId="code")

        await client.session_update(SESSION_ID, update)

        persona.update_acp_current_mode.assert_called_once_with("code")
        persona._sync_awareness_config.assert_called_once()

    async def test_config_option_update_rebuilds_awareness_config(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_config = MagicMock()
        update = ConfigOptionUpdate(
            sessionUpdate="config_option_update", configOptions=[]
        )

        await client.session_update(SESSION_ID, update)

        persona.update_acp_config_options.assert_called_once()
        persona._sync_awareness_config.assert_called_once()

    async def test_usage_update_pushes_awareness_usage(self):
        client, _, persona = _make_client_and_persona()
        client._loading_sessions = {}
        persona._sync_awareness_usage = MagicMock()
        update = UsageUpdate(sessionUpdate="usage_update", used=1, size=2)

        await client.session_update(SESSION_ID, update)

        persona.update_acp_context_usage.assert_called_once_with(update)
        persona._sync_awareness_usage.assert_called_once()


class TestLoadSessionCleanup:
    """Tests for _loading_sessions cleanup on failure."""

    async def test_failed_load_session_removes_task_from_loading_sessions(self):
        """A failed load_session cleans up its task so retries can start fresh."""
        client = object.__new__(JaiAcpClient)
        client.event_loop = asyncio.get_running_loop()
        client._loading_sessions = {}

        persona = MagicMock()
        error = RequestError(-32002, "Resource not found")

        async def _failing_rpc(*args, **kwargs):
            raise error

        client._load_session_rpc = _failing_rpc

        with pytest.raises(RequestError):
            await client.load_session(persona, "stale-session-id")

        assert "stale-session-id" not in client._loading_sessions


def _tool_call(**kwargs) -> ToolCall:
    """Build a ToolCall with sensible defaults for permission tests."""
    kwargs.setdefault("tool_call_id", "tc-1")
    kwargs.setdefault("title", "Edit")
    return ToolCall(**kwargs)


class TestRequestPermissionNotebookGuard:
    """request_permission auto-denies any tool call that targets a Jupyter
    notebook (.ipynb), regardless of which agent sent it or which field
    exposes the path. Notebook edits must go through notebook MCP tools."""

    def _client_and_persona(self, root_dir="/work"):
        client, _, persona = _make_client_and_persona()
        # request_permission reads persona.parent.root_dir when extracting diffs.
        persona.parent.root_dir = root_dir
        # A real logger avoids MagicMock noise but isn't required.
        persona.log = logging.getLogger("test")
        return client, persona

    async def test_denies_when_locations_has_ipynb(self):
        client, persona = self._client_and_persona()
        client._personas_by_session = {SESSION_ID: persona}
        persona.send_message = MagicMock()
        tool_call = _tool_call(
            kind="edit",
            locations=[ToolCallLocation(path="/work/analysis.ipynb")],
        )

        resp = await client.request_permission(
            options=[], session_id=SESSION_ID, tool_call=tool_call
        )

        assert isinstance(resp.outcome, DeniedOutcome)
        # No permission prompt should have been created for the user.
        client._permission_manager.create_request.assert_not_called()
        # The user is told why it was blocked (better UX than a silent hang).
        persona.send_message.assert_called_once()
        assert "MCP tools" in persona.send_message.call_args[0][0]

    async def test_denies_when_edit_content_targets_ipynb(self):
        client, persona = self._client_and_persona()
        client._personas_by_session = {SESSION_ID: persona}
        tool_call = _tool_call(
            kind="edit",
            content=[
                FileEditToolCallContent(
                    type="diff",
                    path="/work/notebook.ipynb",
                    new_text='{"cells": []}',
                )
            ],
        )

        resp = await client.request_permission(
            options=[], session_id=SESSION_ID, tool_call=tool_call
        )

        assert isinstance(resp.outcome, DeniedOutcome)
        client._permission_manager.create_request.assert_not_called()

    async def test_denies_when_raw_input_diff_targets_ipynb(self):
        client, persona = self._client_and_persona()
        client._personas_by_session = {SESSION_ID: persona}
        # Agents like OpenCode send a unified diff string in raw_input.
        tool_call = _tool_call(
            kind="edit",
            raw_input={
                "filepath": "/work/model.ipynb",
                "diff": "@@ -1 +1 @@\n-old\n+new\n",
            },
        )

        resp = await client.request_permission(
            options=[], session_id=SESSION_ID, tool_call=tool_call
        )

        assert isinstance(resp.outcome, DeniedOutcome)
        client._permission_manager.create_request.assert_not_called()

    async def test_denies_when_raw_input_path_targets_ipynb(self):
        """Kiro's direct file tools (strReplace / fsWrite) expose the target as
        raw_input['path'] with no locations/content — the real e2e shape."""
        client, persona = self._client_and_persona()
        client._personas_by_session = {SESSION_ID: persona}
        tool_call = _tool_call(
            kind=None,
            locations=None,
            content=None,
            raw_input={
                "command": "strReplace",
                "path": "/work/for_loop_examples.ipynb",
                "oldStr": "a",
                "newStr": "b",
            },
        )

        resp = await client.request_permission(
            options=[], session_id=SESSION_ID, tool_call=tool_call
        )

        assert isinstance(resp.outcome, DeniedOutcome)
        client._permission_manager.create_request.assert_not_called()

    async def test_non_notebook_edit_is_not_denied_by_guard(self):
        """A .py edit must fall through to the normal permission flow, not be
        auto-denied by the notebook guard."""
        client, persona = self._client_and_persona()
        client._personas_by_session = {SESSION_ID: persona}
        # Make the permission flow resolve immediately as "denied/cancelled"
        # via a completed future, so we can tell the guard did NOT short-circuit
        # (create_request must be called).
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)  # user cancelled → distinguishable from guard deny
        client._permission_manager.create_request = MagicMock(return_value=fut)
        client._tool_call_manager = MagicMock()

        tool_call = _tool_call(
            kind="edit",
            locations=[ToolCallLocation(path="/work/script.py")],
        )

        await client.request_permission(
            options=[], session_id=SESSION_ID, tool_call=tool_call
        )

        # The guard let it through to the normal flow.
        client._permission_manager.create_request.assert_called_once()

    async def test_mcp_notebook_tool_is_not_denied(self):
        """MCP notebook tools (titled 'Running: @<server>/<tool>') are the safe
        path and must NOT be denied, even though they reference an .ipynb."""
        client, persona = self._client_and_persona()
        client._personas_by_session = {SESSION_ID: persona}
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        client._permission_manager.create_request = MagicMock(return_value=fut)
        client._tool_call_manager = MagicMock()

        # Real shape from the logs: MCP insert_cell, path in raw_input.file_path.
        tool_call = _tool_call(
            title="Running: @Jupyter MCP Server/insert_cell",
            kind=None,
            locations=None,
            content=None,
            raw_input={
                "file_path": "/work/for_loop_examples.ipynb",
                "cell_type": "code",
                "content": "print('hi')",
            },
        )

        await client.request_permission(
            options=[], session_id=SESSION_ID, tool_call=tool_call
        )

        # Not denied by the guard — normal permission flow was used.
        client._permission_manager.create_request.assert_called_once()
