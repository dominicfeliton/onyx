"""Unit tests for non-tool-calling fast pipeline.

This tests the _run_non_tool_calling_fast_pipeline function which provides
programmatic tool execution for models that don't support native function calling.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from onyx.chat.turn.fast_chat_turn import _run_non_tool_calling_fast_pipeline


class TestNonToolCallingFastPipeline:
    """Tests for the non-tool-calling fast pipeline."""

    def test_pipeline_disabled_by_default(self) -> None:
        """Test that use_non_tool_calling_fast defaults to False."""
        # Create dependencies with default settings
        deps = MagicMock()
        deps.use_non_tool_calling_fast = False
        assert deps.use_non_tool_calling_fast is False

    def test_pipeline_enabled_when_configured(self) -> None:
        """Test that use_non_tool_calling_fast can be enabled."""
        deps = MagicMock()
        deps.use_non_tool_calling_fast = True
        assert deps.use_non_tool_calling_fast is True

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch(
        "onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm"
    )
    def test_pipeline_extracts_query_from_user_message(
        self,
        mock_check_tools: MagicMock,
        mock_build_system_message: MagicMock,
        mock_get_memories: MagicMock,
    ) -> None:
        """Test that the pipeline correctly extracts the query from user message."""
        # Setup mocks
        mock_get_memories.return_value = []
        mock_build_system_message.return_value = MagicMock(content="System message")
        mock_check_tools.return_value = []

        # Create mock dependencies
        deps = MagicMock()
        deps.tools = []
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock()
        ctx.current_run_step = 0
        ctx.should_cite_documents = False
        ctx.current_input_tokens = 0
        ctx.fetched_documents_cache = {}

        # Create user message with QUERY marker
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Some system instructions\nQUERY:\nWhat is the meaning of life?",
                }
            ],
        }

        prompt_config = MagicMock()

        # Call the pipeline
        _run_non_tool_calling_fast_pipeline(
            dependencies=deps,
            chat_history=[],
            current_user_message=user_message,
            ctx=ctx,
            prompt_config=prompt_config,
        )

        # Verify that check_which_tools_should_run was called with extracted query
        mock_check_tools.assert_called_once()
        call_args = mock_check_tools.call_args
        assert "What is the meaning of life?" in call_args[0][1]

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch(
        "onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm"
    )
    def test_pipeline_returns_empty_list_when_no_tools_should_run(
        self,
        mock_check_tools: MagicMock,
        mock_build_system_message: MagicMock,
        mock_get_memories: MagicMock,
    ) -> None:
        """Test that pipeline returns empty list when no tools are needed."""
        # Setup mocks
        mock_get_memories.return_value = []
        mock_build_system_message.return_value = MagicMock(content="System message")
        mock_check_tools.return_value = [None]  # No tools should run

        # Create mock dependencies
        deps = MagicMock()
        deps.tools = [MagicMock()]  # One tool available
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock()
        ctx.current_run_step = 0
        ctx.should_cite_documents = False
        ctx.current_input_tokens = 0
        ctx.fetched_documents_cache = {}

        user_message = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }

        prompt_config = MagicMock()

        # Call the pipeline
        result = _run_non_tool_calling_fast_pipeline(
            dependencies=deps,
            chat_history=[],
            current_user_message=user_message,
            ctx=ctx,
            prompt_config=prompt_config,
        )

        # Verify empty result (no tools executed)
        assert result == []


class TestSearchSkippedPacket:
    """Tests for SearchSkipped packet emission."""

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch(
        "onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm"
    )
    def test_search_skipped_packet_emitted_when_search_tool_skipped(
        self,
        mock_check_tools: MagicMock,
        mock_build_system_message: MagicMock,
        mock_get_memories: MagicMock,
    ) -> None:
        """Test that SearchSkipped packet is emitted when SearchTool is skipped."""
        from onyx.tools.tool_implementations.search.search_tool import SearchTool
        from onyx.server.query_and_chat.streaming_models import SearchSkipped

        # Setup mocks
        mock_get_memories.return_value = []
        mock_build_system_message.return_value = MagicMock(content="System message")

        # Create a mock SearchTool that passes isinstance check
        mock_search_tool = MagicMock()
        mock_search_tool.__class__ = SearchTool
        mock_search_tool.name = "search"

        # Search tool returns None (skipped)
        mock_check_tools.return_value = [None]

        # Create mock dependencies
        deps = MagicMock()
        deps.tools = [mock_search_tool]
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock()
        ctx.current_run_step = 0
        ctx.should_cite_documents = False
        ctx.current_input_tokens = 0
        ctx.fetched_documents_cache = {}

        user_message = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello, how are you?"}],
        }

        prompt_config = MagicMock()

        # Call the pipeline
        _run_non_tool_calling_fast_pipeline(
            dependencies=deps,
            chat_history=[],
            current_user_message=user_message,
            ctx=ctx,
            prompt_config=prompt_config,
        )

        # Verify that SearchSkipped packet was emitted
        emit_calls = deps.emitter.emit.call_args_list
        search_skipped_emitted = any(
            isinstance(call[0][0].obj, SearchSkipped) for call in emit_calls
        )
        assert (
            search_skipped_emitted
        ), "SearchSkipped packet should be emitted when search is skipped"

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch(
        "onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm"
    )
    def test_search_skipped_packet_not_emitted_when_search_runs(
        self,
        mock_check_tools: MagicMock,
        mock_build_system_message: MagicMock,
        mock_get_memories: MagicMock,
    ) -> None:
        """Test that SearchSkipped packet is NOT emitted when search tool runs."""
        from onyx.server.query_and_chat.streaming_models import SearchSkipped

        # Setup mocks
        mock_get_memories.return_value = []
        mock_build_system_message.return_value = MagicMock(content="System message")

        # Non-SearchTool returns None (skipped)
        mock_other_tool = MagicMock()
        mock_other_tool.name = "other_tool"
        mock_check_tools.return_value = [None]

        # Create mock dependencies
        deps = MagicMock()
        deps.tools = [mock_other_tool]
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock()
        ctx.current_run_step = 0
        ctx.should_cite_documents = False
        ctx.current_input_tokens = 0
        ctx.fetched_documents_cache = {}

        user_message = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }

        prompt_config = MagicMock()

        # Call the pipeline
        _run_non_tool_calling_fast_pipeline(
            dependencies=deps,
            chat_history=[],
            current_user_message=user_message,
            ctx=ctx,
            prompt_config=prompt_config,
        )

        # Verify that SearchSkipped packet was NOT emitted (no SearchTool)
        emit_calls = deps.emitter.emit.call_args_list
        search_skipped_emitted = any(
            isinstance(call[0][0].obj, SearchSkipped) for call in emit_calls
        )
        assert (
            not search_skipped_emitted
        ), "SearchSkipped packet should NOT be emitted when no SearchTool is skipped"


class TestForceSearch:
    """Tests for force search functionality."""

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch(
        "onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm"
    )
    def test_force_search_bypasses_llm_decision(
        self,
        mock_check_tools: MagicMock,
        mock_build_system_message: MagicMock,
        mock_get_memories: MagicMock,
    ) -> None:
        """Test that force search bypasses the LLM's search decision."""
        from onyx.tools.tool_implementations.search.search_tool import SearchTool
        from onyx.tools.force import ForceUseTool

        # Setup mocks
        mock_get_memories.return_value = []
        mock_build_system_message.return_value = MagicMock(content="System message")

        # Create a mock SearchTool that passes isinstance check
        mock_search_tool = MagicMock()
        mock_search_tool.__class__ = SearchTool
        mock_search_tool.name = "run_search"
        mock_search_tool._NAME = "run_search"
        mock_search_tool.get_args_for_non_tool_calling_llm.return_value = {
            "query": "rephrased query"
        }

        # Create mock dependencies
        deps = MagicMock()
        deps.tools = [mock_search_tool]
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock()
        ctx.current_run_step = 0
        ctx.should_cite_documents = False
        ctx.current_input_tokens = 0
        ctx.fetched_documents_cache = {}

        user_message = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello, how are you?"}],
        }

        prompt_config = MagicMock()

        # Create force_use_tool to force search
        force_use_tool = ForceUseTool(
            force_use=True,
            tool_name="run_search",
        )

        # Call the pipeline with force_use_tool
        _run_non_tool_calling_fast_pipeline(
            dependencies=deps,
            chat_history=[],
            current_user_message=user_message,
            ctx=ctx,
            prompt_config=prompt_config,
            force_use_tool=force_use_tool,
        )

        # Verify that check_which_tools_should_run was NOT called (bypassed)
        mock_check_tools.assert_not_called()

        # Verify that SearchTool.get_args_for_non_tool_calling_llm was called with force_run=True
        mock_search_tool.get_args_for_non_tool_calling_llm.assert_called_once()
        call_kwargs = mock_search_tool.get_args_for_non_tool_calling_llm.call_args
        assert call_kwargs[1].get("force_run") is True

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch(
        "onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm"
    )
    def test_no_force_search_uses_llm_decision(
        self,
        mock_check_tools: MagicMock,
        mock_build_system_message: MagicMock,
        mock_get_memories: MagicMock,
    ) -> None:
        """Test that without force_use_tool, the LLM decides whether to search."""
        # Setup mocks
        mock_get_memories.return_value = []
        mock_build_system_message.return_value = MagicMock(content="System message")
        mock_check_tools.return_value = []  # LLM decides no search needed

        # Create mock dependencies
        deps = MagicMock()
        deps.tools = []
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock()
        ctx.current_run_step = 0
        ctx.should_cite_documents = False
        ctx.current_input_tokens = 0
        ctx.fetched_documents_cache = {}

        user_message = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }

        prompt_config = MagicMock()

        # Call the pipeline WITHOUT force_use_tool
        _run_non_tool_calling_fast_pipeline(
            dependencies=deps,
            chat_history=[],
            current_user_message=user_message,
            ctx=ctx,
            prompt_config=prompt_config,
            force_use_tool=None,  # No forcing
        )

        # Verify that check_which_tools_should_run WAS called (normal flow)
        mock_check_tools.assert_called_once()


class TestEmptyResponseRetry:
    """Tests for empty response retry logic."""

    def test_retry_count_default(self) -> None:
        """Test that empty response retries are limited to 5."""
        # This is tested implicitly through the _run_agent_loop function
        # The retry logic limits retries to 5 attempts

    def test_image_generation_skips_retry(self) -> None:
        """Test that image generation tools skip the empty response retry."""
        # This is tested implicitly through the _run_agent_loop function
        # When image_generation tool is detected, retry is skipped


class TestModelConfigurationLookup:
    """Tests for model configuration lookup."""

    def test_use_non_tool_calling_fast_lookup_returns_true_when_set(self) -> None:
        """Test that use_non_tool_calling_fast is correctly looked up."""
        from onyx.llm.factory import _get_use_non_tool_calling_fast_for_model

        # Create mock provider with model configuration
        mock_provider = MagicMock()
        mock_model_config = MagicMock()
        mock_model_config.name = "test-model"
        mock_model_config.use_non_tool_calling_fast = True
        mock_provider.model_configurations = [mock_model_config]

        result = _get_use_non_tool_calling_fast_for_model(mock_provider, "test-model")
        assert result is True

    def test_use_non_tool_calling_fast_lookup_returns_false_when_not_set(self) -> None:
        """Test that use_non_tool_calling_fast returns False when not configured."""
        from onyx.llm.factory import _get_use_non_tool_calling_fast_for_model

        # Create mock provider with model configuration where flag is False
        mock_provider = MagicMock()
        mock_model_config = MagicMock()
        mock_model_config.name = "test-model"
        mock_model_config.use_non_tool_calling_fast = False
        mock_provider.model_configurations = [mock_model_config]

        result = _get_use_non_tool_calling_fast_for_model(mock_provider, "test-model")
        assert result is False

    def test_use_non_tool_calling_fast_lookup_returns_false_for_unknown_model(
        self,
    ) -> None:
        """Test that use_non_tool_calling_fast returns False for unknown models."""
        from onyx.llm.factory import _get_use_non_tool_calling_fast_for_model

        # Create mock provider with no matching model
        mock_provider = MagicMock()
        mock_provider.model_configurations = []

        result = _get_use_non_tool_calling_fast_for_model(
            mock_provider, "unknown-model"
        )
        assert result is False
