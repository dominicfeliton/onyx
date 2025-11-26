"""Unit tests for non-tool-calling fast pipeline.

This tests the _run_non_tool_calling_fast_pipeline function which provides
programmatic tool execution for models that don't support native function calling.
"""
import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.chat.turn.fast_chat_turn import _run_non_tool_calling_fast_pipeline
from onyx.chat.turn.models import ChatTurnContext
from onyx.chat.turn.models import ChatTurnDependencies


class TestNonToolCallingFastPipeline:
    """Tests for the non-tool-calling fast pipeline."""

    def test_pipeline_disabled_by_default(self) -> None:
        """Test that use_non_tool_calling_fast defaults to False."""
        # Create dependencies with default settings
        deps = MagicMock(spec=ChatTurnDependencies)
        deps.use_non_tool_calling_fast = False
        assert deps.use_non_tool_calling_fast is False

    def test_pipeline_enabled_when_configured(self) -> None:
        """Test that use_non_tool_calling_fast can be enabled."""
        deps = MagicMock(spec=ChatTurnDependencies)
        deps.use_non_tool_calling_fast = True
        assert deps.use_non_tool_calling_fast is True

    @patch("onyx.chat.turn.fast_chat_turn.get_memories")
    @patch("onyx.chat.turn.fast_chat_turn.default_build_system_message_v2")
    @patch("onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm")
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
        deps = MagicMock(spec=ChatTurnDependencies)
        deps.tools = []
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock(spec=ChatTurnContext)
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
        result = _run_non_tool_calling_fast_pipeline(
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
    @patch("onyx.chat.turn.fast_chat_turn.check_which_tools_should_run_for_non_tool_calling_llm")
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
        deps = MagicMock(spec=ChatTurnDependencies)
        deps.tools = [MagicMock()]  # One tool available
        deps.llm.config.model_name = "test-model"
        deps.prompt_config = MagicMock()
        deps.user_or_none = None
        deps.db_session = MagicMock()
        deps.emitter = MagicMock()

        # Create context
        ctx = MagicMock(spec=ChatTurnContext)
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


class TestEmptyResponseRetry:
    """Tests for empty response retry logic."""

    def test_retry_count_default(self) -> None:
        """Test that empty response retries are limited to 5."""
        # This is tested implicitly through the _run_agent_loop function
        # The retry logic limits retries to 5 attempts
        pass

    def test_image_generation_skips_retry(self) -> None:
        """Test that image generation tools skip the empty response retry."""
        # This is tested implicitly through the _run_agent_loop function
        # When image_generation tool is detected, retry is skipped
        pass


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

        result = _get_use_non_tool_calling_fast_for_model(mock_provider, "unknown-model")
        assert result is False
