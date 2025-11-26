import json
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import cast
from typing import TYPE_CHECKING
from uuid import UUID

from agents import Agent
from agents import RawResponsesStreamEvent
from agents import RunResultStreaming
from agents import ToolCallItem
from agents.tracing import trace

from onyx.agents.agent_sdk.message_types import AgentSDKMessage
from onyx.agents.agent_sdk.message_types import FunctionCallMessage
from onyx.agents.agent_sdk.message_types import FunctionCallOutputMessage
from onyx.agents.agent_sdk.message_types import InputTextContent
from onyx.agents.agent_sdk.message_types import SystemMessage
from onyx.agents.agent_sdk.message_types import UserMessage
from onyx.agents.agent_sdk.monkey_patches import (
    monkey_patch_convert_tool_choice_to_ignore_openai_hosted_web_search,
)
from onyx.agents.agent_sdk.sync_agent_stream_adapter import SyncAgentStream
from onyx.agents.agent_search.dr.enums import ResearchType
from onyx.agents.agent_search.dr.utils import convert_inference_sections_to_search_docs
from onyx.chat.chat_utils import llm_docs_from_fetched_documents_cache
from onyx.chat.chat_utils import saved_search_docs_from_llm_docs
from onyx.chat.memories import get_memories
from onyx.chat.models import DOCUMENT_CITATION_NUMBER_EMPTY_VALUE
from onyx.chat.models import PromptConfig
from onyx.chat.packet_sniffing import has_had_message_start
from onyx.chat.prompt_builder.answer_prompt_builder import (
    default_build_system_message_v2,
)
from onyx.chat.prune_and_merge import prune_and_merge_sections
from onyx.chat.stop_signal_checker import is_connected
from onyx.chat.stop_signal_checker import reset_cancel_status
from onyx.chat.stream_processing.citation_processing import CitationProcessor
from onyx.chat.stream_processing.utils import map_document_id_order_v2
from onyx.chat.turn.context_handler.citation import (
    assign_citation_numbers_recent_tool_calls,
)
from onyx.chat.turn.context_handler.reminder import maybe_append_reminder
from onyx.chat.turn.infra.chat_turn_event_stream import unified_event_stream
from onyx.chat.turn.models import AgentToolType
from onyx.chat.turn.models import ChatTurnContext
from onyx.chat.turn.models import ChatTurnDependencies
from onyx.chat.turn.models import FetchedDocumentCacheEntry
from onyx.chat.turn.prompts.custom_instruction import build_custom_instructions
from onyx.chat.turn.save_turn import extract_final_answer_from_packets
from onyx.chat.turn.save_turn import save_turn
from onyx.configs.constants import MessageType
from onyx.context.search.models import InferenceSection
from onyx.file_store.models import InMemoryChatFile
from onyx.llm.models import PreviousMessage
from onyx.secondary_llm_flows.query_expansion import history_based_query_rephrase
from onyx.server.query_and_chat.streaming_models import CitationDelta
from onyx.server.query_and_chat.streaming_models import CitationInfo
from onyx.server.query_and_chat.streaming_models import CitationStart
from onyx.server.query_and_chat.streaming_models import CustomToolDelta
from onyx.server.query_and_chat.streaming_models import CustomToolStart
from onyx.server.query_and_chat.streaming_models import MessageDelta
from onyx.server.query_and_chat.streaming_models import MessageStart
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import PacketObj
from onyx.server.query_and_chat.streaming_models import ReasoningDelta
from onyx.server.query_and_chat.streaming_models import ReasoningStart
from onyx.server.query_and_chat.streaming_models import SearchSkipped
from onyx.server.query_and_chat.streaming_models import SearchToolDelta
from onyx.server.query_and_chat.streaming_models import SearchToolStart
from onyx.server.query_and_chat.streaming_models import SectionEnd
from onyx.tools.adapter_v1_to_v2 import force_use_tool_to_function_tool_names
from onyx.tools.adapter_v1_to_v2 import tools_to_function_tools
from onyx.tools.force import filter_tools_for_force_tool_use
from onyx.tools.force import ForceUseTool
from onyx.tools.tool import Tool
from onyx.tools.tool_implementations.search.search_tool import (
    SEARCH_RESPONSE_SUMMARY_ID,
)
from onyx.tools.tool_implementations.search.search_tool import SearchResponseSummary
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations_v2.tool_result_models import (
    LlmInternalSearchResult,
)
from onyx.tools.tool_runner import check_which_tools_should_run_for_non_tool_calling_llm
from onyx.tools.tool_runner import ToolRunner
from onyx.utils.logger import setup_logger

logger = setup_logger()

if TYPE_CHECKING:
    from litellm import ResponseFunctionToolCall

MAX_ITERATIONS = 10


# TODO: We should be able to do this a bit more cleanly since we know the schema
# ahead of time. I'll make sure to do that for when we replace AgentSDKMessage.
def _extract_tokens_from_messages(messages: list[AgentSDKMessage]) -> int:
    from onyx.llm.utils import check_number_of_tokens

    total_input_text_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("output")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            total_input_text_parts.append(text)
            elif isinstance(content, str):
                total_input_text_parts.append(content)
    return check_number_of_tokens("\n".join(total_input_text_parts))


def _run_non_tool_calling_fast_pipeline(
    dependencies: ChatTurnDependencies,
    chat_history: list[AgentSDKMessage],
    current_user_message: UserMessage,
    ctx: ChatTurnContext,
    prompt_config: PromptConfig,
) -> list[AgentSDKMessage]:
    """
    Custom flow for models that don't support native function calling.
    Uses programmatic tool execution instead of LLM tool calling.

    This method is completely isolated from the main Agent SDK pipeline to prevent
    upstream changes from affecting this functionality.

    Returns:
        agent_turn_messages: List of function call and function output messages
    """
    logger.info(
        f"[FAST] Model {dependencies.llm.config.model_name} using non-tool-calling fast pipeline. "
        "Using programmatic tool execution."
    )

    agent_turn_messages: list[AgentSDKMessage] = []

    # Extract query from current user message
    # The user message may contain reminders/system instructions before the actual query.
    # We extract only the text after the "QUERY:" marker to get the actual user query.
    query = ""
    for content_item in current_user_message.get("content", []):
        if isinstance(content_item, dict) and content_item.get("type") == "input_text":
            full_text = content_item.get("text", "")
            # Extract only the text after "QUERY:" marker
            query_marker = "QUERY:\n"
            if query_marker in full_text:
                query = full_text.split(query_marker, 1)[1].strip()
            else:
                # Fallback to full text if no marker found
                query = full_text
            break

    # Convert chat history to PreviousMessage format
    history: list[PreviousMessage] = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue

        # Determine message type
        role = msg.get("role", "")
        if role == "user":
            msg_type = MessageType.USER
        elif role == "assistant":
            msg_type = MessageType.ASSISTANT
        elif role == "system":
            msg_type = MessageType.SYSTEM
        else:
            continue

        # Extract text content
        content = msg.get("content", [])
        text_parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
        elif isinstance(content, str):
            text_parts.append(content)

        if text_parts:
            history.append(
                PreviousMessage(
                    message="\n".join(text_parts),
                    token_count=0,  # Token count not critical here
                    message_type=msg_type,
                    files=[],
                    tool_call=None,
                    refined_answer_improvement=None,
                    research_answer_purpose=None,
                )
            )

    # Calculate existing input tokens BEFORE pruning
    # This fixes the issue where ctx.current_input_tokens was 0
    memories = get_memories(dependencies.user_or_none, dependencies.db_session)
    langchain_system_message = default_build_system_message_v2(
        dependencies.prompt_config,
        dependencies.llm.config,
        memories,
        [],  # No tools in system message for non-tool-calling models
        ctx.should_cite_documents,
    )
    new_system_prompt = SystemMessage(
        role="system",
        content=[
            InputTextContent(
                type="input_text", text=str(langchain_system_message.content)
            )
        ],
    )
    custom_instructions = build_custom_instructions(prompt_config)
    current_messages_for_token_calc = (
        [new_system_prompt]
        + chat_history
        + custom_instructions
        + [current_user_message]
    )
    ctx.current_input_tokens = _extract_tokens_from_messages(
        current_messages_for_token_calc
    )

    logger.info(
        f"[FAST] Calculated existing input tokens: {ctx.current_input_tokens} "
        "(before pruning documents)"
    )

    # Check which tools should run for non-tool-calling LLM
    tool_args_list = check_which_tools_should_run_for_non_tool_calling_llm(
        list(dependencies.tools), query, history, dependencies.llm
    )

    # Check if search was skipped and emit SearchSkipped packet for UI
    for tool, tool_args in zip(dependencies.tools, tool_args_list, strict=False):
        if isinstance(tool, SearchTool) and tool_args is None:
            logger.info("[FAST] Search was skipped - emitting SearchSkipped packet")
            dependencies.emitter.emit(
                Packet(
                    ind=ctx.current_run_step,
                    obj=SearchSkipped(),
                )
            )
            break

    # Force query rephrasing for first query
    # By default, history_based_query_rephrase skips first query when history is empty.
    # We force rephrasing ONLY for first queries to ensure VESPA receives optimized queries.
    # For subsequent queries with history, rephrasing already happened in get_args_for_non_tool_calling_llm.
    if not history:  # Only force rephrase if this is the first query (no history)
        logger.info(
            f"[FAST] First query detected - forcing rephrasing. Original: {query}"
        )
        for i, (tool, tool_args) in enumerate(
            zip(dependencies.tools, tool_args_list, strict=False)
        ):
            if (
                isinstance(tool, SearchTool)
                and tool_args is not None
                and ("query" in tool_args or "query_string" in tool_args)
            ):
                # Get custom prompt from persona (None means use default)
                custom_history_rephrase_prompt = tool.persona.history_rephrase_prompt
                # Force rephrase by setting skip_first_rephrase=False
                rephrase_kwargs: dict = {
                    "query": tool_args.get("query")
                    or tool_args.get("query_string")
                    or "",
                    "history": history,
                    "llm": dependencies.llm,
                    "skip_first_rephrase": False,
                }
                if custom_history_rephrase_prompt:
                    rephrase_kwargs["prompt_template"] = custom_history_rephrase_prompt
                rephrased_query = history_based_query_rephrase(**rephrase_kwargs)
                # Use the same field name that was originally returned
                field_name = "query" if "query" in tool_args else "query_string"
                tool_args_list[i][field_name] = rephrased_query
                logger.info(f"[FAST] Rephrased query for VESPA: {rephrased_query}")
    else:
        # Query already rephrased in get_args_for_non_tool_calling_llm
        for tool, tool_args in zip(dependencies.tools, tool_args_list, strict=False):
            if (
                isinstance(tool, SearchTool)
                and tool_args is not None
                and ("query" in tool_args or "query_string" in tool_args)
            ):
                logger.info(
                    f"[FAST] Query for VESPA (already rephrased): "
                    f"{tool_args.get('query') or tool_args.get('query_string')}"
                )

    # Execute tools that returned arguments
    for tool, tool_args in zip(dependencies.tools, tool_args_list):
        if tool_args is not None:
            logger.info(
                f"[FAST] Executing tool {tool.name} programmatically with args: {tool_args}"
            )
            is_search_tool = isinstance(tool, SearchTool)

            if is_search_tool:
                query_value = tool_args.get("query") or tool_args.get("query_string")
                dependencies.emitter.emit(
                    Packet(
                        ind=ctx.current_run_step,
                        obj=SearchToolStart(
                            type="internal_search_tool_start", is_internet_search=False
                        ),
                    )
                )
                dependencies.emitter.emit(
                    Packet(
                        ind=ctx.current_run_step,
                        obj=SearchToolDelta(
                            type="internal_search_tool_delta",
                            queries=[query_value or ""],
                            documents=[],
                        ),
                    )
                )
            else:
                # UI indicator for non-search tools
                dependencies.emitter.emit(
                    Packet(
                        ind=ctx.current_run_step,
                        obj=CustomToolStart(tool_name=tool.name),
                    )
                )

            # Create a tool runner and execute
            tool_runner = ToolRunner(tool, tool_args)

            # Collect tool responses to extract sections
            tool_responses = list(tool_runner.tool_responses())

            # Extract sections from search response summary
            retrieved_sections: list[InferenceSection] = []
            if is_search_tool:
                for tool_response in tool_responses:
                    if tool_response.id == SEARCH_RESPONSE_SUMMARY_ID:
                        search_response_summary = cast(
                            SearchResponseSummary, tool_response.response
                        )
                        retrieved_sections = search_response_summary.top_sections
                        break

                if retrieved_sections:
                    contextual_pruning_config = getattr(
                        tool, "contextual_pruning_config", None
                    )
                    if contextual_pruning_config is not None:
                        pruned_sections = prune_and_merge_sections(
                            sections=retrieved_sections,
                            section_relevance_list=None,
                            llm_config=dependencies.llm.config,
                            existing_input_tokens=ctx.current_input_tokens,
                            contextual_pruning_config=contextual_pruning_config,
                        )
                    else:
                        logger.info(
                            "[FAST] Skipping pruning for %s (no contextual_pruning_config)",
                            tool.name,
                        )
                        pruned_sections = retrieved_sections
                else:
                    pruned_sections = []

                logger.info(
                    f"[FAST] Pruned {len(retrieved_sections)} sections to {len(pruned_sections)} sections "
                    f"to fit context window (existing tokens: {ctx.current_input_tokens})"
                )

                if pruned_sections:
                    ctx.should_cite_documents = True
                    logger.info("[FAST] Enabled citation instructions for LLM")

                for section in pruned_sections:
                    unique_id = section.center_chunk.document_id
                    if unique_id not in ctx.fetched_documents_cache:
                        ctx.fetched_documents_cache[unique_id] = (
                            FetchedDocumentCacheEntry(
                                inference_section=section,
                                document_citation_number=DOCUMENT_CITATION_NUMBER_EMPTY_VALUE,
                            )
                        )

                search_results = [
                    LlmInternalSearchResult(
                        document_citation_number=DOCUMENT_CITATION_NUMBER_EMPTY_VALUE,
                        title=section.center_chunk.semantic_identifier,
                        excerpt=section.combined_content,
                        metadata=section.center_chunk.metadata,
                        unique_identifier_to_strip_away=section.center_chunk.document_id,
                    )
                    for section in pruned_sections
                ]
                output_payload = [
                    result.model_dump(mode="json") for result in search_results
                ]
            else:
                pruned_sections = []
                search_results = []
                output_payload = tool.build_tool_message_content(*tool_responses)

            # Generate a unique call ID
            call_id = str(uuid.uuid4())

            # Add function call message
            function_call_msg: FunctionCallMessage = {
                "type": "function_call",
                "call_id": call_id,
                "name": tool.name,
                "arguments": json.dumps(tool_args),
            }
            agent_turn_messages.append(function_call_msg)

            if isinstance(output_payload, str):
                output_str = output_payload
            else:
                output_str = json.dumps(output_payload)

            # Add function output message
            function_output_msg: FunctionCallOutputMessage = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_str,
            }
            agent_turn_messages.append(function_output_msg)

            if is_search_tool:
                # Emit search delta with documents to UI
                dependencies.emitter.emit(
                    Packet(
                        ind=ctx.current_run_step,
                        obj=SearchToolDelta(
                            type="internal_search_tool_delta",
                            queries=[tool_args.get("query", "")],
                            documents=convert_inference_sections_to_search_docs(
                                pruned_sections, is_internet=False
                            ),
                        ),
                    )
                )
            else:
                # UI indicator for non-search tools
                delta_payload = (
                    output_payload
                    if isinstance(output_payload, (dict, list, str, int, float, bool))
                    else str(output_payload)
                )
                dependencies.emitter.emit(
                    Packet(
                        ind=ctx.current_run_step,
                        obj=CustomToolDelta(
                            tool_name=tool.name,
                            response_type="text",
                            data=delta_payload,
                            file_ids=None,
                        ),
                    )
                )

            logger.info(
                f"[FAST] Tool {tool.name} execution complete with {len(pruned_sections)} documents"
            )

    return agent_turn_messages


# TODO -- this can be refactored out and played with in evals + normal demo
def _run_agent_loop(
    messages: list[AgentSDKMessage],
    dependencies: ChatTurnDependencies,
    chat_session_id: UUID,
    ctx: ChatTurnContext,
    prompt_config: PromptConfig,
    force_use_tool: ForceUseTool | None = None,
) -> None:
    monkey_patch_convert_tool_choice_to_ignore_openai_hosted_web_search()
    # This should have already been called, but call it again here for good measure.
    # TODO: Get to the root of why sometimes it seems litellm settings aren't configured.
    from onyx.llm.litellm_singleton.config import initialize_litellm

    initialize_litellm()
    chat_history = messages[1:-1]
    current_user_message = cast(UserMessage, messages[-1])
    agent_turn_messages: list[AgentSDKMessage] = []
    last_call_is_final = False
    iteration_count = 0
    empty_response_retried = 0

    # Check if model should use non-tool-calling fast pipeline
    # This is controlled by the use_non_tool_calling_fast model configuration setting
    use_non_tool_calling_fast = dependencies.use_non_tool_calling_fast

    if use_non_tool_calling_fast and dependencies.tools and iteration_count == 0:
        agent_turn_messages = _run_non_tool_calling_fast_pipeline(
            dependencies=dependencies,
            chat_history=chat_history,
            current_user_message=current_user_message,
            ctx=ctx,
            prompt_config=prompt_config,
        )

        # CRITICAL: Assign citation numbers BEFORE the agent loop
        # The LLM needs to see proper citation numbers [1], [2], [3] in the function output,
        # not the placeholder -1 values. If we assign after, the LLM has already generated
        # its response with [-1] citations.
        logger.info("[FAST] Assigning citation numbers before LLM sees the messages")

        citation_result = assign_citation_numbers_recent_tool_calls(
            agent_turn_messages, ctx
        )
        agent_turn_messages = list(citation_result.updated_messages)
        ctx.documents_processed_by_citation_context_handler += (
            citation_result.new_docs_cited
        )
        ctx.tool_calls_processed_by_citation_context_handler += (
            citation_result.num_tool_calls_cited
        )

        logger.info(
            f"[FAST] Pre-assigned citations: {citation_result.new_docs_cited} docs, "
            f"{citation_result.num_tool_calls_cited} tool calls"
        )

    # Normal Agent SDK pipeline
    while not last_call_is_final:
        # Disable tools for non-tool-calling models (tools already executed programmatically)
        if use_non_tool_calling_fast:
            available_tools: Sequence[Tool] = []
        else:
            available_tools = (
                dependencies.tools if iteration_count < MAX_ITERATIONS else []
            )
        if force_use_tool and force_use_tool.force_use:
            available_tools = filter_tools_for_force_tool_use(
                list(available_tools), force_use_tool
            )
        memories = get_memories(dependencies.user_or_none, dependencies.db_session)
        # TODO: The system is rather prompt-cache efficient except for rebuilding the system prompt.
        # The biggest offender is when we hit max iterations and then all the tool calls cannot
        # be cached anymore since the system message will be differ in that it will have no tools.
        langchain_system_message = default_build_system_message_v2(
            dependencies.prompt_config,
            dependencies.llm.config,
            memories,
            available_tools,
            ctx.should_cite_documents,
        )
        new_system_prompt = SystemMessage(
            role="system",
            content=[
                InputTextContent(
                    type="input_text", text=str(langchain_system_message.content)
                )
            ],
        )
        custom_instructions = build_custom_instructions(prompt_config)
        previous_messages = (
            [new_system_prompt]
            + chat_history
            + custom_instructions
            + [current_user_message]
        )
        current_messages = previous_messages + agent_turn_messages
        ctx.current_input_tokens = _extract_tokens_from_messages(current_messages)

        if not available_tools:
            tool_choice = None
        else:
            tool_choice = (
                force_use_tool_to_function_tool_names(force_use_tool, available_tools)
                if iteration_count == 0 and force_use_tool
                else None
            ) or "auto"
        model_settings = replace(dependencies.model_settings, tool_choice=tool_choice)

        agent = Agent(
            name="Assistant",
            model=dependencies.llm_model,
            tools=cast(list[AgentToolType], tools_to_function_tools(available_tools)),
            model_settings=model_settings,
            tool_use_behavior="stop_on_first_tool",
        )
        agent_stream: SyncAgentStream = SyncAgentStream(
            agent=agent,
            input=current_messages,
            context=ctx,
        )
        streamed, tool_call_events = _process_stream(
            agent_stream, chat_session_id, dependencies, ctx
        )

        all_messages_after_stream = streamed.to_input_list()
        new_messages_from_stream = [
            cast(AgentSDKMessage, msg)
            for msg in all_messages_after_stream[len(previous_messages) :]
        ]

        # Handle duplicate function call messages from agent stream for non-tool-calling pipeline
        # The agent stream received function call messages as input and returns them in output.
        # We only want the NEW messages (the LLM's response), not the duplicates.
        if use_non_tool_calling_fast and iteration_count == 0:
            logger.info(
                f"[FAST] Agent stream returned {len(new_messages_from_stream)} messages. "
                f"Types: {[msg.get('type') for msg in new_messages_from_stream]}"
            )

            # Filter out function_call and function_call_output messages (these are duplicates)
            # Keep only the LLM's response (message type)
            llm_response_messages = [
                msg
                for msg in new_messages_from_stream
                if msg.get("type") not in ["function_call", "function_call_output"]
            ]

            # Combine: tool messages (with citations) + LLM response
            agent_turn_messages = agent_turn_messages + llm_response_messages

            logger.info(
                f"[FAST] After filtering duplicates: {len(agent_turn_messages)} total messages. "
                f"Types: {[msg.get('type') for msg in agent_turn_messages]}"
            )
        else:
            agent_turn_messages = new_messages_from_stream

        # Apply context handlers in order:
        # 1. Remove all user messages in the middle (previous reminders)
        agent_turn_messages = [
            msg for msg in agent_turn_messages if msg.get("role") != "user"
        ]

        # 2. Add task prompt reminder
        last_iteration_included_web_search = any(
            tool_call.name == "web_search" for tool_call in tool_call_events
        )
        agent_turn_messages = maybe_append_reminder(
            agent_turn_messages,
            prompt_config,
            ctx.should_cite_documents,
            last_iteration_included_web_search,
        )

        # 3. Assign citation numbers to tool call outputs
        # Instead of doing this complex parsing from the tool call response,
        # I could have just used the ToolCallOutput event from the Agents SDK.
        # TODO: When agent framework is gone, I can just use our ToolCallOutput event.

        # Skip citation assignment for non-tool-calling models
        # We already assigned citations BEFORE the agent loop so the LLM could see them
        if use_non_tool_calling_fast:
            logger.info(
                f"[FAST] Skipping citation assignment (already done before LLM generation). "
                f"Messages: {len(agent_turn_messages)}, types: {[msg.get('type') for msg in agent_turn_messages]}"
            )
        else:
            # Normal tool-calling flow: assign citations after tool execution
            citation_result = assign_citation_numbers_recent_tool_calls(
                agent_turn_messages, ctx
            )
            agent_turn_messages = list(citation_result.updated_messages)
            ctx.documents_processed_by_citation_context_handler += (
                citation_result.new_docs_cited
            )
            ctx.tool_calls_processed_by_citation_context_handler += (
                citation_result.num_tool_calls_cited
            )

        # TODO: Make this configurable on OnyxAgent level
        stopping_tools = ["image_generation"]
        had_image_generation = any(
            tool.name == "image_generation" for tool in tool_call_events
        )
        if len(tool_call_events) == 0 or any(
            tool.name in stopping_tools for tool in tool_call_events
        ):
            # Allow up to 5 retries for empty responses (skip for image gen)
            if had_image_generation:
                last_call_is_final = True
                logger.info(
                    "[FAST] Image generation detected; skipping empty-response retry"
                )
            else:
                current_answer = extract_final_answer_from_packets(
                    dependencies.emitter.packet_history
                )

                if len(current_answer) > 0:
                    last_call_is_final = True  # Got content, exit normally
                elif empty_response_retried < 5:
                    logger.warning(
                        f"[FAST] Empty response on iteration {iteration_count}, "
                        f"retrying ({empty_response_retried + 1}/5)"
                    )
                    empty_response_retried += 1
                    # Don't set last_call_is_final, continue loop
                else:
                    logger.error(
                        f"[FAST] Empty response after {empty_response_retried} retries "
                        f"at iteration {iteration_count}; exiting"
                    )
                    last_call_is_final = True  # Exit and let ValueError happen
        iteration_count += 1


def _fast_chat_turn_core(
    messages: list[AgentSDKMessage],
    dependencies: ChatTurnDependencies,
    chat_session_id: UUID,
    message_id: int,
    research_type: ResearchType,
    prompt_config: PromptConfig,
    force_use_tool: ForceUseTool | None = None,
    # Dependency injectable argument for testing
    starter_context: ChatTurnContext | None = None,
    latest_query_files: list[InMemoryChatFile] | None = None,
) -> None:
    """Core fast chat turn logic that allows overriding global_iteration_responses for testing.

    Args:
        messages: List of chat messages
        dependencies: Chat turn dependencies
        chat_session_id: Chat session ID
        message_id: Message ID
        research_type: Research type
        global_iteration_responses: Optional list of iteration answers to inject for testing
        cited_documents: Optional list of cited documents to inject for testing
    """
    reset_cancel_status(
        chat_session_id,
        dependencies.redis_client,
    )

    ctx = starter_context or ChatTurnContext(
        run_dependencies=dependencies,
        chat_session_id=chat_session_id,
        message_id=message_id,
        chat_files=latest_query_files or [],
    )
    with trace("fast_chat_turn"):
        _run_agent_loop(
            messages=messages,
            dependencies=dependencies,
            chat_session_id=chat_session_id,
            ctx=ctx,
            prompt_config=prompt_config,
            force_use_tool=force_use_tool,
        )
    _emit_citations_for_final_answer(
        dependencies=dependencies,
        ctx=ctx,
    )
    final_answer = extract_final_answer_from_packets(
        dependencies.emitter.packet_history
    )
    # TODO: Make this error handling more robust and not so specific to the qwen ollama cloud case
    # where if it happens to any cloud questions, it hangs on read url
    has_image_generation = any(
        packet.obj.type == "image_generation_tool_delta"
        for packet in dependencies.emitter.packet_history
    )
    # Allow empty final answer if image generation tool was used (it produces images, not text)
    if len(final_answer) == 0 and not has_image_generation:
        raise ValueError(
            """Final answer is empty. Inference provider likely failed to provide
            content packets.
            """
        )
    save_turn(
        db_session=dependencies.db_session,
        message_id=message_id,
        chat_session_id=chat_session_id,
        research_type=research_type,
        model_name=dependencies.llm.config.model_name,
        model_provider=dependencies.llm.config.model_provider,
        iteration_instructions=ctx.iteration_instructions,
        global_iteration_responses=ctx.global_iteration_responses,
        final_answer=final_answer,
        fetched_documents_cache=ctx.fetched_documents_cache,
    )
    dependencies.emitter.emit(
        Packet(ind=ctx.current_run_step, obj=OverallStop(type="stop"))
    )


@unified_event_stream
def fast_chat_turn(
    messages: list[AgentSDKMessage],
    dependencies: ChatTurnDependencies,
    chat_session_id: UUID,
    message_id: int,
    research_type: ResearchType,
    prompt_config: PromptConfig,
    force_use_tool: ForceUseTool | None = None,
    latest_query_files: list[InMemoryChatFile] | None = None,
) -> None:
    """Main fast chat turn function that calls the core logic with default parameters."""
    _fast_chat_turn_core(
        messages,
        dependencies,
        chat_session_id,
        message_id,
        research_type,
        prompt_config,
        force_use_tool=force_use_tool,
        latest_query_files=latest_query_files,
    )


def _process_stream(
    agent_stream: SyncAgentStream,
    chat_session_id: UUID,
    dependencies: ChatTurnDependencies,
    ctx: ChatTurnContext,
) -> tuple[RunResultStreaming, list["ResponseFunctionToolCall"]]:
    from litellm import ResponseFunctionToolCall

    llm_docs = llm_docs_from_fetched_documents_cache(ctx.fetched_documents_cache)
    mapping = map_document_id_order_v2(llm_docs)
    if llm_docs:
        processor = CitationProcessor(
            context_docs=llm_docs,
            doc_id_to_rank_map=mapping,
            stop_stream=None,
        )
    else:
        processor = None
    tool_call_events: list[ResponseFunctionToolCall] = []
    for ev in agent_stream:
        connected = is_connected(
            chat_session_id,
            dependencies.redis_client,
        )
        if not connected:
            _emit_clean_up_packets(dependencies, ctx)
            agent_stream.cancel()
            break
        packets = _default_packet_translation(
            ev, ctx, processor, dependencies.emitter.packet_history
        )
        for packet in packets:
            dependencies.emitter.emit(packet)
        if isinstance(getattr(ev, "item", None), ToolCallItem):
            tool_call_events.append(cast(ResponseFunctionToolCall, ev.item.raw_item))
    if agent_stream.streamed is None:
        raise ValueError("agent_stream.streamed is None")
    return agent_stream.streamed, tool_call_events


# TODO: Maybe in general there's a cleaner way to handle cancellation in the middle of a tool call?
def _emit_clean_up_packets(
    dependencies: ChatTurnDependencies, ctx: ChatTurnContext
) -> None:
    if not (
        dependencies.emitter.packet_history
        and dependencies.emitter.packet_history[-1].obj.type == "message_delta"
    ):
        dependencies.emitter.emit(
            Packet(
                ind=ctx.current_run_step,
                obj=MessageStart(
                    type="message_start", content="Cancelled", final_documents=None
                ),
            )
        )
    dependencies.emitter.emit(
        Packet(ind=ctx.current_run_step, obj=SectionEnd(type="section_end"))
    )


def _emit_citations_for_final_answer(
    dependencies: ChatTurnDependencies,
    ctx: ChatTurnContext,
) -> None:
    index = ctx.current_run_step + 1
    if ctx.citations:
        dependencies.emitter.emit(Packet(ind=index, obj=CitationStart()))
        dependencies.emitter.emit(
            Packet(
                ind=index,
                obj=CitationDelta(citations=ctx.citations),
            )
        )
        dependencies.emitter.emit(Packet(ind=index, obj=SectionEnd(type="section_end")))
    ctx.current_run_step = index


def _default_packet_translation(
    ev: object,
    ctx: ChatTurnContext,
    processor: CitationProcessor | None,
    packet_history: list[Packet],
) -> list[Packet]:
    """Function is a bit messy atm, since there's a bug in OpenAI Agents SDK that
    causes Anthropic packets to be out of order.

    TODO (chris): clean this up once OpenAI Agents SDK is fixed.
    """

    # lazy loading to save memory
    from openai.types.responses import ResponseReasoningSummaryPartAddedEvent
    from openai.types.responses import ResponseReasoningSummaryPartDoneEvent
    from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent

    packets: list[Packet] = []
    obj: PacketObj | None = None
    if isinstance(ev, RawResponsesStreamEvent):
        output_index = getattr(ev.data, "output_index", None)

        # ------------------------------------------------------------
        # Reasoning packets
        # ------------------------------------------------------------
        if isinstance(ev.data, ResponseReasoningSummaryPartAddedEvent):
            packets.append(Packet(ind=ctx.current_run_step, obj=ReasoningStart()))
            ctx.current_output_index = output_index
        elif isinstance(ev.data, ResponseReasoningSummaryTextDeltaEvent):
            packets.append(
                Packet(
                    ind=ctx.current_run_step,
                    obj=ReasoningDelta(reasoning=ev.data.delta),
                )
            )
        elif isinstance(ev.data, ResponseReasoningSummaryPartDoneEvent):
            # only do anything if we haven't already "gone past" this step
            # (e.g. if we've already sent the MessageStart / MessageDelta packets, then we
            # shouldn't do anything)
            if ctx.current_output_index == output_index:
                packets.append(Packet(ind=ctx.current_run_step, obj=SectionEnd()))
                ctx.current_run_step += 1
                ctx.current_output_index = None

        # ------------------------------------------------------------
        # Message packets
        # ------------------------------------------------------------

        # TODO: add this back in. We'd like this to be a simple, dumb translation layer
        # but we can't do that right now since there are weird provider behavior w/ empty
        # `response.content_part.added` packets
        # elif ev.data.type == "response.content_part.added":
        #     retrieved_search_docs = saved_search_docs_from_llm_docs(
        #         ctx.ordered_fetched_documents
        #     )
        #     obj = MessageStart(content="", final_documents=retrieved_search_docs)
        elif ev.data.type == "response.output_text.delta" and len(ev.data.delta) > 0:
            if processor:
                final_answer_piece = ""
                for response_part in processor.process_token(ev.data.delta):
                    if isinstance(response_part, CitationInfo):
                        ctx.citations.append(response_part)
                    else:
                        final_answer_piece += response_part.answer_piece or ""
                obj = MessageDelta(content=final_answer_piece)
            else:
                obj = MessageDelta(content=ev.data.delta)

            needs_start = has_had_message_start(packet_history, ctx.current_run_step)
            if needs_start:
                ctx.current_run_step += 1
                llm_docs_for_message_start = llm_docs_from_fetched_documents_cache(
                    ctx.fetched_documents_cache
                )
                retrieved_search_docs = saved_search_docs_from_llm_docs(
                    llm_docs_for_message_start
                )
                packets.append(
                    Packet(
                        ind=ctx.current_run_step,
                        obj=MessageStart(
                            content="", final_documents=retrieved_search_docs
                        ),
                    )
                )

            packets.append(Packet(ind=ctx.current_run_step, obj=obj))
        elif ev.data.type == "response.content_part.done":
            packets.append(Packet(ind=ctx.current_run_step, obj=SectionEnd()))
            ctx.current_output_index = None

    return packets
