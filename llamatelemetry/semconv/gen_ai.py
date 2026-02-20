"""
llamatelemetry.semconv.gen_ai - Official OpenTelemetry GenAI semantic convention constants.

Based on: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
These constants align with the official OTel GenAI attribute registry.
"""

# ---------------------------------------------------------------------------
# Core GenAI attributes
# ---------------------------------------------------------------------------
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# ---------------------------------------------------------------------------
# Agent attributes
# ---------------------------------------------------------------------------
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"

# ---------------------------------------------------------------------------
# Conversation / session
# ---------------------------------------------------------------------------
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

# ---------------------------------------------------------------------------
# Data source (RAG)
# ---------------------------------------------------------------------------
GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"

# ---------------------------------------------------------------------------
# Content attributes (WARNING: may contain PII - OFF by default)
# ---------------------------------------------------------------------------
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
GEN_AI_PROMPT_NAME = "gen_ai.prompt.name"

# ---------------------------------------------------------------------------
# Request attributes
# ---------------------------------------------------------------------------
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"

# ---------------------------------------------------------------------------
# Response attributes
# ---------------------------------------------------------------------------
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"

# ---------------------------------------------------------------------------
# Tool attributes
# ---------------------------------------------------------------------------
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"

# ---------------------------------------------------------------------------
# Well-known values for gen_ai.operation.name
# ---------------------------------------------------------------------------
OP_CHAT = "chat"
OP_CREATE_AGENT = "create_agent"
OP_EMBEDDINGS = "embeddings"
OP_EXECUTE_TOOL = "execute_tool"
OP_GENERATE_CONTENT = "generate_content"
OP_INVOKE_AGENT = "invoke_agent"
OP_TEXT_COMPLETION = "text_completion"

# ---------------------------------------------------------------------------
# Well-known values for gen_ai.output.type
# ---------------------------------------------------------------------------
OUTPUT_IMAGE = "image"
OUTPUT_JSON = "json"
OUTPUT_SPEECH = "speech"
OUTPUT_TEXT = "text"

# ---------------------------------------------------------------------------
# Well-known values for gen_ai.provider.name
# ---------------------------------------------------------------------------
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_AWS_BEDROCK = "aws.bedrock"
PROVIDER_AZURE_AI_INFERENCE = "azure.ai.inference"
PROVIDER_AZURE_OPENAI = "azure.ai.openai"
PROVIDER_COHERE = "cohere"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_GEMINI = "gcp.gemini"
PROVIDER_GCP_GEN_AI = "gcp.gen_ai"
PROVIDER_VERTEX_AI = "gcp.vertex_ai"
PROVIDER_GROQ = "groq"
PROVIDER_IBM_WATSONX = "ibm.watsonx.ai"
PROVIDER_MISTRAL = "mistral_ai"
PROVIDER_OPENAI = "openai"
PROVIDER_PERPLEXITY = "perplexity"
PROVIDER_XAI = "x_ai"
# Custom providers for llamatelemetry
PROVIDER_LLAMA_CPP = "llama_cpp"
PROVIDER_TRANSFORMERS = "transformers"
PROVIDER_UNSLOTH = "unsloth"

# ---------------------------------------------------------------------------
# Well-known values for gen_ai.token.type
# ---------------------------------------------------------------------------
TOKEN_INPUT = "input"
TOKEN_OUTPUT = "output"

# ---------------------------------------------------------------------------
# Well-known values for gen_ai.tool.type
# ---------------------------------------------------------------------------
TOOL_FUNCTION = "function"
TOOL_EXTENSION = "extension"
TOOL_DATASTORE = "datastore"
