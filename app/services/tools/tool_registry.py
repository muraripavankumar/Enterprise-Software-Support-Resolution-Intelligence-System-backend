from llama_index.core.tools import FunctionTool

from app.services.tools.sql_tool import SQL_TOOL_DESCRIPTION, execute_nl2sql
from app.services.tools.vector_tool import VECTOR_EVIDENCE_TOOL_DESCRIPTION, execute_vector_evidence

SQL_TOOL = FunctionTool.from_defaults(
    async_fn=execute_nl2sql,
    name="structured_support_data_query",
    description=SQL_TOOL_DESCRIPTION,
)

VECTOR_TOOL = FunctionTool.from_defaults(
    async_fn=execute_vector_evidence,
    name="documentation_policy_retrieval",
    description=VECTOR_EVIDENCE_TOOL_DESCRIPTION,
)


def get_tools() -> list[FunctionTool]:
    return [SQL_TOOL, VECTOR_TOOL]
