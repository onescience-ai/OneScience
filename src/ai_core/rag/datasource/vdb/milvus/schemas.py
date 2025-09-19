from ai_core.rag.datasource.vdb.constant import *

SCHEMA_V1 = {
    FIELDS: [
        (TEXT, "str", 5000),
        (TITLE, "str", 200),
        (PARA_SUMMARY, "str", 2000),
        (SEG_ID, "int"),
        (DOC_ID, "str", 40),
    ],
    VEC_FIELDS: [TEXT],
    TEXT_FIELDS: [TEXT],
}

collection_to_schema = {"task_pipeline_desc": SCHEMA_V1}

collection_to_infos = {
    "task_pipeline_desc": "生命科学、地球科学、材料科学等领域的科研任务执行流程说明",
}
