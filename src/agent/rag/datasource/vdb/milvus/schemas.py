from agent.rag.datasource.vdb.constant import *

SCHEMA1 = {
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

COLLECTION_TO_SCHEMA = {
    "science_task_instruction": SCHEMA1,
}

COLLECTION_TO_INFOS = {
    "science_task_instruction": "生命科学、地球科学、材料科学等领域的科研任务执行流程说明",
}
