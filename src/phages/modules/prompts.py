from llama_index import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

llm_system_prompt = """Answer in an direct and concise tone, I am in a hurry. Your audience is an expert, so be
highly specific. If there are ambiguous terms or acronyms, first define them.
"""

answer_prompt_tpl = """Write an answer {answer_length} for the question below based on the provided context. If
the context provides insufficient information, reply "I cannot answer". For each part of
your answer, indicate which sources most support it via valid citation markers at the end
of sentences, like (Example2012). Answer in an unbiased, comprehensive, and scholarly
tone. If the question is subjective, provide an opinionated answer in the concluding 1-2
sentences.

{context}

Query: {query}
Answer:

"""
answer_prompt = PromptTemplate(answer_prompt_tpl, prompt_type=PromptType.CUSTOM)

ask_llm_prompt_tpl = """We are collecting background information for the question/task below.
Provide a brief summary of information you know (about 50 words) that could help answer
the question - do not answer it directly and ignore formatting instructions. It is ok to
not answer, if there is nothing to contribute.
Query: {query}
"""
ask_llm_prompt = PromptTemplate(ask_llm_prompt_tpl, prompt_type=PromptType.CUSTOM)



summary_prompt_tpl = """
Summarize the text below to help answer a question. 
Do not directly answer the question, instead summarize to give evidence to help answer the question. 
Focus on specific details, including numbers, equations, or specific quotes. 
Reply 'Not applicable' if text is irrelevant. 
Use {summary_length}. At the end of your response, provide a score from 0 to 1 on a newline 
indicating relevance to question. Do not explain your score. 
\n\n
{text}\n\n
Excerpt from {citation}\n
Question: {question}\n
Relevant Information Summary:
"""
# TODO: add output parser , output_parser=
summary_prompt = PromptTemplate(summary_prompt_tpl, prompt_type=PromptType.CUSTOM)

qa_prompt_tpl = """Write an answer ({answer_length}) 
    for the question below based on the provided context. 
    If the context provides insufficient information and the question cannot be directly answered, 
    reply 'I cannot answer.'
    For each part of your answer, indicate which sources most support it 
    via valid citation markers at the end of sentences, like (Example2012). \n
    Context (with relevance scores):\n {context}\n
    Question: {question}\n
    Answer: """

qa_prompt = PromptTemplate(qa_prompt_tpl, prompt_type=PromptType.CUSTOM)

select_paper_prompt_tpl = """Select papers that may help answer the question below. 
    Papers are listed as $KEY: $PAPER_INFO. 
    Return a list of keys, separated by commas. 
    'Return None, if no papers are applicable. '
    Choose papers that are relevant, from reputable sources, and timely 
    (if the question requires timely information). \n\n
    Question: {question}\n\n
    Papers: {papers}\n\n
    Selected keys:"""

select_paper_prompt = PromptTemplate(select_paper_prompt_tpl, prompt_type=PromptType.CUSTOM)

citation_prompt_tpl = """Provide the citation for the following text in MLA Format. The year is {year}\n
    {text}\n\n
    Citation:"""

citation_prompt = PromptTemplate(citation_prompt_tpl, prompt_type=PromptType.CUSTOM)

default_system_prompt_tpl = """Answer in a direct and concise tone. 
    Your audience is an expert, so be highly specific. 
    If there are ambiguous terms or acronyms, first define them. """