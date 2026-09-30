from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

import os
from secret import groq_api_key

os.environ["GROQ_API_KEY"] = groq_api_key



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

parser = StrOutputParser()


def generate_restaurant_name_and_items(cuisine):

    

    name_prompt = ChatPromptTemplate.from_template(
        "I want to open a restaurant for {cuisine} food. "
        "Suggest a fancy name for this."
    )

    name_chain = name_prompt | llm | parser

    

    menu_prompt = ChatPromptTemplate.from_template(
        "Suggest some menu items for {restaurant_name}. "
        "Return it as a comma separated string."
    )

    menu_chain = menu_prompt | llm | parser

    

    restaurant_name = name_chain.invoke({
        "cuisine": cuisine
    })

    menu_items = menu_chain.invoke({
        "restaurant_name": restaurant_name
    })

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }


if __name__ == "__main__":
    response = generate_restaurant_name_and_items("Italian")

    print("Restaurant Name:")
    print(response["restaurant_name"])

    print("\nMenu Items:")
    print(response["menu_items"])



from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    groq_api_key=groq_api_key
)


wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=4000
    )
)

@tool
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    Example: 25 + 5 or 100 / 4
    """

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Invalid mathematical expression."




tools = [
    wikipedia,
    calculator
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant.

Use Wikipedia when you need factual information
that is not already known.

Use the calculator tool whenever mathematical
calculation is required.

Give the final answer in simple English.
"""
)


def ask_agent(question):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    return response["messages"][-1].content