import streamlit as st
import langchain_helper


st.title("Restaurant AI Assistant")

cuisine = st.text_input(
    "Enter a cuisine"
)

if cuisine:

    response = langchain_helper.generate_restaurant_name_and_items(
        cuisine
    )

    st.header(response["restaurant_name"])

    menu_items = response["menu_items"].split(",")

    for item in menu_items:
        st.write("-", item)




st.divider()

st.header("AI Agent")

question = st.text_input(
    "Ask the Agent anything",
    key="agent_question"
)

if question:

    answer = langchain_helper.ask_agent(
        question
    )

    st.write(answer)