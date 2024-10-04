import os
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub

# Set your Hugging Face API token here
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


#st.write("Hugging Face API Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize the Hugging Face model
llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7})

comment_prompt = PromptTemplate(
    input_variables=["comment"],
    template="You are a helpful AI that provides relevant responses to Reddit comments. Reply to this comment: {comment}"
)

product_prompt = PromptTemplate(
    input_variables=["product_name"],
    template="Summarize the product: {product_name}"
)


comment_chain = LLMChain(llm=llm, prompt=comment_prompt)
product_chain = LLMChain(llm=llm, prompt=product_prompt)


st.title("Reddit AI Engagement Bot")
st.markdown("### Engage with Reddit comments and promote products")

user_comment = st.text_area("Enter a Reddit comment:")

if st.button("Generate Reply"):
    if user_comment:
        reply = comment_chain.run({"comment": user_comment})
        st.subheader("AI Reply:")
        st.write(reply)


        like_comment = st.radio("Do you like this comment?", ("Yes", "No"))

        if like_comment == "Yes":
            st.success("Comment liked!")

            repost_comment = st.radio("Do you want to repost this comment?", ("Yes", "No"))
            if repost_comment == "Yes":
                st.success("Comment reposted!")


product_name = st.text_input("Enter a product name to promote:")

if st.button("Product promotion Post"):
    if product_name:
        summary = product_chain.run({"product_name": product_name})
        st.subheader("Product Summary:")
        st.write(summary)


st.markdown("---")
st.markdown("### Powered by Langchain and Hugging Face")
