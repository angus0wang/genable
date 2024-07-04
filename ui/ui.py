import streamlit as st
from core.core import Deployer

deployer = Deployer()

def main():
    st.title("LLM Deploy")

    application = st.selectbox("Select Application", ["LLM Application", "Rag Application"])
    config = st.text_input("Config")

    if st.button("Deploy"):
        deployer.deploy(application, config)

    if st.button("Benchmark"):
        scenario = st.text_input("Scenario")
        deployer.benchmark(application, config, scenario)

if __name__ == "__main__":
    main()