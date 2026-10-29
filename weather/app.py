"""Streamlit UI for the LangGraph weather assistant."""

from __future__ import annotations

import streamlit as st

from weather.config import Settings
from weather.graph import WeatherAssistant, build_compiled_graph

st.set_page_config(page_title="Weather Assistant", page_icon="🌦️", layout="wide")

EXAMPLES = [
    "What is the weather in Tunis?",
    "Forecast for Sousse this weekend",
    "Compare Paris and London",
    "Should I go running in Sfax tomorrow?",
    "Weather in Springfield",
]


@st.cache_resource(show_spinner=False)
def get_assistant(provider: str, model: str) -> WeatherAssistant:
    settings = Settings.from_env()
    settings = Settings(
        provider=provider,
        model=model,
        base_url=settings.base_url,
        temperature=0.0,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        geocoding_url=settings.geocoding_url,
        forecast_url=settings.forecast_url,
        default_unit=settings.default_unit,
        forecast_days=settings.forecast_days,
        http_timeout=settings.http_timeout,
        max_retries=settings.max_retries,
        api_key=settings.api_key,
    )
    return WeatherAssistant(build_compiled_graph(settings, with_llm=True), settings)


def main() -> None:
    st.title("🌦️ Weather Assistant with LangGraph")
    st.caption("State machine · conditional routing · retry cycle · checkpointed threads")

    with st.sidebar:
        st.header("Model")
        provider = st.selectbox("Provider", ["ollama", "openai"])
        model = st.text_input("Model", "mistral")
        st.divider()
        st.subheader("Graph")
        st.code(
            "parse → resolve → fetch\n"
            "  ├─ fail → handle_error\n"
            "  ├─ ambiguous → clarify\n"
            "  └─ ok → analyse → respond",
            language="text",
        )
        if st.button("Reset conversation"):
            st.session_state.thread = "ui-session"
            st.session_state.log = []
            st.rerun()

    if "thread" not in st.session_state:
        st.session_state.thread = "ui-session"
    if "log" not in st.session_state:
        st.session_state.log = []

    choice = st.selectbox("Try an example", ["—"] + EXAMPLES)
    question = st.chat_input("Ask about the weather…")
    if choice != "—" and not question:
        question = choice

    if question:
        assistant = get_assistant(provider, model)
        with st.spinner("Running the graph…"):
            try:
                run = assistant.ask(question, st.session_state.thread)
            except Exception as exc:
                st.error(f"Graph failed: {exc}")
                return
        st.session_state.log.append({"question": question, "run": run})

    for entry in st.session_state.log:
        run = entry["run"]
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(run.answer)
            with st.expander("Graph trace"):
                for step in run.trail:
                    st.text(f"· {step}")
                for call in run.state.get("tool_calls", []):
                    st.text(f"↳ {call}")
                if run.state.get("error"):
                    st.warning(run.state["error"])
            if run.status == "clarify":
                st.info("Reply with one of the suggested cities to continue.")

    reports = [e["run"].state.get("report") for e in st.session_state.log]
    reports = [r for r in reports if r]
    if reports:
        st.divider()
        st.subheader("Last observation")
        last = reports[-1]
        columns = st.columns(4)
        columns[0].metric("Temperature", f"{last['temperature']}°")
        columns[1].metric("Feels like", f"{last['apparent_temperature']}°")
        columns[2].metric("Humidity", f"{last['humidity']}%")
        columns[3].metric("Wind", f"{last['wind_speed']} km/h")


if __name__ == "__main__":
    main()
