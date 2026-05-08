"""Gradio chat interface for Vega."""

import gradio as gr
from agent import chat


def respond(message: str, history: list) -> str:
    """Adapter between Gradio's ChatInterface signature and agent.chat()."""
    return chat(history, message)


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Vega",
    description="Markets desk companion. Options, sizing, regime, journal queries.",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
