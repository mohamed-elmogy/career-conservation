from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

load_dotenv()

# ================= PUSH =================

def push(text):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            },
        )
    except:
        pass


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Record user contact details",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record unknown questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
            },
        },
    },
]


# ================= BOT CLASS =================

class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Mohamed Elmogy"

        # Load LinkedIn
        self.linkedin = ""
        try:
            reader = PdfReader("me/linkedin.pdf")
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except:
            pass

        # Load summary
        try:
            with open("me/summary.txt", "r", encoding="utf-8") as f:
                self.summary = f.read()
        except:
            self.summary = ""

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}

            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

    def system_prompt(self):
        return f"""
You are acting as Mohamed Elmogy.
You answer career and technical questions professionally.

Summary:
{self.summary}

LinkedIn:
{self.linkedin}
"""

    def chat(self, message, history):
        messages = [
            {"role": "system", "content": self.system_prompt()},
            *history,
            {"role": "user", "content": message},
        ]

        while True:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
            )

            if response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                results = self.handle_tool_call(msg.tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                return response.choices[0].message.content


# ================= RUN APP =================

me = Me()

demo = gr.ChatInterface(
    me.chat,
    title="Mohamed Elmogy AI Assistant",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

