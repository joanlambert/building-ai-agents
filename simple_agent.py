import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ====================
# 1. Goals
# ====================

GOALS = """
Your goal is to convert qualified leads into sales meetings.

Priorities:
1. Understand what the prospect needs
2. Determine whether our product is relevant
3. Answer questions accurately
4. Follow up when appropriate
5. Try to schedule a sales meeting when there is genuine interest

Never invent product information.
Never send an email unless it helps advance a legitimate sales conversation.
"""

# ==============================
# 2. Additional knowledge
# ==============================

KNOWLEDGE = """
COMPANY:
Acme Analytics

PRODUCT:
Acme helps B2B SaaS companies understand customer churn.

PLANS:
Starter: $99/month
Growth: $499/month
Enterprise: custom pricing

IDEAL CUSTOMER:
B2B SaaS companies with 10-500 employees.

KEY FEATURES:
- churn prediction
- customer health scores
- CRM integration
- automated retention alerts

SALES PROCESS:
Interested prospects should normally be offered a 30-minute demo.
"""

# ======================
# 3. Memory
# ======================

MEMORY = {
    "leads": {
        "jane@example.com": {
            "name": "Jane Doe",
            "company": "The Example Corp",
            "notes": [
                "Downloaded our churn-prevention guide",
            ],
            "last_contact": None
        }
    },
    "events": []
}

def remember_lead(email, info):
    MEMORY["leads"][email] = info

def remember_event(event):
    MEMORY["events"].append({
        "timestamp": datetime.now().isoformat(),
        "event": event,
    })

# ======================
# 4. Tools
# ======================

def lookup_lead(email):
    return MEMORY["leads"].get(email, {"error": "Lead not found"})

def send_email(to, subject, body):
    print("\n======= EMAIL =========")
    print("TO: ", to)
    print("SUBJECT: ", subject)
    print()
    print(body)
    print("=========================\n")

    remember_event(f"Sent email to {to}, subject: {subject}")

    return {
        "status": "sent",
        "to": to,
        "subject": subject
    }

def add_lead_note(email, note):
    lead = MEMORY["leads"].get(email)

    if not lead:
        return {"error": "Lead not found"}

    lead["notes"].append(note)
    remember_event(f"Added CRM note for {email}: {note}")

    return {"status": "success"}

TOOLS = [
    {
        "type": "function",
        "name": "lookup_lead",
        "description": "Look up information about a sales lead",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"}
            },
            "required": ["email"],
            "additionalProperties": False,
        }
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email to a sales prospect",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        }
    },
    {
        "type": "function",
        "name": "add_lead_note",
        "description": "Add a note about a prospect to the CRM",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["email", "note"],
            "additionalProperties": False,
        }
    }
]

TOOL_FUNCTIONS = {
    "lookup_lead": lookup_lead,
    "send_email": send_email,
    "add_lead_note": add_lead_note,
}

# ==========================
# 5. Brain
# ==========================

def build_instructions():
    recent_memory = MEMORY["events"][-20:]
    return f"""
You are an autonomous B2B sales agent.

### GOALS

{GOALS}

### COMPANY KNOWLEDGE

{KNOWLEDGE}

### MEMORY

Recent events:
{json.dumps(recent_memory, indent=2)}

### BEHAVIOR

Think about what needs to happen next.
Use tools whenever an external action or info lookup is required.
Do not claim that an action happened unless the corresponding tool successfully completed it.
When the task has been completed explain the result briefly.
"""

def run_agent(task):
    conversation = [
        {
            "role": "user",
            "content": task
        }
    ]

    while True:
        input("\n[PAUSED] Press enter to continue...\n")

        response = client.responses.create(
            model="gpt-5.4",
            instructions=build_instructions(),
            tools=TOOLS,
            input=conversation,
        )

        conversation.extend(
            item.model_dump(exclude_none=True)
            for item in response.output
        )

        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            return response.output_text

        for call in tool_calls:
            function = TOOL_FUNCTIONS.get(call.name)

            if function is None:
                result = {"error": f"Unknown tool: {call.name}"}

            else:
                arguments = json.loads(call.arguments)

                print(f"Agent calling: {call.name}, {arguments}")

                result = function(**arguments)

            conversation.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })

if __name__ == "__main__":
    end_user_prompt = input("Enter prompt: ")
    result = run_agent(end_user_prompt)

    print("AGENT RESULT:")
    print(result)