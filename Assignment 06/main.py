import asyncio
from prompt_toolkit import PromptSession
from dotenv import load_dotenv
import os
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, AsyncOpenAI, RunContextWrapper, TResponseInputItem, InputGuardrailTripwireTriggered, input_guardrail, GuardrailFunctionOutput
from agents.run import RunConfig
from agents.model_settings import ModelSettings
from pydantic import BaseModel
from typing import List

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found, please check your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

model_settings = ModelSettings(
    temperature=0.7,
    parallel_tool_calls=True 
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=False,
    model_settings=model_settings
)


BOOK_DATABASE = {
    "Atomic Habits": {"author": "James Clear", "copies": 3},
    "Rich Dad Poor Dad": {"author": "Robert T. Kiyosaki", "copies": 2},
    "Think and Grow Rich": {"author": "Napoleon Hill", "copies": 0},
    "The 10X Rule": {"author": "Grant Cardone", "copies": 5}
}


class UserInfo(BaseModel):
    name: str
    member_id: int

class Book(BaseModel):
    name: str
    author: str
    copies: int

class LibraryOutput(BaseModel):
    is_library_output: bool
    reasoning: str

guardrail_agent = Agent(
    name='Library Query Filter',
    instructions='Determine whether the user’s question relates to the library (books, borrowing, availability, etc.). '
                 'Return is_library_output=True if yes, otherwise False, and briefly explain why.',
    output_type=LibraryOutput
)

@input_guardrail
async def library_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_library_output
    )

@function_tool
async def search_book_tool(book_name: str) -> dict:
    """Look up a book in the library collection by its title."""
    book_name_lower = book_name.lower()
    matched_book = next((name for name in BOOK_DATABASE if name.lower() == book_name_lower), None)
    book_exists = matched_book is not None
    return {
        "book_name": matched_book or book_name,
        "exists": book_exists,
        "message": f"'{matched_book or book_name}' {'is' if book_exists else 'is not'} found in the library."
    }

@function_tool
async def check_availability_tool(book_name: str) -> dict:
    """Check how many copies of a book are currently available."""
    book_name_lower = book_name.lower()
    matched_book = next((name for name in BOOK_DATABASE if name.lower() == book_name_lower), None)
    copies = BOOK_DATABASE.get(matched_book, {"copies": 0})["copies"] if matched_book else 0
    return {
        "book_name": matched_book or book_name,
        "copies_available": copies,
        "message": f"There {'are' if copies > 0 else 'are no'} {copies} copies of '{matched_book or book_name}' available."
    }

async def is_book_agent_enabled(self, context: RunContextWrapper[UserInfo]) -> bool:
    return isinstance(context, RunContextWrapper) and isinstance(context.context, UserInfo) and bool(context.context.member_id)

@function_tool(is_enabled=is_book_agent_enabled)
async def book_agent(wrapper: RunContextWrapper[UserInfo]) -> list[Book]:
    """Return a list of all books in the library with their authors and available copies."""
    return [
        Book(name=name, author=info["author"], copies=info["copies"])
        for name, info in BOOK_DATABASE.items()
    ]

def dynamic_instructions(context: RunContextWrapper[UserInfo], agent: Agent[UserInfo]) -> str:
    return (
        f"Hello {context.context.name}, I’m your library assistant. "
        "I can help with book searches, availability checks, and listing all available books. "
        "Use search_book_tool to confirm a book exists, check_availability_tool to see copies left, "
        "and book_agent to browse the catalog. Always treat book names case-insensitively. "
        "For questions about both existence and availability, apply both tools."
    )

main_agent = Agent(
    name="Librarian Agent",
    instructions=dynamic_instructions,
    model=model,
    tools=[book_agent, search_book_tool, check_availability_tool],
    input_guardrails=[library_guardrail]
)

async def get_user_input():
    session = PromptSession("Enter your question (type 'exit' to quit): ")
    return await session.prompt_async()

async def main():
    user_context = UserInfo(name='Tooba Yameen', member_id=432)
    
    while True:
        user_input = await get_user_input()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        try:
            result = await Runner.run(
                main_agent,
                user_input,
                run_config=config,
                context=user_context
            )
            print(result.final_output)
        except InputGuardrailTripwireTriggered:
            print("This doesn’t look like a library-related query. Please ask about books or library services.")
        except Exception as e:
            if "429" in str(e):
                print("⚠️ API quota exceeded. Please check your Gemini API plan and billing details.")
            elif "400" in str(e):
                print("⚠️ Invalid API request. Please review the model setup and try again.")
            else:
                print(f"⚠️ An error occurred: {e}")

Book.model_rebuild()
UserInfo.model_rebuild()
LibraryOutput.model_rebuild()

if __name__ == "__main__":
    asyncio.run(main())
