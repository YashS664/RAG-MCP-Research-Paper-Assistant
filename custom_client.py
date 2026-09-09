import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    command = "python", 
    args = ["mcp_server/server.py"]
)

async def main():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Discover what tools this server offers
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f" - {tool.name}: {tool.description}")
            print()

            # 2. Call list_papers
            result = await session.call_tool("list_papers", arguments={})
            print("list_papers() ->")
            print(result.content[0].text)
            print()

            # 3. Call search_papers
            search_query = "agent tool use" 
            result = await session.call_tool(
                "search_papers", arguments ={"query": search_query}
            )
            print(f"search_papers('{search_query}') ->")
            print(result.content[0].text)
            print()

            # 4. Call answer_question
            question = "How does ReAct combine reasoning and acting?"
            result = await session.call_tool(
                "answer_question", arguments={"query": question}
            )
            print(f"answer_question('{question}') ->")
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())