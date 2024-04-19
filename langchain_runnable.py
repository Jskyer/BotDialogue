from langchain_core.runnables import chain
import asyncio

def reverse_word(word: str):
    return word[::-1]


@chain
async def reverse_words_double(word: str):
    return reverse_word(word) * 2


# await reverse_words_double.ainvoke("1234")
async def test():
    async for event in reverse_words_double.astream_events("1234", version="v1"):
        print(event)

asyncio.run(test())
