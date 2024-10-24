from RagBot import RagBot

API_KEY = "Replace with own OpenAI API_KEY"
prompt = "Replace with own prompt"
agent = RagBot(api_key=API_KEY)

Response = agent.invoke(prompt) 
print(Response)
