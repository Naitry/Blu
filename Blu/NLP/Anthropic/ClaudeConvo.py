from typing import List, \
	Dict, \
	Any

import openai

from Blu.NLP.LLM.LLMConversation import LLMConvo


class ClaudeConvo(LLMConvo):
	def __init__(self,
				 client: openai.OpenAI,
				 model: str = "claude-3-opus-20240229") -> None:
		super().__init__()
		self.client = client
		self.model: str = model

	def addSystemMessage(self,
						 message: str) -> None:
		self.messages.append({
			"role"   : "system",
			"content": message,
			"DT"     : super().currentDateTime()
		})

	def addAssistantMessage(self,
							message: str) -> None:
		self.messages.append({
			"role"   : "assistant",
			"content": message,
			"DT"     : super().currentDateTime()
		})

	def addUserMessage(self,
					   message: str) -> None:
		self.messages.append({
			"role"   : "user",
			"content": message,
			"DT"     : super().currentDateTime()
		})

	def formattedMessages(self) -> List[Dict[str, str]]:
		return [{
			"role"   : msg["role"],
			"content": msg["content"]} for msg in self.messages]

	def requestResponse(self,
						addToConvo: bool = False,
						maxTokens: int = 256) -> str:
		response: Any = self.client.chat.completions.create(model=self.model,
															messages=self.formattedMessages(),
															temperature=1,
															max_tokens=maxTokens,
															top_p=1,
															frequency_penalty=0,
															presence_penalty=0)
		assistantMessage = response.choices[0].message.content
		if addToConvo:
			self.addAssistantMessage(assistantMessage)
		return assistantMessage
