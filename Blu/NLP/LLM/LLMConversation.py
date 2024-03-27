from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime


class LLMConvo(ABC):
	def __init__(self) -> None:
		# role, content, datetime
		self.messages: List[Dict[str, str, str]] = []

	def clearMessages(self) -> None:
		self.messages = []

	@abstractmethod
	def addSystemMessage(self,
						 message: str) -> None:
		pass

	@abstractmethod
	def addAssistantMessage(self,
							message: str) -> None:
		pass

	@abstractmethod
	def addUserMessage(self,
					   message: str) -> None:
		pass

	@abstractmethod
	def requestResponse(self,
						addToConvo: bool = False,
						maxTokens: int = 256) -> str:
		pass

	def currentDateTime(self) -> str:
		return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
