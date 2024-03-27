# for forward type annotation (recursive data structure with annotation)
from __future__ import annotations

# Typing
from abc import ABC

# OpenAI
import openai


class InformationFragment(ABC):
	def __init__(self,
				 name: str = None,
				 body: str = "",
				 embedding: list[float] = None,
				 parent: InformationFragment = None,
				 details: list[InformationFragment] = None):
		self.name: str = name
		self.body: str = body
		self.embedding: list[float] = embedding if embedding is not None else []
		self.parent: InformationFragment = parent
		self.details: list[InformationFragment] = details

	def addDetail(self,
				  detail: InformationFragment) -> None:
		if not self.details:
			self.details = []
		detail.parent = self
		self.details.append(detail)

	def embed(self,
			  client: openai.OpenAI) -> None:
		self.embedding = client.embeddings.create(input=self.body,
												  model="text-embedding-ada-002").data[0].embedding
