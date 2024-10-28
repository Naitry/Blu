from abc import ABC, abstractmethod


class LLMConvo(ABC):
    """
    Abstract base class for managing conversations with Language Learning Models (LLMs)

    This class provides a structure for maintaining a conversation history,
    including system, user, and assistant messages, along with timestamps.
    """

    def __init__(self) -> None:
        """
        Initialize a new conversation with an empty message history.

        The messages list contains dictionaries with keys:
        - role: The speaker's role (system/user/assistant)
        - content: The message content
        - datetime: Timestamp of the message
        """
        self.messages: list[dict[str, str, str]] = []

    @abstractmethod
    def addSystemMessage(self, message: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            message (str): The system message content to be added
        """
        pass

    @abstractmethod
    def addAssistantMessage(self, message: str) -> None:
        """
        Add an assistant's response to the conversation.

        Args:
            message (str): The assistant's message content to be added
        """
        pass

    @abstractmethod
    def addUserMessage(self, message: str) -> None:
        """
        Add a user's message to the conversation.

        Args:
            message (str): The user's message content to be added
        """
        pass

    @abstractmethod
    def requestResponse(self,
                        addToConvo: bool = False,
                        maxTokens: int = 256) -> str:
        """
        Request a response from the LLM based on the current conversation.

        Args:
            addToConvo (bool, optional): Whether to add the response to conversation history.
                                       Defaults to False.
            maxTokens (int, optional): Maximum number of tokens in the response.
                                     Defaults to 256.

        Returns:
            str: The LLM's response
        """
        pass


def dictToConvo(convo_data: dict) -> LLMConvo:
    """
    Convert a dictionary representation of a conversation into an LLMConvo object.

    Args:
        convo_data (dict): Dictionary containing conversation data with a 'messages' key
                          Each message should have 'role' and 'content' keys

    Returns:
        LLMConvo: A conversation object populated with the messages from the dictionary

    Example:
        convo_data = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant response"}
            ]
        }
    """
    convo = LLMConvo()
    message: dict[str, str]
    for message in convo_data.get("messages", []):
        role: str = message.get("role")
        content: str = message.get("content")
        if role == "system":
            convo.addSystemMessage(content)
        elif role == "user":
            convo.addUserMessage(content)
        elif role == "assistant":
            convo.addAssistantMessage(content)
    return convo
