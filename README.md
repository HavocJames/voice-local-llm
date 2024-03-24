# voice-local-llm
using voice interactive local llm ( user -> speech recognition -> text prompt-> local llm -> reply text -> speech synthesis -> user) using vosk, pyttsx3 , llama cpp, mistral 7b

my latest project where I'm leveraging LangChain to host the powerful Mistral 7b model locally, even without a GPU! This groundbreaking initiative allows users to interact seamlessly using Vosk speech recognition technology. Here's how it works:

User Input: The user speaks to the system using Vosk speech recognition.

Text Prompt: Vosk converts the speech into text, creating a text prompt.

LLM Chain Processing: The text prompt is fed into the Mistral 7b model via LangChain for processing.

Reply Text: The model generates a text-based reply based on the input prompt.

Speech Synthesis: Finally, the reply text is converted back into speech using Pyttsx3, which is then delivered back to the user.
