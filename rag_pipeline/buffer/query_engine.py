# /podcast_rag_project/query_engine.py

import ollama
import logging

class QueryEngine:
    """
    Handles query processing, context formatting, and interaction with the Ollama LLM.
    This class orchestrates the final step of the RAG pipeline, bringing together the
    retrieved data and the generative model to produce an answer.
    """

    def __init__(self, vector_store, ollama_model):
        """
        Initializes the QueryEngine.

        Args:
            vector_store (VectorStore): An initialized VectorStore object for the selected episode.
            ollama_model (str): The name of the Ollama model to use for generation.
        """
        self.vector_store = vector_store
        self.ollama_model = ollama_model
        self.llm_client = ollama.Client()
        self.conversation_history = []

    def ask(self, question):
        """
        The main method to process a user's question.
        """
        logging.info(f"Processing question: {question}")
        
        search_results = self.vector_store.search(question, k=5)
        context_str = self._format_context(search_results)
        history_str = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in self.conversation_history])

        # --- Enhanced, Stricter Prompt ---
        prompt = f"""
You are an expert podcast analyst. Your sole mission is to answer questions based *exclusively* on the provided context from a specific podcast episode. You must act as if you have no knowledge outside of this context.

**Strict Instructions:**
1.  **Grounding:** Your entire answer MUST be derived directly from the "Relevant Podcast Context" provided below. Do not add any information, assumptions, or generic statements that are not explicitly supported by the text.
2.  **Specificity:** Be highly specific. Quote or paraphrase key phrases from the segments to support your answer.
3.  **No External Knowledge:** Do not use any general knowledge. If the context doesn't provide an answer, you MUST state that "The information is not available in the provided episode data."
4.  **Synthesis:** Your goal is to synthesize information from multiple retrieved segments and their block summaries to form a coherent, detailed answer.

--- Conversation History ---
{history_str}

{context_str}

--- Current Question ---
User: {question}

--- Your Answer (Grounded *only* in the context above) ---
AI:
"""
        logging.info("Sending strictly-grounded prompt to Ollama for generation.")
        try:
            response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
            answer = response['response']
            
            self.conversation_history.append({"user": question, "ai": answer})
            if len(self.conversation_history) > 5: # Keep history concise
                self.conversation_history.pop(0)

            return answer, search_results
        except Exception as e:
            logging.error(f"Failed to communicate with Ollama: {e}")
            return "Error: Could not connect to the Ollama language model. Is it running?", []

    def _format_context(self, search_results):
        """
        Formats the list of retrieved segments into a rich, human-readable string
        to be injected into the LLM prompt.
        """
        if not search_results:
            return "No relevant context was found in the podcast data."

        formatted_string = "--- Relevant Podcast Context ---\n"
        blocks = {}
        for item in search_results:
            block_id = item.get('block_id', 'N/A')
            if block_id not in blocks:
                blocks[block_id] = {
                    'summary': item.get('block_summary', 'N/A'),
                    'key_points': item.get('block_key_points', []),
                    'insights': item.get('block_insights', {}),
                    'segments': []
                }
            blocks[block_id]['segments'].append(item)

        for block_id, data in blocks.items():
            formatted_string += f"\n[Context Block ID: {block_id}]\n"
            formatted_string += f"Block Summary: \"{data['summary']}\"\n"
            
            insights = data['insights']
            if insights:
                formatted_string += f"Block Theme: {insights.get('theme', 'N/A')}\n"
                formatted_string += f"Block Significance: {insights.get('significance', 'N/A')}\n"

            formatted_string += "\n  --- Relevant Segments in this Block ---\n"
            for seg in data['segments']:
                emotion = seg.get('text_emotion', {}).get('emotion', 'N/A')
                formatted_string += f"  - Segment (Time: {seg.get('start_time'):.2f}s, Speaker: {seg.get('speaker')}, Emotion: {emotion})\n"
                formatted_string += f"    Text: \"{seg.get('text')}\"\n"
            formatted_string += "  ----------------------------------------\n"
        
        return formatted_string
        
    def generate_suggestions(self):
        """
        Generates smart follow-up question suggestions based on the conversation history.
        """
        if not self.conversation_history:
            return "Ask a question first to get suggestions."

        last_interaction = self.conversation_history[-1]
        prompt = f"""
Based on the last user question and AI answer, suggest 3 insightful and relevant follow-up questions.

Last Question: "{last_interaction['user']}"
Last Answer: "{last_interaction['ai']}"

Suggest 3 follow-up questions as a numbered list:
1. ...
2. ...
3. ...
"""
        try:
            response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
            return response['response']
        except Exception as e:
            logging.error(f"Failed to generate suggestions: {e}")
            return "Could not generate suggestions."
