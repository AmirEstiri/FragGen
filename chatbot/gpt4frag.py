"""
Implement the interface of applying GPT-based model to fragranceX data
"""

import os
import argparse

from configure import api_keys, internal_prompt

from typing import List, Dict

from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage
from langchain.chat_models import ChatOpenAI


class GPT4Frag(object):
    def __init__(
        self, model_name: str, solver: str, api_key: str, verbose: bool = False
    ) -> None:
        """
        :param model_name: The name of LLM,
        """

        # Which LLM to use
        self._model_name = model_name  # type: str

        # The large language model instance
        self._llm = None  # type: ChatOpenAI
        self._apikey = api_key  # type: str

        # "You are an expert"
        self._internal_prompt = SystemMessage(content=internal_prompt)

        # The latest LLM response
        self._llm_response = ""  # type: str

        # Log of the whole conversation
        self._global_conversations = []

        # The working directory of the agent
        self._path = "."  # type: str
        self._std_log = "description.log"  # type: str

        # The most recent error message
        self._errmsg = ""  # type: str

        self.verbose = verbose

        return
    
    @staticmethod
    def _log(info):
        print(info)
        # pass

    def _user_says(self) -> None:
        """
        Print header for user input in the log
        :return:
        """

        self._global_conversations.append("\n------------------------")
        self._global_conversations.append("User says: ")
        self._global_conversations.append("------------------------\n")

        return

    def _chatbot_says(self) -> None:
        """
        Print header for LLM input in the log
        :return:
        """

        self._global_conversations.append("\n------------------------")
        self._global_conversations.append("%s says: " % self._model_name)
        self._global_conversations.append("------------------------\n")

        return
    
    def _connect_chatbot(self) -> None:
        """
        Connect to chat bot
        :return:
        """

        self._llm = ChatOpenAI(
            model_name=self._model_name, temperature=0.3, openai_api_key=self._apikey
        )
        return
    
    def _get_frag_data(self) -> None:
        """
        Get problem data from description
        :return: The dictionary with problem and in/out information
        """

        self._data = read_frag_data_from_file(
            os.path.join(self._problem_path, self._std_format)
        )
    
    def dump_conversation(self, path_to_conversation: str = None) -> None:
        if path_to_conversation is None:
            path_to_conversation = self._std_log

        print("Dumping conversation to %s" % path_to_conversation)

        with open(path_to_conversation, "w") as f:
            f.write("\n".join(self._global_conversations))


def read_args():
    """
    Read arguments from command line
    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name")
    parser.add_argument(
        "--doc_path",
        type=str,
        default="fragrancex/fragrances",
        help="Path to documents",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose mode")
    return parser.parse_args()


if __name__ == "__main__":
    # Read arguments from command line
    args = read_args()

    # GPT agent chatbot
    agent = GPT4Frag(
        model_name=args.model,
        api_key=api_keys[0],
        verbose=args.verbose
    )

    try:
        status = agent.solve_problem()
        print("Status: ", status)

    except Exception as e:
        raise e
    
    finally:
        agent.dump_conversation(os.path.join(agent._path, agent._std_log))
