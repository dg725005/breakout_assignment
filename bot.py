from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
dotenv.load_dotenv() # For OpenAI API Key


def load_sop_data(): # Reading the context text file into a variable
    sop_path = "Standard_operating_procedure_for_running_hair_tran.txt" # Used pdf to text converter
    with open(sop_path, "r", encoding="utf-8") as file:
        file_contents = file.read()
    return file_contents

def run_customer_support():
    # Loading sop text into memory
    sop_content = load_sop_data()
    
    # Strict system prompt rules
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a front-desk AI assistant for a aesthetics clinic.\n\n"
            "STRICT RULES:\n"
            "1. Answer questions using ONLY the provided SOP guidelines. Do not assume or guess.\n"
            "2. If the answer is NOT in the SOP, or if the customer makes a complaint, asks a medical question, "
            "or tries to negotiate, reply EXACTLY with: '[ESCALATE] I need to hand you over to an associate.'\n\n"
            f"--- START OF SOP GUIDELINES ---\n{sop_content}\n--- END OF SOP GUIDELINES ---"
        )),
        ("human", "{user_input}")
    ])
    
    # We will use an openAI model to chat
    # An infinite loop chat
    print("Welcome to the Hair Transplant Clinic. Type 'exit' to quit.\n")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    
    # Prompt_template will go to llm, that further goes to a tool that parse the llm's output before the 'final' reply
    chain = prompt_template | llm | StrOutputParser()
    
    
    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ["exit"]:
            print("Ending conversation.")
            break
            
        # response of chain by invoking it
        response = chain.invoke({"user_input": user_input})
        print(f"AI Agent: {response}\n")
        
        # checking if '[ESCALATE]' is in the response
        if "[ESCALATE]" in response:
            print("Conversation has been transferred to a human agent.")
            break

if __name__ == "__main__":
    run_customer_support()