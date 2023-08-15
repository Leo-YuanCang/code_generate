import streamlit as st
import openai
import pickle
import os
import time
import json
from streamlit import session_state

os.environ['http_proxy'] = 'http://127.0.0.1:7966'
os.environ["https_proxy"] = "http://127.0.0.1:7966"
# streamlit run dataset.py --server.port 2323
st.set_page_config(
    page_title='QA Dataset Generator',
    layout="wide",
    page_icon='😅',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


# Load the question library
def load_questions(file_path):
    if not os.path.exists(file_path):
        st.error(f"File {file_path} does not exist")
        print(f"File {file_path} does not exist")
        return []
    else:
        with open(file_path, "r", encoding='utf-8') as file:
            questions = file.readlines()
        return list(set([q.strip() for q in questions if q != '' and q != '\n']))  # Remove duplicates


# Save the question library
def save_questions(file_path, questions):
    with open(file_path, "w", encoding='utf-8') as file:
        for question in questions:
            file.write(question + "\n")


# Generate an answer using the GPT3.5 API
def generate_answer(prompt):
    answer = '114514'
    return answer


def save_answers(temp_answers, just_read=False):
    if just_read:
        if os.path.exists("data.pkl"):
            with open("data.pkl", "rb") as file:
                answers = pickle.load(file)
        else:
            answers = {}
        session_state.all_answers = answers
        return True
    else:
        if os.path.exists("lock"):
            return False
        with open("lock", "w") as lock_file:
            lock_file.write("")
        if os.path.exists("data.pkl"):
            with open("data.pkl", "rb") as file:
                answers = pickle.load(file)
        else:
            answers = {}
        answers.update(temp_answers)  # Overwrite existing answers

        with open("data.pkl", "wb") as file:
            pickle.dump(answers, file)
        if os.path.exists("lock"):
            os.remove("lock")
        session_state.all_answers = answers
        return True


def save_answers_as_json(answers, file_path):
    data = []
    if 1:
        for question, answer in answers.items():  # Modify the output format as needed
            item = {
                "instruction": question,
                "input": "",
                "output": answer
            }
            data.append(item)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    else:
        with open(file_path, "w", encoding="utf-8") as file:
            for question, answer in answers.items():
                item = {"prompt": "Question: " + question + "\n", "completion": answer}
                file.write(json.dumps(item, ensure_ascii=False) + "\n")


def reset_text_area():
    if session_state.text_area_tittle == "Answer: (Leave empty to not save this answer)":
        session_state.text_area_tittle = "Answer: (Leave empty to not save this answer) "
    elif session_state.text_area_tittle == "Answer: (Leave empty to not save this answer) ":
        session_state.text_area_tittle = "Answer: (Leave empty to not save this answer)"


def main():
    st.title("QA Dataset Generator")
    openai.api_key = st.sidebar.text_input("API Key", value='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',
                                           type="password")
    PROMPT = st.sidebar.text_input("Prompt", value="Provide the answer for the following question:")
    if 'temp_answers' not in session_state:
        session_state.temp_answers = {}
    if 'all_answers' not in session_state:
        save_answers(session_state.temp_answers, just_read=True)
        session_state.question_txt = "qustest.txt"
        session_state.answers_json = "answers.json"
        session_state.generated_answer = ""
        session_state.text_area_tittle = "Answer: (Leave empty to not save this answer)"
        session_state.selected_id = 0
    session_state.question_txt = st.sidebar.text_input("Path to the file containing each question",
                                                       value=session_state.question_txt)
    session_state.answers_json = st.sidebar.text_input("Path to save the answers as JSON",
                                                       value=session_state.answers_json)
    if 'questions' not in session_state:
        session_state.questions = load_questions(session_state.question_txt)
    selected_questions = {}
    for q in range(len(session_state.questions)):
        selected_questions[session_state.questions[q]] = q
    selectbox_empty = st.empty()
    selected_question = selectbox_empty.selectbox("Select a question:", session_state.questions,
                                                  index=session_state.selected_id,key=f"selectbox_{session_state.selected_id}")
    if selected_question:
        session_state.selected_id = selected_questions[selected_question]
        selected_question = selectbox_empty.selectbox("Select a question:", session_state.questions,
                                                      index=session_state.selected_id)
        prompt = PROMPT + selected_question
        st.sidebar.write({'Preview': prompt})
        user_answer_empty = st.empty()
        user_answer = user_answer_empty.text_area(session_state.text_area_tittle, session_state.generated_answer,
                                                  height=200)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("Save All Answers (Automatically saves if more than 10)"):
                session_state.selected_id = 0
                for question in session_state.temp_answers:
                    session_state.questions.remove(question)
                save_questions(session_state.question_txt, session_state.questions)
                if save_answers(session_state.temp_answers):
                    st.success("All answers have been saved.")
                    session_state.temp_answers = {}
                else:
                    st.error(
                        "Save failed. Please try again later. Frequent occurrence of this problem may be due to deadlocks. Please delete the data.pkl file and try again.")
                    time.sleep(5)
                st.experimental_rerun()
        with col2:
            if st.button("Generate Answer with GPT"):
                session_state.generated_answer = ''
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0,
                        top_p=1.0,
                        stream=True
                    )
                    event_count = 0
                    for event in response:
                        delta = event["choices"][0]["delta"]
                        if "content" not in delta:
                            continue
                        event_text = event["choices"][0]["delta"]["content"]
                        session_state.generated_answer += event_text
                        if event_count > 10:
                            event_count = 0
                            reset_text_area()
                            user_answer = user_answer_empty.text_area(session_state.text_area_tittle,
                                                                      session_state.generated_answer, height=200)
                        event_count += 1
                    reset_text_area()
                    user_answer = user_answer_empty.text_area(session_state.text_area_tittle,
                                                              session_state.generated_answer, height=200)
                except Exception as e:
                    st.error("Generation failed. Please try again later.")
        with col3:
            if st.button("Confirm Answer (Automatically goes to next)"):
                session_state.generated_answer = ''
                if user_answer != '':
                    session_state.temp_answers[selected_question] = user_answer
                elif selected_question in session_state.temp_answers:
                    del session_state.temp_answers[selected_question]
                reset_text_area()
                user_answer = user_answer_empty.text_area(session_state.text_area_tittle, height=200)
                session_state.selected_id += 1
                if session_state.selected_id >= len(session_state.questions):
                    session_state.selected_id = 0
                st.experimental_rerun()
        with col4:
            if st.button("Previous Question"):
                session_state.generated_answer = ''
                session_state.selected_id -= 1
                if session_state.selected_id < 0:
                    session_state.selected_id = len(session_state.questions) - 1
                reset_text_area()
                user_answer = user_answer_empty.text_area(session_state.text_area_tittle, height=200)
                st.experimental_rerun()
        with col5:
            if st.button("Next Question"):
                session_state.generated_answer = ''
                session_state.selected_id += 1
                if session_state.selected_id >= len(session_state.questions):
                    session_state.selected_id = 0
                reset_text_area()
                user_answer = user_answer_empty.text_area(session_state.text_area_tittle, height=200)
                st.experimental_rerun()
    if st.sidebar.button("Read"):
        session_state.selected_id = 0
        save_answers(session_state.temp_answers, just_read=True)
        session_state.questions = load_questions(session_state.question_txt)
        st.experimental_rerun()
    if st.sidebar.button("Export Saved Answers as JSON"):
        save_answers_as_json(session_state.all_answers, session_state.answers_json)
    st.json({"Unsaved Answers": session_state.temp_answers, "Saved Answers": session_state.all_answers})


if __name__ == "__main__":
    main()
