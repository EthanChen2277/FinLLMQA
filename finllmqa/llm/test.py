import streamlit as st

display_num = answer_num = 0
while answer_num <= 5:
    answer_num += 1
    if answer_num > display_num:
        with st.chat_message('a', avatar='user'):
            place_holder = st.empty()
        display_num = answer_num
    place_holder.markdown(display_num)
