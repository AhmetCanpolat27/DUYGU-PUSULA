from pathlib import Path

path = Path("app.py")
code = path.read_text(encoding="utf-8")

if "def use_example(sentence):" not in code:
    marker = 'st.markdown(\n"""\n<style>'
    insert = '''def use_example(sentence):
    st.session_state.input_text = sentence
    st.session_state.analysis_result = None
    st.session_state.analysis_time = None

st.markdown(
"""
<style>'''
    code = code.replace(marker, insert)

old = '''            if st.button(button_text, key=f"example_{i}", use_container_width=True, type="secondary"):
                st.session_state.input_text = sentence
                st.session_state.analysis_result = None
                st.session_state.analysis_time = None
                st.rerun()
'''

new = '''            st.button(
                button_text,
                key=f"example_{i}",
                use_container_width=True,
                type="secondary",
                on_click=use_example,
                args=(sentence,)
            )
'''

code = code.replace(old, new)

path.write_text(code, encoding="utf-8")
