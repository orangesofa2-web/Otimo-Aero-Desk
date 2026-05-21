if st.session_state.active_engine is None:
    engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
    if engine_match:
        st.session_state.active_engine = engine_match.group(1).upper().replace(" ", "")
        # ... sets up engine ...
        st.rerun()
    else: 
        st.stop()  # <-- HERE IS THE CULPRIT