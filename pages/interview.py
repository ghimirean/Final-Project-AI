"""The mock interview page inside the dashboard app.

    streamlit run dashboard.py      # choose a job -> "Go to the mock interview" opens this page

The page itself is interview_ui.py (which also runs on its own with
`streamlit run interview_ui.py`); this file only makes it a page of the
dashboard app, so st.switch_page can open it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import interview_ui  # noqa: E402

interview_ui.main()
