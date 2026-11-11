import stim as S
_r = S.rows
def rows(): return _r()[:640]
S.rows = rows
import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path("extract2.py", run_name="__main__")
