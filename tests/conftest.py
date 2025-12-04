import warnings

warnings.filterwarnings("ignore", message="PyPDF2 is deprecated", category=DeprecationWarning)

# Import PyPDF2 after configuring the warning filter so the deprecation notice is suppressed globally.
try:
	import PyPDF2  # noqa: F401
except Exception:
	pass
