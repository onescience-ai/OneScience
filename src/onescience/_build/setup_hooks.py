from setuptools.command.build_py import build_py

from onescience._build.af3 import AF3BuildError, build_if_needed, is_strict


class OneScienceBuildPy(build_py):
    def run(self):
        try:
            build_if_needed()
        except AF3BuildError as exc:
            if is_strict():
                raise
            self.announce(f"Skipping AlphaFold3 build: {exc}", level=3)
        super().run()
